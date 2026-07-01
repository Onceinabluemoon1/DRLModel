from typing import Literal, Tuple, Union

import einops as ei
import torch
import torch.nn as nn

PairEncoderOutput = Union[
    torch.Tensor,  # deterministic latents
    Tuple[torch.Tensor, torch.Tensor],  # gaussian latents (mean, logvar)
]


class PairEncoder(nn.Module):
    """(B, A, T, O), (B, A, T, O), (B, A, 1) -> (B, Z) or (B, Z), (B, Z)"""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        latent_dim: int,
        segment_len: int = 1,
        *,
        conv_channels: int = 64,
        gaussian_latent: bool = True,
        encoder_use_labels: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.segment_len = int(segment_len)
        self.use_conv_traj = self.segment_len > 1
        self.gaussian_latent = bool(gaussian_latent)
        self.encoder_use_labels = bool(encoder_use_labels)

        if self.use_conv_traj:
            self.traj_encoder = nn.Sequential(
                nn.Conv1d(self.obs_dim, conv_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(conv_channels, hidden_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
            )
        else:
            traj_in = self.obs_dim * self.segment_len
            self.traj_encoder = nn.Sequential(
                nn.Linear(traj_in, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            )

        pair_feat_dim = 3 * hidden_dim + int(self.encoder_use_labels)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        if self.gaussian_latent:
            self.fc_mean = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            self.fc_z = nn.Linear(hidden_dim, latent_dim)

    def _encode_traj(self, s: torch.Tensor) -> torch.Tensor:
        """(B, A, T, O) -> (B, A, H)"""
        if not self.use_conv_traj:
            feats = ei.rearrange(s, "B A T O -> B A (T O)")
            return self.traj_encoder(feats)
        A = s.shape[1]
        x = ei.rearrange(s, "B A T O -> (B A) O T")
        h = self.traj_encoder(x)  # (B*A, H, T)
        h = h.mean(dim=-1)  # (B*A, H)
        return ei.rearrange(h, "(B A) H -> B A H", A=A)

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y: torch.Tensor,
    ) -> PairEncoderOutput:
        # * siamese encode each traj, fuse labels, pool over A, set MLP
        h1 = self._encode_traj(s=s1)  # (B,A,T,O) -> (B, A, H)
        h2 = self._encode_traj(s=s2)  # (B,A,T,O) -> (B, A, H)
        pair_feats = [h1, h2, h1 - h2]
        if self.encoder_use_labels:
            pair_feats.append(y)
        h_annot = torch.cat(pair_feats, dim=-1)  # (B, A, 3H) or (B, A, 3H+1)
        h_annot = self.pair_mlp(h_annot)  # (B, A, H)
        h_set = h_annot.mean(dim=1)  # (B, H)
        h_ = self.model(h_set)  # (B, H)

        if self.gaussian_latent:
            return self.fc_mean(h_), self.fc_logvar(h_)
        return self.fc_z(h_)  # (B, Z)


def split_obs_z(x: torch.Tensor, *, obs_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, A, T, O+Z)`` -> ``obs (B, A, T, O)``, ``z (B, A, T, Z)``."""
    obs = x[..., :obs_dim]
    z = x[..., obs_dim:]
    return obs, z


class Decoder(nn.Module):
    """r(obs, z) -> scalar reward: (B, A, T, O + Z) -> (B, A, T, 1)"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        rewards = self.model(x)  # (B,A,T,1)
        if vector_rewards:
            raise ValueError(
                "Decoder (mlp) has no per-latent head decomposition; "
                "use a sum-head decoder (linear / odinlinear / slotlinear / ellenlinear / offsetlinear)."
            )
        return rewards


class LinearDecoder(nn.Module):
    """
    Obs trunk once, then linear routing + linear per-dim heads.

    ``r(s, z) = sum_j alpha_j(s) * w_j(s) z_j``

    ``alpha`` from ``softmax`` over gate logits. Each expert is linear in ``z_j``
    given obs features ``h(s)``; no cross-dim ``z`` mixing.
    Input: ``(B, A, T, O+Z)``; output: ``(B, A, T, 1)`` or ``(B, A, T, Z)`` if
    ``vector_rewards=True`` (returns ``alpha_j * r_j``, not pre-routing ``r_j``).
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.gate_temperature = float(gate_temperature)

        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.gate_head = nn.Linear(hidden_dim, latent_dim)
        self.z_weight = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)  # (B,A,T,O), (B,A,T,Z)

        # * obs trunk (once)
        h = self.obs_trunk(obs)  # (B,A,T,O) -> (B,A,T,H)

        # * per-dim linear experts: r_j = w_j(h) z_j
        w = self.z_weight(h)  # (B,A,T,H) -> (B,A,T,Z)
        r_j = w * z  # (B,A,T,Z)

        # * routing weights from obs features only
        gate_logits = self.gate_head(h)  # (B,A,T,H) -> (B,A,T,Z)
        a_j = torch.softmax(gate_logits / self.gate_temperature, dim=-1)  # (B,A,T,Z)
        weighted_r_j = a_j * r_j  # (B,A,T,Z)
        if vector_rewards:
            return weighted_r_j

        rewards = torch.einsum("batz->bat", weighted_r_j).unsqueeze(-1)  # (B,A,T,1)
        return rewards


class OdinLinearDecoder(nn.Module):
    """
    Obs trunk once, then one WN linear row per ``[h(s), z_j]``; direct sum over j.

    ``r(s, z) = sum_j r_j(s, z_j)`` with ``r_j = w̃_j · [h, z_j]`` (ODIN row WN).

    ``z_head`` is ``Linear(H+1, Z, bias=False)`` with ODIN row WN in forward.
    ``decoder_orth_weight`` applies ODIN orth on row-normalized ``z_head.weight``.
    Input: ``(B, A, T, O+Z)``; output: ``(B, A, T, 1)`` or ``(B, A, T, Z)`` if
    ``vector_rewards=True``.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        # gate_temperature kept for factory API parity; unused (no routing head).
        self.gate_temperature = float(gate_temperature)

        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.z_head = nn.Linear(hidden_dim + 1, latent_dim, bias=False)

    def normalized_z_weight(self) -> torch.Tensor:
        """ODIN row WN on ``z_head.weight``: center rows, L2 unit norm. ``(Z, H+1)``."""
        return self.z_head.weight
        # w = self.z_head.weight - self.z_head.weight.mean(dim=-1, keepdim=True)  # (Z, H+1)
        # norms = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (Z, 1)
        # return w / norms

    def orthogonality_loss(self) -> torch.Tensor:
        """ODIN orth: mean |w_i · w_j| on row-normalized ``z_head.weight``, i < j."""
        w_norm = self.normalized_z_weight()  # (Z, H+1)
        gram = w_norm @ w_norm.T  # (Z, Z)
        z = w_norm.shape[0]
        if z <= 1:
            return torch.zeros((), device=w_norm.device, dtype=w_norm.dtype)
        pair_count = z * (z - 1) / 2
        return torch.abs(torch.triu(gram, diagonal=1)).sum() / pair_count

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)  # (B,A,T,O), (B,A,T,Z)

        # * obs trunk (once)
        h = self.obs_trunk(obs)  # (B,A,T,O) -> (B,A,T,H)

        # * one WN row per j on [h, z_j]
        hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=self.latent_dim)
        zinput = ei.rearrange(z, "B A T Z -> B A T Z 1")
        x_j = torch.cat([hinput, zinput], dim=-1)  # (B,A,T,Z,H+1)
        w_norm = self.normalized_z_weight()  # (Z, H+1)
        r_j = torch.einsum("batzd,zd->batz", x_j, w_norm)  # (B,A,T,Z)
        if vector_rewards:
            return r_j

        rewards = r_j.sum(dim=-1).unsqueeze(-1)  # (B,A,T,1)
        return rewards


class SlotLinearDecoder(nn.Module):
    """
    Obs trunk once, then weight-tied per-dim experts on ``[h(s), z_j, onehot(j)]``.

    ``onehot(j)`` is a fixed slot index (no learned embedding) so each latent dim
    keeps a fixed role even when ``z_i = z_j``.

    ``routing="softmax"``: ``r = sum_j alpha_j r_j`` with ``alpha`` from softmax
    over per-slot mix logits (expert outputs reward + logit).

    ``routing="sum"``: ``r = sum_j r_j`` (no softmax routing; like ``odinlinear``).

    Each shared weight-tied expert is an MLP on ``[h, z_j, onehot(j)]``. No cross-dim
    ``z`` mixing. Input: ``(B, A, T, O+Z)``; output: ``(B, A, T, 1)`` or
    ``(B, A, T, Z)`` if ``vector_rewards=True``. With ``routing="softmax"``, vector
    output is ``alpha_j * r_j`` (post-routing), not pre-routing ``r_j``.

    ``decoder_orth_weight`` is ignored for this decoder (no orth penalty yet).
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        gate_temperature: float = 1.0,
        slot_routing: str = "softmax",
    ):
        super().__init__()
        routing = str(slot_routing).lower()
        assert routing in ("softmax", "sum")
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.gate_temperature = float(gate_temperature)
        self.slot_routing = routing
        H, Z = int(hidden_dim), int(latent_dim)

        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.2),
        )
        self.register_buffer(
            name="zindex_onehot", tensor=torch.eye(Z).to(dtype=torch.float32)
        )
        self.r_head = nn.Sequential(
            nn.Linear(H + 1 + Z, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, 2 if self.slot_routing == "softmax" else 1),
        )

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)  # (B,A,T,O), (B,A,T,Z)
        B, A, T, Z = z.shape

        # * obs trunk (once)
        h = self.obs_trunk(obs)  # (B,A,T,O) -> (B,A,T,H)

        # * weight-tied experts on [h, z_j, onehot(j)]
        hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=Z)
        zinput = ei.rearrange(z, "B A T Z -> B A T Z 1")
        jinput = ei.repeat(self.zindex_onehot, "Z C -> B A T Z C", B=B, A=A, T=T)
        head_in = torch.cat([hinput, zinput, jinput], dim=-1)  # (B,A,T,Z,H+1+Z)
        head_out = self.r_head(head_in)  # (B,A,T,Z,1) or (B,A,T,Z,2)
        r_j = head_out[..., 0]  # (B,A,T,Z)

        if self.slot_routing == "sum":
            if vector_rewards:
                return r_j
            rewards = torch.einsum("batz->bat", r_j).unsqueeze(-1)  # (B,A,T,1)
            return rewards

        mix_logits = head_out[..., 1]  # (B,A,T,Z)
        a_j = torch.softmax(mix_logits / self.gate_temperature, dim=-1)  # (B,A,T,Z)
        weighted_r_j = a_j * r_j  # (B,A,T,Z)
        if vector_rewards:
            return weighted_r_j

        rewards = torch.einsum("batz->bat", weighted_r_j).unsqueeze(-1)  # (B,A,T,1)
        return rewards


class EllenLinearDecoder(nn.Module):
    """
    Shared obs trunk, weight-tied readouts on ``[h(s), z_j, onehot(j)]``.

    ``h = obs_trunk(obs)`` once; ``r_j = MLP([h, z_j, onehot(j)])``;
    ``r(s, z) = sum_j r_j``. ``onehot(j)`` fixes slot identity even when
    ``z_i = z_j``.

    Input: ``(B, A, T, O+Z)``; output: ``(B, A, T, 1)`` or ``(B, A, T, Z)`` if
    ``vector_rewards=True``.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        gate_temperature: float = 1.0,
        slot_routing: str = "softmax",
    ):
        del gate_temperature, slot_routing, output_dim  # API parity with other decoders
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        H, Z = int(hidden_dim), int(latent_dim)

        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
        )
        self.register_buffer(
            name="zindex_onehot",
            tensor=torch.eye(Z).to(dtype=torch.float32),
        )
        self.r_head = nn.Sequential(
            nn.Linear(H + 1 + Z, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, 1),
        )

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)  # (B,A,T,O), (B,A,T,Z)
        B, A, T, Z = z.shape

        # * obs trunk (once)
        h = self.obs_trunk(obs)  # (B,A,T,H)

        # * weight-tied readouts on [h, z_j, onehot(j)]
        hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=Z)
        zinput = ei.rearrange(z, "B A T Z -> B A T Z 1")
        jinput = ei.repeat(self.zindex_onehot, "Z C -> B A T Z C", B=B, A=A, T=T)
        head_in = torch.cat([hinput, zinput, jinput], dim=-1)  # (B,A,T,Z,H+1+C)
        r_j = self.r_head(head_in).squeeze(-1)  # (B,A,T,Z)
        if vector_rewards:
            return r_j

        return r_j.sum(dim=-1, keepdim=True)  # (B,A,T,1)


class OffsetLinearDecoder(nn.Module):
    """
    Shared obs trunk, weight-tied ``(a_j, b_j)`` from ``[h(s), onehot(j)]``.

    ``r_j = b_j(h) + a_j(h) * z_j``; ``r(s, z) = sum_j r_j``. ``onehot(j)`` fixes
    slot identity even when ``z_i = z_j``.

    Input: ``(B, A, T, O+Z)``; output: ``(B, A, T, 1)`` or ``(B, A, T, Z)`` if
    ``vector_rewards=True``.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        gate_temperature: float = 1.0,
        slot_routing: str = "softmax",
    ):
        del gate_temperature, slot_routing, output_dim  # API parity with other decoders
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        H, Z = int(hidden_dim), int(latent_dim)

        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
        )
        self.register_buffer(
            name="zindex_onehot",
            tensor=torch.eye(Z).to(dtype=torch.float32),
        )
        self.affine_head = nn.Sequential(
            nn.Linear(H + Z, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, 2),
        )

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)  # (B,A,T,O), (B,A,T,Z)
        B, A, T, Z = z.shape

        # * obs trunk (once)
        h = self.obs_trunk(obs)  # (B,A,T,H)

        # * weight-tied (a_j, b_j) on [h, onehot(j)]
        hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=Z)
        jinput = ei.repeat(self.zindex_onehot, "Z C -> B A T Z C", B=B, A=A, T=T)
        head_in = torch.cat([hinput, jinput], dim=-1)  # (B,A,T,Z,H+C)
        out = self.affine_head(head_in)  # (B,A,T,Z,2)
        a_j, b_j = out[..., 0], out[..., 1]  # (B,A,T,Z)
        r_j = b_j + a_j * z  # (B,A,T,Z)
        if vector_rewards:
            return r_j

        return r_j.sum(dim=-1, keepdim=True)  # (B,A,T,1)


SUM_HEAD_DECODER_TYPES: frozenset[str] = frozenset(
    {
        "linear",
        "odinlinear",
        "slotlinear",
        "ellenlinear",
        "offsetlinear",
    }
)


def decoder_sums_latent_heads(*, decoder_type: str) -> bool:
    """Decoders with ``r = sum_j r_j`` (or routed sum) over per-latent heads."""
    return str(decoder_type).lower() in SUM_HEAD_DECODER_TYPES


def reward_decoder_column_orth_penalty(*, decoder: nn.Module) -> torch.Tensor:
    """ODIN orth penalty on ``OdinLinearDecoder`` only (no-op for other decoders)."""
    if isinstance(decoder, OdinLinearDecoder):
        return decoder.orthogonality_loss()
    p = next(decoder.parameters())
    return torch.zeros((), device=p.device, dtype=p.dtype)


DecoderType = Literal[
    "mlp", "linear", "odinlinear", "slotlinear", "ellenlinear", "offsetlinear"
]
DECODER_TYPES: tuple[str, ...] = (
    "mlp",
    "linear",
    "odinlinear",
    "slotlinear",
    "ellenlinear",
    "offsetlinear",
)


def make_reward_decoder(
    *,
    decoder_type: DecoderType | str = "mlp",
    obs_dim: int,
    latent_dim: int,
    hidden_dim: int,
    output_dim: int = 1,
    decoder_input: int | None = None,
    gate_temperature: float = 1.0,
    slot_routing: str = "softmax",
) -> nn.Module:
    """
    Factory for per-step reward decoders ``(B, A, T, O+Z) -> (B, A, T, 1)`` or
    ``(B, A, T, Z)`` when ``vector_rewards=True`` (sum-head decoders only).

    ``mlp``: concat ``[obs, z]`` MLP (legacy ``Decoder``).
    ``linear``: shared obs trunk, linear ``alpha_j(s)`` routing, ``r_j = w_j(s) z_j``.
    ``odinlinear``: per-dim ``[h, z_j]`` via ``z_head`` + ODIN row WN; direct sum.
    ``slotlinear``: weight-tied MLP experts on ``[h, z_j, onehot(j)]``; ``slot_routing`` is
      ``softmax`` (mix logits) or ``sum`` (direct ``sum_j r_j``).
    ``ellenlinear``: weight-tied MLP readouts on ``[h, z_j, onehot(j)]``; ``r = sum_j r_j``.
    ``offsetlinear``: weight-tied ``(a_j, b_j)`` from ``[h, onehot(j)]``;
      ``r_j = b_j(h) + a_j(h) z_j``; ``r = sum_j r_j``.
    """
    kind = str(decoder_type).lower()
    if kind not in DECODER_TYPES:
        raise ValueError(f"Unknown decoder_type={decoder_type!r}; choose {DECODER_TYPES}")

    if kind == "mlp":
        in_dim = int(decoder_input if decoder_input is not None else obs_dim + latent_dim)
        return Decoder(input_dim=in_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    expected_in = int(obs_dim + latent_dim)
    if decoder_input is not None and int(decoder_input) != expected_in:
        raise ValueError(
            f"decoder_type={kind!r} requires decoder_input=obs_dim+latent_dim "
            f"({expected_in}); got {decoder_input}. Use decoder_type='mlp' otherwise."
        )

    decoder_kwargs = dict(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        gate_temperature=gate_temperature,
    )
    if kind == "linear":
        return LinearDecoder(**decoder_kwargs)
    if kind == "odinlinear":
        return OdinLinearDecoder(**decoder_kwargs)
    if kind == "slotlinear":
        return SlotLinearDecoder(**decoder_kwargs, slot_routing=slot_routing)
    if kind == "ellenlinear":
        return EllenLinearDecoder(**decoder_kwargs)
    if kind == "offsetlinear":
        return OffsetLinearDecoder(**decoder_kwargs)
    raise RuntimeError(f"Unhandled decoder_type={kind!r}")
