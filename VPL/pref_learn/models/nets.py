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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def linear_column_orthogonality_loss(*, weight: torch.Tensor) -> torch.Tensor:
    """Off-diagonal penalty on ``W W^T`` for ``Linear`` weight ``(Z, H)``."""
    gram = weight @ weight.T  # (Z, Z)
    off_diag = gram - torch.diag(torch.diag(gram))
    return (off_diag**2).sum()


def reward_decoder_column_orth_penalty(*, decoder: nn.Module) -> torch.Tensor:
    """Sum column-orth penalties over structured decoder readout layers."""
    terms: list[torch.Tensor] = []
    if isinstance(decoder, LinearDecoder):
        terms.append(linear_column_orthogonality_loss(weight=decoder.z_weight.weight))
        terms.append(linear_column_orthogonality_loss(weight=decoder.gate_head.weight))
    if isinstance(decoder, OdinLinearDecoder):
        terms.append(linear_column_orthogonality_loss(weight=decoder.z_head.weight))
    if not terms:
        p = next(decoder.parameters())
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return sum(terms)


class LinearDecoder(nn.Module):
    """
    Obs trunk once, then linear routing + linear per-dim heads.

    ``r(s, z) = sum_j alpha_j(s) * w_j(s) z_j``

    ``alpha`` from ``softmax`` over gate logits. Each expert is linear in ``z_j``
    given obs features ``h(s)``; no cross-dim ``z`` mixing.
    Input: ``(B, A, T, O+Z)``; output: ``(B, A, T, 1)``.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)  # (B,A,T,O), (B,A,T,Z)

        # * obs trunk (once)
        h = self.obs_trunk(obs)  # (B,A,T,O) -> (B,A,T,H)

        # * routing weights from obs features only
        gate_logits = self.gate_head(h)  # (B,A,T,H) -> (B,A,T,Z)
        a_j = torch.softmax(gate_logits / self.gate_temperature, dim=-1)  # (B,A,T,Z)

        # * per-dim linear experts: r_j = w_j(h) z_j
        w = self.z_weight(h)  # (B,A,T,H) -> (B,A,T,Z)
        r_j = w * z  # (B,A,T,Z)
        rewards = torch.einsum("batz,batz->bat", a_j, r_j).unsqueeze(-1)  # (B,A,T,1)
        return rewards


class OdinLinearDecoder(nn.Module):
    """
    Obs trunk once, then one linear head per ``[h(s), z_j]``; direct sum over j.

    ``r(s, z) = sum_j r_j(s, z_j)``

    Each ``r_j`` uses its own row in ``z_head`` (shape ``(Z, H+1)``); no cross-dim
    ``z`` mixing and no softmax routing. ``decoder_orth_weight`` penalizes
    off-diagonal ``W W^T`` on ``z_head``.
    Input: ``(B, A, T, O+Z)``; output: ``(B, A, T, 1)``.
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
        self.z_head = nn.Linear(hidden_dim + 1, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)  # (B,A,T,O), (B,A,T,Z)

        # * obs trunk (once)
        h = self.obs_trunk(obs)  # (B,A,T,O) -> (B,A,T,H)

        # * one linear head per j on [h, z_j]; z_head rows are orth-regularizable
        x_j = torch.cat(
            [
                ei.repeat(h, "B A T H -> B A T Z H", Z=self.latent_dim),
                ei.rearrange(z, "B A T Z -> B A T Z 1"),
            ],
            dim=-1,
        )  # (B,A,T,Z,H+1)
        W = self.z_head.weight  # (Z, H+1)
        r_j = torch.einsum("batzd,zd->batz", x_j, W)  # (B,A,T,Z)
        if self.z_head.bias is not None:
            r_j = r_j + self.z_head.bias

        rewards = r_j.sum(dim=-1).unsqueeze(-1)  # (B,A,T,1)
        return rewards


class SlotLinearDecoder(nn.Module):
    """
    Obs trunk once, then weight-tied per-dim experts on ``[h(s), z_j]``.

    ``r(s, z) = sum_j alpha_j(s, z_j) * r_j(s, z_j)``

    Each shared expert maps ``[h, z_j] -> (scalar_reward, mix_logit)``; ``alpha``
    from ``softmax`` over ``mix_logit``. Still linear in ``z_j`` per expert; no
    cross-dim ``z`` mixing.
    Input: ``(B, A, T, O+Z)``; output: ``(B, A, T, 1)``.

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
        self.expert_head = nn.Linear(hidden_dim + 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)  # (B,A,T,O), (B,A,T,Z)

        # * obs trunk (once)
        h = self.obs_trunk(obs)  # (B,A,T,O) -> (B,A,T,H)

        # * weight-tied experts: [h, z_j] -> (r_j, logit_j)
        expert_in = torch.cat(
            [
                ei.repeat(h, "B A T H -> B A T Z H", Z=self.latent_dim),
                ei.rearrange(z, "B A T Z -> B A T Z 1"),
            ],
            dim=-1,
        )  # (B,A,T,Z,H+1)
        expert_out = self.expert_head(expert_in)  # (B,A,T,Z,2)
        r_j, a_j = expert_out[..., 0], expert_out[..., 1]  # (B,A,T,Z)
        a_j = torch.softmax(a_j / self.gate_temperature, dim=-1)  # (B,A,T,Z)

        rewards = torch.einsum("batz,batz->bat", a_j, r_j).unsqueeze(-1)  # (B,A,T,1)
        return rewards


DecoderType = Literal["mlp", "linear", "odinlinear", "slotlinear"]
DECODER_TYPES: tuple[str, ...] = ("mlp", "linear", "odinlinear", "slotlinear")


def make_reward_decoder(
    *,
    decoder_type: DecoderType | str = "mlp",
    obs_dim: int,
    latent_dim: int,
    hidden_dim: int,
    output_dim: int = 1,
    decoder_input: int | None = None,
    gate_temperature: float = 1.0,
) -> nn.Module:
    """
    Factory for per-step reward decoders ``(B, A, T, O+Z) -> (B, A, T, 1)``.

    ``mlp``: concat ``[obs, z]`` MLP (legacy ``Decoder``).
    ``linear``: shared obs trunk, linear ``alpha_j(s)`` routing, ``r_j = w_j(s) z_j``.
    ``odinlinear``: per-dim ``[h, z_j]`` linear heads, direct sum over j.
    ``slotlinear``: weight-tied ``[h, z_j]`` experts for ``r_j`` and mix logits.
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
        return SlotLinearDecoder(**decoder_kwargs)
    raise RuntimeError(f"Unhandled decoder_type={kind!r}")
