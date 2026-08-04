from dataclasses import dataclass
from typing import Literal

import einops as ei
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_obs_z(x: torch.Tensor, *, obs_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, A, T, O+Z)`` -> ``obs (B, A, T, O)``, ``z (B, A, T, Z)``."""
    obs = x[..., :obs_dim]
    z = x[..., obs_dim:]
    return obs, z


class MultiplicativeDecoder(nn.Module):
    """Deterministic ``theta_j(s) = softplus(w_j(h))``; ``r = sum_j theta_j(s) z_j``.

    Optional ``trunk_layernorm`` on ``h``. Sparsity is always on pre-projection ``theta``.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        *,
        sparsity_mode: str = "elementwise",
        softplus_beta: float = 1.0,
        trunk_layernorm: bool = False,
    ):
        super().__init__()
        mode = str(sparsity_mode).lower()
        assert mode in ("elementwise", "group")
        self.sparsity_mode = mode
        self.softplus_beta = float(softplus_beta)
        self.trunk_layernorm = bool(trunk_layernorm)
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        H, Z = int(hidden_dim), int(latent_dim)

        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
        )
        self.trunk_ln = (
            nn.LayerNorm(H, elementwise_affine=False) if self.trunk_layernorm else None
        )
        self.w_head = nn.Linear(H, Z, bias=False)
        self.last_w_sparsity: torch.Tensor | None = None

    def theta_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """``softplus(w(h))`` from observations only; ``(B, A, T, Z)``."""
        h = self.obs_trunk(obs)  # (B, A, T, O) -> (B, A, T, H)
        if self.trunk_ln is not None:
            h = self.trunk_ln(h)
        return F.softplus(self.w_head(h), beta=self.softplus_beta)

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)
        # * trunk: (B, A, T, O) → (B, A, T, H)
        theta = self.theta_from_obs(obs)  # (B, A, T, Z)
        if self.sparsity_mode == "group":
            head_energy = (theta**2).mean(dim=(0, 1, 2))
            self.last_w_sparsity = torch.sqrt(head_energy + 1e-12).mean()
        else:
            self.last_w_sparsity = theta.mean()
        # * reward: (B, A, T, Z) = (B, A, T, Z) * (B, A, T, Z)
        r_j = theta * z
        if vector_rewards:
            return r_j
        return torch.einsum("batz->bat", r_j).unsqueeze(-1)


class OdinDecoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        H, Z = int(hidden_dim), int(latent_dim)
        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
        )
        self.z_head = nn.Linear(H + 1, Z, bias=False)

    def normalized_z_weight(self) -> torch.Tensor:
        return self.z_head.weight

    def orthogonality_loss(self) -> torch.Tensor:
        w_norm = self.normalized_z_weight()
        gram = w_norm @ w_norm.T
        z = w_norm.shape[0]
        if z <= 1:
            return torch.zeros((), device=w_norm.device, dtype=w_norm.dtype)
        pair_count = z * (z - 1) / 2
        return torch.abs(torch.triu(gram, diagonal=1)).sum() / pair_count

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)
        h = self.obs_trunk(obs)
        hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=self.latent_dim)
        zinput = ei.rearrange(z, "B A T Z -> B A T Z 1")
        x_j = torch.cat([hinput, zinput], dim=-1)
        w_norm = self.normalized_z_weight()
        r_j = torch.einsum("batzd,zd->batz", x_j, w_norm)
        if vector_rewards:
            return r_j
        return r_j.sum(dim=-1).unsqueeze(-1)


def _cond_per_index_inputs(
    *,
    h: torch.Tensor,
    z: torch.Tensor,
    zindex_onehot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand trunk ``h`` and build per-latent-index head inputs."""
    B, A, T, Z = z.shape
    hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=Z)
    zinput = ei.rearrange(z, "B A T Z -> B A T Z 1")
    jinput = ei.repeat(zindex_onehot, "Z C -> B A T Z C", B=B, A=A, T=T)
    head_in = torch.cat([hinput, zinput, jinput], dim=-1)  # (B, A, T, Z, H+1+Z)
    return hinput, zinput, jinput, head_in


class ConditionalFiLMHead(nn.Module):
    """SPL-style adaptive latent conditioning: FiLM on ``h`` from ``(z_j, j)``."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        latent_dim: int,
        out_dim: int,
    ):
        super().__init__()
        H, Z = int(hidden_dim), int(latent_dim)
        self.h_proj = nn.Sequential(
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
        )
        self.film = nn.Linear(1 + Z, 2 * H)
        self.out = nn.Linear(H, out_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        with torch.no_grad():
            self.film.bias[:H] = 1.0

    def forward(
        self,
        *,
        hinput: torch.Tensor,
        zinput: torch.Tensor,
        jinput: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.h_proj(hinput)  # (B, A, T, Z, H)
        cond = torch.cat([zinput, jinput], dim=-1)  # (B, A, T, Z, 1+Z)
        gamma, beta = self.film(cond).chunk(chunks=2, dim=-1)  # each (B, A, T, Z, H)
        hidden = gamma * hidden + beta  # (B, A, T, Z, H)
        return self.out(hidden)  # (B, A, T, Z, out_dim)


def _make_cond_reward_head(
    *,
    hidden_dim: int,
    latent_dim: int,
    out_dim: int,
    use_rhead_film: bool,
) -> nn.Module:
    H, Z = int(hidden_dim), int(latent_dim)
    if use_rhead_film:
        return ConditionalFiLMHead(hidden_dim=H, latent_dim=Z, out_dim=out_dim)
    return nn.Sequential(
        nn.Linear(H + 1 + Z, H),
        nn.LeakyReLU(0.2),
        nn.Linear(H, out_dim),
    )


def _cond_head_forward(
    *,
    head: nn.Module,
    hinput: torch.Tensor,
    zinput: torch.Tensor,
    jinput: torch.Tensor,
    head_in: torch.Tensor,
) -> torch.Tensor:
    if isinstance(head, ConditionalFiLMHead):
        return head(hinput=hinput, zinput=zinput, jinput=jinput)
    return head(head_in)


def _group_head_reward_sparsity(r_j: torch.Tensor) -> torch.Tensor:
    """Group L1 on per-head reward magnitudes; ``r_j`` is ``(B, A, T, Z)``."""
    head_energy = (r_j**2).mean(dim=(0, 1, 2))  # (Z,)
    return torch.sqrt(head_energy + 1e-12).mean()


CondRewardAct = Literal["linear", "sigmoid", "softplus", "categorical"]


def _cond_head_out_dim(
    *,
    reward_act: CondRewardAct,
    use_gate: bool,
    n_bins: int,
) -> int:
    if reward_act == "categorical":
        return int(n_bins)
    if reward_act == "sigmoid" and use_gate:
        return 2
    return 1


class ConditionalDecoder(nn.Module):
    """Per-index MLP head on ``(h, z_j, j)``; variant selected by flags / ``reward_act``."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        *,
        use_rhead_film: bool = False,
        use_gate: bool = False,
        reward_act: CondRewardAct = "linear",
        softplus_beta: float = 1.0,
        r_max: float = 1.0,
        n_bins: int = 11,
    ):
        super().__init__()
        act = str(reward_act).lower()
        assert act in ("linear", "sigmoid", "softplus", "categorical")
        if act == "categorical":
            assert int(n_bins) >= 2
        if act != "sigmoid":
            assert not use_gate, "use_gate only applies to reward_act=sigmoid (condsig)"
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.use_rhead_film = bool(use_rhead_film)
        self.use_gate = bool(use_gate)
        self.reward_act: CondRewardAct = act  # type: ignore[assignment]
        self.softplus_beta = float(softplus_beta)
        self.r_max = float(r_max)
        self.n_bins = int(n_bins)
        H, Z = int(hidden_dim), int(latent_dim)
        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
        )
        self.register_buffer(
            name="zindex_onehot", tensor=torch.eye(Z).to(dtype=torch.float32)
        )
        head_out_dim = _cond_head_out_dim(
            reward_act=self.reward_act,
            use_gate=self.use_gate,
            n_bins=self.n_bins,
        )
        self.head = _make_cond_reward_head(
            hidden_dim=H,
            latent_dim=Z,
            out_dim=head_out_dim,
            use_rhead_film=self.use_rhead_film,
        )
        if self.reward_act == "categorical":
            K = self.n_bins
            pos_bins = torch.linspace(r_max / (K - 1), r_max, K - 1)
            self.register_buffer(
                name="bins",
                tensor=torch.cat([torch.zeros(1), pos_bins], dim=0).to(
                    dtype=torch.float32
                ),
            )
            self._init_categorical_head_bias()
        self.last_head_probs: torch.Tensor | None = None
        self.last_w_sparsity: torch.Tensor | None = None

    def _init_categorical_head_bias(self) -> None:
        # * init mass near bin 0
        if not self.use_rhead_film:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
            with torch.no_grad():
                self.head[-1].bias[0] = 2.0
        else:
            nn.init.zeros_(self.head.out.weight)
            nn.init.zeros_(self.head.out.bias)
            with torch.no_grad():
                self.head.out.bias[0] = 2.0

    def _per_head_rewards(
        self,
        *,
        head_out: torch.Tensor,
    ) -> torch.Tensor:
        if self.reward_act == "categorical":
            logits = head_out  # (B, A, T, Z, K)
            p_j = F.softmax(logits, dim=-1)
            self.last_head_probs = p_j
            return ei.einsum(p_j, self.bins, "B A T Z K, K -> B A T Z")
        if self.reward_act == "sigmoid" and self.use_gate:
            raw = head_out[..., 0]
            gate = torch.sigmoid(head_out[..., 1])
            return gate * torch.sigmoid(raw) * self.r_max
        raw = head_out.squeeze(-1)  # (B, A, T, Z)
        if self.reward_act == "sigmoid":
            return torch.sigmoid(raw) * self.r_max
        if self.reward_act == "softplus":
            return F.softplus(raw, beta=self.softplus_beta)
        return raw

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)
        h = self.obs_trunk(obs)  # (B, A, T, H)
        hinput, zinput, jinput, head_in = _cond_per_index_inputs(
            h=h,
            z=z,
            zindex_onehot=self.zindex_onehot,
        )
        head_out = _cond_head_forward(
            head=self.head,
            hinput=hinput,
            zinput=zinput,
            jinput=jinput,
            head_in=head_in,
        )
        if self.reward_act != "categorical":
            self.last_head_probs = None
        r_j = self._per_head_rewards(head_out=head_out)  # (B, A, T, Z)
        self.last_w_sparsity = _group_head_reward_sparsity(r_j=r_j)
        if vector_rewards:
            return r_j
        return r_j.sum(dim=-1, keepdim=True)


class SlotDecoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
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
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)
        B, A, T, Z = z.shape
        h = self.obs_trunk(obs)
        hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=Z)
        zinput = ei.rearrange(z, "B A T Z -> B A T Z 1")
        jinput = ei.repeat(self.zindex_onehot, "Z C -> B A T Z C", B=B, A=A, T=T)
        head_in = torch.cat([hinput, zinput, jinput], dim=-1)
        head_out = self.r_head(head_in)
        r_j = head_out[..., 0]
        if self.slot_routing == "sum":
            if vector_rewards:
                return r_j
            return torch.einsum("batz->bat", r_j).unsqueeze(-1)
        mix_logits = head_out[..., 1]
        a_j = torch.softmax(mix_logits / self.gate_temperature, dim=-1)
        weighted_r_j = a_j * r_j
        if vector_rewards:
            return weighted_r_j
        return torch.einsum("batz->bat", weighted_r_j).unsqueeze(-1)


SUM_HEAD_DECODER_TYPES: frozenset[str] = frozenset(
    {
        "multi",
        "cond",
        "condsig",
        "condsp",
        "condcat",
        "slot",
        "odin",
    }
)


def decoder_sums_latent_heads(*, decoder_type: str) -> bool:
    return normalize_decoder_type(decoder_type=decoder_type) in SUM_HEAD_DECODER_TYPES


def reward_decoder_column_orth_penalty(*, decoder: nn.Module) -> torch.Tensor:
    if isinstance(decoder, OdinDecoder):
        return decoder.orthogonality_loss()
    p = next(decoder.parameters())
    return torch.zeros((), device=p.device, dtype=p.dtype)


def reward_decoder_theta_sparsity(*, decoder: nn.Module) -> torch.Tensor:
    if isinstance(decoder, (MultiplicativeDecoder, ConditionalDecoder)):
        if decoder.last_w_sparsity is not None:
            return decoder.last_w_sparsity
    p = next(decoder.parameters())
    return torch.zeros((), device=p.device, dtype=p.dtype)


DecoderType = Literal[
    "multi",
    "cond",
    "condsig",
    "condsp",
    "condcat",
    "slot",
    "odin",
]
DECODER_TYPES: tuple[str, ...] = (
    "multi",
    "cond",
    "condsig",
    "condsp",
    "condcat",
    "slot",
    "odin",
)

_DECODER_TYPE_ALIASES: dict[str, str] = {
    "poslinear": "multi",
    "ellenlinear": "cond",
    "condsoftplus": "condsp",
    "condsigmoid": "condsig",
    "slotlinear": "slot",
    "odinlinear": "odin",
}


def normalize_decoder_type(*, decoder_type: str) -> str:
    kind = str(decoder_type).lower()
    return _DECODER_TYPE_ALIASES.get(kind, kind)


@dataclass
class RewardDecoderConfig:
    """Architecture + hyperparameters for ``make_reward_decoder``."""

    decoder_type: str = "multi"
    obs_dim: int = 0
    latent_dim: int = 0
    hidden_dim: int = 128
    decoder_input: int | None = None

    # multi
    sparsity_mode: str = "elementwise"
    softplus_beta: float = 1.0
    trunk_layernorm: bool = False

    # cond (+ condsig / condsp / condcat via ``make_reward_decoder``)
    use_rhead_film: bool = False
    r_max: float = 1.0
    n_bins: int = 11
    # slot
    gate_temperature: float = 1.0
    slot_routing: str = "softmax"

    @classmethod
    def from_train(
        cls,
        *,
        decoder_type: str,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        decoder_input: int | None = None,
        sparsity_mode: str = "elementwise",
        softplus_beta: float = 1.0,
        trunk_layernorm: bool = False,
        use_rhead_film: bool = False,
        r_max: float = 1.0,
        n_bins: int = 11,
        gate_temperature: float = 1.0,
        slot_routing: str = "softmax",
    ) -> "RewardDecoderConfig":
        return cls(
            decoder_type=normalize_decoder_type(decoder_type=str(decoder_type)),
            obs_dim=int(obs_dim),
            latent_dim=int(latent_dim),
            hidden_dim=int(hidden_dim),
            decoder_input=decoder_input,
            sparsity_mode=str(sparsity_mode),
            softplus_beta=float(softplus_beta),
            trunk_layernorm=bool(trunk_layernorm),
            use_rhead_film=bool(use_rhead_film),
            r_max=float(r_max),
            n_bins=int(n_bins),
            gate_temperature=float(gate_temperature),
            slot_routing=str(slot_routing),
        )


_COND_DECODER_KINDS: dict[str, tuple[CondRewardAct, bool]] = {
    "cond": ("linear", False),
    "condsig": ("sigmoid", True),
    "condsp": ("softplus", False),
    "condcat": ("categorical", False),
}


def make_reward_decoder(*, config: RewardDecoderConfig) -> nn.Module:
    kind = normalize_decoder_type(decoder_type=config.decoder_type)
    if kind not in DECODER_TYPES:
        raise ValueError(
            f"Unknown decoder_type={config.decoder_type!r}; choose {DECODER_TYPES}"
        )

    expected_in = int(config.obs_dim + config.latent_dim)
    if config.decoder_input is not None and int(config.decoder_input) != expected_in:
        raise ValueError(
            f"decoder_type={kind!r} requires decoder_input=obs_dim+latent_dim "
            f"({expected_in}); got {config.decoder_input}."
        )

    core = dict(
        obs_dim=config.obs_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    )
    if kind == "multi":
        return MultiplicativeDecoder(
            **core,
            sparsity_mode=config.sparsity_mode,
            softplus_beta=config.softplus_beta,
            trunk_layernorm=config.trunk_layernorm,
        )
    if kind in _COND_DECODER_KINDS:
        reward_act, use_gate = _COND_DECODER_KINDS[kind]
        return ConditionalDecoder(
            obs_dim=config.obs_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            use_rhead_film=config.use_rhead_film,
            use_gate=use_gate,
            reward_act=reward_act,
            softplus_beta=config.softplus_beta,
            r_max=config.r_max,
            n_bins=config.n_bins,
        )
    if kind == "slot":
        return SlotDecoder(
            **core,
            gate_temperature=config.gate_temperature,
            slot_routing=config.slot_routing,
        )
    if kind == "odin":
        return OdinDecoder(**core)
    raise RuntimeError(f"Unhandled decoder_type={kind!r}")
