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
        rewards = self.model(x)
        if vector_rewards:
            raise ValueError(
                "Decoder (mlp) has no per-latent head decomposition; "
                "use a sum-head decoder (linear / odinlinear / slotlinear / ellenlinear / offsetlinear)."
            )
        return rewards


class LinearDecoder(nn.Module):
    """``r(s, z) = sum_j w_j(s) z_j``."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        gate_temperature: float = 1.0,
    ):
        del gate_temperature, output_dim
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
        self.w_head = nn.Linear(H, Z)

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)
        h = self.obs_trunk(obs)
        w = self.w_head(h)
        r_j = w * z
        if vector_rewards:
            return r_j
        return torch.einsum("batz->bat", r_j).unsqueeze(-1)


class PosLinearDecoder(nn.Module):
    """Deterministic ``theta_j(s) = softplus(w_j(h))``; ``r = sum_j theta_j(s) z_j``."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        gate_temperature: float = 1.0,
        sparsity_mode: str = "elementwise",
        softplus_beta: float = 1.0,
    ):
        del gate_temperature, output_dim
        super().__init__()
        mode = str(sparsity_mode).lower()
        assert mode in ("elementwise", "group")
        self.sparsity_mode = mode
        self.softplus_beta = float(softplus_beta)
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
        self.w_head = nn.Linear(H, Z)
        self.last_w_sparsity: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)
        h = self.obs_trunk(obs)
        theta = F.softplus(self.w_head(h), beta=self.softplus_beta)
        if self.sparsity_mode == "group":
            head_energy = (theta**2).mean(dim=(0, 1, 2))
            self.last_w_sparsity = torch.sqrt(head_energy + 1e-12).mean()
        else:
            self.last_w_sparsity = theta.mean()
        r_j = theta * z
        if vector_rewards:
            return r_j
        return torch.einsum("batz->bat", r_j).unsqueeze(-1)


class OdinLinearDecoder(nn.Module):
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
        self.z_head = nn.Linear(hidden_dim + 1, latent_dim, bias=False)

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


class SlotLinearDecoder(nn.Module):
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


class EllenLinearDecoder(nn.Module):
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
        del gate_temperature, slot_routing, output_dim
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
            name="zindex_onehot", tensor=torch.eye(Z).to(dtype=torch.float32)
        )
        self.r_head = nn.Sequential(
            nn.Linear(H + 1 + Z, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, 1),
        )

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)
        B, A, T, Z = z.shape
        h = self.obs_trunk(obs)
        hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=Z)
        zinput = ei.rearrange(z, "B A T Z -> B A T Z 1")
        jinput = ei.repeat(self.zindex_onehot, "Z C -> B A T Z C", B=B, A=A, T=T)
        head_in = torch.cat([hinput, zinput, jinput], dim=-1)
        r_j = self.r_head(head_in).squeeze(-1)
        if vector_rewards:
            return r_j
        return r_j.sum(dim=-1, keepdim=True)


class OffsetLinearDecoder(nn.Module):
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
        del gate_temperature, slot_routing, output_dim
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
            name="zindex_onehot", tensor=torch.eye(Z).to(dtype=torch.float32)
        )
        self.affine_head = nn.Sequential(
            nn.Linear(H + Z, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, 2),
        )

    def forward(self, x: torch.Tensor, *, vector_rewards: bool = False) -> torch.Tensor:
        obs, z = split_obs_z(x=x, obs_dim=self.obs_dim)
        B, A, T, Z = z.shape
        h = self.obs_trunk(obs)
        hinput = ei.repeat(h, "B A T H -> B A T Z H", Z=Z)
        jinput = ei.repeat(self.zindex_onehot, "Z C -> B A T Z C", B=B, A=A, T=T)
        head_in = torch.cat([hinput, jinput], dim=-1)
        out = self.affine_head(head_in)
        a_j, b_j = out[..., 0], out[..., 1]
        r_j = b_j + a_j * z
        if vector_rewards:
            return r_j
        return r_j.sum(dim=-1, keepdim=True)


SUM_HEAD_DECODER_TYPES: frozenset[str] = frozenset(
    {
        "linear",
        "poslinear",
        "odinlinear",
        "slotlinear",
        "ellenlinear",
        "offsetlinear",
    }
)


def decoder_sums_latent_heads(*, decoder_type: str) -> bool:
    return str(decoder_type).lower() in SUM_HEAD_DECODER_TYPES


def reward_decoder_column_orth_penalty(*, decoder: nn.Module) -> torch.Tensor:
    if isinstance(decoder, OdinLinearDecoder):
        return decoder.orthogonality_loss()
    p = next(decoder.parameters())
    return torch.zeros((), device=p.device, dtype=p.dtype)


def reward_decoder_theta_sparsity(*, decoder: nn.Module) -> torch.Tensor:
    if isinstance(decoder, PosLinearDecoder) and decoder.last_w_sparsity is not None:
        return decoder.last_w_sparsity
    p = next(decoder.parameters())
    return torch.zeros((), device=p.device, dtype=p.dtype)


DecoderType = Literal[
    "mlp",
    "linear",
    "poslinear",
    "odinlinear",
    "slotlinear",
    "ellenlinear",
    "offsetlinear",
]
DECODER_TYPES: tuple[str, ...] = (
    "mlp",
    "linear",
    "poslinear",
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
    sparsity_mode: str = "elementwise",
    softplus_beta: float = 1.0,
) -> nn.Module:
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
    if kind == "poslinear":
        return PosLinearDecoder(
            **decoder_kwargs,
            sparsity_mode=sparsity_mode,
            softplus_beta=softplus_beta,
        )
    if kind == "odinlinear":
        return OdinLinearDecoder(**decoder_kwargs)
    if kind == "slotlinear":
        return SlotLinearDecoder(**decoder_kwargs, slot_routing=slot_routing)
    if kind == "ellenlinear":
        return EllenLinearDecoder(**decoder_kwargs)
    if kind == "offsetlinear":
        return OffsetLinearDecoder(**decoder_kwargs)
    raise RuntimeError(f"Unhandled decoder_type={kind!r}")
