from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x

        out = self.conv1(x)
        out = out.transpose(1, 2)  # (B, T, C)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)  # (B, C, T)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)

        return residual + out


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        dilations = [2**i for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.input_proj(x)  # (B, H, T)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)  # (B, T, H)
        x = self.out_norm(x)
        x = x.mean(dim=1)  # (B, H)
        return x


class LabelEncoder(nn.Module):
    def __init__(
        self, num_classes: int, embed_dim: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B,) long
        e = self.embedding(y)
        return self.mlp(e)


class TemporalDecoder(nn.Module):
    """
    Decodes a latent z into a fixed-length sequence using
    latent projection -> temporal tensor -> TCN-style refinement.
    """

    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * seq_len),
            nn.GELU(),
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))

        dilations = [2**i for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        bsz = z.shape[0]
        x = self.latent_proj(z).view(bsz, self.seq_len, self.hidden_dim)
        x = x + self.pos_embedding

        x = x.transpose(1, 2)  # (B, H, T)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (B, T, H)

        x = self.out_norm(x)
        x = self.out_proj(x)  # (B, T, output_dim)
        return x


@dataclass
class HopperVAEConfig:
    obs_dim: int
    act_dim: int
    seq_len: int = 200
    hidden_dim: int = 128
    num_tcn_blocks: int = 4
    kernel_size: int = 5
    dropout: float = 0.1
    label_embed_dim: int = 16
    label_hidden_dim: int = 64
    latent_dim: int = 32
    num_label_classes: int = 2
    pref_hidden_dim: int = 128


class HopperPairVAE(nn.Module):
    """
    Option A:
      - one shared latent z for the whole pair (x1, x2, y)
      - reconstruct x1_obs, x1_act, x2_obs, x2_act
      - reconstruct preference y
    """

    def __init__(self, config: HopperVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.state_encoder = TemporalEncoder(
            input_dim=config.obs_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_tcn_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.action_encoder = TemporalEncoder(
            input_dim=config.act_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_tcn_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.label_encoder = LabelEncoder(
            num_classes=config.num_label_classes,
            embed_dim=config.label_embed_dim,
            hidden_dim=config.label_hidden_dim,
            dropout=config.dropout,
        )

        # pair features:
        # h1_obs, h1_act, h2_obs, h2_act, diffs, label_h
        pair_dim = (
            config.hidden_dim * 4  # h1_obs, h1_act, h2_obs, h2_act
            + config.hidden_dim * 2  # obs diff, act diff
            + config.label_hidden_dim
        )

        self.posterior_mlp = nn.Sequential(
            nn.Linear(pair_dim, config.pref_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pref_hidden_dim, config.pref_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.mu_head = nn.Linear(config.pref_hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.pref_hidden_dim, config.latent_dim)

        # Four trajectory decoders
        self.x1_obs_decoder = TemporalDecoder(
            latent_dim=config.latent_dim,
            seq_len=config.seq_len,
            hidden_dim=config.hidden_dim,
            output_dim=config.obs_dim,
            num_blocks=config.num_tcn_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.x1_act_decoder = TemporalDecoder(
            latent_dim=config.latent_dim,
            seq_len=config.seq_len,
            hidden_dim=config.hidden_dim,
            output_dim=config.act_dim,
            num_blocks=config.num_tcn_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.x2_obs_decoder = TemporalDecoder(
            latent_dim=config.latent_dim,
            seq_len=config.seq_len,
            hidden_dim=config.hidden_dim,
            output_dim=config.obs_dim,
            num_blocks=config.num_tcn_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.x2_act_decoder = TemporalDecoder(
            latent_dim=config.latent_dim,
            seq_len=config.seq_len,
            hidden_dim=config.hidden_dim,
            output_dim=config.act_dim,
            num_blocks=config.num_tcn_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )

        # Preference decoder head
        self.pref_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.pref_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pref_hidden_dim, config.pref_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pref_hidden_dim, 1),
        )

    def encode_pair(
        self,
        x1_obs: torch.Tensor,
        x1_act: torch.Tensor,
        x2_obs: torch.Tensor,
        x2_act: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h1_obs = self.state_encoder(x1_obs)
        h1_act = self.action_encoder(x1_act)
        h2_obs = self.state_encoder(x2_obs)
        h2_act = self.action_encoder(x2_act)
        hy = self.label_encoder(y)

        h = torch.cat(
            [
                h1_obs,
                h1_act,
                h2_obs,
                h2_act,
                h1_obs - h2_obs,
                h1_act - h2_act,
                hy,
            ],
            dim=-1,
        )

        h = self.posterior_mlp(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1_obs_recon = self.x1_obs_decoder(z)
        x1_act_recon = self.x1_act_decoder(z)
        x2_obs_recon = self.x2_obs_decoder(z)
        x2_act_recon = self.x2_act_decoder(z)
        pref_logit = self.pref_head(z).squeeze(-1)

        return {
            "x1_obs_recon": x1_obs_recon,
            "x1_act_recon": x1_act_recon,
            "x2_obs_recon": x2_obs_recon,
            "x2_act_recon": x2_act_recon,
            "pref_logit": pref_logit,
            "pref_prob": torch.sigmoid(pref_logit),
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode_pair(
            x1_obs=batch["x1_obs"],
            x1_act=batch["x1_act"],
            x2_obs=batch["x2_obs"],
            x2_act=batch["x2_act"],
            y=batch["y"],
        )
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            **decoded,
        }


def kl_divergence_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # returns mean KL over batch
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.sum(dim=-1)  # sum over latent dim
    return kl.mean()
