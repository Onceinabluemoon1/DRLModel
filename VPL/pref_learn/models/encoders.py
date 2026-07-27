from typing import Tuple, Union

import einops as ei
import torch
import torch.nn as nn
import torch.nn.functional as F

PairEncoderOutput = Union[
    torch.Tensor,  # deterministic latents
    Tuple[torch.Tensor, torch.Tensor],  # gaussian (mean, logvar)
]

# * Legacy encS ckpts: concat source OH into pair_mlp (repeat across A).
# Current/correct: False — append sources after set mean into self.model.
REPEAT_SOURCES_ACROSS_A = False


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
        encode_labels: bool = True,
        encode_sources: bool = False,
        source_dims: list[int] | None = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.segment_len = int(segment_len)
        self.use_conv_traj = self.segment_len > 1
        self.gaussian_latent = bool(gaussian_latent)
        self.encode_labels = bool(encode_labels)
        self.encode_sources = bool(encode_sources)
        if self.encode_sources:
            assert source_dims is not None and len(source_dims) > 0
            self.source_dims = [int(d) for d in source_dims]
            self.source_feat_dim = int(sum(self.source_dims))
        else:
            self.source_dims = []
            self.source_feat_dim = 0

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

        # * REPEAT_SOURCES_ACROSS_A: sources in pair_mlp; else after set mean in self.model
        pair_src = self.source_feat_dim if REPEAT_SOURCES_ACROSS_A else 0
        set_src = 0 if REPEAT_SOURCES_ACROSS_A else self.source_feat_dim
        pair_feat_dim = 3 * hidden_dim + int(self.encode_labels) + pair_src
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        set_in_dim = hidden_dim + set_src
        self.model = nn.Sequential(
            nn.Linear(set_in_dim, hidden_dim),
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
        h = self.traj_encoder(x)
        h = h.mean(dim=-1)
        return ei.rearrange(h, "(B A) H -> B A H", A=A)

    def _encode_sources(self, *, source: torch.Tensor) -> torch.Tensor:
        """(B, n_sources) -> (B, sum(source_dims))."""
        feats = []
        for i, src_dim in enumerate(self.source_dims):
            src_oh = F.one_hot(source[:, i].long(), src_dim).float()
            feats.append(src_oh)
        return torch.cat(feats, dim=-1)

    def _encode_set(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y: torch.Tensor,
        *,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Preference-set trunk: ``(B, A, ...)`` -> ``(B, H)``."""
        h1 = self._encode_traj(s=s1)
        h2 = self._encode_traj(s=s2)
        pair_feats = [h1, h2, h1 - h2]
        if self.encode_labels:
            pair_feats.append(y)
        src: torch.Tensor | None = None
        if self.encode_sources:
            assert source is not None, "encode_sources=True but `source` is None"
            src = self._encode_sources(source=source)
            if REPEAT_SOURCES_ACROSS_A:
                A = int(h1.shape[1])
                pair_feats.append(ei.repeat(src, "B D -> B A D", A=A))
        h_annot = torch.cat(pair_feats, dim=-1)
        h_annot = self.pair_mlp(h_annot)
        h_set = h_annot.mean(dim=1)
        if self.encode_sources and src is not None and not REPEAT_SOURCES_ACROSS_A:
            h_set = torch.cat([h_set, src], dim=-1)
        return self.model(h_set)

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y: torch.Tensor,
        *,
        source: torch.Tensor | None = None,
    ) -> PairEncoderOutput:
        h_ = self._encode_set(s1=s1, s2=s2, y=y, source=source)
        if self.gaussian_latent:
            return self.fc_mean(h_), self.fc_logvar(h_)
        return self.fc_z(h_)
