from typing import Tuple, Union

import einops as ei
import torch
import torch.nn as nn

PairEncoderOutput = Union[
    torch.Tensor,  # deterministic latents
    Tuple[torch.Tensor, torch.Tensor],  # gaussian latents (mean, logvar)
]


class PairEncoderMLP(nn.Module):
    """(B, A, T, O), (B, A, T, O), (B, A, 1) -> (B, Z) or (B, Z), (B, Z)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        *,
        gaussian_latent: bool = True,
    ):
        super().__init__()
        self.gaussian_latent = bool(gaussian_latent)

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
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

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y: torch.Tensor,
    ) -> PairEncoderOutput:
        s1_ = ei.rearrange(s1, "B A T O -> B A (T O)")
        s2_ = ei.rearrange(s2, "B A T O -> B A (T O)")
        encoder_input = torch.cat([s1_, s2_, y], dim=-1)  # (B, A, (2*T*O + 1))
        x = ei.rearrange(encoder_input, "B A feats -> B (A feats)")
        h_ = self.model(x)  # (B, H)
        if self.gaussian_latent:
            return self.fc_mean(h_), self.fc_logvar(h_)
        return self.fc_z(h_)  # (B, Z)


class PairEncoder1DConv(nn.Module):
    """Siamese 1D-conv pair encoder: shared trunk on each traj, mean over T and A.

    (B, A, T, O), (B, A, T, O), (B, A, 1) -> (B, Z) or (B, Z), (B, Z)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        latent_dim: int,
        conv_channels: int = 64,
        *,
        gaussian_latent: bool = True,
    ):
        super().__init__()
        self.gaussian_latent = bool(gaussian_latent)
        self.traj_conv = nn.Sequential(
            nn.Conv1d(obs_dim, conv_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(conv_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        pair_feat_dim = 3 * hidden_dim + 1
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
        A = s.shape[1]
        x = ei.rearrange(s, "B A T O -> (B A) O T")
        h = self.traj_conv(x)  # (B*A, H, T)
        h = h.mean(dim=-1)  # (B*A, H)
        h = ei.rearrange(h, "(B A) H -> B A H", A=A)
        return h

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y: torch.Tensor,
    ) -> PairEncoderOutput:
        # * siamese encode each segment
        h1 = self._encode_traj(s=s1)  # (B,A,T,O) -> (B,A,H)
        h2 = self._encode_traj(s=s2)  # (B,A,T,O) -> (B,A,H)

        # * fuse label per annotation, then mean pool over A (not mean over raw y)
        h_annot = torch.cat([h1, h2, h1 - h2, y], dim=-1)  # (B, A, 3H+1)
        h_annot = self.pair_mlp(h_annot)  # (B, A, H)
        h_set = h_annot.mean(dim=1)  # (B, H)

        h_ = self.model(h_set)
        if self.gaussian_latent:
            return self.fc_mean(h_), self.fc_logvar(h_)
        return self.fc_z(h_)  # (B, Z)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x_hat = self.model(x)
        return x_hat
