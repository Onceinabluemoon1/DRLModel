import einops as ei
import torch
import torch.nn as nn


class LatentEncoder(nn.Module):
    """Preference-set encoder to continuous latents pre-quantization."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_z(self.model(x))


class PairEncoderMLP(nn.Module):
    """(B, A, T, O), (B, A, T, O), (B, A, 1) -> (B, Z), (B, Z)"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(PairEncoderMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s1_ = ei.rearrange(s1, "B A T O -> B A (T O)")
        s2_ = ei.rearrange(s2, "B A T O -> B A (T O)")
        encoder_input = torch.cat([s1_, s2_, y], dim=-1)  # (B, A, (2*T*O + 1))
        x = ei.rearrange(encoder_input, "B A feats -> B (A feats)")

        h_ = self.model(x)
        mean = self.FC_mean(h_)
        logvar = self.FC_logvar(h_)

        return mean, logvar


class PairEncoder1DConv(nn.Module):
    """Siamese 1D-conv pair encoder: shared trunk on each traj, mean over T and A.

    (B, A, T, O), (B, A, T, O), (B, A, 1) -> (B, Z), (B, Z)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        latent_dim: int,
        conv_channels: int = 64,
    ):
        super().__init__()
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
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # * siamese encode each segment
        h1 = self._encode_traj(s=s1)  # (B,A,T,O) -> (B,A,H)
        h2 = self._encode_traj(s=s2)  # (B,A,T,O) -> (B,A,H)

        # * fuse label per annotation, then mean pool over A (not mean over raw y)
        h_annot = torch.cat([h1, h2, h1 - h2, y], dim=-1)  # (B, A, 3H+1)
        h_annot = self.pair_mlp(h_annot)  # (B, A, H)
        h_set = h_annot.mean(dim=1)  # (B, H)

        h_ = self.model(h_set)
        mean = self.fc_mean(h_)
        logvar = self.fc_logvar(h_)
        return mean, logvar


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
