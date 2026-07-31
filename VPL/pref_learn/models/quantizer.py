import torch
import torch.nn as nn


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for rounding."""
    z_rounded = torch.round(z)
    return z + (z_rounded - z).detach()


class FSQQuantizer(nn.Module):
    """
    Finite-scalar quantization on ``[-1, 1]`` with the same bin count per dim.

    Ported from https://github.com/kylehkhsu/tripod/blob/main/disentangle/nn/quantizer.py
    """

    def __init__(self, num_latents: int, num_values_per_latent: int):
        super().__init__()
        assert num_values_per_latent >= 2
        self.num_latents = int(num_latents)
        self.num_values_per_latent = int(num_values_per_latent)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: ``(..., D)`` continuous pre-quantization latents.

        Returns:
            ``z_c``: bounded continuous latents in ``[-1, 1]``.
            ``z_q``: STE-quantized latents in ``[-1, 1]`` on a fixed grid.
            ``z_bins``: integer bin indices in ``{0, ..., V-1}``.
        """
        # ? ---- activation bounding ----
        z_c = torch.tanh(z)  # FSQ. `*3` is added on, need to change below.
        # z_c = 2.0 * torch.sigmoid(1.6 * z) - 1.0  # iFSQ
        # ? ---- activation bounding ----
        v = self.num_values_per_latent
        z_bins = ((z_c + 1.0) / 2.0) * (v - 1)
        z_bins = round_ste(z_bins)
        z_q = 2.0 * z_bins / (v - 1) - 1.0
        return {"z_c": z_c, "z_q": z_q, "z_bins": z_bins}
