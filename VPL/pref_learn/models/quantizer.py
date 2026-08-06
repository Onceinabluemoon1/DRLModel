import torch
import torch.nn as nn


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for rounding."""
    z_rounded = torch.round(z)
    return z + (z_rounded - z).detach()


def fsq_per_head_bin_entropy(
    z_bins: torch.Tensor,
    *,
    num_bins: int,
) -> torch.Tensor:
    """Shannon entropy of per-dim FSQ bin histograms; ``(Z,)`` in nats."""
    v = int(num_bins)
    z_idx = z_bins.round().long().clamp(min=0, max=v - 1)
    one_hot = torch.nn.functional.one_hot(z_idx, num_classes=v).to(dtype=z_bins.dtype)
    counts = one_hot.sum(dim=0)  # (Z, V)
    p = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
    return -(p * log_p).sum(dim=-1)


class FSQQuantizer(nn.Module):
    """
    Finite-scalar quantization on ``[-R, R]`` with STE binning.

    Default bounding is ``R·tanh(z)``; set ``use_ifsq=True`` for iFSQ
    ``R·(2σ(αz)-1)`` (``α=1.6`` default; see https://arxiv.org/abs/2601.17124).

    Ported from https://github.com/kylehkhsu/tripod/blob/main/disentangle/nn/quantizer.py
    """

    def __init__(
        self,
        num_latents: int,
        num_values_per_latent: int,
        *,
        bound_half_width: float = 1.0,
        use_ifsq: bool = False,
        ifsq_alpha: float = 1.6,
    ):
        super().__init__()
        assert num_values_per_latent >= 2
        assert float(bound_half_width) > 0.0
        self.num_latents = int(num_latents)
        self.num_values_per_latent = int(num_values_per_latent)
        self.bound_half_width = float(bound_half_width)
        self.use_ifsq = bool(use_ifsq)
        self.ifsq_alpha = float(ifsq_alpha)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: ``(..., D)`` continuous pre-quantization latents.

        Returns:
            ``z_c``: bounded continuous latents in ``[-R, R]``.
            ``z_q``: STE-quantized latents on the ``V``-level grid in ``[-R, R]``.
            ``z_bins``: integer bin indices in ``{0, ..., V-1}``.
        """
        R = self.bound_half_width
        v = self.num_values_per_latent
        if self.use_ifsq:
            z_c = R * (2.0 * torch.sigmoid(self.ifsq_alpha * z) - 1.0)
        else:
            z_c = R * torch.tanh(z)
        z_bins = ((z_c + R) / (2.0 * R)) * (v - 1)
        z_bins = round_ste(z_bins)
        z_q = 2.0 * R * z_bins / (v - 1) - R
        return {"z_c": z_c, "z_q": z_q, "z_bins": z_bins}
