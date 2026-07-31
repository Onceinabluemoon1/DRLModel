"""
FSQ latent autoencoder for preference learning.

Like QLAE (https://github.com/kylehkhsu/latent_quantization/blob/main/disentangle/models/quantized_ae.py)
but uses FSQ straight-through quantization instead of VQ — no quantization or commitment loss.
"""

import einops as ei
import torch
import torch.nn as nn
import torch.nn.functional as F
from pref_learn.models.nets import (
    PairEncoder,
    RewardDecoderConfig,
    make_reward_decoder,
    reward_decoder_column_orth_penalty,
    reward_decoder_theta_sparsity,
)
from pref_learn.models.quantizer import FSQQuantizer


class QLAE(nn.Module):
    """
    Preference-learning autoencoder with finite-scalar quantized latents.

    Loss is Bradley-Terry reconstruction only; FSQ gradients flow via STE.
    """

    discrete_latents: bool = True

    def __init__(
        self,
        encoder_input: int,
        decoder_input: int,
        latent_dim: int,
        hidden_dim: int,
        annotation_size: int,
        segment_len: int,
        obs_dim: int,
        num_quantized_values: int = 10,
        reward_scaling: float = 1.0,
        loss_aggregation: str = "mean",
        decoder_config: RewardDecoderConfig | None = None,
        encode_labels: bool = True,
        encode_sources: bool = False,
        source_dims: list[int] | None = None,
        decoder_orth_weight: float = 0.0,
        decoder_sparsity_weight: float = 0.0,
    ):
        super().__init__()
        assert loss_aggregation in ("sum", "mean")
        if decoder_config is None:
            decoder_config = RewardDecoderConfig.from_train(
                decoder_type="multi",
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                decoder_input=decoder_input,
            )
        self.encode_labels = bool(encode_labels)
        self.encode_sources = bool(encode_sources)
        self.encoder = PairEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            segment_len=segment_len,
            gaussian_latent=False,
            encode_labels=encode_labels,
            encode_sources=encode_sources,
            source_dims=source_dims,
        )
        self.quantizer = FSQQuantizer(
            num_latents=latent_dim,
            num_values_per_latent=num_quantized_values,
        )
        self.decoder = make_reward_decoder(config=decoder_config)
        self.decoder_config = decoder_config
        self.decoder_type = str(decoder_config.decoder_type).lower()
        self.decoder_orth_weight = float(decoder_orth_weight)
        self.decoder_sparsity_weight = float(decoder_sparsity_weight)
        self.latent_dim = latent_dim
        self.annotation_size = annotation_size
        self.segment_len = segment_len
        self.num_quantized_values = int(num_quantized_values)
        self.scaling = reward_scaling
        self.loss_agg = loss_aggregation

    def encode(self, s1: torch.Tensor, s2: torch.Tensor, y: torch.Tensor, source=None):
        """
        Returns quantized latents for downstream use and continuous pre-quant latents.

        Output:
            z_q: (B, D)
            z_c: (B, D)
        """
        z_c = self.encode_continuous(s1=s1, s2=s2, y=y, source=source)
        z_q = self.quantizer(z=z_c)["z_q"]
        return z_q, z_c

    def encode_continuous(
        self, s1: torch.Tensor, s2: torch.Tensor, y: torch.Tensor, source=None
    ) -> torch.Tensor:
        return self.encoder(s1=s1, s2=s2, y=y, source=source)  # (B, D)

    def quantize(self, z_c: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.quantizer(z=z_c)

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        decoder_input = torch.cat([obs, z], dim=-1)
        return self.decoder(decoder_input)

    def _as_batl_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 2:
            return ei.rearrange(obs, "B O -> B 1 1 O")
        if obs.ndim == 3:
            return ei.rearrange(obs, "B T O -> B 1 T O")
        return obs

    def _probabilistic_reward_batch(
        self,
        s_a: torch.Tensor,
        z: torch.Tensor,
        comparison_obs: torch.Tensor,
    ) -> torch.Tensor:
        s_a = self._as_batl_obs(obs=s_a)
        s_b = self._as_batl_obs(obs=comparison_obs)
        m_n, _, t_n, _ = s_a.shape
        n_n = int(s_b.shape[0])

        s_a_pair = ei.repeat(s_a, "M 1 T O -> M N T O", N=n_n)
        s_b_pair = ei.repeat(s_b, "N 1 T O -> M N T O", M=m_n)
        z_pair = ei.repeat(z, "M Z -> M N T Z", N=n_n, T=t_n)

        r_a = self.decode(s_a_pair, z_pair).sum(dim=-2) / self.scaling
        r_b = self.decode(s_b_pair, z_pair).sum(dim=-2) / self.scaling
        p_pref = torch.sigmoid(r_a - r_b)
        return p_pref.mean(dim=1)

    def get_probabilistic_reward(
        self,
        s_a: torch.Tensor,
        z: torch.Tensor,
        comparison_obs: torch.Tensor,
        *,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        if z.shape[0] == 1 and s_a.shape[0] > 1:
            z = ei.repeat(z, "1 Z -> M Z", M=int(s_a.shape[0]))
        m_n = int(s_a.shape[0])
        if m_n <= chunk_size:
            return self._probabilistic_reward_batch(
                s_a=s_a, z=z, comparison_obs=comparison_obs
            )
        chunks: list[torch.Tensor] = []
        for start in range(0, m_n, chunk_size):
            end = min(start + chunk_size, m_n)
            chunks.append(
                self._probabilistic_reward_batch(
                    s_a=s_a[start:end],
                    z=z[start:end],
                    comparison_obs=comparison_obs,
                )
            )
        return torch.cat(chunks, dim=0)

    def reconstruction_loss(self, label, logits) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, label, reduction=self.loss_agg)

    def accuracy(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def latent_metrics(self, z_c: torch.Tensor, z_q: torch.Tensor) -> dict[str, float]:
        return {
            "enc_zc_cross_std": z_c.std(dim=0).mean().item(),
            "enc_zq_cross_std": z_q.std(dim=0).mean().item(),
            "enc_zc_abs": z_c.abs().mean().item(),
            "enc_zq_unique_mean": len(torch.unique(z_q, dim=0)) / max(z_q.shape[0], 1),
        }

    def forward_btl(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y: torch.Tensor,
        source=None,
    ):
        """
        Encode, quantize, and compute Bradley-Terry logits.

        Returns:
            recon_loss, logits, labels, p_hat, z_c, z_q
        """
        z_c = self.encode_continuous(s1=s1, s2=s2, y=y, source=source)  # (B, D)
        z_q = self.quantize(z_c=z_c)["z_q"]  # (B, D)

        z = ei.repeat(z_q, "B D -> B A T D", A=self.annotation_size, T=self.segment_len)
        r0 = self.decode(s1, z)
        r1 = self.decode(s2, z)
        r_hat1 = r0.sum(dim=-2) / self.scaling
        r_hat2 = r1.sum(dim=-2) / self.scaling
        logits = ei.rearrange(r_hat1 - r_hat2, "B A 1 -> (B A) 1")
        p_hat = torch.sigmoid(logits)
        labels = ei.rearrange(y, "B A 1 -> (B A) 1")
        recon_loss = self.reconstruction_loss(label=labels, logits=logits)
        return recon_loss, logits, labels, p_hat, z_c, z_q

    def forward(self, s1, s2, y, source=None):  # (B, A, T, O), (B, A, 1)
        recon_loss, _, labels, p_hat, z_c, z_q = self.forward_btl(
            s1=s1, s2=s2, y=y, source=source
        )
        total_loss = recon_loss
        orth_metrics: dict[str, float] = {}
        if self.decoder_orth_weight > 0.0:
            orth_loss = reward_decoder_column_orth_penalty(decoder=self.decoder)
            total_loss = total_loss + self.decoder_orth_weight * orth_loss
            orth_metrics["decoder_orth_loss"] = orth_loss.item()
        if self.decoder_sparsity_weight > 0.0:
            sparsity_loss = reward_decoder_theta_sparsity(decoder=self.decoder)
            total_loss = total_loss + self.decoder_sparsity_weight * sparsity_loss
            orth_metrics["decoder_sparsity_loss"] = sparsity_loss.item()
        acc = self.accuracy(labels, p_hat)
        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "accuracy": acc.item(),
            "num_quantized_values": self.num_quantized_values,
            **self.latent_metrics(z_c=z_c, z_q=z_q),
            **orth_metrics,
        }
        return total_loss, metrics

    def sample_posterior(self, s1, s2, y, source=None):
        z_q, z_c = self.encode(s1, s2, y, source=source)
        return z_q, z_c, z_q

    def update_posteriors(self, posteriors, biased_latents):
        self.posteriors = posteriors
        self.biased_latents = biased_latents


def _qlae_uses_weight_decay(*, name: str, model: nn.Module) -> bool:
    if name.startswith("quantizer.") or name.endswith(".bias"):
        return False
    # if getattr(model, "decoder_type", None) == "slot" and name.startswith(
    #     "decoder.obs_trunk."
    # ):
    #     return False
    return True


def build_qlae_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    AdamW with Kyle Hsu QLAE-style param groups (``latent_quantization`` ``ae.AE.construct_optimizer``):
    ``weight_decay`` on encoder/decoder weights; no decay on biases or quantizer.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _qlae_uses_weight_decay(name=name, model=model):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
