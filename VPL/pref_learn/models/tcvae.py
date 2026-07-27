import math

import einops as ei
import torch
import torch.distributions as D
from pref_learn.models.nets import (
    reward_decoder_column_orth_penalty,
    reward_decoder_theta_sparsity,
)
from pref_learn.models.utils import logvar_to_std
from pref_learn.models.vae import VAEModel
from torch import logsumexp as lse


class TCVAEModel(VAEModel):
    """
    β-TC-VAE reward model.

    Decomposes the ELBO KL term into mutual information (MI), total correlation
    (TC), and dimension-wise KL (DWKL), following Chen et al. and
    https://github.com/kylehkhsu/latent_quantization/blob/main/disentangle/models/tcvae.py
    """

    def __init__(
        self,
        encoder_input: int,
        decoder_input: int,
        latent_dim: int,
        hidden_dim: int,
        annotation_size: int,
        segment_len: int,
        obs_dim: int,
        dataset_size: int,
        tc_weight: float = 1.0,
        mi_weight: float = 1.0,
        dwkl_weight: float = 1.0,
        learned_prior: bool = False,
        flow_prior: bool = False,
        annealer=None,
        reward_scaling: float = 1.0,
        discrete_latents: bool = False,
        loss_aggregation: str = "mean",
        normalize_kl: bool = True,
        decoder_type: str = "mlp",
        decoder_gate_temperature: float = 1.0,
        decoder_slot_routing: str = "softmax",
        encode_labels: bool = True,
        encode_sources: bool = False,
        source_dims: list[int] | None = None,
        decoder_orth_weight: float = 0.0,
        decoder_sparsity_weight: float = 0.0,
        decoder_sparsity_mode: str = "elementwise",
        decoder_softplus_beta: float = 1.0,
    ):
        if flow_prior:
            raise NotImplementedError("TCVAEModel does not support flow_prior.")
        super().__init__(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            annotation_size=annotation_size,
            segment_len=segment_len,
            obs_dim=obs_dim,
            kl_weight=1.0,
            learned_prior=learned_prior,
            flow_prior=False,
            annealer=annealer,
            reward_scaling=reward_scaling,
            discrete_latents=discrete_latents,
            loss_aggregation=loss_aggregation,
            normalize_kl=normalize_kl,
            decoder_type=decoder_type,
            decoder_gate_temperature=decoder_gate_temperature,
            decoder_slot_routing=decoder_slot_routing,
            encode_labels=encode_labels,
            encode_sources=encode_sources,
            source_dims=source_dims,
            decoder_orth_weight=decoder_orth_weight,
            decoder_sparsity_weight=decoder_sparsity_weight,
            decoder_sparsity_mode=decoder_sparsity_mode,
            decoder_softplus_beta=decoder_softplus_beta,
        )
        self.dataset_size = int(dataset_size)
        self.tc_weight = tc_weight
        self.mi_weight = mi_weight
        self.dwkl_weight = dwkl_weight

    @property
    def kl_normalizer(self) -> float:
        """
        Scale latent terms vs mean-over-(B,A) recon.

        MI/TC/DWKL already sum over latent dim D (β-TC-VAE convention), so only
        compensate for one posterior per batch row vs `A` preference pairs — not D/A.
        """
        if self.normalize_kl:
            return 1.0 / self.annotation_size
        return 1.0

    @staticmethod
    def log_q_z_given_x_perdim(
        z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """``log q(z|x)`` per latent dim; shapes ``(..., D)``."""
        return D.Normal(mu, sigma).log_prob(z)

    def log_p_z(self, z: torch.Tensor) -> torch.Tensor:
        """``log p(z)`` summed over dims; ``(B, D)`` -> ``(B,)``."""
        prior_mean = self.prior_mean if self.learned_prior else 0.0
        prior_std = logvar_to_std(self.prior_logvar) if self.learned_prior else 1.0
        p_z = D.Normal(prior_mean, prior_std)
        return p_z.log_prob(z).sum(-1)

    def tcvae_latent_loss(
        self,
        z: torch.Tensor,  # (B, D)
        mu: torch.Tensor,  # (B, D)
        logvar: torch.Tensor,  # (B, D)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Minibatch estimate of β-TC-VAE latent terms.

        Returns aggregated MI, TC, and DWKL (sum or mean over batch per ``loss_aggregation``).
        i := encoder latent sample index (B)
        n := dataset sample for aggregate posterior q(z) (estimated using B)
        j := latent dimension index (D)
        """
        std = logvar_to_std(logvar)  # (B, D)
        N, B, D = self.dataset_size, z.shape[0], z.shape[1]
        log_norm = math.log(B * N)  # bs * N

        # log q(z_i,j | x_n): (B, B, D), pairwise log density, per latent dim
        log_q_zi_giv_xn_perdim = self.log_q_z_given_x_perdim(
            z=ei.rearrange(z, "B D -> B 1 D"),
            mu=ei.rearrange(mu, "B D -> 1 B D"),
            sigma=ei.rearrange(std, "B D -> 1 B D"),
        )
        idx = torch.arange(z.shape[0], device=z.device)
        log_q_z_giv_x_perdim = log_q_zi_giv_xn_perdim[idx, idx]  # (B, D)
        log_q_z_giv_x = log_q_z_giv_x_perdim.sum(-1)  # (B,)

        # * MWS: minibatch weighted sampling
        log_q_z = lse(log_q_zi_giv_xn_perdim.sum(-1), 1) - log_norm  # (B,)
        log_prodk_q_zk = (lse(log_q_zi_giv_xn_perdim, 1) - log_norm).sum(-1)  # (B,)
        log_p_z = self.log_p_z(z=z)  # (B,)

        mi_per_sample = log_q_z_giv_x - log_q_z  # (B,)
        tc_per_sample = log_q_z - log_prodk_q_zk  # (B,)
        dwkl_per_sample = log_prodk_q_zk - log_p_z  # (B,)
        mi_loss = ei.reduce(mi_per_sample, "b ->", self.loss_agg)
        tc_loss = ei.reduce(tc_per_sample, "b ->", self.loss_agg)
        dwkl_loss = ei.reduce(dwkl_per_sample, "b ->", self.loss_agg)
        return mi_loss, tc_loss, dwkl_loss

    def forward(self, s1, s2, y, source=None):  # (B, A, T, O), (B, A, 1)
        mean, logvar = self.encode(s1, s2, y, source=source)  # (B, D), (B, D)
        z = self.reparameterization(mean=mean, std=logvar_to_std(logvar))  # (B, D)
        N, B, D = self.dataset_size, mean.shape[0], mean.shape[1]

        # * Bradley-Terry likelihood
        z_dec = ei.repeat(z, "B Z -> B A T Z", A=self.annotation_size, T=self.segment_len)
        r0 = self.decode(s1, z_dec)  # (B, A, T, O+Z) -> (B, A, T, 1)
        r1 = self.decode(s2, z_dec)  # (B, A, T, O+Z) -> (B, A, T, 1)
        r_hat1 = r0.sum(-2) / self.scaling  # (B, A, 1)
        r_hat2 = r1.sum(-2) / self.scaling  # (B, A, 1)
        logits = ei.rearrange(r_hat1 - r_hat2, "B A 1 -> (B A) 1")
        p_hat = torch.sigmoid(logits)
        labels = ei.rearrange(y, "B A 1 -> (B A) 1")

        # loss computation
        recon_loss = self.reconstruction_loss(label=labels, logits=logits)
        mi_loss, tc_loss, dwkl_loss = self.tcvae_latent_loss(z, mean, logvar)

        tc_scale = self.annealer.slope() if self.annealer else 1.0
        tc_weight = self.tc_weight * tc_scale
        latent_loss = self.kl_normalizer * (
            self.mi_weight * mi_loss + tc_weight * tc_loss + self.dwkl_weight * dwkl_loss
        )
        total_loss = recon_loss + latent_loss
        orth_metrics: dict[str, float] = {}
        if self.decoder_orth_weight > 0.0:
            orth_loss = reward_decoder_column_orth_penalty(decoder=self.decoder)
            total_loss = total_loss + self.decoder_orth_weight * orth_loss
            orth_metrics["decoder_orth_loss"] = orth_loss.item()
        if self.decoder_sparsity_weight > 0.0:
            sparsity_loss = reward_decoder_theta_sparsity(decoder=self.decoder)
            total_loss = total_loss + self.decoder_sparsity_weight * sparsity_loss
            orth_metrics["decoder_sparsity_loss"] = sparsity_loss.item()
        # latent_loss mean-pools over D → VAEModel-style D/A scale, not β-TC 1/A
        vanilla_kld_loss = self.latent_loss(mean, logvar)  # KL(q(z|x) || p(z))
        vanilla_kl_norm = (
            self.kl_weight * self.latent_dim / self.annotation_size
            if self.normalize_kl
            else self.kl_weight
        )
        vanilla_elbo_loss = recon_loss + vanilla_kl_norm * vanilla_kld_loss

        acc = self.accuracy(labels, p_hat)
        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "mi_loss": mi_loss.item(),
            "tc_loss": tc_loss.item(),
            "dwkl_loss": dwkl_loss.item(),
            "latent_loss": latent_loss.item(),
            "accuracy": acc.item(),
            "tc_weight": tc_weight,
            "mi_weight": self.mi_weight,
            "dwkl_weight": self.dwkl_weight,
            # * diagnostic
            "frac_recon2total": recon_loss.item() / total_loss.item(),
            "frac_mi2bound": mi_loss.item() / math.log(N),
            "frac_tc2bound": tc_loss.item() / (math.log(N) * (D - 1)),
            # vanilla elbo metrics
            "vanilla_kld_loss": vanilla_kld_loss.item(),
            "vanilla_elbo_loss": vanilla_elbo_loss.item(),
            "vanilla_kl_normalizer": vanilla_kl_norm,
            "vanilla_frac_recon2total": recon_loss.item() / vanilla_elbo_loss.item(),
            **self.encoder_cross_std_metrics(mean=mean, logvar=logvar),
            **orth_metrics,
        }

        return total_loss, metrics
