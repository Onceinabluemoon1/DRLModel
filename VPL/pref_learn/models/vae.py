import einops as ei
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from pref_learn.models.flow import Flow
from pref_learn.models.nets import (
    PairEncoder,
    make_reward_decoder,
    reward_decoder_column_orth_penalty,
)
from pref_learn.models.utils import Annealer, logvar_to_std


class VAEModel(nn.Module):
    discrete_latents: bool = False

    def __init__(
        self,
        encoder_input,  # legacy flat size A*(2*T*O+1); MLP path uses obs_dim*segment_len
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        segment_len,
        obs_dim: int,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        discrete_latents: bool = False,
        loss_aggregation: str = "mean",
        normalize_kl: bool = True,
        decoder_type: str = "mlp",
        decoder_gate_temperature: float = 1.0,
        encoder_use_labels: bool = True,
        decoder_orth_weight: float = 0.0,
    ):
        super(VAEModel, self).__init__()
        assert loss_aggregation in ("sum", "mean")
        self.discrete_latents = bool(discrete_latents)
        self.encoder_use_labels = bool(encoder_use_labels)
        self.encoder = PairEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            segment_len=segment_len,
            gaussian_latent=True,
            encoder_use_labels=encoder_use_labels,
        )
        self.decoder = make_reward_decoder(
            decoder_type=decoder_type,
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            decoder_input=decoder_input,
            gate_temperature=decoder_gate_temperature,
        )
        self.decoder_type = str(decoder_type).lower()
        self.decoder_orth_weight = float(decoder_orth_weight)
        self.latent_dim = latent_dim
        self.prior_mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.prior_logvar = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.annotation_size = annotation_size
        self.segment_len = segment_len
        self.learned_prior = learned_prior

        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, "radial", 4)

        self.kl_weight = kl_weight
        self.annealer: Annealer = annealer
        self.scaling = reward_scaling
        self.loss_agg = loss_aggregation
        self.normalize_kl = normalize_kl

    def encode(self, s1, s2, y):
        """
        Input:
            traj1: (B, A, T, O)
            traj2: (B, A, T, O)
            y: (B, A, 1)
        Output:
            mean: (B, Z)
            logvar: (B, Z)
        """
        mean, logvar = self.encoder(s1=s1, s2=s2, y=y)  # (B, Z), (B, Z)
        return mean, logvar

    def decode(self, obs, z) -> torch.Tensor:
        """
        per-step reward decoding conditioned on latent z
        z typically broadcasted as repeat(z, "B Z -> B A T Z")
        (B, A, T, O + Z) -> (B, A, T, 1)
        """
        decoder_input = torch.cat([obs, z], dim=-1)  # (B, A, T, O + Z)
        r = self.decoder(decoder_input)  # (B, A, T, 1)
        return r

    def _as_batl_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """``(..., O)`` -> ``(B, 1, T, O)`` for BTL segment decoding."""
        if obs.ndim == 2:
            return ei.rearrange(obs, "B O -> B 1 1 O")
        if obs.ndim == 3:
            return ei.rearrange(obs, "B T O -> B 1 T O")
        return obs  # (B, A, T, O)

    def _probabilistic_reward_batch(
        self,
        s_a: torch.Tensor,
        z: torch.Tensor,
        comparison_obs: torch.Tensor,
    ) -> torch.Tensor:
        """``E_{s_B}[p(y=1 | s_A, s_B, z)]`` for one aligned batch."""
        s_a = self._as_batl_obs(obs=s_a)  # (M, 1, T, O)
        s_b = self._as_batl_obs(obs=comparison_obs)  # (N, 1, T, O)
        m_n, _, t_n, _ = s_a.shape
        n_n = int(s_b.shape[0])

        s_a_pair = ei.repeat(s_a, "M 1 T O -> M N T O", N=n_n)
        s_b_pair = ei.repeat(s_b, "N 1 T O -> M N T O", M=m_n)
        z_pair = ei.repeat(z, "M Z -> M N T Z", N=n_n, T=t_n)

        r_a = self.decode(s_a_pair, z_pair).sum(dim=-2) / self.scaling  # (M, N, 1)
        r_b = self.decode(s_b_pair, z_pair).sum(dim=-2) / self.scaling
        p_pref = torch.sigmoid(r_a - r_b)  # (M, N, 1)
        return p_pref.mean(dim=1)  # (M, 1)

    def get_probabilistic_reward(
        self,
        s_a: torch.Tensor,
        z: torch.Tensor,
        comparison_obs: torch.Tensor,
        *,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """
        Scaled reward from VPL Sec. 4.1:
        ``r(s_A, z) = E_{s_B}[p(y=1 | s_A, s_B, z)]``.

        Args:
            s_a: ``(M, O)`` or ``(M, T, O)`` query states / segments.
            z: ``(M, Z)`` latents aligned with ``s_a`` (or ``(1, Z)`` broadcast).
            comparison_obs: ``(N, O)`` or ``(N, T, O)`` pool of states from data.
        """
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
        return torch.cat(chunks, dim=0)  # (M, 1)

    def get_reward(self, x):
        r = self.decoder(x)  # (B, A, T, 1)
        return r

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std).to(mean.device)
        z = mean + std * epsilon
        return z

    def transform_flow(self, mean, std) -> torch.Tensor:
        """Transform mean and std to z using flow model."""
        epsilon = torch.randn_like(std).to(mean.device)
        z = mean + std * epsilon
        return self.flow(z)

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    @property
    def kl_normalizer(self) -> float:
        """
        D / A makes sense when latent loss is averaged over D
        1 / A makes sense when latent loss is summed over D
        """
        if self.normalize_kl:
            return self.latent_dim / self.annotation_size
        return 1.0

    def reconstruction_loss(self, label, logits):
        """
        label: ((B A), 1)
        logits: ((B A), 1), Bradley-Terry logit r(s1) - r(s2)

        reduce over B * A queries
        """
        return F.binary_cross_entropy_with_logits(logits, label, reduction=self.loss_agg)

    def latent_loss(self, mean, logvar):
        """
        KL(q(z|x) || p(z))

        mean: (B, Z), note that posterior takes a set of annotations
        logvar: (B, Z)

        reduction: sum or mean over latent dims, mean over batch

        """
        # * KL(q(z|x) || p(z))
        prior_mean = self.prior_mean if self.learned_prior else 0.0
        prior_std = logvar_to_std(self.prior_logvar) if self.learned_prior else 1.0
        p_z = D.Normal(prior_mean, prior_std)
        q_z_giv_x = D.Normal(mean, logvar_to_std(logvar))

        kl_per_sample = D.kl_divergence(q_z_giv_x, p_z).mean(dim=-1)  # (B,Z) -> (B,)
        kl = ei.reduce(kl_per_sample, "b ->", self.loss_agg)

        #! old Implementation was missing a 1/2 factor, so KL is 2x the above
        # if self.learned_prior:
        #     kl = -torch.sum(
        #         1
        #         + (logvar - self.prior_logvar)
        #         - (logvar - self.prior_logvar).exp()
        #         - (mean - self.prior_mean).pow(2) / (self.prior_logvar.exp())
        #     )
        # else:
        #     kl = -torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp())
        return kl

    def encoder_cross_std_metrics(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> dict[str, float]:
        """
        Cross-sample spread of ``q(z|x)`` params; low values indicate collapsed latents.

        mean, logvar: (B, D)
        """
        return {
            "enc_mean_cross_std": mean.std(dim=0).mean().item(),
            "enc_logvar_cross_std": logvar.std(dim=0).mean().item(),
            "enc_mean_abs": mean.abs().mean().item(),
        }

    def forward(self, s1, s2, y):  # (B, A, T, O), (B, A, 1)
        mean, logvar = self.encode(s1, s2, y)  # (B, Z), (B, Z)
        std = logvar_to_std(logvar)

        if self.flow_prior:
            z, log_det = self.transform_flow(mean, std)
        else:
            z = self.reparameterization(mean, std)  # (B, Z)
            log_det = None

        # bradley-terry likelihood
        z = ei.repeat(z, "B Z -> B A T Z", A=self.annotation_size, T=self.segment_len)
        r0 = self.decode(s1, z)  # (B, A, T, O+Z) -> (B, A, T, 1)
        r1 = self.decode(s2, z)  # (B, A, T, O+Z) -> (B, A, T, 1)
        r_hat1 = r0.sum(dim=-2) / self.scaling  # (B, A, 1)
        r_hat2 = r1.sum(dim=-2) / self.scaling  # (B, A, 1)
        logits = ei.rearrange(r_hat1 - r_hat2, "B A 1 -> (B A) 1")
        p_hat = torch.sigmoid(logits)
        labels = ei.rearrange(y, "B A 1 -> (B A) 1")

        # loss computation
        recon_loss = self.reconstruction_loss(label=labels, logits=logits)
        kl_loss = self.latent_loss(mean, logvar)

        kl_scale = self.annealer.slope() if self.annealer else 1.0
        kl_weight = kl_scale * self.kl_weight * self.kl_normalizer
        total_loss = recon_loss + kl_weight * kl_loss
        if self.flow_prior:
            total_loss = total_loss - ei.reduce(log_det, "b 1 ->", self.loss_agg)
        orth_metrics: dict[str, float] = {}
        if self.decoder_orth_weight > 0.0:
            orth_loss = reward_decoder_column_orth_penalty(decoder=self.decoder)
            total_loss = total_loss + self.decoder_orth_weight * orth_loss
            orth_metrics["decoder_orth_loss"] = orth_loss.item()

        acc = self.accuracy(labels, p_hat)
        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kld_loss": kl_loss.item(),
            "accuracy": acc.item(),
            "kl_weight": kl_weight,
            "frac_recon2total": recon_loss.item() / total_loss.item(),
            **self.encoder_cross_std_metrics(mean=mean, logvar=logvar),
            **orth_metrics,
        }

        return total_loss, metrics

    def sample_prior(self, size):
        dev = next(self.parameters()).device
        eps = torch.randn(size, self.latent_dim).to(dev)
        if self.learned_prior:
            std = logvar_to_std(self.prior_logvar)
            z = eps * std + self.prior_mean
        elif self.flow_prior:
            z, _ = self.flow(eps)
        else:
            z = eps
        return z

    def sample_posterior(self, s1, s2, y):
        mean, logvar = self.encode(s1, s2, y)
        z = self.reparameterization(mean, logvar_to_std(logvar))
        return mean, logvar, z

    def update_posteriors(self, posteriors, biased_latents):
        self.posteriors = posteriors
        self.biased_latents = biased_latents


# TODO: remove this class
class VAEClassifier(VAEModel):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        segment_len,
        obs_dim: int,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        discrete_latents: bool = False,
        loss_aggregation: str = "sum",
        normalize_kl: bool = True,
    ):
        super(VAEClassifier, self).__init__(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            annotation_size=annotation_size,
            segment_len=segment_len,
            obs_dim=obs_dim,
            kl_weight=kl_weight,
            learned_prior=learned_prior,
            flow_prior=flow_prior,
            annealer=annealer,
            reward_scaling=reward_scaling,
            discrete_latents=discrete_latents,
            loss_aggregation=loss_aggregation,
            normalize_kl=normalize_kl,
        )

    def forward(self, s1, s2, y):  # (B, A, T, O), (B, A, 1)
        mean, logvar = self.encode(s1, s2, y)  # (B, Z), (B, Z)
        std = logvar_to_std(logvar)

        if self.flow_prior:
            z, log_det = self.transform_flow(mean, std)
        else:
            z = self.reparameterization(mean, std)  # (B, Z)
            log_det = None
        # z: broadcast same latent to each (annotation, segment) pair
        z = ei.repeat(z, "B Z -> B A T Z", A=self.annotation_size, T=self.segment_len)

        dec_input = torch.cat([s1, s2, z], dim=-1)  # (B, A, T, 2*O+Z)
        logits = ei.rearrange(self.decoder(dec_input), "B A T 1 -> (B A T) 1")
        p_hat = torch.sigmoid(logits)
        labels = ei.rearrange(y, "B A 1 -> (B A) 1")

        recon_loss = self.reconstruction_loss(label=labels, logits=logits)
        latent_loss = self.latent_loss(mean, logvar)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        kl_weight = kl_weight * self.kl_normalizer
        total_loss = recon_loss + kl_weight * latent_loss

        if self.flow_prior:
            total_loss = total_loss - ei.reduce(log_det, "b 1 ->", self.loss_agg)

        acc = self.accuracy(labels, p_hat)
        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "latent_loss": latent_loss.item(),
            "accuracy": acc.item(),
            "kl_weight": kl_weight,
            "frac_recon2total": recon_loss.item() / total_loss.item(),
            **self.encoder_cross_std_metrics(mean=mean, logvar=logvar),
        }

        return total_loss, metrics

    def decode(self, x, y, z):  # B x S, N x S, B x Z
        n_n = int(y.shape[0])
        b_n = int(x.shape[0])
        x = ei.repeat(x, "B S -> B N S", N=n_n)
        z = ei.repeat(z, "B Z -> B N Z", N=n_n)
        y = ei.repeat(y, "N S -> B N S", B=b_n)
        x = torch.cat([x, y, z], dim=-1)  # (B, N, 2S + Z)
        x = F.sigmoid(self.decoder(x))  # (B, N, 1)
        return x[:, :, 0].mean(dim=-1)  # (B,)
