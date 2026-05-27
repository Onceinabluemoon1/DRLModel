import math
import os
import pickle

import einops as ei
import numpy as np
import torch
import torch.nn as nn
from pref_learn.models.flow import Flow


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = self.model(x)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var


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


class VAEModel(nn.Module):
    discrete_latents: bool = False

    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        segment_len,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        discrete_latents: bool = False,
    ):
        super(VAEModel, self).__init__()
        self.discrete_latents = bool(discrete_latents)
        self.Encoder = Encoder(encoder_input, hidden_dim, latent_dim)
        self.Decoder = Decoder(decoder_input, hidden_dim, 1)
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.log_var = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.annotation_size = annotation_size
        self.segment_len = segment_len
        self.learned_prior = learned_prior

        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, "radial", 4)

        self.kl_weight = kl_weight
        self.annealer = annealer
        self.scaling = reward_scaling

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def encode(self, s1, s2, y):
        """
        Input:
            traj1: (B, A, T, O)
            traj2: (B, A, T, O)
            y: (B, A, 1)
        Output:
            mean: (B, Z)
            log_var: (B, Z)
        """

        B, A, T, O = s1.shape
        s1_ = ei.rearrange(s1, "B A T O -> B A (T O)")
        s2_ = ei.rearrange(s2, "B A T O -> B A (T O)")
        y = ei.rearrange(y, "B A 1 -> B A 1")

        encoder_input = torch.cat([s1_, s2_, y], dim=-1)  # (B, A, (2*T*O + 1))
        encoder_input = encoder_input.view(B, -1)  # (B, A*(2*T*O + 1))
        mean, log_var = self.Encoder(encoder_input)  # (B, Z), (B, Z)
        return mean, log_var

    def decode(self, obs, z):
        decoder_input = torch.cat([obs, z], dim=-1)  # (B, A, T, O + Z)
        r = self.Decoder(decoder_input)  # (B, A, T, 1)
        return r

    def _as_batl_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """``(..., O)`` -> ``(B, 1, T, O)`` for BTL segment decoding."""
        if obs.ndim == 2:
            return obs.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, O)
        if obs.ndim == 3:
            return obs.unsqueeze(1)  # (B, 1, T, O)
        return obs  # (B, A, T, O)

    def _broadcast_z_for_decode(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """``z: (B, Z)`` -> ``(B, A, T, Z)`` matching ``obs``."""
        if z.ndim != 2:
            return z
        _, a_n, t_n, _ = obs.shape
        return ei.repeat(z, "B Z -> B A T Z", A=a_n, T=t_n)

    def segment_return(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """BTL segment score ``sum_t r(s, t, z) / scaling`` with shape ``(B, A, 1)``."""
        obs = self._as_batl_obs(obs=obs)
        z = self._broadcast_z_for_decode(z=z, obs=obs)
        r = self.decode(obs, z)  # (B, A, T, 1)
        return r.sum(dim=-2) / self.scaling  # (B, A, 1)

    def preference_probability(
        self,
        s_a: torch.Tensor,
        s_b: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """``p(y=1 | s_A, s_B, z) = sigmoid(R(s_A, z) - R(s_B, z))``; shape ``(B, A, 1)``."""
        r_a = self.segment_return(obs=s_a, z=z)  # (B, A, 1)
        r_b = self.segment_return(obs=s_b, z=z)
        return torch.sigmoid(r_a - r_b)

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
            z = z.expand(s_a.shape[0], -1)
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
        r = self.Decoder(x)  # (B, A, T, 1)
        return r

    def transform(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def latent_loss(self, mean, log_var):
        if self.learned_prior:
            kl = -torch.sum(
                1
                + (log_var - self.log_var)
                - (log_var - self.log_var).exp()
                - (mean - self.mean).pow(2) / (self.log_var.exp())
            )
        else:
            kl = -torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp())
        return kl

    def forward(self, s1, s2, y):  # (B, A, T, O), (B, A, 1)
        mean, log_var = self.encode(s1, s2, y)  # (B, Z), (B, Z)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # (B, Z)
            log_det = None
        # z: broadcast same latent to each (annotation, segment) pair
        z = ei.repeat(z, "B Z -> B A T Z", A=self.annotation_size, T=self.segment_len)

        # bradley-terry likelihood
        r0 = self.decode(s1, z)  # (B, A, T, O+Z) -> (B, A, T, 1)
        r1 = self.decode(s2, z)  # (B, A, T, O+Z) -> (B, A, T, 1)
        r_hat1 = r0.sum(axis=2) / self.scaling  # (B, A, 1)
        r_hat2 = r1.sum(axis=2) / self.scaling  # (B, A, 1)
        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2).view(-1, 1)  # (B*A, 1)
        labels = y.view(-1, 1)  # (B, A, 1) -> (B*A, 1)

        recon_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        kl_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        total_loss = recon_loss + kl_weight * kl_loss

        if self.flow_prior:
            total_loss = total_loss - torch.sum(log_det)

        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kld_loss": kl_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
            # diagnostics
            "ratio_recon_kld": recon_loss.item() / kl_loss.item(),
        }

        return total_loss, metrics

    def sample_prior(self, size):
        z = torch.randn(size, self.latent_dim).to(next(self.parameters()).device)
        if self.learned_prior:
            z = z * torch.exp(0.5 * self.log_var) + self.mean
        elif self.flow_prior:
            z, _ = self.flow(z)
        return z

    def sample_posterior(self, s1, s2, y):
        mean, log_var = self.encode(s1, s2, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return mean, log_var, z

    def update_posteriors(self, posteriors, biased_latents):
        self.posteriors = posteriors
        self.biased_latents = biased_latents


class VAEClassifier(VAEModel):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        segment_len,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        discrete_latents: bool = False,
    ):
        super(VAEClassifier, self).__init__(
            encoder_input,
            decoder_input,
            latent_dim,
            hidden_dim,
            annotation_size,
            segment_len,
            kl_weight,
            learned_prior,
            flow_prior,
            annealer,
            reward_scaling,
            discrete_latents=discrete_latents,
        )

    def forward(self, s1, s2, y):  # (B, A, T, O), (B, A, 1)
        mean, log_var = self.encode(s1, s2, y)  # (B, Z), (B, Z)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # (B, Z)
            log_det = None
        # z: broadcast same latent to each (annotation, segment) pair
        z = ei.repeat(z, "B Z -> B A T Z", A=self.annotation_size, T=self.segment_len)

        decoder_input = torch.cat([s1, s2, z], dim=-1)  # (B, A, T, 2*O + Z)
        p_hat = self.Decoder(decoder_input).view(-1, 1)  # (B, A, T, 1) -> (B*A*T, 1)
        p_hat = torch.nn.functional.sigmoid(p_hat).view(-1, 1)  # (B*A*T, 1)
        labels = y.view(-1, 1)  # (B, A, T) -> (B*A*T, 1)

        recon_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        kl_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        total_loss = recon_loss + kl_weight * kl_loss

        if self.flow_prior:
            total_loss = total_loss - torch.sum(log_det)

        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kld_loss": kl_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
            # diagnostics
            "ratio_recon_kld": recon_loss.item() / kl_loss.item(),
            "frac_recon2total": recon_loss.item() / total_loss.item(),
        }

        return total_loss, metrics

    def decode(self, x, y, z):  # B x S, N x S, B x Z
        x = x[:, None].repeat(1, y.shape[0], 1)  # B x N x S
        z = z[:, None].repeat(1, y.shape[0], 1)  # B x N x Z
        y = y[None].repeat(x.shape[0], 1, 1)  # B x N x S
        x = torch.cat([x, y, z], dim=-1)  # B x N x (2S + Z)
        x = torch.nn.functional.sigmoid(self.Decoder(x))  # B x N x 1
        return x[:, :, 0].mean(dim=-1)  # (B, )
