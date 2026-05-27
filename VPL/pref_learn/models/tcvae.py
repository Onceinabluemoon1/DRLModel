import einops as ei
import torch
from pref_learn.models.vae import VAEModel
from torch.distributions import Normal


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
        dataset_size: int,
        tc_weight: float = 1.0,
        mi_weight: float = 1.0,
        dwkl_weight: float = 1.0,
        learned_prior: bool = False,
        flow_prior: bool = False,
        annealer=None,
        reward_scaling: float = 1.0,
        discrete_latents: bool = False,
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
            kl_weight=1.0,
            learned_prior=learned_prior,
            flow_prior=False,
            annealer=annealer,
            reward_scaling=reward_scaling,
            discrete_latents=discrete_latents,
        )
        self.dataset_size = int(dataset_size)
        self.tc_weight = tc_weight
        self.mi_weight = mi_weight
        self.dwkl_weight = dwkl_weight

    @staticmethod
    def log_q_z_given_x_k(
        z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """``log q(z|x)`` per latent dim; shapes ``(..., Z)``."""
        return Normal(loc=mu, scale=sigma).log_prob(z)

    @staticmethod
    def log_q_z_given_x(
        z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """``log q(z|x)`` summed over dims; ``(B, Z)`` -> ``(B,)``."""
        return TCVAEModel.log_q_z_given_x_k(z=z, mu=mu, sigma=sigma).sum(dim=-1)

    def log_p_z(self, z: torch.Tensor) -> torch.Tensor:
        """``log p(z)`` summed over dims; ``(B, Z)`` -> ``(B,)``."""
        if self.learned_prior:
            prior = Normal(loc=self.mean, scale=self.log_var.mul(0.5).exp())
        else:
            prior = Normal(loc=0.0, scale=1.0)
        return prior.log_prob(z).sum(dim=-1)

    def tcvae_latent_loss(
        self,
        z: torch.Tensor,  # (B, Z)
        mu: torch.Tensor,  # (B, Z)
        log_var: torch.Tensor,  # (B, Z)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Minibatch estimate of β-TC-VAE latent terms.

        Returns summed MI, TC, and DWKL (same reduction as ``VAEModel.latent_loss``).
        """
        bs = int(z.shape[0])
        n = int(self.dataset_size)
        sigma = log_var.mul(0.5).exp()  # (B, Z)

        # log q(z_i,k | x_j): (B, B, Z)
        z_i = z.unsqueeze(1)  # (B, 1, Z)
        mu_j = mu.unsqueeze(0)  # (1, B, Z)
        sigma_j = sigma.unsqueeze(0)  # (1, B, Z)
        log_q_zi_given_xj_k = self.log_q_z_given_x_k(z=z_i, mu=mu_j, sigma=sigma_j)

        log_norm = torch.log(z.new_tensor(float(bs * n)))
        log_q_z = (
            torch.logsumexp(log_q_zi_given_xj_k.sum(dim=-1), dim=1) - log_norm
        )  # (B,)
        log_prodk_q_zk = (
            torch.logsumexp(log_q_zi_given_xj_k, dim=1).sum(dim=-1) - log_norm
        )  # (B,)

        log_q_z_given_x = self.log_q_z_given_x(z=z, mu=mu, sigma=sigma)  # (B,)
        log_p_z = self.log_p_z(z=z)  # (B,)

        mi_loss = (log_q_z_given_x - log_q_z).sum()
        tc_loss = (log_q_z - log_prodk_q_zk).sum()
        dwkl_loss = (log_prodk_q_zk - log_p_z).sum()
        return mi_loss, tc_loss, dwkl_loss

    def forward(self, s1, s2, y):
        mean, log_var = self.encode(s1, s2, y)  # (B, Z), (B, Z)
        z = self.reparameterization(mean=mean, var=torch.exp(0.5 * log_var))  # (B, Z)

        # * Bradley-Terry likelihood
        z_dec = ei.repeat(z, "B Z -> B A T Z", A=self.annotation_size, T=self.segment_len)
        r0 = self.decode(s1, z_dec)  # (B, A, T, 1)
        r1 = self.decode(s2, z_dec)  # (B, A, T, 1)
        r_hat1 = r0.sum(axis=2) / self.scaling  # (B, A, 1)
        r_hat2 = r1.sum(axis=2) / self.scaling  # (B, A, 1)
        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2).view(-1, 1)  # (B*A, 1)
        labels = y.view(-1, 1)  # (B*A, 1)

        recon_loss = self.reconstruction_loss(labels, p_hat)
        mi_loss, tc_loss, dwkl_loss = self.tcvae_latent_loss(z, mean, log_var)
        tc_weight = self.annealer.slope() if self.annealer else self.tc_weight
        latent_loss = (
            self.mi_weight * mi_loss + tc_weight * tc_loss + self.dwkl_weight * dwkl_loss
        )
        total_loss = recon_loss + latent_loss
        accuracy = self.accuracy(labels, p_hat)

        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "mi_loss": mi_loss.item(),
            "tc_loss": tc_loss.item(),
            "dwkl_loss": dwkl_loss.item(),
            "latent_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "tc_weight": tc_weight,
            # diagnostics
            "ratio_recon_latent": recon_loss.item() / latent_loss.item(),
            "frac_recon2total": recon_loss.item() / total_loss.item(),
        }

        return total_loss, metrics
