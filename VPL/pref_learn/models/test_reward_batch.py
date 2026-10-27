"""Batch shapes for policy relabel: flat (B, O) + (B, Z) -> decoder (B, A, T, 1)."""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pref_learn.models.decoders import broadcast_obs_and_latent
from pref_learn.models.qlae import QLAE
from pref_learn.models.vae import VAEModel


def _vae(**kwargs):
    defaults = dict(
        encoder_input=1,
        decoder_input=7,
        latent_dim=3,
        hidden_dim=8,
        annotation_size=2,
        segment_len=1,
        obs_dim=4,
    )
    defaults.update(kwargs)
    return VAEModel(**defaults)


def _qlae():
    return QLAE(
        encoder_input=1,
        decoder_input=7,
        latent_dim=3,
        hidden_dim=8,
        annotation_size=2,
        segment_len=1,
        obs_dim=4,
        num_quantized_values=5,
    )


class BroadcastObsLatentTest(unittest.TestCase):
    def test_flat_policy_batch(self):
        obs = torch.randn(5, 4)
        z = torch.randn(5, 3)
        obs_b, z_b = broadcast_obs_and_latent(obs, z)
        self.assertEqual(tuple(obs_b.shape), (5, 1, 1, 4))
        self.assertEqual(tuple(z_b.shape), (5, 1, 1, 3))

    def test_segment_and_single_latent(self):
        obs = torch.randn(2, 7, 4)
        z = torch.randn(1, 3)
        obs_b, z_b = broadcast_obs_and_latent(obs, z)
        self.assertEqual(tuple(obs_b.shape), (2, 1, 7, 4))
        self.assertEqual(tuple(z_b.shape), (2, 1, 7, 3))

    def test_training_layout_unchanged(self):
        obs = torch.randn(2, 3, 4, 5)
        z = torch.randn(2, 3, 4, 6)
        obs_b, z_b = broadcast_obs_and_latent(obs, z)
        self.assertTrue(obs_b is obs)
        self.assertTrue(z_b is z)


class RewardBatchInterfaceTest(unittest.TestCase):
    def test_vae_flat_get_reward_matches_concat(self):
        model = _vae()
        model.eval()
        obs = torch.randn(6, 4)
        z = torch.randn(6, 3)
        with torch.no_grad():
            paired = model.get_reward(obs, z)
            legacy = model.get_reward(torch.cat([obs, z], dim=-1))
        self.assertEqual(tuple(paired.shape), (6, 1, 1, 1))
        self.assertTrue(torch.allclose(paired, legacy))
        self.assertEqual(paired.reshape(-1).shape, (6,))
        seg = torch.randn(2, 5, 7)
        with torch.no_grad():
            seg_r = model.get_reward(seg)
        self.assertEqual(tuple(seg_r.shape), (2, 1, 5, 1))

    def test_vae_training_forward_keeps_bat_layout(self):
        model = _vae()
        s1 = torch.randn(2, 2, 1, 4)
        s2 = torch.randn(2, 2, 1, 4)
        y = torch.zeros(2, 2, 1)
        z = torch.randn(2, 2, 1, 3)
        decoded = model.decode(s1, z)
        self.assertEqual(tuple(decoded.shape), (2, 2, 1, 1))
        loss, metrics = model(s1, s2, y)
        self.assertEqual(tuple(loss.shape), ())
        self.assertIn("total_loss", metrics)

    def test_vae_prob_alias_and_single_z(self):
        model = _vae()
        model.eval()
        obs = torch.randn(4, 4)
        z = torch.randn(1, 3)
        comp = torch.randn(5, 4)
        with torch.no_grad():
            r = model.get_probabilistic_reward(s_a=obs, z=z, comparison_obs=comp)
            alias = model.get_prob_reward(s_a=obs, z=z, comparison_obs=comp)
        self.assertEqual(tuple(r.shape), (4, 1))
        self.assertTrue(torch.allclose(r, alias))

    def test_qlae_flat_and_training_batch(self):
        model = _qlae()
        model.eval()
        obs = torch.randn(3, 4)
        z = torch.randn(3, 3)
        with torch.no_grad():
            r = model.get_reward(obs, z)
        self.assertEqual(tuple(r.shape), (3, 1, 1, 1))
        s1 = torch.randn(2, 2, 1, 4)
        s2 = torch.randn(2, 2, 1, 4)
        y = torch.ones(2, 2, 1)
        loss, metrics = model(s1, s2, y)
        self.assertEqual(tuple(loss.shape), ())
        self.assertIn("accuracy", metrics)
        comp = torch.randn(4, 4)
        with torch.no_grad():
            prob = model.get_prob_reward(s_a=obs, z=z, comparison_obs=comp)
        self.assertEqual(tuple(prob.shape), (3, 1))


if __name__ == "__main__":
    unittest.main()
