#!/usr/bin/env python3
"""Extract posterior latent means for a saved VPL VAE/VAEClassifier checkpoint.

Saves:
- posterior_means.npy   shape [num_modes, num_samples_per_mode, latent_dim]
- flat_latents.npy      shape [num_modes * num_samples_per_mode, latent_dim]
- mode_labels.npy       shape [num_modes * num_samples_per_mode]
- latents_with_labels.npz  keys: latents, labels
- latent_pca.png
- latent_tsne.png
- metrics.txt
"""
import argparse
import os

import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import jaxrl_m.envs  # noqa: F401
from pref_learn.models.utils import get_all_posterior, get_datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--set-length", type=int, default=16)
    p.add_argument("--num-samples-per-mode", type=int, default=500)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def safe_perplexity(perplexity, n):
    if n < 4:
        raise ValueError("Need at least 4 samples for t-SNE")
    return min(perplexity, max(2.0, min(float(n - 1), float(max(5, n // 3)))))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    _, _, _, eval_dataset, set_length, len_query, loaded_obs_dim = get_datasets(
        args.dataset,
        observation_dim=obs_dim,
        action_dim=action_dim,
        batch_size=args.batch_size,
        set_length=args.set_length,
    )

    reward_model = torch.load(args.ckpt, map_location=device, weights_only=False)
    reward_model = reward_model.to(device)
    reward_model.eval()

    means = get_all_posterior(
        env=env,
        reward_model=reward_model,
        dataset=eval_dataset,
        num_samples=args.num_samples_per_mode,
    )

    num_modes, num_samples, latent_dim = means.shape
    flat_latents = means.reshape(-1, latent_dim)
    mode_labels = np.repeat(np.arange(num_modes), num_samples)

    np.save(os.path.join(args.outdir, "posterior_means.npy"), means)
    np.save(os.path.join(args.outdir, "flat_latents.npy"), flat_latents)
    np.save(os.path.join(args.outdir, "mode_labels.npy"), mode_labels)
    np.savez(
        os.path.join(args.outdir, "latents_with_labels.npz"),
        latents=flat_latents,
        labels=mode_labels,
    )

    X = StandardScaler().fit_transform(flat_latents)

    pca = PCA(n_components=2, random_state=args.seed)
    pca_2d = pca.fit_transform(X)
    np.save(os.path.join(args.outdir, "pca_2d.npy"), pca_2d)

    tsne = TSNE(
        n_components=2,
        random_state=args.seed,
        perplexity=safe_perplexity(args.perplexity, len(X)),
        init="pca",
        learning_rate="auto",
    )
    tsne_2d = tsne.fit_transform(X)
    np.save(os.path.join(args.outdir, "tsne_2d.npy"), tsne_2d)

    for name, xy in [("latent_pca.png", pca_2d), ("latent_tsne.png", tsne_2d)]:
        plt.figure(figsize=(6, 5))
        for mode in range(num_modes):
            idx = mode_labels == mode
            plt.scatter(xy[idx, 0], xy[idx, 1], s=12, alpha=0.7, label=f"mode {mode}")
        plt.legend()
        plt.title(name.replace("_", " ").replace(".png", ""))
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, name), dpi=200)
        plt.close()

    kmeans = KMeans(n_clusters=num_modes, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    ari = adjusted_rand_score(mode_labels, cluster_labels)

    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(f"ckpt={args.ckpt}\n")
        f.write(f"dataset={args.dataset}\n")
        f.write(f"env={args.env}\n")
        f.write(f"obs_dim={obs_dim}\n")
        f.write(f"action_dim={action_dim}\n")
        f.write(f"set_length={set_length}\n")
        f.write(f"len_query={len_query}\n")
        f.write(f"loaded_obs_dim={loaded_obs_dim}\n")
        f.write(f"num_modes={num_modes}\n")
        f.write(f"num_samples_per_mode={num_samples}\n")
        f.write(f"latent_dim={latent_dim}\n")
        f.write(f"pca_explained_variance_ratio={pca.explained_variance_ratio_.tolist()}\n")
        f.write(f"kmeans_ari={ari}\n")

    print("saved to:", args.outdir)
    print("means shape:", means.shape)
    print("KMeans ARI:", ari)


if __name__ == "__main__":
    main()
