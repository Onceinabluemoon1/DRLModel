# save as VPL/scripts/extract_raven_latents.py

import os
import numpy as np
import torch
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import jaxrl_m.envs
from pref_learn.models.utils import get_datasets, get_all_posterior

CKPT_PATH = "./VPL/logs3/sort-easy-v0/VAEClassifier/sort_final_reward_results/s42/best_model.pt"
DATASET_PATH = "./VPL/pref_datasets/sort-easy-v0/relabelled_queries_num5000_q1_s16"
OUT_DIR = "./VPL/latent_analysis/sort-easy-v0/s42"
ENV_NAME = "sort-easy-v0"

BATCH_SIZE = 256
SET_LENGTH = 16
NUM_SAMPLES_PER_MODE = 1000  # start here; increase later if you want

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

env = gym.make(ENV_NAME)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

_, _, train_dataset, eval_dataset, set_length, len_query, loaded_obs_dim = get_datasets(
    DATASET_PATH,
    observation_dim=obs_dim,
    action_dim=action_dim,
    batch_size=BATCH_SIZE,
    set_length=SET_LENGTH,
)

print("obs_dim:", obs_dim)
print("action_dim:", action_dim)
print("set_length:", set_length)
print("len_query:", len_query)
print("loaded_obs_dim:", loaded_obs_dim)
print("num_modes:", env.get_num_modes())

reward_model = torch.load(CKPT_PATH, map_location=device, weights_only=False)
reward_model = reward_model.to(device)
reward_model.eval()

# Shape: [num_modes, num_samples, latent_dim]
means = get_all_posterior(
    env=env,
    reward_model=reward_model,
    dataset=eval_dataset,
    num_samples=NUM_SAMPLES_PER_MODE,
)

print("means shape:", means.shape)

num_modes, num_samples, latent_dim = means.shape
flat_latents = means.reshape(-1, latent_dim)
mode_labels = np.repeat(np.arange(num_modes), num_samples)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0, perplexity=30)
tsne_2d = tsne.fit_transform(flat_latents)

plt.figure(figsize=(6, 5))
for mode in range(num_modes):
    idx = mode_labels == mode
    plt.scatter(tsne_2d[idx, 0], tsne_2d[idx, 1], s=12, alpha=0.7, label=f"mode {mode}")
plt.legend()
plt.title("Raven latent t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latent_tsne.png"), dpi=200)
plt.close()

np.save(os.path.join(OUT_DIR, "posterior_means.npy"), means)
np.save(os.path.join(OUT_DIR, "flat_latents.npy"), flat_latents)
np.save(os.path.join(OUT_DIR, "mode_labels.npy"), mode_labels)

# PCA visualization
pca = PCA(n_components=2, random_state=0)
pca_2d = pca.fit_transform(flat_latents)
np.save(os.path.join(OUT_DIR, "pca_2d.npy"), pca_2d)

plt.figure(figsize=(6, 5))
for mode in range(num_modes):
    idx = mode_labels == mode
    plt.scatter(pca_2d[idx, 0], pca_2d[idx, 1], s=12, alpha=0.7, label=f"mode {mode}")
plt.legend()
plt.title("Raven latent PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latent_pca.png"), dpi=200)
plt.close()

# Simple clustering check
kmeans = KMeans(n_clusters=num_modes, random_state=0, n_init=10)
cluster_labels = kmeans.fit_predict(flat_latents)
ari = adjusted_rand_score(mode_labels, cluster_labels)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"num_modes={num_modes}\n")
    f.write(f"num_samples_per_mode={num_samples}\n")
    f.write(f"latent_dim={latent_dim}\n")
    f.write(f"pca_explained_variance_ratio={pca.explained_variance_ratio_.tolist()}\n")
    f.write(f"kmeans_ari={ari}\n")

print("saved to:", OUT_DIR)
print("KMeans ARI:", ari)