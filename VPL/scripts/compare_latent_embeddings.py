#!/usr/bin/env python3
"""Compare latent spaces across models with PCA and t-SNE.

Expected input:
- One or more latent files, usually .npz files saved from your model.
- Each file should contain a latent matrix under one of these keys:
    mu, latents, z, embeddings, features
- Optional labels for coloring under one of these keys:
    labels, annotator_id, annotator_ids, user_id, user_ids, cluster, clusters

Typical use:
python compare_latent_embeddings.py \
  --inputs baseline=baseline_latents.npz beta4=beta4_latents.npz beta5=beta5_latents.npz \
  --outdir latent_viz

Recommended practice:
- Use the same datapoints / users in every file.
- For VAEs, use posterior means (mu), not random z samples, for cleaner comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

LATENT_KEYS = ["mu", "latents", "z", "embeddings", "features"]
LABEL_KEYS = [
    "labels",
    "annotator_id",
    "annotator_ids",
    "user_id",
    "user_ids",
    "cluster",
    "clusters",
]
ID_KEYS = ["ids", "sample_ids", "user_ids", "annotator_ids"]


@dataclass
class LoadedLatents:
    name: str
    latents: np.ndarray
    labels: Optional[np.ndarray]
    ids: Optional[np.ndarray]
    path: str
    latent_key: str
    label_key: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Pairs like name=path.npz. Example: baseline=a.npz beta4=b.npz beta5=c.npz",
    )
    parser.add_argument("--outdir", type=str, default="latent_viz")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument(
        "--title", type=str, default="Latent-space comparison: disentangled vs non-disentangled"
    )
    parser.add_argument(
        "--normalize-globally",
        action="store_true",
        help=(
            "Fit one StandardScaler on the concatenation of all models before PCA/t-SNE. "
            "Default behavior standardizes each model separately."
        ),
    )
    parser.add_argument(
        "--labels-key",
        type=str,
        default=None,
        help="Force a specific label key from the npz files.",
    )
    return parser.parse_args()



def parse_input_pairs(pairs: List[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Bad input '{item}'. Expected name=path.")
        name, path = item.split("=", 1)
        if not name or not path:
            raise ValueError(f"Bad input '{item}'. Expected name=path.")
        parsed.append((name, path))
    return parsed



def _pick_first_key(container, keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in container:
            return key
    return None



def load_file(name: str, path: str, forced_label_key: Optional[str] = None) -> LoadedLatents:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        latent_key = _pick_first_key(data, LATENT_KEYS)
        if latent_key is None:
            raise KeyError(
                f"Could not find a latent matrix in {path}. Expected one of: {LATENT_KEYS}. Found: {list(data.keys())}"
            )
        label_key = forced_label_key if forced_label_key in data else _pick_first_key(data, LABEL_KEYS)
        id_key = _pick_first_key(data, ID_KEYS)

        latents = np.asarray(data[latent_key])
        labels = np.asarray(data[label_key]) if label_key is not None else None
        ids = np.asarray(data[id_key]) if id_key is not None else None
    elif path.endswith(".npy"):
        latents = np.asarray(np.load(path, allow_pickle=True))
        labels = None
        ids = None
        latent_key = "npy_array"
        label_key = None
    else:
        raise ValueError(f"Unsupported file type for {path}. Use .npz or .npy")

    if latents.ndim != 2:
        raise ValueError(f"Latents in {path} must have shape [N, D], got {latents.shape}")

    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(latents):
            raise ValueError(
                f"Label count mismatch in {path}: {len(labels)} labels for {len(latents)} latent rows"
            )

    if ids is not None:
        ids = np.asarray(ids).reshape(-1)
        if len(ids) != len(latents):
            raise ValueError(
                f"ID count mismatch in {path}: {len(ids)} ids for {len(latents)} latent rows"
            )

    return LoadedLatents(
        name=name,
        latents=latents,
        labels=labels,
        ids=ids,
        path=path,
        latent_key=latent_key,
        label_key=label_key,
    )



def safe_perplexity(requested: float, n_samples: int) -> float:
    if n_samples < 4:
        raise ValueError("Need at least 4 samples for t-SNE.")
    upper = max(2.0, min(float(n_samples - 1), float(max(5, n_samples // 3))))
    return min(requested, upper)



def standardize_models(models: List[LoadedLatents], normalize_globally: bool) -> Dict[str, np.ndarray]:
    if normalize_globally:
        scaler = StandardScaler()
        all_latents = np.concatenate([m.latents for m in models], axis=0)
        scaler.fit(all_latents)
        return {m.name: scaler.transform(m.latents) for m in models}

    output: Dict[str, np.ndarray] = {}
    for m in models:
        scaler = StandardScaler()
        output[m.name] = scaler.fit_transform(m.latents)
    return output



def embed_model(latents: np.ndarray, perplexity: float, random_state: int) -> Dict[str, np.ndarray]:
    pca = PCA(n_components=2, random_state=random_state)
    pca_xy = pca.fit_transform(latents)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    tsne_xy = tsne.fit_transform(latents)

    return {
        "pca_xy": pca_xy,
        "tsne_xy": tsne_xy,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_components": pca.components_,
    }



def make_label_colors(labels: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
    if labels is None:
        return None, None
    labels = np.asarray(labels).reshape(-1)
    unique = [str(x) for x in np.unique(labels)]
    mapping = {lab: i for i, lab in enumerate(unique)}
    numeric = np.array([mapping[str(x)] for x in labels])
    return numeric, mapping



def plot_single_model(
    model: LoadedLatents,
    emb: Dict[str, np.ndarray],
    outdir: str,
    point_size: float,
) -> None:
    numeric_labels, label_mapping = make_label_colors(model.labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(model.name)

    for ax, xy, plot_name in [
        (axes[0], emb["pca_xy"], "PCA"),
        (axes[1], emb["tsne_xy"], "t-SNE"),
    ]:
        if numeric_labels is None:
            ax.scatter(xy[:, 0], xy[:, 1], s=point_size, alpha=0.8)
        else:
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=numeric_labels, s=point_size, alpha=0.8)
            if len(label_mapping) <= 20:
                handles, _ = scatter.legend_elements()
                labels_txt = list(label_mapping.keys())
                ax.legend(handles, labels_txt, title=model.label_key or "labels", fontsize=7)
        ax.set_title(plot_name)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

    fig.savefig(os.path.join(outdir, f"{model.name}_pca_tsne.png"), dpi=200)
    plt.close(fig)



def plot_grid(
    models: List[LoadedLatents],
    embeddings: Dict[str, Dict[str, np.ndarray]],
    outdir: str,
    point_size: float,
    title: str,
) -> None:
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4.5 * len(models)), constrained_layout=True)
    if len(models) == 1:
        axes = np.array([axes])
    fig.suptitle(title)

    for row, model in enumerate(models):
        emb = embeddings[model.name]
        numeric_labels, label_mapping = make_label_colors(model.labels)
        for col, (xy, plot_name) in enumerate([
            (emb["pca_xy"], "PCA"),
            (emb["tsne_xy"], "t-SNE"),
        ]):
            ax = axes[row, col]
            if numeric_labels is None:
                ax.scatter(xy[:, 0], xy[:, 1], s=point_size, alpha=0.8)
            else:
                scatter = ax.scatter(xy[:, 0], xy[:, 1], c=numeric_labels, s=point_size, alpha=0.8)
                if row == 0 and len(label_mapping) <= 20:
                    handles, _ = scatter.legend_elements()
                    ax.legend(handles, list(label_mapping.keys()), title=model.label_key or "labels", fontsize=7)
            ax.set_title(f"{model.name} — {plot_name}")
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")

    fig.savefig(os.path.join(outdir, "comparison_grid.png"), dpi=220)
    plt.close(fig)



def write_summary(
    models: List[LoadedLatents],
    scaled: Dict[str, np.ndarray],
    embeddings: Dict[str, Dict[str, np.ndarray]],
    outdir: str,
    perplexity: float,
) -> None:
    summary = {"perplexity": perplexity, "models": {}}
    for model in models:
        X = scaled[model.name]
        cov = np.cov(X, rowvar=False)
        offdiag = cov - np.diag(np.diag(cov))
        summary["models"][model.name] = {
            "path": model.path,
            "latent_key": model.latent_key,
            "label_key": model.label_key,
            "n_samples": int(X.shape[0]),
            "latent_dim": int(X.shape[1]),
            "pca_explained_variance_ratio": embeddings[model.name]["pca_explained_variance_ratio"].tolist(),
            "mean_abs_offdiag_covariance": float(np.mean(np.abs(offdiag))),
        }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)



def check_alignment(models: List[LoadedLatents]) -> None:
    lengths = [len(m.latents) for m in models]
    if len(set(lengths)) != 1:
        print("[warn] Models have different numbers of samples. Comparison plots still work, but matched visual comparison is weaker.")
        return

    ids_present = all(m.ids is not None for m in models)
    if ids_present:
        base = models[0].ids
        for m in models[1:]:
            if not np.array_equal(base, m.ids):
                print("[warn] Model IDs are not aligned row-by-row. Use the same datapoints in the same order for a cleaner comparison.")
                break



def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    parsed_pairs = parse_input_pairs(args.inputs)
    models = [load_file(name, path, forced_label_key=args.labels_key) for name, path in parsed_pairs]
    check_alignment(models)

    scaled = standardize_models(models, normalize_globally=args.normalize_globally)

    embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    used_perplexity: Optional[float] = None
    for model in models:
        perp = safe_perplexity(args.perplexity, len(model.latents))
        used_perplexity = perp if used_perplexity is None else min(used_perplexity, perp)

    assert used_perplexity is not None

    for model in models:
        embeddings[model.name] = embed_model(
            latents=scaled[model.name],
            perplexity=used_perplexity,
            random_state=args.random_state,
        )
        plot_single_model(model, embeddings[model.name], args.outdir, args.point_size)

    plot_grid(models, embeddings, args.outdir, args.point_size, args.title)
    write_summary(models, scaled, embeddings, args.outdir, used_perplexity)

    print(f"Saved plots to: {args.outdir}")
    print(f"Used t-SNE perplexity: {used_perplexity}")
    print("Files:")
    for model in models:
        print(f"  - {os.path.join(args.outdir, model.name + '_pca_tsne.png')}")
    print(f"  - {os.path.join(args.outdir, 'comparison_grid.png')}")
    print(f"  - {os.path.join(args.outdir, 'summary.json')}")


if __name__ == "__main__":
    main()
