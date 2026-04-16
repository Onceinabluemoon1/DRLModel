
import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LATENT_KEYS = ("mu", "latents", "z", "embeddings", "features")
LABEL_KEYS = (
    "labels",
    "annotator_id",
    "annotator_ids",
    "user_id",
    "user_ids",
    "cluster",
    "clusters",
    "mode_labels",
)
ID_KEYS = ("ids", "id", "query_ids", "sample_ids")


def parse_inputs(items: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        parsed.append((name, path))
    return parsed


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    payload = {k: data[k] for k in data.files}
    return payload


def get_first_key(payload: Dict[str, np.ndarray], keys: Tuple[str, ...]):
    for key in keys:
        if key in payload:
            return key
    return None


def load_latents(path: str):
    payload = load_npz(path)
    latent_key = get_first_key(payload, LATENT_KEYS)
    if latent_key is None:
        raise KeyError(f"No latent key found in {path}. Tried: {LATENT_KEYS}")
    latents = np.asarray(payload[latent_key])

    if latents.ndim != 2:
        raise ValueError(f"{path}: expected latents [N, D], got shape {latents.shape}")

    label_key = get_first_key(payload, LABEL_KEYS)
    labels = None if label_key is None else np.asarray(payload[label_key]).reshape(-1)

    id_key = get_first_key(payload, ID_KEYS)
    ids = None if id_key is None else np.asarray(payload[id_key]).reshape(-1)

    return latents, labels, ids, payload


def mean_abs_offdiag(matrix: np.ndarray) -> float:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(np.abs(matrix[mask])))


def participation_ratio(eigvals: np.ndarray) -> float:
    eigvals = np.asarray(eigvals)
    if np.allclose(np.sum(eigvals), 0):
        return 0.0
    denom = np.sum(eigvals ** 2)
    if np.isclose(denom, 0):
        return 0.0
    return float((np.sum(eigvals) ** 2) / denom)


def compute_metrics(latents: np.ndarray) -> Dict[str, object]:
    centered = latents - np.mean(latents, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    # Guard against zero-variance dims in corrcoef.
    std = np.std(centered, axis=0, ddof=1)
    safe = std > 1e-12
    corr = np.eye(latents.shape[1], dtype=np.float64)
    if np.any(safe):
        safe_latents = centered[:, safe]
        safe_corr = np.corrcoef(safe_latents, rowvar=False)
        corr[np.ix_(safe, safe)] = safe_corr

    eigvals = np.linalg.eigvalsh(cov)[::-1]
    total = float(np.sum(eigvals))
    explained = (eigvals / total) if total > 0 else np.zeros_like(eigvals)

    diag = np.diag(cov)
    metrics = {
        "covariance": cov,
        "correlation": corr,
        "eigenvalues": eigvals,
        "explained_variance_ratio": explained,
        "latent_dim": int(latents.shape[1]),
        "n_points": int(latents.shape[0]),
        "mean_abs_offdiag_cov": mean_abs_offdiag(cov),
        "mean_abs_offdiag_corr": mean_abs_offdiag(corr),
        "identity_gap_fro": float(np.linalg.norm(cov - np.eye(cov.shape[0]), ord="fro") / cov.shape[0]),
        "diag_mean": float(np.mean(diag)),
        "diag_std": float(np.std(diag)),
        "trace": float(np.trace(cov)),
        "top1_var_ratio": float(explained[0]) if len(explained) >= 1 else 0.0,
        "top2_var_ratio": float(np.sum(explained[:2])) if len(explained) >= 2 else float(np.sum(explained)),
        "effective_rank_pr": participation_ratio(eigvals),
    }
    return metrics


def maybe_linear_probe(latents: np.ndarray, labels: np.ndarray):
    if labels is None:
        return None
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except Exception:
        return None

    x_train, x_test, y_train, y_test = train_test_split(
        latents, labels, test_size=0.3, random_state=42, stratify=labels
    )
    clf = LogisticRegression(max_iter=5000, multi_class="auto")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return float(accuracy_score(y_test, pred))


def plot_cov_heatmaps(results: Dict[str, Dict[str, object]], outdir: str):
    names = list(results.keys())
    n = len(names)
    ncols = 3 if n >= 3 else n
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows))
    axes = np.array(axes).reshape(-1)
    vmax = max(float(np.max(np.abs(results[name]["covariance"]))) for name in names)
    vmax = max(vmax, 1e-6)

    for ax, name in zip(axes, names):
        cov = results[name]["covariance"]
        im = ax.imshow(cov, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{name} covariance")
        ax.set_xlabel("latent dim")
        ax.set_ylabel("latent dim")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[:len(names)], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "covariance_heatmaps.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmaps(results: Dict[str, Dict[str, object]], outdir: str):
    names = list(results.keys())
    n = len(names)
    ncols = 3 if n >= 3 else n
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, names):
        corr = results[name]["correlation"]
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_title(f"{name} correlation")
        ax.set_xlabel("latent dim")
        ax.set_ylabel("latent dim")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[:len(names)], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "correlation_heatmaps.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_eigen_spectra(results: Dict[str, Dict[str, object]], outdir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        eigvals = res["eigenvalues"]
        ax.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", linewidth=1.5, label=name)
    ax.set_title("Latent covariance eigenvalue spectra")
    ax.set_xlabel("principal component index")
    ax.set_ylabel("eigenvalue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_eigenvalues.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        ratio = np.asarray(res["explained_variance_ratio"])
        ax.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio), marker="o", linewidth=1.5, label=name)
    ax.set_title("Cumulative explained variance")
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance ratio")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_cumulative_variance.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary_rows(results: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        row = {
            "model": name,
            "n_points": res["n_points"],
            "latent_dim": res["latent_dim"],
            "mean_abs_offdiag_cov": res["mean_abs_offdiag_cov"],
            "mean_abs_offdiag_corr": res["mean_abs_offdiag_corr"],
            "identity_gap_fro": res["identity_gap_fro"],
            "diag_mean": res["diag_mean"],
            "diag_std": res["diag_std"],
            "trace": res["trace"],
            "top1_var_ratio": res["top1_var_ratio"],
            "top2_var_ratio": res["top2_var_ratio"],
            "effective_rank_pr": res["effective_rank_pr"],
        }
        if "linear_probe_acc" in res and res["linear_probe_acc"] is not None:
            row["linear_probe_acc"] = res["linear_probe_acc"]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Lower is better for these.
    for col in ("mean_abs_offdiag_cov", "mean_abs_offdiag_corr", "identity_gap_fro", "diag_std"):
        df[f"rank_{col}"] = df[col].rank(method="min", ascending=True)

    # Higher is better for these.
    if "linear_probe_acc" in df.columns:
        df["rank_linear_probe_acc"] = df["linear_probe_acc"].rank(method="min", ascending=False)

    # Heuristic diagonality rank.
    diag_rank_cols = [
        "rank_mean_abs_offdiag_cov",
        "rank_mean_abs_offdiag_corr",
        "rank_identity_gap_fro",
        "rank_diag_std",
    ]
    df["rank_diagonality"] = df[diag_rank_cols].mean(axis=1)

    if "rank_linear_probe_acc" in df.columns:
        df["rank_balanced"] = 0.5 * df["rank_diagonality"] + 0.5 * df["rank_linear_probe_acc"]

    df = df.sort_values(
        "rank_balanced" if "rank_balanced" in df.columns else "rank_diagonality"
    ).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze latent covariance geometry across beta runs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Pairs like beta1=path/to/latents.npz")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results = {}
    alignment = []
    first_ids = None
    first_n = None

    for name, path in parse_inputs(args.inputs):
        latents, labels, ids, _ = load_latents(path)
        metrics = compute_metrics(latents)
        metrics["path"] = path
        probe = maybe_linear_probe(latents, labels)
        metrics["linear_probe_acc"] = probe
        results[name] = metrics

        if first_n is None:
            first_n = latents.shape[0]
        alignment.append({"model": name, "n_points": int(latents.shape[0]), "has_ids": ids is not None})
        if first_ids is None and ids is not None:
            first_ids = ids
        elif first_ids is not None and ids is not None:
            metrics["ids_match_first"] = bool(np.array_equal(first_ids, ids))

    plot_cov_heatmaps(results, args.outdir)
    plot_corr_heatmaps(results, args.outdir)
    plot_eigen_spectra(results, args.outdir)

    summary_df = build_summary_rows(results)
    summary_csv = os.path.join(args.outdir, "latent_geometry_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    serializable = {
        "alignment": alignment,
        "models": {},
        "ranking_csv": summary_csv,
    }
    for name, res in results.items():
        serializable["models"][name] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in res.items()
            if k not in {"covariance", "correlation"}  # heatmaps are saved separately
        }

    with open(os.path.join(args.outdir, "latent_geometry_summary.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Saved heatmaps and eigen plots to: {args.outdir}")


if __name__ == "__main__":
    main()
