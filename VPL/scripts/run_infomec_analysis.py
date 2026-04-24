import argparse
import json
import os
import pickle
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pref_learn.utils.infomec import compute_infomec
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--split", choices=["all", "train", "eval"], default="eval")
    p.add_argument(
        "--source-keys",
        default="auto",
        help="Comma-separated source fields. Use 'auto' to prefer source_variable, then source_mode, then annotator_id.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--set-length", type=int, default=-1)
    p.add_argument("--discrete-latents", action="store_true")
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def split_indices(n: int, split: str) -> np.ndarray:
    eval_size = int(0.1 * n)
    train_size = n - eval_size
    if split == "all":
        return np.arange(n)
    if split == "train":
        return np.arange(train_size)
    return np.arange(train_size, n)


def maybe_truncate(arr: np.ndarray, set_length: int) -> np.ndarray:
    if set_length < 0 or arr.ndim < 2:
        return arr
    return arr[:, :set_length]


def resolve_source_keys(batch: dict, source_keys_arg: str) -> List[str]:
    available = set(batch.keys())
    aliases = {
        "source_variable": ["source_variable", "source_mode", "annotator_id"],
        "source_mode": ["source_mode", "source_variable", "annotator_id"],
        "annotator_id": ["annotator_id", "source_variable", "source_mode"],
    }

    if source_keys_arg.strip().lower() == "auto":
        for candidates in (["source_variable"], ["source_mode"], ["annotator_id"]):
            if candidates[0] in available:
                return candidates
        raise KeyError(
            "Could not infer source keys automatically. Expected one of source_variable, source_mode, or annotator_id. "
            f"Available keys include: {sorted(batch.keys())[:40]}"
        )

    resolved = []
    for raw_key in [k.strip() for k in source_keys_arg.split(",") if k.strip()]:
        if raw_key in available:
            resolved.append(raw_key)
            continue
        for candidate in aliases.get(raw_key, []):
            if candidate in available:
                resolved.append(candidate)
                break
        else:
            raise KeyError(
                f"Source key '{raw_key}' not found in dataset. Available keys include: {sorted(batch.keys())[:40]}"
            )
    return resolved


def infer_source_columns(
    raw: np.ndarray, key: str, num_bins: int
) -> Tuple[np.ndarray, List[str]]:
    arr = np.asarray(raw)

    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr2 = arr[..., 0]
        if np.all(arr2 == arr2[:, [0]]):
            arr = arr2[:, 0][:, None]
        else:
            # Majority class per set for discrete-looking inputs.
            if np.allclose(arr2, np.round(arr2)):
                majority = []
                for row in arr2:
                    vals, counts = np.unique(row.astype(np.int64), return_counts=True)
                    majority.append(vals[np.argmax(counts)])
                arr = np.asarray(majority)[:, None]
            else:
                arr = np.mean(arr2, axis=1, keepdims=True)
    elif arr.ndim > 2:
        raise ValueError(f"Unsupported source shape for key={key}: {arr.shape}")

    columns = []
    names = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if np.issubdtype(col.dtype, np.floating) and not np.allclose(col, np.round(col)):
            quantiles = np.linspace(0, 1, num_bins + 1)
            edges = np.unique(np.quantile(col, quantiles))
            if len(edges) <= 2:
                binned = np.zeros_like(col, dtype=np.int32)
            else:
                binned = np.digitize(col, edges[1:-1], right=False).astype(np.int32)
            columns.append(binned)
            names.append(f"{key}[{j}]_binned")
        else:
            columns.append(np.round(col).astype(np.int32))
            names.append(key if arr.shape[1] == 1 else f"{key}[{j}]")
    return np.stack(columns, axis=1), names


def batched_encode(model, obs1, obs2, labels, batch_size, device):
    means = []
    for start in range(0, len(obs1), batch_size):
        end = min(start + batch_size, len(obs1))
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)
        o2 = torch.from_numpy(obs2[start:end]).float().to(device)
        y = torch.from_numpy(labels[start:end]).float().to(device)
        with torch.no_grad():
            mean, _ = model.encode(o1, o2, y)
        means.append(mean.cpu().numpy())
    return np.concatenate(means, axis=0)


def save_heatmap(
    mat: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    title: str,
    outpath: str,
):
    plt.figure(figsize=(max(6, len(col_names) * 0.35), max(3, len(row_names) * 0.55)))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im)
    plt.xticks(np.arange(len(col_names)), col_names, rotation=90)
    plt.yticks(np.arange(len(row_names)), row_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def maybe_save_lda(latents: np.ndarray, y: np.ndarray, outpath: str, title: str):
    classes = np.unique(y)
    if len(classes) < 2:
        return
    n_comp = 1 if len(classes) == 2 else 2
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    proj = lda.fit_transform(latents, y)
    plt.figure(figsize=(6, 5))
    if n_comp == 1:
        plt.scatter(proj[:, 0], np.zeros_like(proj[:, 0]), c=y, s=12, alpha=0.8)
        plt.yticks([])
        plt.xlabel("LD1")
    else:
        plt.scatter(proj[:, 0], proj[:, 1], c=y, s=12, alpha=0.8)
        plt.xlabel("LD1")
        plt.ylabel("LD2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def maybe_save_pca(latents: np.ndarray, y: np.ndarray, outpath: str, title: str):
    if latents.ndim != 2 or latents.shape[1] < 2:
        return None
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(latents)
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], c=y, s=12, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return pca.explained_variance_ratio_.tolist()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.dataset, "rb") as f:
        batch = pickle.load(f)

    n = len(batch["observations"])
    idx = split_indices(n, args.split)

    obs1 = maybe_truncate(np.asarray(batch["observations"])[idx], args.set_length)
    obs2 = maybe_truncate(np.asarray(batch["observations_2"])[idx], args.set_length)
    labels = maybe_truncate(np.asarray(batch["labels"])[idx], args.set_length)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = torch.load(args.ckpt, map_location=device, weights_only=False).to(device)
    model.eval()

    latents = batched_encode(model, obs1, obs2, labels, args.batch_size, device)

    source_keys = resolve_source_keys(batch, args.source_keys)
    source_cols = []
    source_names = []
    for key in source_keys:
        cols, names = infer_source_columns(
            np.asarray(batch[key])[idx], key, args.num_bins
        )
        source_cols.append(cols)
        source_names.extend(names)
    sources = np.concatenate(source_cols, axis=1)

    result = compute_infomec(
        sources=sources, latents=latents, discrete_latents=args.discrete_latents
    )

    np.save(os.path.join(args.outdir, "latents.npy"), latents)
    np.save(os.path.join(args.outdir, "sources.npy"), sources)
    np.save(os.path.join(args.outdir, "nmi.npy"), result.nmi)

    latent_names = [f"z{j}" for j in range(latents.shape[1])]
    save_heatmap(
        result.nmi,
        source_names,
        latent_names,
        title="InfoMEC normalized MI (sources x latents)",
        outpath=os.path.join(args.outdir, "infomec_nmi_heatmap.png"),
    )

    first_source = sources[:, 0]
    pca_var = maybe_save_pca(
        latents,
        first_source,
        os.path.join(args.outdir, "pca_by_first_source.png"),
        f"PCA colored by {source_names[0]}",
    )
    maybe_save_lda(
        latents,
        first_source,
        os.path.join(args.outdir, "lda_by_first_source.png"),
        f"LDA by {source_names[0]}",
    )

    metrics = result.to_dict()
    metrics.update(
        {
            "source_keys": source_keys,
            "source_names": source_names,
            "num_samples": int(len(idx)),
            "num_sources": int(sources.shape[1]),
            "latent_dim": int(latents.shape[1]),
            "split": args.split,
            "pca_explained_variance_ratio": pca_var,
            "note": (
                "InfoM becomes trivial when there is only one source column; in that case "
                "InfoE/InfoC and the NMI heatmap are more informative than the scalar InfoM value."
            ),
        }
    )
    with open(os.path.join(args.outdir, "infomec_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Also save a compact CSV-like text file for quick beta sweeps.
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"infom={metrics['infom']:.6f}\n")
        f.write(f"infoc={metrics['infoc']:.6f}\n")
        f.write(f"infoe={metrics['infoe']:.6f}\n")
        f.write(
            f"linear_probe_accuracy_per_source={metrics['linear_probe_accuracy_per_source']}\n"
        )
        f.write(
            f"active_latents={int(np.sum(result.active_latents))}/{len(result.active_latents)}\n"
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
