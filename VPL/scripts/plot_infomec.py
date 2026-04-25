import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from pref_learn.utils.infomec import InfoMECResult, compute_infomec
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Fixed colors for InfoM / InfoE / InfoC so legends match when a series is flat or empty.
INFOMEC_COLOR_INFOM: str = "tab:blue"
INFOMEC_COLOR_INFOE: str = "tab:orange"
INFOMEC_COLOR_INFOC: str = "tab:green"


def epoch_checkpoint_prefers_log_x(epochs: List[int]) -> bool:
    """Log-scale epoch x-axis when spans orders of magnitude and bunches toward small epochs.

    Uses only strictly positive epochs for the dynamic-range check (so ``model_0.pt`` is ok).
    Same heuristic shape as :func:`run_infomec_sweep.beta_sweep_prefers_log_x`.
    """
    e = np.sort(np.unique(np.asarray(epochs, dtype=np.float64)))
    if e.size < 3 or np.any(e < 0) or not np.all(np.isfinite(e)):
        return False
    e_pos = e[e > 0]
    if e_pos.size < 2:
        return False
    dynamic_range = float(e_pos[-1] / e_pos[0])
    if dynamic_range < 10.0:
        return False
    median = float(np.median(e))
    b0, b1 = float(e[0]), float(e[-1])
    linear_mid = 0.5 * (b0 + b1)
    skewed_toward_small = median < 0.22 * linear_mid + 0.78 * b0
    if skewed_toward_small:
        return True
    return dynamic_range >= 50.0


def apply_checkpoint_epoch_x_scale(
    ax_left: Axes,
    epochs: List[int],
    beta_x_axis: Literal["auto", "linear", "log"],
) -> None:
    """Set x-scale for InfoMEC vs checkpoint epoch (linear, log, or symlog if epoch 0 is present)."""
    e = np.asarray(epochs, dtype=np.float64)
    if beta_x_axis == "auto":
        use_log = epoch_checkpoint_prefers_log_x(epochs)
    elif beta_x_axis == "log":
        use_log = True
    else:
        use_log = False
    if not use_log:
        return
    if np.any(e < 0):
        raise ValueError(
            "log epoch axis requires non-negative checkpoint indices; "
            f"got epochs={epochs!r}."
        )
    base_lbl = "epoch (from checkpoint name)"
    if np.any(e <= 0):
        e_pos = e[e > 0]
        linthresh = max(1.0, float(np.min(e_pos)) * 0.1) if e_pos.size else 1.0
        ax_left.set_xscale("symlog", linthresh=linthresh)
        ax_left.set_xlabel(base_lbl + " — symlog scale (linear near 0)")
    else:
        ax_left.set_xscale("log")
        ax_left.set_xlabel(base_lbl + " — log scale")
    ax_left.grid(alpha=0.3, which="both")


@dataclass
class AnalysisCLI:
    """InfoMEC / NMI for one training directory (variant.json + checkpoint)."""

    run_dir: Path
    """Directory with variant.json and the checkpoint (see ckpt-name)."""

    ckpt_name: str = "best_model.pt"
    """Checkpoint file name inside run_dir."""

    dataset: Optional[Path] = None
    """Override dataset pickle; default from variant.json dataset_path."""

    split: Literal["all", "train", "eval"] = "eval"
    """Which part of the preference dataset to use."""

    source_keys: str = "auto"
    """Comma-separated source fields, or 'auto' (source_variable, source_mode, annotator_id)."""

    batch_size: Optional[int] = None
    """Encoder batch size; default from variant.json."""

    set_length: Optional[int] = None
    """Truncate set dimension; default from variant.json set_size."""

    discrete_latents: bool = False
    """If True, treat latents as discrete for InfoMEC."""

    num_bins: int = 10
    """Histogram bins when discretizing continuous sources."""

    device: str = "auto"
    """torch device name, or 'auto'."""

    epoch_checkpoint_plot: bool = True
    """If True, plot InfoMEC vs epoch for model_<n>.pt checkpoints in run_dir."""

    beta_x_axis: Literal["auto", "linear", "log"] = "auto"
    """X-axis for the checkpoint-epoch InfoMEC curve PNG: ``auto`` (heuristic), or ``linear`` / ``log``. Same flag name as sweep (this script has no β sweep)."""


_MODEL_CKPT_RE = re.compile(r"^model_(\d+)\.pt$")


def gather_epoch_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    """Paths like model_0.pt, model_450.pt under run_dir; sorted by epoch."""
    pairs: List[Tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        m = _MODEL_CKPT_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def draw_infomec_epoch_trajectory_axes(
    ax_left: Axes,
    ax_right: Axes,
    epochs: List[int],
    infom: np.ndarray,
    infoe: np.ndarray,
    infoc: np.ndarray,
    n_active: np.ndarray,
    latent_dim: int,
    *,
    title: Optional[str] = None,
    show_legend: bool = True,
    legend_fontsize: int = 8,
) -> None:
    """Draw InfoM/E/C + active latents (twin) on existing axes; reused by sweep subplots."""
    infom_a = np.asarray(infom, dtype=np.float64)
    infoe_a = np.asarray(infoe, dtype=np.float64)
    infoc_a = np.asarray(infoc, dtype=np.float64)
    n_active_plot = np.asarray(n_active, dtype=np.float64)
    L = max(int(latent_dim), 1)
    n_active_norm = n_active_plot / float(L)

    ax_left.plot(
        epochs,
        infom_a,
        label="InfoM",
        marker="o",
        markersize=3,
        color=INFOMEC_COLOR_INFOM,
    )
    ax_left.plot(
        epochs,
        infoe_a,
        label="InfoE",
        marker="o",
        markersize=3,
        color=INFOMEC_COLOR_INFOE,
    )
    ax_left.plot(
        epochs,
        infoc_a,
        label="InfoC",
        marker="o",
        markersize=3,
        color=INFOMEC_COLOR_INFOC,
    )
    ax_left.set_xlabel("epoch (from checkpoint name)")
    ax_left.set_ylabel("InfoM / InfoE / InfoC ([0, 1])")
    ax_left.set_ylim(-0.02, 1.02)
    ax_left.grid(alpha=0.3)
    if title:
        ax_left.set_title(title, fontsize=9)

    ax_right.plot(
        epochs,
        n_active_norm,
        linestyle="--",
        color="black",
        linewidth=0.9,
        label="Active latents #",
    )
    ax_right.set_ylabel("Active latents (count)")
    ax_right.set_ylim(-0.02, 1.02)
    if L <= 12:
        tick_pos = np.arange(L + 1, dtype=np.float64) / float(L)
        tick_lbl = [str(i) for i in range(L + 1)]
    else:
        n_ticks = 11
        tick_pos = np.linspace(0.0, 1.0, n_ticks)
        tick_lbl = [str(int(round(t * L))) for t in tick_pos]
    ax_right.set_yticks(tick_pos)
    ax_right.set_yticklabels(tick_lbl)

    if show_legend:
        h1, lab1 = ax_left.get_legend_handles_labels()
        h2, lab2 = ax_right.get_legend_handles_labels()
        ax_left.legend(h1 + h2, lab1 + lab2, loc="best", fontsize=legend_fontsize)


def plot_infomec_epoch_trajectory(
    epochs: List[int],
    infom: np.ndarray,
    infoe: np.ndarray,
    infoc: np.ndarray,
    n_active: np.ndarray,
    latent_dim: int,
    outpath: Path,
    *,
    beta_x_axis: Literal["auto", "linear", "log"] = "auto",
) -> None:
    """Line plot: x=epoch; left y = InfoM/E/C in [0,1]; right y = same scale with active = n_active/L (ticks show counts)."""
    fig, ax_left = plt.subplots(figsize=(8, 4.5))
    ax_right = ax_left.twinx()
    draw_infomec_epoch_trajectory_axes(
        ax_left,
        ax_right,
        epochs,
        infom,
        infoe,
        infoc,
        n_active,
        latent_dim,
    )
    apply_checkpoint_epoch_x_scale(ax_left, epochs, beta_x_axis)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def collect_epoch_checkpoint_infomec_series(
    run_dir: Path,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    batch_size: int,
    device: torch.device,
    discrete_latents: bool,
) -> Optional[Dict[str, Any]]:
    ckpts = gather_epoch_checkpoints(run_dir)
    if not ckpts:
        return None

    epochs_list: List[int] = []
    infoms: List[float] = []
    infoes: List[float] = []
    infocs: List[float] = []
    actives: List[int] = []
    latent_dim: Optional[int] = None

    for ep, ckpt_path in ckpts:
        model = torch.load(ckpt_path, map_location=device, weights_only=False).to(device)
        model.eval()
        latents = batched_encode(model, obs1, obs2, labels, batch_size, device)
        if latent_dim is None:
            latent_dim = int(latents.shape[1])
        result = compute_infomec(
            sources=sources, latents=latents, discrete_latents=discrete_latents
        )
        epochs_list.append(ep)
        infoms.append(float(result.infom))
        infoes.append(float(result.infoe))
        infocs.append(float(result.infoc))
        actives.append(int(np.sum(result.active_latents)))

    assert latent_dim is not None
    return {
        "checkpoints": [str(p) for _, p in ckpts],
        "epochs": epochs_list,
        "infom": infoms,
        "infoe": infoes,
        "infoc": infocs,
        "n_active_latents": actives,
        "latent_dim": latent_dim,
    }


def save_infomec_epoch_series_artifacts(
    payload: Optional[Dict[str, Any]],
    outdir: Path,
    beta_x_axis: Literal["auto", "linear", "log"] = "auto",
) -> None:
    """Write infomec_over_epoch_checkpoints.png + infomec_epoch_checkpoint_curve.json if payload is set."""
    if payload is None:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    latent_dim = int(payload["latent_dim"])
    plot_infomec_epoch_trajectory(
        payload["epochs"],
        np.asarray(payload["infom"], dtype=np.float64),
        np.asarray(payload["infoe"], dtype=np.float64),
        np.asarray(payload["infoc"], dtype=np.float64),
        np.asarray(payload["n_active_latents"], dtype=np.float64),
        latent_dim,
        outdir / "infomec_over_epoch_checkpoints.png",
        beta_x_axis=beta_x_axis,
    )

    with open(outdir / "infomec_epoch_checkpoint_curve.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_epoch_checkpoint_infomec_curve(
    run_dir: Path,
    outdir: Path,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    batch_size: int,
    device: torch.device,
    discrete_latents: bool,
    *,
    beta_x_axis: Literal["auto", "linear", "log"] = "auto",
) -> None:
    payload = collect_epoch_checkpoint_infomec_series(
        run_dir,
        obs1,
        obs2,
        labels,
        sources,
        batch_size,
        device,
        discrete_latents,
    )
    save_infomec_epoch_series_artifacts(payload, outdir, beta_x_axis=beta_x_axis)


def save_eval_artifacts_after_encode(
    outdir: Path,
    run_dir: Path,
    latents: np.ndarray,
    sources: np.ndarray,
    result: InfoMECResult,
    source_names: List[str],
    obs1: np.ndarray,
    batch_size: int,
    variant: dict,
    dataset_path: Path,
    split: str,
    source_keys: List[str],
    eval_timestamp: str,
) -> Dict[str, Any]:
    """Persist latents, NMI heatmap, PCA/LDA, infomec_metrics.json, summary.txt under outdir.

    latents: (N, L) — N preference sets after split, L latent dim.
    sources: (N, S) — S binned/discrete source columns.
    obs1: (N, T, ...) or truncated set dim T per variant set_size.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    n_active_latents = int(np.sum(result.active_latents))
    latent_dim = int(latents.shape[1])

    np.save(outdir / "latents.npy", latents)
    np.save(outdir / "sources.npy", sources)
    np.save(outdir / "nmi.npy", result.nmi)

    latent_names = [f"z{j}" for j in range(latents.shape[1])]
    save_heatmap(
        result.nmi,
        source_names,
        latent_names,
        title="InfoMEC normalized MI (sources x latents)",
        outpath=str(outdir / "infomec_nmi_heatmap.png"),
    )

    first_source = sources[:, 0]
    pca_var = maybe_save_pca(
        latents,
        first_source,
        str(outdir / "pca_by_first_source.png"),
        f"PCA colored by {source_names[0]}",
    )
    maybe_save_lda(
        latents,
        first_source,
        str(outdir / "lda_by_first_source.png"),
        f"LDA by {source_names[0]}",
    )

    metrics = result.to_dict()
    metrics.update(
        {
            "run_dir": str(run_dir),
            "eval_dir": str(outdir),
            "eval_timestamp": eval_timestamp,
            "dataset_path": str(dataset_path),
            "env": variant.get("env"),
            "batch_size_used": batch_size,
            "set_length_used": int(variant["set_size"]),
            "source_keys": source_keys,
            "source_names": source_names,
            "num_samples": int(len(obs1)),
            "num_sources": int(sources.shape[1]),
            "latent_dim": int(latents.shape[1]),
            "split": split,
            "pca_explained_variance_ratio": pca_var,
        }
    )
    if (outdir / "infomec_epoch_checkpoint_curve.json").is_file():
        metrics["infomec_epoch_checkpoint_curve_json"] = str(
            outdir / "infomec_epoch_checkpoint_curve.json"
        )
        metrics["infomec_over_epoch_checkpoints_png"] = str(
            outdir / "infomec_over_epoch_checkpoints.png"
        )
    with open(outdir / "infomec_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"infom={metrics['infom']:.6f}\n")
        f.write(f"infoc={metrics['infoc']:.6f}\n")
        f.write(f"infoe={metrics['infoe']:.6f}\n")
        f.write(f"source_explicitness={metrics['source_explicitness']}\n")
        f.write(f"active_latents={n_active_latents}/{latent_dim}\n")

    return metrics


def _infer_vpl_root(run_dir: Path) -> Path:
    """Ancestor of run_dir that contains pref_learn/ (repo VPL layout)."""
    resolved = run_dir.resolve()
    for anc in [resolved, *resolved.parents]:
        if (anc / "pref_learn").is_dir():
            return anc
    return Path.cwd()


def resolve_dataset_path(
    run_dir: Path, dataset_rel: str, override: Optional[Path]
) -> Path:
    if override is not None:
        p = override.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--dataset not found: {p}")
        return p

    rel = dataset_rel.strip()
    if rel.startswith("./"):
        rel = rel[2:]

    root = _infer_vpl_root(run_dir)
    cand = (root / rel).resolve()
    if cand.is_file():
        return cand

    cand_cwd = (Path.cwd() / rel).resolve()
    if cand_cwd.is_file():
        return cand_cwd

    raise FileNotFoundError(
        f"Could not find dataset for variant dataset_path={dataset_rel!r}. "
        f"Tried {cand} and {cand_cwd}. Set --dataset or run from the VPL root."
    )


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


def plot_nmi_heatmap_on_ax(
    ax: Axes,
    mat: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    title: str,
    title_fontsize: int = 8,
) -> Any:
    """Draw NMI heatmap on an existing axes (for sweep grid); mirrors save_heatmap styling."""
    im = ax.imshow(mat, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=6)
    ax.set_title(title, fontsize=title_fontsize)
    return im


def resolve_eval_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def prepare_infomec_eval_arrays(
    run_dir: Path,
    *,
    split: str,
    source_keys: str,
    num_bins: int,
    batch_size: Optional[int],
    set_length: Optional[int],
    dataset_override: Optional[Path],
    device_str: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    int,
    dict,
    Path,
    torch.device,
]:
    """Load dataset + build observation tensors and source matrix for one training run_dir."""
    run_dir = run_dir.expanduser().resolve()
    variant_path = run_dir / "variant.json"
    if not variant_path.is_file():
        raise FileNotFoundError(f"Missing variant.json under run_dir: {variant_path}")
    with open(variant_path, encoding="utf-8") as vf:
        meta = json.load(vf)
    variant = meta["variant"]
    bs = int(variant["batch_size"]) if batch_size is None else int(batch_size)
    sl = int(variant["set_size"]) if set_length is None else int(set_length)
    dataset_path = resolve_dataset_path(
        run_dir, str(variant["dataset_path"]), dataset_override
    )
    with open(dataset_path, "rb") as f:
        batch = pickle.load(f)
    n = len(batch["observations"])
    idx = split_indices(n, split)
    obs1 = maybe_truncate(np.asarray(batch["observations"])[idx], sl)
    obs2 = maybe_truncate(np.asarray(batch["observations_2"])[idx], sl)
    labels = maybe_truncate(np.asarray(batch["labels"])[idx], sl)
    device = resolve_eval_device(device_str)
    sk = resolve_source_keys(batch, source_keys)
    source_cols = []
    source_names: List[str] = []
    for key in sk:
        cols, names = infer_source_columns(np.asarray(batch[key])[idx], key, num_bins)
        source_cols.append(cols)
        source_names.extend(names)
    sources = np.concatenate(source_cols, axis=1)
    return (
        obs1,
        obs2,
        labels,
        sources,
        source_names,
        sk,
        bs,
        variant,
        dataset_path,
        device,
    )


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


def main(args: AnalysisCLI) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    variant_path = run_dir / "variant.json"
    ckpt_path = run_dir / args.ckpt_name

    if not variant_path.is_file():
        raise FileNotFoundError(f"Missing variant.json under run_dir: {variant_path}")

    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = run_dir / "evals" / eval_timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    (
        obs1,
        obs2,
        labels,
        sources,
        source_names,
        source_keys,
        batch_size,
        variant,
        dataset_path,
        device,
    ) = prepare_infomec_eval_arrays(
        run_dir,
        split=args.split,
        source_keys=args.source_keys,
        num_bins=args.num_bins,
        batch_size=args.batch_size,
        set_length=args.set_length,
        dataset_override=args.dataset,
        device_str=args.device,
    )

    if args.epoch_checkpoint_plot:
        run_epoch_checkpoint_infomec_curve(
            run_dir,
            outdir,
            obs1,
            obs2,
            labels,
            sources,
            batch_size,
            device,
            args.discrete_latents,
            beta_x_axis=args.beta_x_axis,
        )

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint {args.ckpt_name!r} under run_dir: {ckpt_path}"
        )

    model = torch.load(ckpt_path, map_location=device, weights_only=False).to(device)
    model.eval()

    latents = batched_encode(model, obs1, obs2, labels, batch_size, device)

    result = compute_infomec(
        sources=sources, latents=latents, discrete_latents=args.discrete_latents
    )

    metrics = save_eval_artifacts_after_encode(
        outdir,
        run_dir,
        latents,
        sources,
        result,
        source_names,
        obs1,
        batch_size,
        variant,
        dataset_path,
        args.split,
        source_keys,
        eval_timestamp,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(tyro.cli(AnalysisCLI))
