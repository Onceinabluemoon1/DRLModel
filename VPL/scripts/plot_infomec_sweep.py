#!/usr/bin/env python3
"""Aggregate InfoMEC / NMI plots across subruns under a sweep training directory."""

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_infomec as plt_info  # noqa: E402
from pref_learn.utils.infomec import compute_infomec  # noqa: E402


@dataclass
class SweepCLI:
    """Aggregate InfoMEC / NMI across beta subdirs under sweep-root."""

    sweep_root: Path
    """Parent directory; each child with variant.json + best_model.pt is one run."""

    outdir: Optional[Path] = None
    """Aggregate PNG/CSV/manifest; default <sweep-root>/evals/<timestamp>/."""

    split: Literal["all", "train", "eval"] = "eval"
    """Dataset split for metrics (same as single-run analysis)."""

    source_keys: str = "auto"
    """Comma-separated source fields or 'auto'."""

    batch_size: Optional[int] = None
    """Override batch size; default from each variant.json."""

    set_length: Optional[int] = None
    """Override set truncation; default from each variant.json."""

    discrete_latents: bool = False
    """If True, treat latents as discrete for InfoMEC."""

    num_bins: int = 10
    """Histogram bins when discretizing continuous sources."""

    device: str = "auto"
    """torch device name, or 'auto'."""

    beta_x_axis: Literal["auto", "linear", "log"] = "auto"
    """``auto`` / ``linear`` / ``log`` for aggregate β plot; same mode for per-run checkpoint-epoch curves (PNG + epoch grid)."""


def beta_from_variant(variant: dict) -> float:
    return float(variant.get("kl_weight", 0.0))


def beta_sweep_prefers_log_x(betas: np.ndarray) -> bool:
    """Use log-scaled β only when values span orders of magnitude and bunch toward small β.

    Keeps linear x for comparable, evenly spaced β (e.g. 0.2 … 2.0).
    """
    b = np.sort(np.unique(np.asarray(betas, dtype=np.float64)))
    if b.size < 3 or np.any(b <= 0) or not np.all(np.isfinite(b)):
        return False
    dynamic_range = float(b[-1] / b[0])
    if dynamic_range < 10.0:
        return False
    median = float(np.median(b))
    b0, b1 = float(b[0]), float(b[-1])
    linear_mid = 0.5 * (b0 + b1)
    # Most runs packed on the low-β side of the linear range (e.g. 0.01…100 with many points <10).
    skewed_toward_small = median < 0.22 * linear_mid + 0.78 * b0
    if skewed_toward_small:
        return True
    return dynamic_range >= 50.0


def discover_sweep_run_dirs(sweep_root: Path) -> List[Path]:
    root = sweep_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--sweep-root is not a directory: {root}")

    def sort_key(p: Path) -> tuple:
        with open(p / "variant.json", encoding="utf-8") as vf:
            v = json.load(vf)["variant"]
        return (beta_from_variant(v), p.name)

    dirs = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and (p / "variant.json").is_file()
        and (p / "best_model.pt").is_file()
    ]
    return sorted(dirs, key=sort_key)


def _add_active_latents_twin_beta_sweep(
    ax_left: plt.Axes,
    betas: np.ndarray,
    n_active: np.ndarray,
    latent_dims: np.ndarray,
) -> None:
    """Twin y for active counts; y plotted as n_active/L per run; ticks use max L."""
    Lmax = int(np.max(latent_dims)) if latent_dims.size else 1
    Lmax = max(Lmax, 1)
    n_norm = n_active.astype(np.float64) / np.maximum(latent_dims.astype(np.float64), 1.0)
    ax_right = ax_left.twinx()
    ax_right.plot(
        betas,
        n_norm,
        linestyle="--",
        color="black",
        linewidth=0.9,
        label="Active latents #",
    )
    ax_right.set_ylabel("Active latents (count)")
    ax_right.set_ylim(-0.02, 1.02)
    if Lmax <= 12:
        tick_pos = np.arange(Lmax + 1, dtype=np.float64) / float(Lmax)
        tick_lbl = [str(i) for i in range(Lmax + 1)]
    else:
        n_ticks = 11
        tick_pos = np.linspace(0.0, 1.0, n_ticks)
        tick_lbl = [str(int(round(t * Lmax))) for t in tick_pos]
    ax_right.set_yticks(tick_pos)
    ax_right.set_yticklabels(tick_lbl)
    h1, lab1 = ax_left.get_legend_handles_labels()
    h2, lab2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, lab1 + lab2, loc="best", fontsize=8)


def main(args: SweepCLI) -> None:
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(args.sweep_root).expanduser().resolve()
    outdir: Path
    if args.outdir is not None:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = sweep_root / "evals_sweep" / eval_ts
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_sweep_run_dirs(sweep_root)
    if not run_dirs:
        raise SystemExit(
            f"No subruns with variant.json + best_model.pt under {args.sweep_root}"
        )

    device = plt_info.resolve_eval_device(args.device)
    runs_data: List[Dict[str, Any]] = []

    for rd in run_dirs:
        with open(rd / "variant.json", encoding="utf-8") as vf:
            variant = json.load(vf)["variant"]
        beta = beta_from_variant(variant)

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
            _dev,
        ) = plt_info.prepare_infomec_eval_arrays(
            rd,
            split=args.split,
            source_keys=args.source_keys,
            num_bins=args.num_bins,
            batch_size=args.batch_size,
            set_length=args.set_length,
            dataset_override=None,
            device_str=args.device,
        )

        ckpt = rd / "best_model.pt"
        model = torch.load(ckpt, map_location=device, weights_only=False).to(device)
        model.eval()
        latents = plt_info.batched_encode(model, obs1, obs2, labels, batch_size, device)
        result = compute_infomec(
            sources=sources, latents=latents, discrete_latents=args.discrete_latents
        )
        latent_dim = int(latents.shape[1])
        latent_names = [f"z{j}" for j in range(latent_dim)]
        n_active = int(np.sum(result.active_latents))

        epoch_series: Optional[Dict[str, Any]] = (
            plt_info.collect_epoch_checkpoint_infomec_series(
                rd,
                obs1,
                obs2,
                labels,
                sources,
                batch_size,
                device,
                args.discrete_latents,
            )
        )

        per_run_eval = rd / "evals" / eval_ts
        plt_info.save_infomec_epoch_series_artifacts(
            epoch_series, per_run_eval, beta_x_axis=args.beta_x_axis
        )
        plt_info.save_eval_artifacts_after_encode(
            per_run_eval,
            rd,
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
            eval_ts,
        )

        runs_data.append(
            {
                "run_dir": str(rd),
                "name": rd.name,
                "beta": beta,
                "eval_dir": str(per_run_eval),
                "nmi": np.asarray(result.nmi),
                "source_names": source_names,
                "latent_names": latent_names,
                "infom": float(result.infom),
                "infoe": float(result.infoe),
                "infoc": float(result.infoc),
                "n_active": n_active,
                "latent_dim": latent_dim,
                "epoch_series": epoch_series,
            }
        )

    n = len(runs_data)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    # Short row height: NMI heatmaps are wide; tight vertical size per subplot.
    col_w_in, row_h_in = 3.2, 2.05
    fig_h, axes = plt.subplots(
        nrows, ncols, figsize=(col_w_in * ncols, row_h_in * nrows), squeeze=False
    )
    for i, r in enumerate(runs_data):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        title = (
            f"β={r['beta']:.4g}\n"
            f"InfoM={r['infom']:.3f}  InfoC={r['infoc']:.3f}\n"
            f"{r['n_active']}/{r['latent_dim']} active latents\n"
            f"{r['name']}"
        )
        plt_info.plot_nmi_heatmap_on_ax(
            ax,
            r["nmi"],
            r["source_names"],
            r["latent_names"],
            title,
            title_fontsize=7,
        )
    for j in range(i + 1, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)
    fig_h.suptitle("NMI heatmaps (best_model.pt)", fontsize=11, y=1.02)
    plt.tight_layout()
    fig_h.savefig(outdir / "aggregate_nmi_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig_h)

    betas = np.array([r["beta"] for r in runs_data], dtype=np.float64)
    infoms = np.array([r["infom"] for r in runs_data], dtype=np.float64)
    infoes = np.array([r["infoe"] for r in runs_data], dtype=np.float64)
    infocs = np.array([r["infoc"] for r in runs_data], dtype=np.float64)
    n_act = np.array([r["n_active"] for r in runs_data], dtype=np.float64)
    Ls = np.array([r["latent_dim"] for r in runs_data], dtype=np.int64)

    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    ax_b.plot(
        betas,
        infoms,
        marker="o",
        label="InfoM",
        color=plt_info.INFOMEC_COLOR_INFOM,
    )
    ax_b.plot(
        betas,
        infoes,
        marker="o",
        label="InfoE",
        color=plt_info.INFOMEC_COLOR_INFOE,
    )
    ax_b.plot(
        betas,
        infocs,
        marker="o",
        label="InfoC",
        color=plt_info.INFOMEC_COLOR_INFOC,
    )
    if args.beta_x_axis == "auto":
        use_log_beta = beta_sweep_prefers_log_x(betas)
    elif args.beta_x_axis == "log":
        use_log_beta = True
    else:
        use_log_beta = False
    if use_log_beta and np.any(betas <= 0):
        raise ValueError(
            "log β axis requires strictly positive kl_weight for all runs; "
            f"got betas={betas!r}. Use --beta-x-axis linear or fix variants."
        )
    if use_log_beta:
        ax_b.set_xscale("log")
    ax_b.set_xlabel(
        "β (kl_weight from variant.json)" + (" — log scale" if use_log_beta else "")
    )
    ax_b.set_ylabel("InfoM / InfoE / InfoC ([0, 1])")
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.grid(alpha=0.3, which=("both" if use_log_beta else "major"))
    _add_active_latents_twin_beta_sweep(ax_b, betas, n_act, Ls)
    L_unique = np.unique(Ls)
    if L_unique.size == 1:
        ld_part = f"L={int(L_unique[0])}"
    else:
        ld_part = "L=" + ",".join(str(int(x)) for x in sorted(L_unique.tolist()))
    fig_b.suptitle(f"InfoMEC vs β (best_model.pt), {ld_part}", fontsize=11)
    plt.tight_layout()
    fig_b.savefig(outdir / "aggregate_infomec_vs_beta.png", dpi=200, bbox_inches="tight")
    plt.close(fig_b)

    ncols_e = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows_e = math.ceil(n / ncols_e)
    fig_e, axes_e = plt.subplots(
        nrows_e, ncols_e, figsize=(4.0 * ncols_e, 3.5 * nrows_e), squeeze=False
    )
    last_i = 0
    for i, r in enumerate(runs_data):
        row, col = divmod(i, ncols_e)
        ax_l = axes_e[row][col]
        ax_r = ax_l.twinx()
        ep = r["epoch_series"]
        if ep is not None:
            plt_info.draw_infomec_epoch_trajectory_axes(
                ax_l,
                ax_r,
                ep["epochs"],
                np.asarray(ep["infom"], dtype=np.float64),
                np.asarray(ep["infoe"], dtype=np.float64),
                np.asarray(ep["infoc"], dtype=np.float64),
                np.asarray(ep["n_active_latents"], dtype=np.float64),
                int(ep["latent_dim"]),
                title=f"β={r['beta']:.4g}",
                legend_fontsize=6,
            )
            plt_info.apply_checkpoint_epoch_x_scale(ax_l, ep["epochs"], args.beta_x_axis)
        else:
            ax_l.set_title(f"β={r['beta']:.4g} (no model_*.pt)")
            ax_l.set_ylim(0, 1)
        last_i = i
    for j in range(last_i + 1, nrows_e * ncols_e):
        row, col = divmod(j, ncols_e)
        axes_e[row][col].set_visible(False)
    fig_e.suptitle("InfoMEC vs epoch (per-run checkpoints)", fontsize=11, y=1.01)
    plt.tight_layout()
    fig_e.savefig(
        outdir / "aggregate_infomec_vs_epoch_grid.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig_e)

    csv_path = outdir / "sweep_runs_summary.csv"
    fieldnames = [
        "name",
        "beta",
        "run_dir",
        "eval_dir",
        "infom",
        "infoe",
        "infoc",
        "n_active",
        "latent_dim",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs_data:
            w.writerow(
                {
                    "name": r["name"],
                    "beta": r["beta"],
                    "run_dir": r["run_dir"],
                    "eval_dir": r["eval_dir"],
                    "infom": r["infom"],
                    "infoe": r["infoe"],
                    "infoc": r["infoc"],
                    "n_active": r["n_active"],
                    "latent_dim": r["latent_dim"],
                }
            )

    manifest = {
        "sweep_root": str(sweep_root),
        "eval_timestamp": eval_ts,
        "aggregate_outdir": str(outdir),
        "runs": [
            {
                "name": r["name"],
                "beta": r["beta"],
                "run_dir": r["run_dir"],
                "eval_dir": r["eval_dir"],
                "infom": r["infom"],
                "infoe": r["infoe"],
                "infoc": r["infoc"],
                "n_active": r["n_active"],
                "latent_dim": r["latent_dim"],
            }
            for r in runs_data
        ],
    }
    with open(outdir / "sweep_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Per-run evals (timestamp {eval_ts}) under each subdir: .../evals/{eval_ts}/")
    print(f"Wrote {outdir / 'aggregate_nmi_heatmaps.png'}")
    print(f"Wrote {outdir / 'aggregate_infomec_vs_beta.png'}")
    print(f"Wrote {outdir / 'aggregate_infomec_vs_epoch_grid.png'}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(SweepCLI))
