#!/usr/bin/env python3
"""Aggregate InfoMEC / NMI plots across subruns under a sweep training directory."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib
import tyro
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_infomec as plt_infomec  # noqa: E402
from pref_learn.utils.disentangle_sweep import discover_sweep_subruns, sweep_from_variant
from pref_learn.utils.infomec import compute_infomec  # noqa: E402


@dataclass
class SweepCLI:
    """Aggregate InfoMEC / NMI across beta subdirs under ``input_dir``."""

    input_dir: Path
    """Sweep root with ``subruns/<task>/variant.json`` (+ checkpoint). Legacy flat subruns also supported."""

    model_selection: plt_infomec.ModelSelection = plt_infomec.ModelSelection.best
    """``best``: ``best_model.pt``; ``final``: latest ``checkpoints/model_<epoch>.pt``."""

    outdir: Optional[Path] = None
    """Aggregate PNG/CSV/manifest; default ``<input_dir>/evals_sweep/<timestamp>_model=<best|final>/``."""

    split: Literal["all", "train", "eval"] = "all"
    """Dataset split for metrics (same as single-run analysis)."""

    batch_size: Optional[int] = None
    """Override batch size; default from each variant.json."""

    set_length: Optional[int] = None
    """Override set truncation; default from each variant.json."""

    num_bins: int = 10
    """Histogram bins when discretizing continuous sources."""

    device: str = "auto"
    """torch device name, or 'auto'."""

    nmi_heatmap_active_dots: bool = True
    """If True, mark active latents on aggregate and per-run NMI heatmaps (red dots). Tyro: ``--no-nmi-heatmap-active-dots`` turns this off."""

    latent_cluster_method: plt_infomec.LatentClusterMethod = (
        plt_infomec.LatentClusterMethod.tsne
    )
    """2D latent embedding colored by ``source_mode``: ``tsne``, ``umap``, or ``none`` to skip."""

    tsne_perplexity: Optional[float] = None
    """t-SNE perplexity; default min(30, max(2, N//3)) for N samples."""

    tsne_max_iter: Optional[int] = None
    """t-SNE ``max_iter``; sklearn default 1000 when unset. Use e.g. ``50`` for fast debug sweeps."""

    umap_n_neighbors: Optional[int] = None
    """UMAP ``n_neighbors``; default min(15, N-1) with N>=3."""

    umap_min_dist: Optional[float] = None
    """UMAP ``min_dist``; default 0.1."""

    learned_reward_mode: Literal["raw", "prob"] = "prob"
    """Ignored when the model supports probabilistic rewards (both prob+raw are saved)."""

    learned_reward_max_comp: int = 512
    """Max comparison states ``s_B`` for prob learned rewards."""


def beta_from_variant(variant: dict) -> float:
    return sweep_from_variant(variant=variant)[1]


def discover_sweep_run_dirs(
    input_dir: Path,
    *,
    model_selection: plt_infomec.ModelSelection,
) -> List[Path]:
    def sort_key(p: Path) -> tuple:
        with open(p / "variant.json", encoding="utf-8") as vf:
            v = json.load(vf)["variant"]
        return (beta_from_variant(v), p.name)

    dirs = [
        p
        for p in discover_sweep_subruns(input_dir)
        if plt_infomec.run_dir_has_checkpoint(p, model_selection=model_selection)
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
        label="# active latents",
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

    h_left, lab_left = ax_left.get_legend_handles_labels()
    h_right, lab_right = ax_right.get_legend_handles_labels()
    infomec_h: list = []
    infomec_l: list = []
    nmi_h: list = []
    nmi_l: list = []
    for h, lab in zip(h_left, lab_left):
        if lab == "max NMI":
            nmi_h.append(h)
            nmi_l.append(lab)
        else:
            infomec_h.append(h)
            infomec_l.append(lab)
    ax_left.legend(
        infomec_h + h_right + nmi_h,
        infomec_l + lab_right + nmi_l,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=5,
        fontsize=8,
        frameon=True,
    )


def _effective_sweep_eval_settings(
    run_dir: Path,
    *,
    batch_size: Optional[int],
    set_length: Optional[int],
) -> Tuple[str, int, int]:
    """Resolved dataset path + batch/set sizes used for InfoMEC eval (must match across a sweep)."""
    with open(run_dir / "variant.json", encoding="utf-8") as vf:
        variant = json.load(vf)["variant"]
    dataset_path = plt_infomec.resolve_infomec_dataset_path(
        run_dir, variant, override=None
    )
    bs = int(variant["batch_size"]) if batch_size is None else int(batch_size)
    sl = int(variant["set_size"]) if set_length is None else int(set_length)
    return str(dataset_path.resolve()), bs, sl


def assert_sweep_runs_share_eval_data(
    run_dirs: List[Path],
    *,
    batch_size: Optional[int],
    set_length: Optional[int],
) -> None:
    if len(run_dirs) <= 1:
        return
    ref = _effective_sweep_eval_settings(
        run_dirs[0], batch_size=batch_size, set_length=set_length
    )
    for run_dir in run_dirs[1:]:
        fp = _effective_sweep_eval_settings(
            run_dir, batch_size=batch_size, set_length=set_length
        )
        if fp != ref:
            raise ValueError(
                "Sweep subruns must share infomec dataset_path, batch_size, and set_size for "
                f"shared eval arrays; {run_dirs[0].name} has {ref!r}, "
                f"{run_dir.name} has {fp!r}"
            )


def _color_log_beta_one_tick_red(ax: Axes) -> None:
    """If β=1 (10^0) appears as a major x tick, draw that label in red (log-β sweeps)."""
    for loc, tlab in zip(ax.get_xticks(), ax.get_xticklabels()):
        if loc > 0 and np.isclose(loc, 1.0, rtol=0.0, atol=1e-9):
            tlab.set_color("red")


def main(args: SweepCLI) -> None:
    eval_ts = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_model={args.model_selection.value}"
    )
    input_dir = args.input_dir.expanduser().resolve()
    outdir: Path
    if args.outdir is not None:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = input_dir / "evals_sweep" / eval_ts
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_sweep_run_dirs(input_dir, model_selection=args.model_selection)
    if not run_dirs:
        ckpt_hint = (
            "best_model.pt"
            if args.model_selection == plt_infomec.ModelSelection.best
            else "checkpoints/model_<epoch>.pt"
        )
        raise SystemExit(f"No subruns with variant.json + {ckpt_hint} under {input_dir}")

    ckpt_display = plt_infomec.model_selection_display_label(args.model_selection)

    device = plt_infomec.resolve_eval_device(args.device)
    assert_sweep_runs_share_eval_data(
        run_dirs,
        batch_size=args.batch_size,
        set_length=args.set_length,
    )
    (
        obs1,  # (Q, Qset, T, O)
        obs2,
        labels,  # (Q, Qset, T)
        sources,  # (Q, S)
        source_names,
        source_keys,
        batch_size,
        _shared_variant,
        dataset_path,
        _dev,
    ) = plt_infomec.prepare_infomec_eval_arrays(
        run_dirs[0],
        split=args.split,
        num_bins=args.num_bins,
        batch_size=args.batch_size,
        set_length=args.set_length,
        dataset_override=None,
        device_str=args.device,
    )
    mode_color = plt_infomec.source_mode_labels_for_split(
        dataset_path=dataset_path, split=args.split
    )
    mode_names = plt_infomec.source_mode_names_from_dataset(dataset_path=dataset_path)
    print(
        f"Loaded shared eval split={args.split!r} pref_dataset="
        f"{_shared_variant['pref_dataset']!r} once from {dataset_path} "
        f"({len(obs1)} queries)"
    )

    runs_data: List[Dict[str, Any]] = []

    n_runs = len(run_dirs)
    cluster_part = args.latent_cluster_method.value
    print(
        f"\nFound {n_runs} runs to process "
        f"(device={device}, latent cluster={cluster_part}, ckpt={ckpt_display})..."
    )
    for i, run_dir in enumerate(run_dirs, start=1):
        with open(run_dir / "variant.json", encoding="utf-8") as vf:
            variant = json.load(vf)["variant"]
        sweep_param_name, beta, sweep_param_label = sweep_from_variant(variant=variant)

        ckpt_path, ckpt_label, selected_epoch, model_selection_tag = (
            plt_infomec.resolve_run_checkpoint(
                run_dir, model_selection=args.model_selection
            )
        )
        model = torch.load(ckpt_path, map_location=device, weights_only=False).to(device)
        model.eval()
        discrete_latents = plt_infomec.resolve_discrete_latents(
            model=model, variant=variant
        )
        latents = plt_infomec.batched_encode(
            model, obs1, obs2, labels, batch_size, device
        )  # (Q, Z)
        result = compute_infomec(
            sources=sources, latents=latents, discrete_latents=discrete_latents
        )
        latent_dim = int(latents.shape[1])
        latent_names = [f"z{j}" for j in range(latent_dim)]
        n_varies, n_inform, n_active, _latent_dim = (
            plt_infomec.latent_mask_counts_from_result(result)
        )

        epoch_series: Optional[Dict[str, Any]] = (
            plt_infomec.collect_epoch_checkpoint_infomec_series(
                run_dir,
                obs1,
                obs2,
                labels,
                sources,
                batch_size,
                device,
                discrete_latents,
            )
        )
        run_title = plt_infomec.format_infomec_run_title(
            infom=float(result.infom),
            infoe=float(result.infoe),
            infoc=float(result.infoc),
            n_varies=n_varies,
            n_inform=n_inform,
            n_active=n_active,
            latent_dim=latent_dim,
            run_label=run_dir.name,
            ckpt_name=ckpt_label,
            best_epoch=selected_epoch,
            best_selection=model_selection_tag,
        )

        per_run_eval = plt_infomec.eval_run_outdir(
            input_dir=run_dir, eval_timestamp=eval_ts, kind="subsweep"
        )
        comment = str(variant.get("comment", run_dir.name)).strip()
        plt_infomec.save_infomec_epoch_series_artifacts(
            epoch_series,
            per_run_eval,
            title=run_title,
            best_epoch=selected_epoch,
            best_selection=model_selection_tag,
        )
        plt_infomec.save_eval_artifacts_after_encode(
            per_run_eval,
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
            eval_ts,
            run_title=run_title,
            nmi_heatmap_active_dots=args.nmi_heatmap_active_dots,
            latent_cluster_method=args.latent_cluster_method,
            tsne_perplexity=args.tsne_perplexity,
            tsne_max_iter=args.tsne_max_iter,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            model=model,
            obs2=obs2,
            labels=labels,
            device=device,
            learned_reward_mode=args.learned_reward_mode,
            learned_reward_max_comp=args.learned_reward_max_comp,
            learned_reward_sweep_outdir=outdir,
            learned_reward_sweep_comment=comment,
        )

        runs_data.append(
            {
                "run_dir": str(run_dir),
                "name": run_dir.name,
                "beta": beta,
                "sweep_param_name": sweep_param_name,
                "sweep_param_value": beta,
                "sweep_param_label": sweep_param_label,
                "eval_dir": str(per_run_eval),
                "run_title": run_title,
                "nmi": np.asarray(result.nmi),
                "source_names": source_names,
                "latent_names": latent_names,
                "infom": float(result.infom),
                "infoe": float(result.infoe),
                "infoc": float(result.infoc),
                "max_nmi": float(result.max_nmi),
                "n_varies": n_varies,
                "n_inform": n_inform,
                "n_active": n_active,
                "active_latents": np.asarray(result.active_latents, dtype=bool),
                "latent_dim": latent_dim,
                "epoch_series": epoch_series,
                "selected_epoch": selected_epoch,
                "ckpt_label": ckpt_label,
                "model_selection_tag": model_selection_tag,
                **(
                    {"latents": latents}
                    if plt_infomec.latent_cluster_enabled(
                        method=args.latent_cluster_method
                    )
                    else {}
                ),
            }
        )
        print(
            f"======= [{i}/{n_runs}] Finished {sweep_param_label}={beta} ({run_dir.name}) ========"
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
        title = plt_infomec.format_infomec_run_title(
            infom=float(r["infom"]),
            infoe=float(r["infoe"]),
            infoc=float(r["infoc"]),
            n_varies=int(r["n_varies"]),
            n_inform=int(r["n_inform"]),
            n_active=int(r["n_active"]),
            latent_dim=int(r["latent_dim"]),
            run_label=r["name"],
            ckpt_name=str(r["ckpt_label"]),
            best_epoch=r.get("selected_epoch"),
            best_selection=str(r["model_selection_tag"]),
        )
        al = r["active_latents"] if args.nmi_heatmap_active_dots else None
        plt_infomec.plot_nmi_heatmap_on_ax(
            ax,
            r["nmi"],
            r["source_names"],
            r["latent_names"],
            title,
            title_fontsize=7,
            active_latents=al,
        )
        # plt_infomec.plot_nmi_heatmap_on_ax(
        #     ax,
        #     r["nmi"],
        #     r["source_names"],
        #     r["latent_names"],
        #     title,
        #     title_fontsize=7,
        #     active_latents=r["active_latents"],
        # )
    for j in range(i + 1, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)
    fig_h.suptitle(f"NMI heatmaps ({ckpt_display})", fontsize=11, y=1.02)
    plt.tight_layout()
    fig_h.savefig(outdir / "aggregate_nmi_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig_h)

    projection_grid_path = (
        outdir / f"aggregate_{args.latent_cluster_method.value}_by_source_mode.png"
    )
    if plt_infomec.latent_cluster_enabled(method=args.latent_cluster_method):
        plt_infomec.save_aggregate_latent_projection_sweep_grid(
            runs_data,
            projection_grid_path,
            mode_color,
            method=args.latent_cluster_method,
            mode_names=mode_names,
            tsne_perplexity=args.tsne_perplexity,
            tsne_max_iter=args.tsne_max_iter,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            ckpt_label=ckpt_display,
        )

    betas = np.array([r["beta"] for r in runs_data], dtype=np.float64)
    infoms = np.array([r["infom"] for r in runs_data], dtype=np.float64)
    infoes = np.array([r["infoe"] for r in runs_data], dtype=np.float64)
    infocs = np.array([r["infoc"] for r in runs_data], dtype=np.float64)
    max_nmis = np.array([r["max_nmi"] for r in runs_data], dtype=np.float64)
    n_act = np.array([r["n_active"] for r in runs_data], dtype=np.float64)
    Ls = np.array([r["latent_dim"] for r in runs_data], dtype=np.int64)

    L_unique = np.unique(Ls)
    if L_unique.size == 1:
        ld_part = f"L={int(L_unique[0])}"
    else:
        ld_part = "L=" + ",".join(str(int(x)) for x in sorted(L_unique.tolist()))
    sweep_param_label = runs_data[0]["sweep_param_label"] if runs_data else "β"
    supt = f"InfoMEC vs {sweep_param_label} ({ckpt_display}), {ld_part}"

    beta_plot_path = outdir / "aggregate_infomec_vs_beta_log.png"
    if np.any(betas <= 0):
        print(f"Skip {beta_plot_path.name}: all β must be >0 for log x; got {betas!r}")
    else:
        fig_b, ax_b = plt.subplots(figsize=(8, 5))
        ax_b.plot(
            betas,
            infoms,
            marker="o",
            label="InfoM",
            color=plt_infomec.InfoMECCurveColor.infom,
        )
        ax_b.plot(
            betas,
            infoes,
            marker="o",
            label="InfoE",
            color=plt_infomec.InfoMECCurveColor.infoe,
        )
        ax_b.plot(
            betas,
            infocs,
            marker="o",
            label="InfoC",
            color=plt_infomec.InfoMECCurveColor.infoc,
        )
        ax_b.plot(
            betas,
            max_nmis,
            linestyle=":",
            color="gray",
            linewidth=1.2,
            label="max NMI",
        )
        ax_b.set_xscale("log")
        ax_b.set_xlabel(
            f"{runs_data[0]['sweep_param_label']} ({runs_data[0]['sweep_param_name']}) — log scale"
        )
        ax_b.grid(alpha=0.3, which="both")
        ax_b.set_ylabel("InfoM / InfoE / InfoC / max NMI ([0, 1])")
        ax_b.set_ylim(-0.02, 1.02)
        _add_active_latents_twin_beta_sweep(ax_b, betas, n_act, Ls)
        fig_b.suptitle(supt, fontsize=11)
        fig_b.tight_layout(rect=[0, 0.1, 1, 0.99])
        fig_b.canvas.draw()
        _color_log_beta_one_tick_red(ax_b)
        fig_b.savefig(
            beta_plot_path,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.12,
        )
        plt.close(fig_b)

    ncols_e = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows_e = math.ceil(n / ncols_e)
    fig_e, axes_e = plt.subplots(
        nrows_e, ncols_e, figsize=(4.0 * ncols_e, 3.5 * nrows_e), squeeze=False
    )
    epoch_xmax = 0
    for r in runs_data:
        ep = r.get("epoch_series")
        if ep is not None and ep.get("epochs"):
            epoch_xmax = max(epoch_xmax, max(int(e) for e in ep["epochs"]))
    first_legend_ax_pair: Optional[Tuple[Any, Any]] = None
    last_i = 0
    for i, r in enumerate(runs_data):
        row, col = divmod(i, ncols_e)
        ax_l = axes_e[row][col]
        ax_r = ax_l.twinx()
        ep = r["epoch_series"]
        if ep is not None:
            sub_title = plt_infomec.format_infomec_run_title(
                infom=float(r["infom"]),
                infoe=float(r["infoe"]),
                infoc=float(r["infoc"]),
                n_varies=int(r["n_varies"]),
                n_inform=int(r["n_inform"]),
                n_active=int(r["n_active"]),
                latent_dim=int(r["latent_dim"]),
                run_label=r["name"],
                ckpt_name=str(r["ckpt_label"]),
                best_epoch=r.get("selected_epoch"),
                best_selection=str(r["model_selection_tag"]),
            )
            plt_infomec.draw_infomec_epoch_trajectory_axes(
                ax_l,
                ax_r,
                ep["epochs"],
                np.asarray(ep["infom"], dtype=np.float64),
                np.asarray(ep["infoe"], dtype=np.float64),
                np.asarray(ep["infoc"], dtype=np.float64),
                np.asarray(ep["n_active_latents"], dtype=np.float64),
                int(ep["latent_dim"]),
                title=sub_title,
                show_legend=False,
                selected_epoch=r.get("selected_epoch"),
            )
            if first_legend_ax_pair is None:
                first_legend_ax_pair = (ax_l, ax_r)
        else:
            ax_l.set_title(
                f"{r['sweep_param_label']}={r['sweep_param_value']:.4g} (no model_*.pt)"
            )
            ax_l.set_ylim(0, 1)
        if epoch_xmax > 0:
            ax_l.set_xlim(0, epoch_xmax)
        last_i = i
    for j in range(last_i + 1, nrows_e * ncols_e):
        row, col = divmod(j, ncols_e)
        axes_e[row][col].set_visible(False)
    if first_legend_ax_pair is not None:
        ax0, ax1 = first_legend_ax_pair
        lines_leg = ax0.get_lines() + ax1.get_lines()
        leg_labels = [ln.get_label() for ln in lines_leg]
        fig_e.legend(
            lines_leg,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=4,
            fontsize=8,
            frameon=True,
        )
    fig_e.suptitle(
        f"InfoMEC vs epoch (best selected by {plt_infomec.BEST_CHECKPOINT_SELECTION})",
        fontsize=11,
        y=1.01,
    )
    fig_e.tight_layout(rect=[0, 0.05, 1, 0.99])
    fig_e.savefig(
        outdir / "aggregate_infomec_vs_epoch_grid.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig_e)

    csv_path = outdir / "sweep_runs_summary.csv"
    fieldnames = [
        "name",
        "sweep_param_name",
        "sweep_param_value",
        "sweep_param_label",
        "beta",
        "run_dir",
        "eval_dir",
        "infom",
        "infoe",
        "infoc",
        "max_nmi",
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
                    "sweep_param_name": r["sweep_param_name"],
                    "sweep_param_value": r["sweep_param_value"],
                    "sweep_param_label": r["sweep_param_label"],
                    "beta": r["beta"],
                    "run_dir": r["run_dir"],
                    "eval_dir": r["eval_dir"],
                    "infom": r["infom"],
                    "infoe": r["infoe"],
                    "infoc": r["infoc"],
                    "max_nmi": r["max_nmi"],
                    "n_active": r["n_active"],
                    "latent_dim": r["latent_dim"],
                }
            )

    manifest = {
        "input_dir": str(input_dir),
        "eval_timestamp": eval_ts,
        "model_selection": args.model_selection.value,
        "aggregate_outdir": str(outdir),
        "sweep_param_name": runs_data[0]["sweep_param_name"] if runs_data else None,
        "sweep_param_label": runs_data[0]["sweep_param_label"] if runs_data else None,
        "runs": [
            {
                "name": r["name"],
                "sweep_param_name": r["sweep_param_name"],
                "sweep_param_value": r["sweep_param_value"],
                "sweep_param_label": r["sweep_param_label"],
                "beta": r["beta"],
                "run_dir": r["run_dir"],
                "eval_dir": r["eval_dir"],
                "infom": r["infom"],
                "infoe": r["infoe"],
                "infoc": r["infoc"],
                "max_nmi": r["max_nmi"],
                "n_active": r["n_active"],
                "latent_dim": r["latent_dim"],
            }
            for r in runs_data
        ],
    }
    with open(outdir / "sweep_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Per-run evals (timestamp {eval_ts}) under each subdir: "
        f".../evals_subsweep/{eval_ts}/"
    )
    for reward_subdir in ("learned_rewards_mode_prob", "learned_rewards_mode_raw"):
        learned_rewards_dir = outdir / reward_subdir
        if learned_rewards_dir.is_dir() and any(learned_rewards_dir.glob("*.png")):
            print(f"Wrote learned-reward copies under {learned_rewards_dir}/")
    print(f"Wrote {outdir / 'aggregate_nmi_heatmaps.png'}")
    if plt_infomec.latent_cluster_enabled(method=args.latent_cluster_method):
        print(f"Wrote {projection_grid_path}")
    if not np.any(betas <= 0):
        print(f"Wrote {beta_plot_path}")
    print(f"Wrote {outdir / 'aggregate_infomec_vs_epoch_grid.png'}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(SweepCLI))
