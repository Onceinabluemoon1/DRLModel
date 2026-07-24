#!/usr/bin/env python3
"""Heatmap grids for beta × latent_dim sweeps from local W&B summaries + InfoMEC."""

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_VPL_ROOT = _SCRIPT_DIR.parent
if str(_VPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_VPL_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_infomec as plt_infomec  # noqa: E402
from pref_learn.utils.disentangle_sweep import (  # noqa: E402
    discover_sweep_subruns,
    sweep_from_variant,
)
from pref_learn.utils.infomec import compute_infomec  # noqa: E402

WandbMetricSpec = tuple[str, str, bool] | None
WandbMetricRow = list[WandbMetricSpec]

VAE_WANDB_METRICS: list[tuple[str, str, bool]] = [
    ("recon_loss", "train/recon_loss", False),
    ("frac_recon2total", "train/frac_recon2total", False),
    ("kld_loss", "train/kld_loss", True),
    ("enc_mean_cross_std", "train/enc_mean_cross_std", True),
]

TCVAE_WANDB_METRICS: list[tuple[str, str, bool]] = [
    ("recon_loss", "train/recon_loss", False),
    ("vanilla_frac_recon2total", "train/vanilla_frac_recon2total", False),
    ("vanilla_kld_loss", "train/vanilla_kld_loss", True),
    ("enc_mean_cross_std", "train/enc_mean_cross_std", True),
]

QLAE_WANDB_METRICS: WandbMetricRow = [
    ("recon_loss", "train/recon_loss", False),
    ("enc_zc_cross_std", "train/enc_zc_cross_std", True),
    ("enc_zq_cross_std", "train/enc_zq_cross_std", True),
    ("enc_zq_unique_mean", "train/enc_zq_unique_mean", False),
]

INFOMEC_METRICS: list[tuple[str, str, bool]] = [
    ("infom", "InfoM", False),
    ("infoe", "InfoE", False),
    ("infoc", "InfoC", False),
]

MODEL_WANDB_METRICS: dict[str, WandbMetricRow] = {
    "VAE": VAE_WANDB_METRICS,
    "TCVAE": TCVAE_WANDB_METRICS,
    "QLAE": QLAE_WANDB_METRICS,
}

SWEEP_METRICS: list[tuple[str, str, bool]] = VAE_WANDB_METRICS + INFOMEC_METRICS

WANDEB_METRICS = VAE_WANDB_METRICS


@dataclass
class LatentSweepHeatmapCLI:
    """Plot train-metric heatmaps over (beta, latent_dim) for a latentsweep directory."""

    input_dir: Path
    """Sweep root with ``subruns/<task>/`` subdirs. Legacy flat subruns also supported."""

    outdir: Optional[Path] = None
    """Default ``<input_dir>/evals_latentgrid/<timestamp>/``."""

    annotate: bool = False
    """Write numeric values in each cell (off by default)."""

    dpi: int = 200

    model_selection: plt_infomec.ModelSelection = plt_infomec.ModelSelection.final
    """Checkpoint for InfoMEC encode: ``best_model.pt`` or final epoch checkpoint."""

    split: Literal["all", "train", "eval"] = "all"
    """InfoMEC dataset split (same as ``plot_infomec_sweep``)."""

    batch_size: Optional[int] = None
    """Override encoder batch size; default from each variant.json."""

    set_length: Optional[int] = None
    """Override set truncation; default from each variant.json set_size."""

    num_bins: int = 10
    """Histogram bins when discretizing continuous sources for InfoMEC."""

    device: str = "auto"
    """Torch device for InfoMEC encode, or ``auto``."""

    infomec_from_dir: Optional[Path] = None
    """Reuse ``infomec/<subrun>/infomec_metrics.json`` from a prior ``evals_latentgrid/`` run."""


def _wandb_summary_path(run_dir: Path) -> Path:
    return run_dir / "wandb" / "latest-run" / "files" / "wandb-summary.json"


def discover_subruns(input_dir: Path) -> list[Path]:
    return discover_sweep_subruns(input_dir)


def normalized_kl_weight(
    *,
    kl_weight: float,
    latent_dim: int,
    annotation_size: int,
) -> float:
    """Effective KL weight when ``normalize_kl=True`` (matches ``VAEModel.kl_normalizer``)."""
    return kl_weight * latent_dim / annotation_size


def annotation_size_from_rows(rows: list[dict]) -> int:
    ann = int(rows[0]["annotation_size"])
    for row in rows[1:]:
        if int(row["annotation_size"]) != ann:
            raise ValueError(
                "Mixed annotation_size across sweep subruns; "
                f"expected one value, also saw {row['annotation_size']}."
            )
    return ann


def resolve_wandb_metrics(model_type: str) -> WandbMetricRow:
    return MODEL_WANDB_METRICS.get(model_type, VAE_WANDB_METRICS)


def _active_wandb_metrics(wandb_metrics: WandbMetricRow) -> list[tuple[str, str, bool]]:
    return [metric for metric in wandb_metrics if metric is not None]


def load_sweep_rows(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in discover_subruns(input_dir):
        summary_path = _wandb_summary_path(run_dir=run_dir)
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing W&B summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        variant = json.loads((run_dir / "variant.json").read_text(encoding="utf-8"))[
            "variant"
        ]
        model_type = str(variant.get("model_type", ""))
        wandb_metrics = resolve_wandb_metrics(model_type=model_type)
        annotation_size = int(
            variant.get("resolved_set_size", variant.get("set_size", -1))
        )
        if annotation_size <= 0:
            raise ValueError(
                f"Missing resolved_set_size in {run_dir / 'variant.json'}; "
                "needed for sweep metadata."
            )
        sweep_param_name, sweep_param_value, sweep_param_label = sweep_from_variant(
            variant=variant
        )
        latent_dim = int(variant["latent_dim"])
        row = {
            "run_dir": str(run_dir),
            "comment": str(variant.get("comment", run_dir.name)),
            "model_type": model_type,
            "latent_dim": latent_dim,
            "sweep_param_name": sweep_param_name,
            "sweep_param_value": float(sweep_param_value),
            "sweep_param_label": sweep_param_label,
            "annotation_size": annotation_size,
            "wandb_step": int(summary.get("_step", -1)),
        }
        if "weight_decay" in variant:
            row["weight_decay"] = float(variant["weight_decay"])
        if sweep_param_name == "kl_weight":
            kl_weight = float(sweep_param_value)
            row["kl_weight"] = kl_weight
            row["normalized_kl_weight"] = normalized_kl_weight(
                kl_weight=kl_weight,
                latent_dim=latent_dim,
                annotation_size=annotation_size,
            )
        for field, wandb_key, _log_scale in _active_wandb_metrics(wandb_metrics):
            if wandb_key not in summary:
                raise KeyError(f"{wandb_key} not in {summary_path}")
            row[field] = float(summary[wandb_key])
        rows.append(row)
    return rows


def _effective_infomec_eval_settings(
    run_dir: Path,
    *,
    batch_size: Optional[int],
    set_length: Optional[int],
) -> tuple[str, int, int]:
    variant = json.loads((run_dir / "variant.json").read_text(encoding="utf-8"))[
        "variant"
    ]
    dataset_path = plt_infomec.resolve_infomec_dataset_path(
        run_dir, variant, override=None
    )
    bs = int(variant["batch_size"]) if batch_size is None else int(batch_size)
    sl = int(variant["set_size"]) if set_length is None else int(set_length)
    return str(dataset_path.resolve()), bs, sl


def _assert_infomec_eval_settings_match(
    run_dirs: list[Path],
    *,
    batch_size: Optional[int],
    set_length: Optional[int],
) -> None:
    if len(run_dirs) <= 1:
        return
    ref = _effective_infomec_eval_settings(
        run_dirs[0], batch_size=batch_size, set_length=set_length
    )
    for run_dir in run_dirs[1:]:
        fp = _effective_infomec_eval_settings(
            run_dir, batch_size=batch_size, set_length=set_length
        )
        if fp != ref:
            raise ValueError(
                "Sweep subruns must share infomec dataset_path, batch_size, and set_size; "
                f"{run_dirs[0].name} has {ref!r}, {run_dir.name} has {fp!r}"
            )


def enrich_rows_with_infomec(
    rows: list[dict],
    *,
    model_selection: plt_infomec.ModelSelection,
    split: Literal["all", "train", "eval"],
    batch_size: Optional[int],
    set_length: Optional[int],
    num_bins: int,
    device_str: str,
    outdir: Path,
) -> list[dict]:
    """Encode each subrun checkpoint once and attach InfoM / InfoE / InfoC to ``rows``."""
    row_by_run_dir = {Path(r["run_dir"]).resolve(): r for r in rows}
    run_dirs = [
        Path(r["run_dir"]).resolve()
        for r in rows
        if plt_infomec.run_dir_has_checkpoint(
            Path(r["run_dir"]), model_selection=model_selection
        )
    ]
    if not run_dirs:
        ckpt_hint = (
            "best_model.pt"
            if model_selection == plt_infomec.ModelSelection.best
            else "checkpoints/model_<epoch>.pt"
        )
        raise RuntimeError(f"No subruns with {ckpt_hint} for InfoMEC eval")

    _assert_infomec_eval_settings_match(
        run_dirs, batch_size=batch_size, set_length=set_length
    )
    (
        obs1,
        obs2,
        labels,
        sources,
        _source_names,
        _source_keys,
        encode_batch_size,
        _shared_variant,
        dataset_path,
        query_source,
        device,
    ) = plt_infomec.prepare_infomec_eval_arrays(
        run_dirs[0],
        split=split,
        num_bins=num_bins,
        batch_size=batch_size,
        set_length=set_length,
        dataset_override=None,
        device_str=device_str,
    )
    print(
        f"InfoMEC eval split={split!r} dataset={dataset_path} "
        f"({len(obs1)} queries, device={device})"
    )

    infomec_root = outdir / "infomec"
    infomec_root.mkdir(parents=True, exist_ok=True)
    for run_dir in tqdm(run_dirs, desc="InfoMEC runs", unit="run"):
        row = row_by_run_dir[run_dir.resolve()]
        ckpt_path, ckpt_label, selected_epoch, model_selection_tag = (
            plt_infomec.resolve_run_checkpoint(run_dir, model_selection=model_selection)
        )
        model = torch.load(ckpt_path, map_location=device, weights_only=False).to(device)
        model.eval()
        variant = json.loads((run_dir / "variant.json").read_text(encoding="utf-8"))[
            "variant"
        ]
        discrete_latents = plt_infomec.resolve_discrete_latents(
            model=model, variant=variant
        )
        latents = plt_infomec.batched_encode(
            model, obs1, obs2, labels, encode_batch_size, device,
            source=query_source,
        )
        result = compute_infomec(
            sources=sources, latents=latents, discrete_latents=discrete_latents
        )
        row["infom"] = float(result.infom)
        row["infoe"] = float(result.infoe)
        row["infoc"] = float(result.infoc)
        row["infomec_max_nmi"] = float(result.max_nmi)
        row["infomec_ckpt"] = ckpt_label
        row["infomec_epoch"] = selected_epoch
        row["infomec_model_selection"] = model_selection_tag

        per_run_out = infomec_root / run_dir.name
        per_run_out.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "run_dir": str(run_dir),
            "comment": row["comment"],
            "model_type": row.get("model_type"),
            "sweep_param_name": row.get("sweep_param_name"),
            "sweep_param_value": row.get("sweep_param_value"),
            "sweep_param_label": row.get("sweep_param_label"),
            "latent_dim": row["latent_dim"],
            "infomec_split": split,
            "infomec_dataset": str(dataset_path),
            "checkpoint": ckpt_label,
            "selected_epoch": selected_epoch,
            "model_selection": model_selection_tag,
            **result.to_dict(),
        }
        if "kl_weight" in row:
            metrics_payload["kl_weight"] = row["kl_weight"]
            metrics_payload["normalized_kl_weight"] = row["normalized_kl_weight"]
        (per_run_out / "infomec_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2),
            encoding="utf-8",
        )
        tqdm.write(
            f"{run_dir.name}: InfoM={result.infom:.3f} "
            f"InfoE={result.infoe:.3f} InfoC={result.infoc:.3f} ({ckpt_label})"
        )

    missing = [
        r["run_dir"]
        for r in rows
        if "infom" not in r or "infoe" not in r or "infoc" not in r
    ]
    if missing:
        raise RuntimeError(
            "InfoMEC missing for some sweep rows (no usable checkpoint?): "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    return rows


def load_infomec_rows_from_dir(
    rows: list[dict],
    *,
    infomec_root: Path,
) -> list[dict]:
    """Attach InfoM/E/C from existing ``infomec_metrics.json`` files (no re-encode)."""
    root = infomec_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--infomec-from-dir is not a directory: {root}")

    missing: list[str] = []
    for row in rows:
        run_name = Path(row["run_dir"]).name
        metrics_path = root / run_name / "infomec_metrics.json"
        if not metrics_path.is_file():
            missing.append(run_name)
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row["infom"] = float(payload["infom"])
        row["infoe"] = float(payload["infoe"])
        row["infoc"] = float(payload["infoc"])
        row["infomec_max_nmi"] = float(payload["max_nmi"])
        row["infomec_ckpt"] = str(payload.get("checkpoint", ""))
        row["infomec_epoch"] = payload.get("selected_epoch")
        row["infomec_model_selection"] = str(payload.get("model_selection", ""))

    if missing:
        raise RuntimeError(
            "Missing infomec_metrics.json for subruns: "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    return rows


def build_raw_grid(
    rows: list[dict],
    *,
    value_key: str,
) -> tuple[np.ndarray, list[float], list[int]]:
    sweep_vals = sorted({float(r["sweep_param_value"]) for r in rows})
    lds = sorted({int(r["latent_dim"]) for r in rows})
    grid = np.full((len(sweep_vals), len(lds)), np.nan, dtype=np.float64)
    lookup = {
        (float(r["sweep_param_value"]), int(r["latent_dim"])): float(r[value_key])
        for r in rows
    }
    for i, sweep_val in enumerate(sweep_vals):
        for j, ld in enumerate(lds):
            grid[i, j] = lookup.get((sweep_val, ld), np.nan)
    return grid, sweep_vals, lds


def _format_beta(beta: float) -> str:
    if beta >= 1.0 and float(beta).is_integer():
        return str(int(beta))
    return f"{beta:g}"


def _norm_for_grid(
    grid: np.ndarray,
    *,
    field: str,
    log_scale: bool,
) -> Normalize | LogNorm:
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    if field == "frac_recon2total" or field in {"infom", "infoe", "infoc"}:
        return Normalize(vmin=0.0, vmax=1.0)
    if field == "recon_loss":
        vmax = float(np.max(finite))
        if vmax <= 0.0:
            vmax = 1.0
        return Normalize(vmin=0.0, vmax=vmax)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if log_scale:
        vmin = max(vmin, 1e-12)
        if vmax <= vmin:
            vmax = vmin * 10.0
        return LogNorm(vmin=vmin, vmax=vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def _maybe_annotate_cells(
    ax: plt.Axes,
    *,
    grid: np.ndarray,
    norm: Normalize | LogNorm,
    annotate: bool,
    x_of: Callable[[int, int], float],
    y_of: Callable[[int, int], float],
) -> None:
    if not annotate:
        return
    n0, n1 = grid.shape
    for i in range(n0):
        for j in range(n1):
            val = grid[i, j]
            if np.isnan(val):
                continue
            val_norm = (val - norm.vmin) / (norm.vmax - norm.vmin)  # type: ignore[attr-defined]
            txt_color = "white" if val_norm < 0.55 else "black"
            ax.text(
                x_of(i=i, j=j),
                y_of(i=i, j=j),
                f"{val:.3g}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=6,
            )


def plot_heatmap_ax_raw(
    ax: plt.Axes,
    *,
    grid: np.ndarray,
    sweep_vals: list[float],
    lds: list[int],
    field: str,
    title: str,
    sweep_ylabel: str,
    log_scale: bool,
    show_xlabel: bool,
    show_ylabel: bool,
    annotate: bool,
    x_tick_labels: list[str] | None = None,
    xlabel: str | None = None,
) -> plt.cm.ScalarMappable:
    norm = _norm_for_grid(grid, field=field, log_scale=log_scale)
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=norm,
        interpolation="nearest",
    )
    n_x = grid.shape[1]
    ax.set_xticks(np.arange(n_x))
    ax.set_xticklabels(
        x_tick_labels if x_tick_labels is not None else [str(ld) for ld in lds]
    )
    ax.set_yticks(np.arange(len(sweep_vals)))
    ax.set_yticklabels([_format_beta(b) for b in sweep_vals])
    if show_xlabel:
        ax.set_xlabel(xlabel if xlabel is not None else r"Size of $z$")
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(sweep_ylabel)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_title(title)
    _maybe_annotate_cells(
        ax=ax,
        grid=grid,
        norm=norm,
        annotate=annotate,
        x_of=lambda i, j: j,
        y_of=lambda i, j: i,
    )
    return im


def save_metrics_figure(
    *,
    rows: list[dict],
    out_png: Path,
    annotate: bool,
    dpi: int,
    wandb_metrics: WandbMetricRow,
    sweep_ylabel: str,
) -> dict[str, dict]:
    sweep_vals = sorted({float(r["sweep_param_value"]) for r in rows})
    lds = sorted({int(r["latent_dim"]) for r in rows})
    nrows = 2
    ncols = 4

    fig_w = max(14.0, 0.95 * len(lds) * ncols)
    fig_h = max(9.0, 0.45 * len(sweep_vals) * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    def _plot_metric_ax(
        *,
        ax: plt.Axes,
        field: str,
        title: str,
        log_scale: bool,
        show_xlabel: bool,
        show_ylabel: bool,
    ) -> tuple[plt.cm.ScalarMappable, np.ndarray]:
        grid, _, _ = build_raw_grid(rows, value_key=field)
        finite = grid[np.isfinite(grid)]
        im = plot_heatmap_ax_raw(
            ax,
            grid=grid,
            sweep_vals=sweep_vals,
            lds=lds,
            field=field,
            title=title,
            sweep_ylabel=sweep_ylabel,
            log_scale=log_scale,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            annotate=annotate,
        )
        return im, finite

    metric_meta: dict[str, dict] = {}
    first_wandb_col = next(
        (col_i for col_i, metric in enumerate(wandb_metrics) if metric is not None),
        0,
    )
    for col_i, metric in enumerate(wandb_metrics):
        ax = axes[0, col_i]
        if metric is None:
            ax.set_visible(False)
            continue
        field, title, log_scale = metric
        im, finite = _plot_metric_ax(
            ax=ax,
            field=field,
            title=title,
            log_scale=log_scale,
            show_xlabel=False,
            show_ylabel=col_i == first_wandb_col,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        metric_meta[field] = {
            "wandb_key": title,
            "log_scale": log_scale,
            "vmin": 0.0
            if field == "recon_loss"
            else (float(np.min(finite)) if finite.size else None),
            "vmax": float(np.max(finite)) if finite.size else None,
        }

    for col_i, (field, title, log_scale) in enumerate(INFOMEC_METRICS):
        ax = axes[1, col_i]
        im, finite = _plot_metric_ax(
            ax=ax,
            field=field,
            title=title,
            log_scale=log_scale,
            show_xlabel=True,
            show_ylabel=col_i == 0,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        metric_meta[field] = {
            "wandb_key": title,
            "log_scale": log_scale,
            "vmin": 0.0
            if field == "recon_loss"
            else (float(np.min(finite)) if finite.size else None),
            "vmax": float(np.max(finite)) if finite.size else None,
        }

    axes[1, 3].set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return metric_meta


def _write_csv(
    *,
    rows: list[dict],
    csv_path: Path,
    wandb_metrics: WandbMetricRow,
) -> None:
    metric_fields = [
        field for field, _, _ in _active_wandb_metrics(wandb_metrics) + INFOMEC_METRICS
    ]
    extra_fields = [
        "infomec_max_nmi",
        "infomec_ckpt",
        "infomec_epoch",
        "infomec_model_selection",
    ]
    header = [
        "sweep_param_name",
        "sweep_param_value",
        "sweep_param_label",
        "latent_dim",
        "weight_decay",
        "annotation_size",
        *metric_fields,
        *extra_fields,
        "wandb_step",
        "comment",
        "run_dir",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in sorted(
            rows,
            key=lambda x: (
                x.get("weight_decay", 0.0),
                x["sweep_param_value"],
                x["latent_dim"],
            ),
        ):
            writer.writerow(
                [
                    r.get("sweep_param_name", ""),
                    r.get("sweep_param_value", ""),
                    r.get("sweep_param_label", ""),
                    r["latent_dim"],
                    r.get("weight_decay", ""),
                    r["annotation_size"],
                    *[r.get(field, "") for field in metric_fields],
                    *[r.get(field, "") for field in extra_fields],
                    r["wandb_step"],
                    r["comment"],
                    r["run_dir"],
                ]
            )


def main(args: LatentSweepHeatmapCLI) -> None:
    input_dir = args.input_dir.expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else input_dir / "evals_latentgrid" / ts
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_sweep_rows(input_dir)
    if not rows:
        raise RuntimeError(f"No subruns with variant.json under {input_dir}")

    if args.infomec_from_dir is not None:
        rows = load_infomec_rows_from_dir(
            rows,
            infomec_root=args.infomec_from_dir,
        )
        infomec_outdir = str(args.infomec_from_dir.expanduser().resolve())
    else:
        rows = enrich_rows_with_infomec(
            rows,
            model_selection=args.model_selection,
            split=args.split,
            batch_size=args.batch_size,
            set_length=args.set_length,
            num_bins=args.num_bins,
            device_str=args.device,
            outdir=outdir,
        )
        infomec_outdir = str(outdir / "infomec")
    infomec_evaluated = True

    betas = sorted({float(r["sweep_param_value"]) for r in rows})
    lds = sorted({int(r["latent_dim"]) for r in rows})
    annotation_size = annotation_size_from_rows(rows=rows)
    model_type = str(rows[0].get("model_type", ""))
    wandb_metrics = resolve_wandb_metrics(model_type=model_type)
    sweep_ylabel = str(rows[0].get("sweep_param_label", "sweep"))

    png_path = outdir / "latentgrid_metrics_train.png"
    metric_meta = save_metrics_figure(
        rows=rows,
        out_png=png_path,
        annotate=args.annotate,
        dpi=args.dpi,
        wandb_metrics=wandb_metrics,
        sweep_ylabel=sweep_ylabel,
    )
    print(f"Wrote {png_path}")

    csv_path = outdir / "latentgrid_metrics_train.csv"
    _write_csv(rows=rows, csv_path=csv_path, wandb_metrics=wandb_metrics)
    print(f"Wrote {csv_path}")

    manifest = {
        "input_dir": str(input_dir),
        "grid_csv": str(csv_path),
        "n_runs": len(rows),
        "model_type": model_type,
        "sweep_param_name": rows[0].get("sweep_param_name"),
        "sweep_param_label": sweep_ylabel,
        "sweep_values": betas,
        "latent_dims": lds,
        "annotation_size": annotation_size,
        "infomec_evaluated": infomec_evaluated,
        "infomec_model_selection": args.model_selection.value,
        "infomec_split": args.split,
        "infomec_outdir": infomec_outdir,
        "figure": str(png_path),
        "metrics": metric_meta,
    }
    manifest_path = outdir / "latentgrid_metrics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main(tyro.cli(LatentSweepHeatmapCLI))
