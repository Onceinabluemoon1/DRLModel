#!/usr/bin/env python3
"""Per-head reward maps for latentgrid sweeps (sum-over-Z decoders only)."""

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import einops as ei
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_VPL_ROOT = _SCRIPT_DIR.parent
if str(_VPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_VPL_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_infomec as plt_infomec  # noqa: E402
from pref_learn.models.decoders import MultiplicativeDecoder  # noqa: E402
from pref_learn.models.nets import (  # noqa: E402
    decoder_sums_latent_heads,
    normalize_decoder_type,
)
from pref_learn.models.qlae import QLAE  # noqa: E402
from pref_learn.models.quantizer import fsq_per_head_bin_entropy  # noqa: E402
from pref_learn.models.utils import logvar_to_std  # noqa: E402
from pref_learn.models.vae import VAEModel  # noqa: E402
from pref_learn.utils.disentangle_sweep import (  # noqa: E402
    discover_sweep_subruns,
    sweep_from_variant,
)
from pref_learn.utils.plot_utils import (  # noqa: E402
    latent_index_latex,
    observation_xy_limits,
    per_head_l_nuisance,
    plot_observation_rewards_by_annotator_mode,
)

MAX_HEAD_PANELS = 16
HEAD_GRID_NCOLS = 4

REWARD_MODE_OUTDIR: dict[str, str] = {
    "raw": "raw_rewards",
    "btl": "btl_rewards",
}

# sweep_param_name -> filename prefix (e.g. kl_weight -> beta)
SWEEP_PARAM_FILENAME_PREFIX: dict[str, str] = {
    "kl_weight": "beta",
    "weight_decay": "wd",
    "tc_weight": "tc",
}


class RewardMapMode(str, Enum):
    raw = "raw"
    btl = "btl"
    both = "both"


class RewardMapPlotXY(str, Enum):
    ee = "ee"
    obj = "obj"


class RewardMapBtlUnit(str, Enum):
    frame = "frame"
    traj = "traj"


class BtlTrajScatter(str, Enum):
    last = "last"
    random = "random"


class BtlLogitScaling(str, Enum):
    train = "train"
    per_head = "per_head"


def sweep_param_filename_prefix(*, sweep_param_name: str) -> str:
    return SWEEP_PARAM_FILENAME_PREFIX.get(str(sweep_param_name), str(sweep_param_name))


def rewardmap_run_dir_tag(
    *,
    ts: str,
    sweep_param_name: str,
    sweep_param_value: float,
    btl_unit: RewardMapBtlUnit,
    plot_xy: RewardMapPlotXY,
    btl_logit_scaling: BtlLogitScaling,
) -> str:
    """``<timestamp>_<prefix>=<value>_btlUnit=…_xy=…_btlLogitScaling=…``."""
    prefix = sweep_param_filename_prefix(sweep_param_name=sweep_param_name)
    return (
        f"{ts}_{prefix}={sweep_param_value:g}_"
        f"btlUnit={btl_unit.value}_xy={plot_xy.value}_"
        f"btlLogitScaling={btl_logit_scaling.value}"
    )


def rewardmap_metrics_fname_suffix(
    *,
    recon_loss: Optional[float] = None,
    infom: Optional[float] = None,
    infoe: Optional[float] = None,
    infoc: Optional[float] = None,
) -> str:
    """``_recon=…_M=…_E=…_C=…`` appended after ``nz=…`` in reward-map PNG names."""
    parts: list[str] = []
    if recon_loss is not None:
        parts.append(f"recon={recon_loss:.3g}")
    if infom is not None:
        parts.append(f"M={infom:.3g}")
    if infoe is not None:
        parts.append(f"E={infoe:.3g}")
    if infoc is not None:
        parts.append(f"C={infoc:.3g}")
    if not parts:
        return ""
    return "_" + "_".join(parts)


def rewardmap_out_png(
    *,
    outdir: Path,
    sweep_param_name: str,
    sweep_param_value: float,
    latent_dim: int,
    reward_mode: Literal["raw", "btl"],
    recon_loss: Optional[float] = None,
    infom: Optional[float] = None,
    infoe: Optional[float] = None,
    infoc: Optional[float] = None,
) -> Path:
    """Flat non-sort layout: ``…/{mode}_{prefix}=<val>_nz=<nz>_recon=….png``."""
    mode_dir = outdir / REWARD_MODE_OUTDIR[reward_mode]
    prefix = sweep_param_filename_prefix(sweep_param_name=sweep_param_name)
    fname = f"{reward_mode}_{prefix}={sweep_param_value:g}_nz={latent_dim}"
    fname += rewardmap_metrics_fname_suffix(
        recon_loss=recon_loss,
        infom=infom,
        infoe=infoe,
        infoc=infoc,
    )
    return mode_dir / f"{fname}.png"


def sort_easy_panel_stem(*, latent_dim: int, panel_idx: int, panel_name: str) -> str:
    """``nz=4_00left`` / ``nz=4_08aggregate``."""
    return f"nz={latent_dim}_{int(panel_idx):02d}{panel_name}"


def sort_easy_nz_png(
    *,
    outdir: Path,
    reward_mode: Literal["raw", "btl"],
    latent_dim: int,
    stem: str,
    recon_loss: Optional[float] = None,
    infom: Optional[float] = None,
    infoe: Optional[float] = None,
    infoc: Optional[float] = None,
) -> Path:
    """``…/btl_rewards/nz=4_recon=…_M=…_E=…_C=…/<stem>.png``."""
    nz_dir = f"nz={latent_dim}" + rewardmap_metrics_fname_suffix(
        recon_loss=recon_loss,
        infom=infom,
        infoe=infoe,
        infoc=infoc,
    )
    return outdir / REWARD_MODE_OUTDIR[reward_mode] / nz_dir / f"{stem}.png"


def infomec_path_metrics(
    *,
    infomec: dict[str, float] | None,
    recon_loss: float | None,
) -> dict[str, float | None]:
    if infomec is None:
        return {
            "recon_loss": recon_loss,
            "infom": None,
            "infoe": None,
            "infoc": None,
        }
    return {
        "recon_loss": recon_loss,
        "infom": float(infomec["infom"]),
        "infoe": float(infomec["infoe"]),
        "infoc": float(infomec["infoc"]),
    }


def aligned_pref_mode_labels(
    *,
    dataset_path: Path,
    split: str,
    n_queries: int,
    viz_data_frac: float,
    viz_data_seed: int,
) -> np.ndarray:
    """``source_mode`` labels aligned with ``prepare_preference_eval_arrays`` subsample."""
    mode_labels = plt_infomec.source_mode_labels_for_split(
        dataset_path=dataset_path,
        split=split,
    )
    n = int(len(mode_labels))
    frac_f = float(viz_data_frac)
    if frac_f >= 1.0 or n <= 1:
        if n != n_queries:
            raise ValueError(
                f"mode_labels len {n} != n_queries {n_queries} without subsample"
            )
        return mode_labels
    k = max(1, int(round(frac_f * n)))
    rng = np.random.default_rng(seed=viz_data_seed)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    out = mode_labels[idx]
    if len(out) != n_queries:
        raise ValueError(f"mode_labels subsample len {len(out)} != n_queries {n_queries}")
    return out


def raw_reward_standardization_values(
    *,
    rewards: np.ndarray,
    summed_rewards: np.ndarray,
    latent_dim: int,
) -> np.ndarray:
    """Pooled per-head + summed-head scalars for raw mean/std (matches ``global_raw_reward_limits``)."""
    n_active = min(int(latent_dim), MAX_HEAD_PANELS)
    if rewards.size == 0 or n_active <= 0:
        return np.empty((0,), dtype=np.float64)
    flat = rewards.reshape(-1, rewards.shape[-1])[:, :n_active]
    summed = np.asarray(summed_rewards, dtype=np.float64).reshape(-1)
    return np.concatenate([flat.reshape(-1), summed], axis=0)


def raw_reward_standardization_stats(
    *,
    rewards: np.ndarray,
    summed_rewards: np.ndarray,
    latent_dim: int,
) -> tuple[float, float]:
    """Mean/std over active head outputs and summed-head rewards in ``rewards`` / ``summed_rewards``."""
    vals = raw_reward_standardization_values(
        rewards=rewards,
        summed_rewards=summed_rewards,
        latent_dim=latent_dim,
    )
    if vals.size == 0:
        return 0.0, 1.0
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std <= 0.0 or not np.isfinite(std):
        std = 1.0
    return mean, std


def standardize_raw_reward_arrays(
    *,
    rewards: np.ndarray,
    summed_rewards: np.ndarray,
    mean: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(r - mean) / std`` on per-head rewards and summed-head panel values."""
    scale = float(std)
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    shift = float(mean)
    return (rewards - shift) / scale, (summed_rewards - shift) / scale


def global_raw_reward_limits(
    *,
    rewards: np.ndarray,
    latent_dim: int,
) -> tuple[float, float]:
    """Min/max over active head columns and their sum on the full (mixed) reward tensor."""
    n_active = min(int(latent_dim), MAX_HEAD_PANELS)
    if rewards.size == 0 or n_active <= 0:
        return 0.0, 1.0
    flat = rewards.reshape(-1, rewards.shape[-1])[:, :n_active]
    summed = flat.sum(axis=-1)
    all_vals = np.concatenate([flat.reshape(-1), summed], axis=0)
    return float(np.min(all_vals)), float(np.max(all_vals))


@dataclass
class LatentGridRewardMapCLI:
    """Per-head reward scatter grids for one disentanglement hyperparameter slice."""

    input_dir: Path
    """Latentgrid sweep root with ``subruns/<task>/``."""

    sweep_param_value: Optional[float] = None
    """Disentanglement HP slice; default auto from ``evals_latentgrid`` metrics."""

    latentgrid_metrics_dir: Optional[Path] = None
    """``evals_latentgrid/<timestamp>/`` with ``latentgrid_metrics_train.csv``; default latest."""

    recon_top_k: int = 4
    """Auto HP: among rows with ≥1 passing cell, keep up to this many lowest mean-passing-``recon_loss`` HPs, then pick max InfoM over passing cells."""

    recon_loss_threshold: Optional[float] = None
    """Auto HP: row survives if any cell has ``recon_loss`` ≤ threshold; top-k ranks by mean recon over passing cells only; InfoM tie-break uses passing cells."""

    outdir: Optional[Path] = None
    """Default ``<input_dir>/evals_rewardmap/<ts>_<prefix>=<value>_btlUnit=…_xy=…/``."""

    reward_mode: RewardMapMode = RewardMapMode.both
    """``raw``: decoder head output; ``btl``: BTL preference using one head only."""

    model_selection: plt_infomec.ModelSelection = plt_infomec.ModelSelection.final
    """Checkpoint: ``best_model.pt`` or final epoch checkpoint."""

    split: Literal["all", "train", "eval"] = "all"
    """Dataset split for reward maps (training preference pickle)."""

    viz_data_frac: float = 1.0
    """Random query subsample fraction in ``(0, 1]`` for scatter viz (full data at 1.0)."""

    viz_data_seed: int = 0
    """RNG seed for ``viz_data_frac`` subsampling."""

    plot_xy: RewardMapPlotXY = RewardMapPlotXY.ee
    """Scatter plane for SortEnv: ``ee`` = ``O[1]`` vs ``O[0]`` (base at top); ``obj`` = ``O[3:5]``."""

    btl_unit: RewardMapBtlUnit = RewardMapBtlUnit.frame
    """``frame``: one sampled frame per segment for BTL; ``traj``: segment BTL (sum over ``T``)."""

    btl_traj_scatter: BtlTrajScatter = BtlTrajScatter.last
    """``last``/``random`` frame per ``(q,s)``: xy scatter if ``btl_unit=traj``; BTL ``s_A`` if ``btl_unit=frame``."""

    batch_size: Optional[int] = None
    """Encoder batch size; default from variant.json."""

    set_length: Optional[int] = None
    """Preference-set truncation; default from variant.json ``set_size``."""

    device: str = "auto"
    """Torch device for encode/decode, or ``auto``."""

    prob_max_comp: int = 512
    """Max comparison states for probabilistic per-head rewards."""

    prob_chunk_size: int = 64
    """Query-state chunk size for probabilistic per-head decode."""

    prob_comp_chunk_size: int = 64
    """Comparison-pool chunk size for probabilistic per-head decode."""

    btl_logit_scaling: BtlLogitScaling = BtlLogitScaling.train
    """``train``: model ``reward_scaling``; ``per_head``: divide by ``nz`` (traj) or ``T·nz`` (frame)."""

    raw_reward_standardize: bool = False
    """Raw stratified maps: ``(r-mean)/std`` per pref-mode subset; aggregate uses full-dataset mean/std."""

    dpi: int = 150

    sweep_param_rtol: float = 1e-5
    sweep_param_atol: float = 1e-8


def sweep_param_matches(
    *,
    value: float,
    target: float,
    rtol: float,
    atol: float,
) -> bool:
    return bool(np.isclose(float(value), float(target), rtol=rtol, atol=atol))


def resolve_latentgrid_metrics_csv(
    *,
    input_dir: Path,
    metrics_dir: Optional[Path],
) -> Path:
    """``evals_latentgrid/<timestamp>/latentgrid_metrics_train.csv``."""
    if metrics_dir is not None:
        csv_path = metrics_dir.expanduser().resolve() / "latentgrid_metrics_train.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing latentgrid metrics CSV: {csv_path}")
        return csv_path

    evals_root = input_dir / "evals_latentgrid"
    if not evals_root.is_dir():
        raise FileNotFoundError(
            f"No evals_latentgrid/ under {input_dir}; run plot_latentgrid_heatmap.py first."
        )
    for ts_dir in sorted(
        (p for p in evals_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    ):
        csv_path = ts_dir / "latentgrid_metrics_train.csv"
        if csv_path.is_file():
            return csv_path
    raise FileNotFoundError(
        f"No latentgrid_metrics_train.csv under {evals_root}; "
        "run plot_latentgrid_heatmap.py first."
    )


def resolve_latentgrid_infomec_root(
    *,
    input_dir: Path,
    metrics_dir: Optional[Path],
) -> Path:
    """``evals_latentgrid/<timestamp>/infomec/`` (same eval as metrics CSV)."""
    csv_path = resolve_latentgrid_metrics_csv(
        input_dir=input_dir,
        metrics_dir=metrics_dir,
    )
    infomec_root = csv_path.parent / "infomec"
    if not infomec_root.is_dir():
        raise FileNotFoundError(
            f"Missing InfoMEC cache: {infomec_root}; run plot_latentgrid_heatmap.py first."
        )
    return infomec_root


def load_infomec_metrics_for_run(
    *,
    infomec_root: Path,
    run_dir: Path,
) -> dict[str, float]:
    metrics_path = infomec_root / run_dir.name / "infomec_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing infomec_metrics.json: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "infom": float(payload["infom"]),
        "infoe": float(payload["infoe"]),
        "infoc": float(payload["infoc"]),
    }


def load_latentgrid_metrics_rows(*, csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_latentgrid_recon_loss_by_run_dir(*, csv_path: Path) -> dict[str, float]:
    """Lookup train ``recon_loss`` by resolved ``run_dir`` and by subrun basename."""
    lookup: dict[str, float] = {}
    for row in load_latentgrid_metrics_rows(csv_path=csv_path):
        run_dir = Path(row["run_dir"]).expanduser().resolve()
        recon = float(row["recon_loss"])
        lookup[str(run_dir)] = recon
        #* basename / comment: same keying as InfoMEC cache (survives moved sweep roots)
        lookup[run_dir.name] = recon
        comment = str(row.get("comment", "")).strip()
        if comment:
            lookup[comment] = recon
    return lookup


def lookup_latentgrid_recon_loss(
    *,
    recon_by_run_dir: dict[str, float],
    run_dir: Path,
) -> float | None:
    """Resolve recon for a subrun; prefer absolute path, then folder name."""
    resolved = str(run_dir.expanduser().resolve())
    if resolved in recon_by_run_dir:
        return float(recon_by_run_dir[resolved])
    name = run_dir.name
    if name in recon_by_run_dir:
        return float(recon_by_run_dir[name])
    return None


def _hp_row_stats(
    *,
    cells: list[tuple[int, float, float]],
    recon_loss_threshold: Optional[float],
) -> dict[str, Any]:
    """Per-HP stats from ``(latent_dim, recon_loss, infom)`` cells (one per ``nz``)."""
    if not cells:
        raise RuntimeError("HP row has no metric cells")

    recons = [recon for _, recon, _ in cells]
    infoms = [infom for _, _, infom in cells]
    avg_recon_loss = float(mean(recons))
    max_infom_all = float(max(infoms))
    min_recon_loss = float(min(recons))

    if recon_loss_threshold is None:
        return {
            "avg_recon_loss": avg_recon_loss,
            "mean_passing_recon": avg_recon_loss,
            "min_recon_loss": min_recon_loss,
            "max_infom": max_infom_all,
            "max_infom_all": max_infom_all,
            "n_cells": len(cells),
            "n_cells_passing": len(cells),
            "passing_nz": sorted(nz for nz, _, _ in cells),
        }

    thr = float(recon_loss_threshold)
    passing_cells = [(nz, recon, infom) for nz, recon, infom in cells if recon <= thr]
    passing_nz = sorted(nz for nz, _, _ in passing_cells)
    passing_recons = [recon for _, recon, _ in passing_cells]
    passing_infoms = [infom for _, _, infom in passing_cells]
    n_passing = len(passing_cells)
    mean_passing_recon = float(mean(passing_recons)) if n_passing > 0 else float("inf")
    max_infom = float(max(passing_infoms)) if n_passing > 0 else float("-inf")

    return {
        "avg_recon_loss": avg_recon_loss,
        "mean_passing_recon": mean_passing_recon,
        "min_recon_loss": min_recon_loss,
        "max_infom": max_infom,
        "max_infom_all": max_infom_all,
        "n_cells": len(cells),
        "n_cells_passing": n_passing,
        "passing_nz": passing_nz,
    }


def select_sweep_param_value(
    *,
    rows: list[dict[str, str]],
    recon_top_k: int,
    recon_loss_threshold: Optional[float] = None,
) -> tuple[float, dict[str, Any]]:
    """Keep rows with ≥1 passing cell, top-k by mean passing recon, then max InfoM on passing cells."""
    by_hp: dict[float, list[tuple[int, float, float]]] = {}
    for row in rows:
        hp = float(row["sweep_param_value"])
        nz = int(row["latent_dim"])
        recon = float(row["recon_loss"])
        infom = float(row["infom"])
        by_hp.setdefault(hp, []).append((nz, recon, infom))

    if not by_hp:
        raise RuntimeError("No sweep rows in latentgrid metrics CSV")

    row_stats = {
        hp: _hp_row_stats(
            cells=cells,
            recon_loss_threshold=recon_loss_threshold,
        )
        for hp, cells in by_hp.items()
    }

    def recon_rank(stats: dict[str, Any]) -> float:
        return float(stats["mean_passing_recon"])

    # * threshold filter: row survives if any cell passes (optional)
    filtered_stats = row_stats
    if recon_loss_threshold is not None:
        filtered_stats = {
            hp: stats
            for hp, stats in row_stats.items()
            if int(stats["n_cells_passing"]) > 0
        }
        if not filtered_stats:
            best_hp, best_stats = min(
                row_stats.items(),
                key=lambda item: (item[1]["min_recon_loss"], item[0]),
            )
            raise RuntimeError(
                f"No HP rows with any cell recon_loss <= {recon_loss_threshold:g}; "
                f"closest is {best_hp:g} "
                f"(min_recon_loss={best_stats['min_recon_loss']:.6g})"
            )

    k = min(int(recon_top_k), len(filtered_stats))
    recon_candidates = sorted(
        filtered_stats.items(),
        key=lambda item: (recon_rank(item[1]), item[0]),
    )[:k]

    def effective_max_infom(stats: dict[str, float | int]) -> float:
        if recon_loss_threshold is None:
            return float(stats["max_infom"])
        if int(stats["n_cells_passing"]) > 0:
            return float(stats["max_infom"])
        return float("-inf")

    selected_hp, selected_stats = max(
        recon_candidates,
        key=lambda item: (effective_max_infom(item[1]), -recon_rank(item[1])),
    )
    used_infom_fallback = recon_loss_threshold is not None and effective_max_infom(
        selected_stats
    ) == float("-inf")
    if used_infom_fallback:
        selected_hp, selected_stats = max(
            recon_candidates,
            key=lambda item: (
                float(item[1]["max_infom_all"]),
                -recon_rank(item[1]),
            ),
        )

    if recon_loss_threshold is not None:
        method = (
            f"any_cell_recon<={recon_loss_threshold:g}, "
            f"lowest-{k}-of-{len(filtered_stats)}_mean_passing_recon then "
            f"max_infom(recon_loss<={recon_loss_threshold:g})"
        )
        if used_infom_fallback:
            method += " [fallback: max_infom_all, no threshold-passing cells in top-k]"
    else:
        method = f"lowest-{k}-avg_recon_loss then max_infom"

    selection_info: dict[str, Any] = {
        "method": method,
        "recon_top_k": k,
        "recon_loss_threshold": recon_loss_threshold,
        "n_threshold_passing": len(filtered_stats),
        "infom_fallback_all_cells": used_infom_fallback,
        "row_stats": {f"{hp:g}": stats for hp, stats in row_stats.items()},
        "threshold_passing": [
            {"sweep_param_value": hp, **stats}
            for hp, stats in sorted(
                filtered_stats.items(),
                key=lambda item: (recon_rank(item[1]), item[0]),
            )
        ],
        "recon_candidates": [
            {"sweep_param_value": hp, **stats} for hp, stats in recon_candidates
        ],
        "selected": {"sweep_param_value": selected_hp, **selected_stats},
    }
    return float(selected_hp), selection_info


def resolve_sweep_param_value(
    *,
    input_dir: Path,
    sweep_param_value: Optional[float],
    latentgrid_metrics_dir: Optional[Path],
    recon_top_k: int,
    recon_loss_threshold: Optional[float] = None,
) -> tuple[float, dict[str, Any]]:
    if sweep_param_value is not None:
        return float(sweep_param_value), {
            "method": "manual",
            "sweep_param_value": float(sweep_param_value),
        }

    csv_path = resolve_latentgrid_metrics_csv(
        input_dir=input_dir,
        metrics_dir=latentgrid_metrics_dir,
    )
    metric_rows = load_latentgrid_metrics_rows(csv_path=csv_path)
    selected_hp, selection_info = select_sweep_param_value(
        rows=metric_rows,
        recon_top_k=recon_top_k,
        recon_loss_threshold=recon_loss_threshold,
    )
    selection_info["latentgrid_metrics_csv"] = str(csv_path)
    selected = selection_info["selected"]
    recon_key = (
        "mean_passing_recon" if recon_loss_threshold is not None else "avg_recon_loss"
    )
    infom_msg = f"max_infom={selected['max_infom']:.4g}"
    if recon_loss_threshold is not None:
        passing_nz = selected.get("passing_nz", [])
        infom_msg += (
            f" (passing_nz={passing_nz}, "
            f"n_cells_passing={selected['n_cells_passing']}/{selected['n_cells']}, "
            f"max_infom_all={selected['max_infom_all']:.4g})"
        )
    print(
        f"Auto-selected {selection_info['method']}: "
        f"sweep_param_value={selected_hp:g} "
        f"({recon_key}={selected[recon_key]:.6g}, {infom_msg}) "
        f"from {csv_path.parent.name}"
    )
    for cand in selection_info["recon_candidates"]:
        hp = float(cand["sweep_param_value"])
        cand_infom = (
            f"max_infom={cand['max_infom']:.4g}"
            if cand["max_infom"] != float("-inf")
            else f"max_infom=— (max_infom_all={cand['max_infom_all']:.4g})"
        )
        passing_nz = cand.get("passing_nz", [])
        print(
            f"  candidate {hp:g}: {recon_key}={cand[recon_key]:.6g}, "
            f"{cand_infom}, "
            f"n_cells_passing={cand['n_cells_passing']}/{cand['n_cells']}"
            + (" ← selected" if np.isclose(hp, selected_hp) else "")
        )
    return selected_hp, selection_info


def load_subrun_metadata(*, run_dir: Path) -> dict[str, Any]:
    variant_path = run_dir / "variant.json"
    if not variant_path.is_file():
        raise FileNotFoundError(f"Missing variant.json: {variant_path}")
    variant = json.loads(variant_path.read_text(encoding="utf-8"))["variant"]
    sweep_param_name, sweep_param_value, sweep_param_label = sweep_from_variant(
        variant=variant
    )
    decoder_type = normalize_decoder_type(
        decoder_type=str(variant.get("decoder_type", "multi"))
    )
    return {
        "run_dir": run_dir.resolve(),
        "comment": str(variant.get("comment", run_dir.name)),
        "model_type": str(variant.get("model_type", "")),
        "decoder_type": decoder_type,
        "latent_dim": int(variant["latent_dim"]),
        "sweep_param_name": sweep_param_name,
        "sweep_param_value": float(sweep_param_value),
        "sweep_param_label": sweep_param_label,
        "env": str(variant.get("env", "")),
        "obs_dim": int(variant["obs_dim"]) if "obs_dim" in variant else None,
        "reward_scaling": float(variant.get("reward_scaling", 1.0)),
    }


def filter_subruns_for_sweep_value(
    *,
    input_dir: Path,
    sweep_param_value: float,
    model_selection: plt_infomec.ModelSelection,
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_sweep_subruns(input_dir):
        meta = load_subrun_metadata(run_dir=run_dir)
        if not sweep_param_matches(
            value=meta["sweep_param_value"],
            target=sweep_param_value,
            rtol=rtol,
            atol=atol,
        ):
            continue
        if not plt_infomec.run_dir_has_checkpoint(
            run_dir, model_selection=model_selection
        ):
            continue
        rows.append(meta)
    rows.sort(key=lambda row: int(row["latent_dim"]))
    return rows


def resolve_reward_map_layout(
    *,
    env_name: str,
    plot_xy: RewardMapPlotXY,
    obs_dim: int | None,
) -> dict[str, Any]:
    if plot_xy == RewardMapPlotXY.obj:
        if obs_dim is not None and obs_dim < 5:
            raise ValueError(
                f"plot_xy=obj requires obs_dim>=5 (SortEnv object xyz); got {obs_dim}"
            )
        scatter_s = 0.25
        title_fontsize = 9
        if plt_infomec.is_ravens_sort_env(env_name):
            layout = plt_infomec.sort_learned_reward_map_layout(
                env_name=env_name, n_modes=1
            )
            scatter_s = layout.scatter_s
            title_fontsize = layout.title_fontsize
        return dict(
            x_idx=3,
            y_idx=4,
            axis_xlabel="obj x",
            axis_ylabel="obj y",
            invert_yaxis=False,
            scatter_s=scatter_s,
            title_fontsize=title_fontsize,
        )

    if plt_infomec.is_ravens_sort_env(env_name):
        layout = plt_infomec.sort_learned_reward_map_layout(env_name=env_name, n_modes=1)
        return dict(
            x_idx=layout.x_idx,
            y_idx=layout.y_idx,
            axis_xlabel=layout.axis_xlabel,
            axis_ylabel=layout.axis_ylabel,
            invert_yaxis=layout.invert_yaxis,
            scatter_s=layout.scatter_s,
            title_fontsize=layout.title_fontsize,
        )
    return dict(
        x_idx=0,
        y_idx=1,
        axis_xlabel="obs[0]",
        axis_ylabel="obs[1]",
        invert_yaxis=False,
        scatter_s=0.25,
        title_fontsize=9,
    )


def encode_query_latents(
    *,
    model: torch.nn.Module,
    obs1: torch.Tensor,
    obs2: torch.Tensor,
    labels: torch.Tensor,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    enc_kw: dict[str, torch.Tensor] = {}
    if plt_infomec.model_encode_sources(model=model):
        if source is None:
            raise ValueError("Model has encode_sources=True but source is None.")
        enc_kw["source"] = source
    if isinstance(model, QLAE):
        z_q, _ = model.encode(obs1, obs2, labels, **enc_kw)
        return z_q
    mean, _ = model.encode(obs1, obs2, labels, **enc_kw)
    return mean


@torch.no_grad()
def batched_per_head_elbo_stats(
    *,
    model: VAEModel,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    source: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-head KL plus strata mean/std of encoder ``μ``; arrays ``(Z,)``."""
    import torch.distributions as D

    q_n = int(len(obs1))
    z_dim = int(model.latent_dim)
    kl_sum = torch.zeros(z_dim, device=device)
    mean_chunks: list[torch.Tensor] = []
    prior_std = model.prior_std().to(device=device)  # (Z,)
    prior_mean = model.prior_mean.to(device=device)  # (Z,)

    for start in range(0, q_n, batch_size):
        end = min(start + batch_size, q_n)
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)
        o2 = torch.from_numpy(obs2[start:end]).float().to(device)
        y = torch.from_numpy(labels[start:end]).float().to(device)
        enc_kw: dict[str, torch.Tensor] = {}
        if plt_infomec.model_encode_sources(model=model):
            if source is None:
                raise ValueError("Model has encode_sources=True but source is None.")
            enc_kw["source"] = (
                torch.from_numpy(source[start:end]).long().to(device)
            )
        mean, logvar = model.encode(o1, o2, y, **enc_kw)  # (B, Z), (B, Z)
        post_std = logvar_to_std(logvar)
        q_z = D.Normal(mean, post_std)
        p_z = D.Normal(prior_mean, prior_std)
        kl = D.kl_divergence(q_z, p_z)  # (B, Z)
        kl_sum = kl_sum + kl.sum(dim=0)
        mean_chunks.append(mean)

    assert mean_chunks
    means = torch.cat(mean_chunks, dim=0)  # (Q, Z)
    n_total = int(means.shape[0])
    assert n_total > 0
    post_mean = means.mean(dim=0)
    if n_total > 1:
        post_mean_std = means.std(dim=0, unbiased=True)
    else:
        post_mean_std = torch.zeros(z_dim, device=device)
    return {
        "kl_per_head": (kl_sum / n_total).detach().cpu().numpy(),
        "prior_std": prior_std.detach().cpu().numpy(),
        "post_mean": post_mean.detach().cpu().numpy(),
        "post_mean_std": post_mean_std.detach().cpu().numpy(),
    }


@torch.no_grad()
def batched_per_head_zq_std(
    *,
    model: QLAE,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    source: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-head ``z_q`` std and FSQ bin entropy over full query split; ``(Z,)``."""
    q_n = int(len(obs1))
    z_dim = int(model.latent_dim)
    zq_chunks: list[torch.Tensor] = []
    z_bins_chunks: list[torch.Tensor] = []

    for start in range(0, q_n, batch_size):
        end = min(start + batch_size, q_n)
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)
        o2 = torch.from_numpy(obs2[start:end]).float().to(device)
        y = torch.from_numpy(labels[start:end]).float().to(device)
        enc_kw: dict[str, torch.Tensor] = {}
        if plt_infomec.model_encode_sources(model=model):
            if source is None:
                raise ValueError("Model has encode_sources=True but source is None.")
            enc_kw["source"] = (
                torch.from_numpy(source[start:end]).long().to(device)
            )
        z_c = model.encode_continuous(o1, o2, y, **enc_kw)  # (B, Z)
        quant = model.quantize(z_c)
        z_q = quant["z_q"]
        z_bins = quant["z_bins"]
        zq_chunks.append(z_q)
        z_bins_chunks.append(z_bins)

    assert zq_chunks
    z_q_all = torch.cat(zq_chunks, dim=0)  # (Q, Z)
    z_bins_all = torch.cat(z_bins_chunks, dim=0)  # (Q, Z)
    n_total = int(z_q_all.shape[0])
    zq_mean = z_q_all.mean(dim=0)
    if n_total > 1:
        zq_std = z_q_all.std(dim=0, unbiased=False)
    else:
        zq_std = torch.zeros(z_dim, device=device)
    zq_bin_entropy = fsq_per_head_bin_entropy(
        z_bins_all,
        num_bins=int(model.num_quantized_values),
    )
    return {
        "zq_std_per_head": zq_std.detach().cpu().numpy(),
        "zq_mean_per_head": zq_mean.detach().cpu().numpy(),
        "zq_bin_entropy_per_head": zq_bin_entropy.detach().cpu().numpy(),
    }


# def std_post_mean_across_strata(
#     *,
#     per_strata_post_mean: list[np.ndarray],
# ) -> np.ndarray | None:
#     """Std of per-strata ``\bar{\mu}_j`` across pref modes; ``(Z,)``."""
#     if not per_strata_post_mean:
#         return None
#     stacked = np.stack(
#         [np.asarray(m, dtype=np.float64).reshape(-1) for m in per_strata_post_mean],
#         axis=0,
#     )  # (S, Z)
#     if stacked.shape[0] == 1:
#         return np.zeros(stacked.shape[1], dtype=np.float64)
#     return stacked.std(axis=0, ddof=1)


@torch.no_grad()
def batched_per_head_theta_on_obs1(
    *,
    decoder: MultiplicativeDecoder,
    obs1: np.ndarray,
    batch_size: int,
    device: torch.device,
    btl_unit: RewardMapBtlUnit,
    scatter_t_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Per-head ``theta(s)=softplus(w(h))`` on ``obs1``; shape matches reward-map layout."""
    q_n = int(len(obs1))
    s_n = int(obs1.shape[1])
    t_n = int(obs1.shape[2])
    z_dim = int(decoder.latent_dim)
    n_heads = min(z_dim, MAX_HEAD_PANELS)
    traj_mode = btl_unit == RewardMapBtlUnit.traj
    frame_scatter = btl_unit == RewardMapBtlUnit.frame and scatter_t_idx is not None
    segment_theta = traj_mode or frame_scatter
    if segment_theta:
        thetas = np.full((q_n, s_n, n_heads), np.nan, dtype=np.float64)
    else:
        thetas = np.full((q_n, s_n, t_n, n_heads), np.nan, dtype=np.float64)

    for start in range(0, q_n, batch_size):
        end = min(start + batch_size, q_n)
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)  # (B, S, T, O)
        b_n = end - start
        theta = decoder.theta_from_obs(o1)  # (B, S, T, Z)
        if traj_mode:
            thetas[start:end, :, :n_heads] = (
                theta[..., :n_heads].sum(dim=-2).detach().cpu().numpy()
            )
            continue
        if frame_scatter:
            assert scatter_t_idx is not None
            b_idx = torch.arange(b_n, device=device)[:, None]
            s_idx = torch.arange(s_n, device=device)[None, :]
            t_idx = (
                torch.from_numpy(scatter_t_idx[start:end])
                .long()
                .to(device=device)
            )
            theta_pick = theta[b_idx, s_idx, t_idx, :n_heads]  # (B, S, Z)
            thetas[start:end, :, :n_heads] = theta_pick.detach().cpu().numpy()
            continue
        thetas[start:end, :, :, :n_heads] = theta[..., :n_heads].detach().cpu().numpy()
    return thetas


@torch.no_grad()
def batched_mean_theta_per_head(
    *,
    decoder: MultiplicativeDecoder,
    obs1: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Mean ``softplus(w(h))`` per head over all ``(Q,S,T)`` in ``obs1``; ``(Z,)``."""
    q_n = int(len(obs1))
    z_dim = int(decoder.latent_dim)
    n_heads = min(z_dim, MAX_HEAD_PANELS)
    theta_sum = torch.zeros(n_heads, device=device)
    n_total = 0

    for start in range(0, q_n, batch_size):
        end = min(start + batch_size, q_n)
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)  # (B, S, T, O)
        theta = decoder.theta_from_obs(o1)  # (B, S, T, Z)
        b_n, s_n, t_n, _ = theta.shape
        theta_sum = theta_sum + theta[..., :n_heads].sum(dim=(0, 1, 2))
        n_total += b_n * s_n * t_n

    assert n_total > 0
    mean_active = (theta_sum / float(n_total)).detach().cpu().numpy()
    out = np.full(z_dim, np.nan, dtype=np.float64)
    out[:n_heads] = mean_active
    return out


def resolve_btl_logit_scaling(
    *,
    train_scaling: float,
    btl_logit_scaling: BtlLogitScaling,
    btl_unit: RewardMapBtlUnit,
    segment_len: int,
    latent_dim: int,
) -> float:
    """Eval-only BTL scaling for per-head prob maps vs summed-head training."""
    if btl_logit_scaling == BtlLogitScaling.train:
        return float(train_scaling)
    denom = float(latent_dim)
    if btl_unit == RewardMapBtlUnit.frame:
        denom *= float(segment_len)
    if denom <= 0.0:
        raise ValueError(
            f"btl_logit_scaling=per_head needs T>0 and nz>0; got T={segment_len}, nz={latent_dim}"
        )
    return float(train_scaling) / denom


def _probabilistic_reward_heads_chunk(
    *,
    model: torch.nn.Module,
    s_a: torch.Tensor,
    z: torch.Tensor,
    comparison_obs: torch.Tensor,
    scaling: float,
    comp_chunk_size: int,
    frame_logit_scale: float = 1.0,
) -> torch.Tensor:
    """``E_{s_B}[p(y=1 | s_A, s_B, z)]`` per head; one query chunk. ``(M, Z)``."""
    s_a = model._as_batl_obs(obs=s_a)  # (M, 1, T, O)
    s_b = model._as_batl_obs(obs=comparison_obs)  # (N, 1, T, O)
    m_n, _, t_n, _ = s_a.shape
    n_n = int(s_b.shape[0])
    z_dim = int(z.shape[-1])
    p_sum = torch.zeros((m_n, z_dim), device=s_a.device, dtype=s_a.dtype)
    logit_scale = float(frame_logit_scale)

    for n_start in range(0, n_n, comp_chunk_size):
        n_end = min(n_start + comp_chunk_size, n_n)
        s_b_chunk = s_b[n_start:n_end]  # (N_c, 1, T, O)
        n_c = int(s_b_chunk.shape[0])

        s_a_pair = ei.repeat(s_a, "M 1 T O -> M N T O", N=n_c)
        s_b_pair = ei.repeat(s_b_chunk, "N 1 T O -> M N T O", M=m_n)
        z_pair = ei.repeat(z, "M Z -> M N T Z", N=n_c, T=t_n)

        x_a = torch.cat([s_a_pair, z_pair], dim=-1)  # (M, N_c, T, O+Z)
        x_b = torch.cat([s_b_pair, z_pair], dim=-1)
        r_a_heads = model.decoder(x_a, vector_rewards=True)  # (M, N_c, T, Z)
        r_b_heads = model.decoder(x_b, vector_rewards=True)

        r_a_j = r_a_heads.sum(dim=-2) / scaling  # (M, N_c, Z)
        r_b_j = r_b_heads.sum(dim=-2) / scaling
        p_pref = torch.sigmoid(logit_scale * (r_a_j - r_b_j))  # (M, N_c, Z)
        p_sum += p_pref.sum(dim=1)  # (M, Z)

    return p_sum / float(n_n)  # (M, Z)


def _probabilistic_reward_heads_batch(
    *,
    model: torch.nn.Module,
    s_a: torch.Tensor,
    z: torch.Tensor,
    comparison_obs: torch.Tensor,
    scaling: float,
    chunk_size: int,
    comp_chunk_size: int,
    frame_logit_scale: float = 1.0,
) -> torch.Tensor:
    """``E_{s_B}[p(y=1 | s_A, s_B, z)]`` for each head. ``(M, Z)``."""
    m_n = int(s_a.shape[0])
    if m_n <= chunk_size:
        return _probabilistic_reward_heads_chunk(
            model=model,
            s_a=s_a,
            z=z,
            comparison_obs=comparison_obs,
            scaling=scaling,
            comp_chunk_size=comp_chunk_size,
            frame_logit_scale=frame_logit_scale,
        )

    chunks: list[torch.Tensor] = []
    for start in range(0, m_n, chunk_size):
        end = min(start + chunk_size, m_n)
        chunks.append(
            _probabilistic_reward_heads_chunk(
                model=model,
                s_a=s_a[start:end],
                z=z[start:end],
                comparison_obs=comparison_obs,
                scaling=scaling,
                comp_chunk_size=comp_chunk_size,
                frame_logit_scale=frame_logit_scale,
            )
        )
    return torch.cat(chunks, dim=0)  # (M, Z)


def last_traj_scatter_frame_indices(
    *,
    q_n: int,
    s_n: int,
    t_n: int,
) -> np.ndarray:
    """Last frame ``t=T-1`` per ``(q,s)``. ``(Q,S)``."""
    return np.full((q_n, s_n), t_n - 1, dtype=np.int64)


def random_traj_scatter_frame_indices(
    *,
    q_n: int,
    s_n: int,
    t_n: int,
) -> np.ndarray:
    """Random frame per ``(q,s)``. ``(Q,S)``."""
    rng = np.random.default_rng()
    return rng.integers(low=0, high=t_n, size=(q_n, s_n))


def resolve_btl_traj_scatter_indices(
    *,
    q_n: int,
    s_n: int,
    t_n: int,
    mode: BtlTrajScatter,
) -> np.ndarray:
    """Pick one frame index per ``(q,s)`` for traj xy or frame BTL. ``(Q,S)``."""
    if mode == BtlTrajScatter.last:
        return last_traj_scatter_frame_indices(q_n=q_n, s_n=s_n, t_n=t_n)
    return random_traj_scatter_frame_indices(q_n=q_n, s_n=s_n, t_n=t_n)


def scatter_obs_at_traj_frames(
    *,
    obs1: np.ndarray,
    scatter_t_idx: np.ndarray,
) -> np.ndarray:
    """``obs1`` ``(Q,S,T,O)`` → ``(Q·S,O)`` at ``scatter_t_idx[q,s]`` per segment."""
    q_n, s_n, _, o_dim = obs1.shape
    q_idx = np.arange(q_n)[:, None]
    s_idx = np.arange(s_n)[None, :]
    return obs1[q_idx, s_idx, scatter_t_idx, :].reshape(q_n * s_n, o_dim)


def gather_obs1_at_frame_indices(
    *,
    obs1: torch.Tensor,
    scatter_t_idx: np.ndarray,
) -> torch.Tensor:
    """``obs1`` ``(B,S,T,O)`` → ``(B·S,O)`` at ``scatter_t_idx[b,s]`` per segment."""
    b_n, s_n, _, o_dim = obs1.shape
    b_idx = torch.arange(b_n, device=obs1.device)[:, None]
    s_idx = torch.arange(s_n, device=obs1.device)[None, :]
    t_idx = torch.from_numpy(np.asarray(scatter_t_idx)).long().to(device=obs1.device)
    return obs1[b_idx, s_idx, t_idx, :].reshape(b_n * s_n, o_dim)


def batched_per_head_rewards_on_obs1(
    *,
    model: torch.nn.Module,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    reward_mode: Literal["raw", "btl"],
    btl_unit: RewardMapBtlUnit = RewardMapBtlUnit.frame,
    btl_logit_scaling: BtlLogitScaling = BtlLogitScaling.train,
    comparison_obs: np.ndarray | None = None,
    prob_max_comp: int = 512,
    prob_chunk_size: int = 64,
    prob_comp_chunk_size: int = 64,
    source: np.ndarray | None = None,
    scatter_t_idx: np.ndarray | None = None,
    precomputed_theta: np.ndarray | None = None,
) -> np.ndarray:
    """Per-head rewards on ``obs1``; ``(Q,S,Z)`` if traj/sampled-frame prob else ``(Q,S,T,Z)`` or ``(Q,S,Z)`` raw traj."""
    q_n = int(len(obs1))
    s_n = int(obs1.shape[1])
    t_n = int(obs1.shape[2])
    o_dim = int(obs1.shape[-1])
    z_dim = int(model.latent_dim)
    n_heads = min(z_dim, MAX_HEAD_PANELS)
    traj_mode = btl_unit == RewardMapBtlUnit.traj
    frame_sampled_prob = (
        btl_unit == RewardMapBtlUnit.frame
        and reward_mode == "btl"
        and scatter_t_idx is not None
    )
    segment_rewards = traj_mode or frame_sampled_prob
    if segment_rewards:
        rewards = np.full((q_n, s_n, n_heads), np.nan, dtype=np.float64)
    else:
        rewards = np.full((q_n, s_n, t_n, n_heads), np.nan, dtype=np.float64)

    comp_t: torch.Tensor | None = None
    if reward_mode == "btl":
        assert comparison_obs is not None
        comp_t = torch.from_numpy(comparison_obs).float().to(device)

    scaling = float(getattr(model, "scaling", 1.0))
    frame_logit_scale = 1.0
    if reward_mode == "btl":
        scaling = resolve_btl_logit_scaling(
            train_scaling=scaling,
            btl_logit_scaling=btl_logit_scaling,
            btl_unit=btl_unit,
            segment_len=t_n,
            latent_dim=z_dim,
        )
        if btl_unit == RewardMapBtlUnit.frame:
            frame_logit_scale = float(t_n)

    for start in range(0, q_n, batch_size):
        end = min(start + batch_size, q_n)
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)  # (B, S, T, O)
        o2 = torch.from_numpy(obs2[start:end]).float().to(device)
        y = torch.from_numpy(labels[start:end]).float().to(device)
        b_n = end - start
        u: torch.Tensor | None = None
        if plt_infomec.model_encode_sources(model=model):
            if source is None:
                raise ValueError("Model has encode_sources=True but source is None.")
            u = torch.from_numpy(source[start:end]).long().to(device)
        with torch.no_grad():
            z_q = encode_query_latents(
                model=model,
                obs1=o1,
                obs2=o2,
                labels=y,
                source=u,
            )  # (B, Z)
            z = ei.repeat(z_q, "B Z -> B S T Z", S=s_n, T=t_n)
            if reward_mode == "raw":
                dec = model.decoder
                if precomputed_theta is not None:
                    theta_t = (
                        torch.from_numpy(np.asarray(precomputed_theta)[start:end])
                        .float()
                        .to(device)
                    )
                    if theta_t.ndim == 3:
                        z_seg = ei.repeat(z_q[:, :n_heads], "B Z -> B S Z", S=s_n)
                        r_seg = theta_t[..., :n_heads] * z_seg  # (B, S, Z)
                        rewards[start:end, :, :n_heads] = r_seg.detach().cpu().numpy()
                    else:
                        r_heads = theta_t[..., :n_heads] * z[..., :n_heads]
                        rewards[start:end, :, :, :n_heads] = (
                            r_heads.detach().cpu().numpy()
                        )
                elif isinstance(dec, MultiplicativeDecoder):
                    theta = dec.theta_from_obs(o1)  # (B, S, T, Z)
                    r_heads = theta[..., :n_heads] * z[..., :n_heads]
                    if traj_mode:
                        rewards[start:end, :, :n_heads] = (
                            r_heads.sum(dim=-2).detach().cpu().numpy()
                        )
                    else:
                        rewards[start:end, :, :, :n_heads] = (
                            r_heads.detach().cpu().numpy()
                        )
                else:
                    x = torch.cat([o1, z], dim=-1)  # (B, S, T, O+Z)
                    r_heads = dec(x, vector_rewards=True)
                    if traj_mode:
                        rewards[start:end, :, :n_heads] = (
                            r_heads[..., :n_heads].sum(dim=-2).detach().cpu().numpy()
                        )
                    else:
                        rewards[start:end, :, :, :n_heads] = (
                            r_heads[..., :n_heads].detach().cpu().numpy()
                        )
                continue

            assert comp_t is not None
            if traj_mode:
                seg_flat = o1.reshape(b_n * s_n, t_n, o_dim)
                z_flat = ei.repeat(z_q, "B Z -> (B S) Z", S=s_n)
                r_flat = _probabilistic_reward_heads_batch(
                    model=model,
                    s_a=seg_flat,
                    z=z_flat,
                    comparison_obs=comp_t,
                    scaling=scaling,
                    chunk_size=prob_chunk_size,
                    comp_chunk_size=prob_comp_chunk_size,
                    frame_logit_scale=frame_logit_scale,
                )  # (B*S, Z)
                rewards[start:end, :, :n_heads] = (
                    r_flat[:, :n_heads].reshape(b_n, s_n, n_heads).detach().cpu().numpy()
                )
                continue

            if frame_sampled_prob:
                assert scatter_t_idx is not None
                o_flat = gather_obs1_at_frame_indices(
                    obs1=o1,
                    scatter_t_idx=scatter_t_idx[start:end],
                )  # (B*S, O)
                z_flat = ei.repeat(z_q, "B Z -> (B S) Z", S=s_n)
                r_flat = _probabilistic_reward_heads_batch(
                    model=model,
                    s_a=o_flat,
                    z=z_flat,
                    comparison_obs=comp_t,
                    scaling=scaling,
                    chunk_size=prob_chunk_size,
                    comp_chunk_size=prob_comp_chunk_size,
                    frame_logit_scale=frame_logit_scale,
                )  # (B*S, Z)
                rewards[start:end, :, :n_heads] = (
                    r_flat[:, :n_heads].reshape(b_n, s_n, n_heads).detach().cpu().numpy()
                )
                continue

            o_flat = o1.reshape(b_n * s_n * t_n, o_dim)
            z_flat = ei.repeat(z_q, "B Z -> (B S T) Z", S=s_n, T=t_n)
            r_flat = _probabilistic_reward_heads_batch(
                model=model,
                s_a=o_flat,
                z=z_flat,
                comparison_obs=comp_t,
                scaling=scaling,
                chunk_size=prob_chunk_size,
                comp_chunk_size=prob_comp_chunk_size,
                frame_logit_scale=frame_logit_scale,
            )  # (B*S*T, Z)
            rewards[start:end, :, :, :n_heads] = (
                r_flat[:, :n_heads].reshape(b_n, s_n, t_n, n_heads).detach().cpu().numpy()
            )
    return rewards


def batched_summed_reward_on_obs1(
    *,
    model: torch.nn.Module,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    reward_mode: Literal["raw", "btl"],
    btl_unit: RewardMapBtlUnit = RewardMapBtlUnit.frame,
    comparison_obs: np.ndarray | None = None,
    prob_chunk_size: int = 64,
    source: np.ndarray | None = None,
    scatter_t_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Summed-head rewards on ``obs1``; ``(Q,S)`` if traj/sampled-frame prob else ``(Q,S,T)`` or ``(Q,S)`` raw traj."""
    q_n = int(len(obs1))
    s_n = int(obs1.shape[1])
    t_n = int(obs1.shape[2])
    o_dim = int(obs1.shape[-1])
    traj_mode = btl_unit == RewardMapBtlUnit.traj
    frame_sampled_prob = (
        btl_unit == RewardMapBtlUnit.frame
        and reward_mode == "btl"
        and scatter_t_idx is not None
    )
    segment_rewards = traj_mode or frame_sampled_prob
    if segment_rewards:
        rewards = np.full((q_n, s_n), np.nan, dtype=np.float64)
    else:
        rewards = np.full((q_n, s_n, t_n), np.nan, dtype=np.float64)

    comp_t: torch.Tensor | None = None
    if reward_mode == "btl":
        assert comparison_obs is not None
        comp_t = torch.from_numpy(comparison_obs).float().to(device)

    for start in range(0, q_n, batch_size):
        end = min(start + batch_size, q_n)
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)  # (B, S, T, O)
        o2 = torch.from_numpy(obs2[start:end]).float().to(device)
        y = torch.from_numpy(labels[start:end]).float().to(device)
        b_n = end - start
        u: torch.Tensor | None = None
        if plt_infomec.model_encode_sources(model=model):
            if source is None:
                raise ValueError("Model has encode_sources=True but source is None.")
            u = torch.from_numpy(source[start:end]).long().to(device)
        with torch.no_grad():
            z_q = encode_query_latents(
                model=model,
                obs1=o1,
                obs2=o2,
                labels=y,
                source=u,
            )  # (B, Z)
            z = ei.repeat(z_q, "B Z -> B S T Z", S=s_n, T=t_n)
            if reward_mode == "raw":
                x = torch.cat([o1, z], dim=-1)  # (B, S, T, O+Z)
                r = model.decoder(x, vector_rewards=False)  # (B, S, T, 1)
                if traj_mode:
                    rewards[start:end, :] = (
                        r.sum(dim=-2).squeeze(-1).detach().cpu().numpy()
                    )
                else:
                    rewards[start:end, :, :] = r.squeeze(-1).detach().cpu().numpy()
                continue

            assert comp_t is not None
            if traj_mode:
                seg_flat = o1.reshape(b_n * s_n, t_n, o_dim)
                z_flat = ei.repeat(z_q, "B Z -> (B S) Z", S=s_n)
                r_flat = model.get_probabilistic_reward(
                    s_a=seg_flat,
                    z=z_flat,
                    comparison_obs=comp_t,
                    chunk_size=prob_chunk_size,
                )  # (B*S, 1)
                rewards[start:end, :] = r_flat.reshape(b_n, s_n).detach().cpu().numpy()
                continue

            if frame_sampled_prob:
                assert scatter_t_idx is not None
                o_flat = gather_obs1_at_frame_indices(
                    obs1=o1,
                    scatter_t_idx=scatter_t_idx[start:end],
                )  # (B*S, O)
                z_flat = ei.repeat(z_q, "B Z -> (B S) Z", S=s_n)
                r_flat = model.get_probabilistic_reward(
                    s_a=o_flat,
                    z=z_flat,
                    comparison_obs=comp_t,
                    chunk_size=prob_chunk_size,
                )  # (B*S, 1)
                rewards[start:end, :] = r_flat.reshape(b_n, s_n).detach().cpu().numpy()
                continue

            o_flat = o1.reshape(b_n * s_n * t_n, o_dim)
            z_flat = ei.repeat(z_q, "B Z -> (B S T) Z", S=s_n, T=t_n)
            r_flat = model.get_probabilistic_reward(
                s_a=o_flat,
                z=z_flat,
                comparison_obs=comp_t,
                chunk_size=prob_chunk_size,
            )  # (B*S*T, 1)
            rewards[start:end, :, :] = (
                r_flat.reshape(b_n, s_n, t_n).detach().cpu().numpy()
            )
    return rewards


def head_order_by_kl_desc(
    *,
    kl_per_head: np.ndarray | None,
    head_indices: list[int],
) -> list[int]:
    """Sort ``head_indices`` by descending per-head KL (stable tie-break on index)."""
    if kl_per_head is None or not head_indices:
        return list(head_indices)
    kl = np.asarray(kl_per_head, dtype=np.float64).reshape(-1)

    def sort_key(h: int) -> tuple[float, int]:
        if h < kl.shape[0]:
            return (-float(kl[h]), h)
        return (0.0, h)

    return sorted(head_indices, key=sort_key)


def scatter_head_order(
    *,
    kl_per_head: np.ndarray | None,
    n_active: int,
    n_show: int,
) -> list[int]:
    """Panel order for reward scatters: active heads by descending KL, then inactive."""
    active = [h for h in range(n_show) if h < n_active]
    inactive = [h for h in range(n_show) if h >= n_active]
    return head_order_by_kl_desc(kl_per_head=kl_per_head, head_indices=active) + inactive


def elbo_bar_head_order(*, kl_per_head: np.ndarray | None) -> list[int] | None:
    """Panel order for ELBO bar plots: all heads by descending KL."""
    if kl_per_head is None:
        return None
    kl = np.asarray(kl_per_head, dtype=np.float64).reshape(-1)
    return head_order_by_kl_desc(
        kl_per_head=kl_per_head,
        head_indices=list(range(kl.shape[0])),
    )


def flatten_obs_and_head_rewards(
    *,
    obs1: np.ndarray,
    rewards: np.ndarray,
    latent_dim: int,
    scatter_t_idx: np.ndarray | None = None,
    n_panels: int = MAX_HEAD_PANELS,
    head_order: list[int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build per-head panel lists: shared obs, per-head scalar reward."""
    use_scatter_frame = scatter_t_idx is not None and rewards.ndim == 3
    if use_scatter_frame:
        obs_flat = scatter_obs_at_traj_frames(obs1=obs1, scatter_t_idx=scatter_t_idx)
        rw_flat = rewards.reshape(-1, rewards.shape[-1])
    else:
        obs_flat = ei.rearrange(obs1, "q s t o -> (q s t) o")
        rw_flat = ei.rearrange(rewards, "q s t z -> (q s t) z")
    n_active = min(int(latent_dim), MAX_HEAD_PANELS)
    compact = int(n_panels) < MAX_HEAD_PANELS
    n_show = min(n_active, int(n_panels)) if compact else MAX_HEAD_PANELS
    order = head_order if head_order is not None else list(range(n_show))
    per_head_obs: list[np.ndarray] = []
    per_head_r: list[np.ndarray] = []
    for head_idx in order:
        if head_idx < n_active:
            per_head_obs.append(obs_flat)
            per_head_r.append(rw_flat[:, head_idx : head_idx + 1])
        else:
            per_head_obs.append(np.empty((0, obs_flat.shape[1]), dtype=np.float64))
            per_head_r.append(np.empty((0, 1), dtype=np.float64))
    return per_head_obs, per_head_r


def append_summed_head_reward_panel(
    *,
    obs1: np.ndarray,
    rewards: np.ndarray,
    per_head_obs: list[np.ndarray],
    per_head_r: list[np.ndarray],
    panel_titles: list[str],
    summed_rewards: np.ndarray,
    scatter_t_idx: np.ndarray | None = None,
) -> int:
    """Append one panel with summed-head rewards; returns its panel index."""
    use_scatter_frame = scatter_t_idx is not None and rewards.ndim == 3
    if use_scatter_frame:
        obs_flat = scatter_obs_at_traj_frames(obs1=obs1, scatter_t_idx=scatter_t_idx)
    else:
        obs_flat = ei.rearrange(obs1, "q s t o -> (q s t) o")
    summed = np.asarray(summed_rewards)
    if summed.ndim == 2:
        rw_flat = summed.reshape(-1, 1)
    elif summed.ndim == 3:
        rw_flat = ei.rearrange(summed, "q s t -> (q s t) 1")
    else:
        raise ValueError(
            f"summed_rewards must be (Q,S) or (Q,S,T); got shape {summed.shape}"
        )
    per_head_obs.append(obs_flat)
    per_head_r.append(rw_flat)
    panel_titles.append(r"$\sum_i r_i$")
    return len(panel_titles) - 1


def head_panel_titles(
    *,
    latent_dim: int,
    n_panels: int = MAX_HEAD_PANELS,
    kl_per_head: np.ndarray | None = None,
    score_panel_prefix: str = "KL",
    head_order: list[int] | None = None,
) -> list[str]:
    n_active = min(int(latent_dim), MAX_HEAD_PANELS)
    compact = int(n_panels) < MAX_HEAD_PANELS
    n_show = min(n_active, int(n_panels)) if compact else MAX_HEAD_PANELS
    kl = None if kl_per_head is None else np.asarray(kl_per_head, dtype=np.float64).reshape(-1)
    order = head_order if head_order is not None else list(range(n_show))
    titles: list[str] = []
    for head_idx in order:
        if head_idx < n_active:
            z_label = latent_index_latex(head_idx=head_idx)
            if kl is not None and head_idx < kl.shape[0]:
                titles.append(f"{z_label} ({score_panel_prefix}={kl[head_idx]:.3g})")
            else:
                titles.append(z_label)
        else:
            titles.append(f"{latent_index_latex(head_idx=head_idx)} (inactive)")
    return titles


def format_infomec_suptitle_suffix(
    *,
    infomec: dict[str, float],
    recon_loss: float | None = None,
) -> str:
    recon_part = f"recon={recon_loss:.3g} " if recon_loss is not None else ""
    return (
        f"{recon_part}"
        f"InfoM={infomec['infom']:.3g} "
        f"InfoE={infomec['infoe']:.3g} "
        f"InfoC={infomec['infoc']:.3g}"
    )


def format_reward_map_suptitle(
    *,
    meta: dict[str, Any],
    reward_mode: Literal["raw", "btl"],
    ckpt_label: str,
    split: str,
    btl_unit: RewardMapBtlUnit,
    infomec: dict[str, float] | None = None,
    pref_mode_name: str | None = None,
    n_queries: int | None = None,
    recon_loss: float | None = None,
    raw_reward_standardize: bool = False,
    raw_standardize_scope: Literal["subset", "global"] | None = None,
) -> str:
    param_label = meta["sweep_param_label"]
    param_val = meta["sweep_param_value"]
    lines = [
        f"Per-head reward ({reward_mode}, btl={btl_unit.value})",
        str(meta["env"]),
        f"{param_label}={param_val:g}, nz={meta['latent_dim']}, dec={meta['decoder_type']}",
        f"split={split}, {ckpt_label}",
    ]
    if pref_mode_name is not None:
        q_msg = f", n_queries={n_queries}" if n_queries is not None else ""
        lines.append(f"pref_mode={pref_mode_name} (segment A only{q_msg})")
    if reward_mode == "raw" and raw_reward_standardize:
        scope_msg = (
            "full dataset"
            if raw_standardize_scope == "global"
            else "pref-mode subset"
        )
        lines.append(f"raw standardized: (r-mean)/std ({scope_msg} stats)")
    if infomec is not None:
        lines.append(
            format_infomec_suptitle_suffix(infomec=infomec, recon_loss=recon_loss)
        )
    return "\n".join(lines)


def save_per_head_reward_figure(
    *,
    obs1: np.ndarray,
    rewards: np.ndarray,
    meta: dict[str, Any],
    reward_mode: Literal["raw", "btl"],
    ckpt_label: str,
    split: str,
    out_png: Path,
    dpi: int,
    plot_xy: RewardMapPlotXY,
    btl_unit: RewardMapBtlUnit,
    scatter_t_idx: np.ndarray | None = None,
    infomec: dict[str, float] | None = None,
    pref_mode_name: str | None = None,
    n_queries: int | None = None,
    reward_vmin: float | None = None,
    reward_vmax: float | None = None,
    n_panels: int = MAX_HEAD_PANELS,
    include_summed_head_panel: bool = False,
    summed_rewards: np.ndarray | None = None,
    recon_loss: float | None = None,
    elbo_kl_per_head: np.ndarray | None = None,
    head_order_score_ylabel: str | None = None,
    head_order_score_title: str | None = None,
    head_order_score_panel_prefix: str | None = None,
    qlae_bin_entropy_per_head: np.ndarray | None = None,
    elbo_post_mean: np.ndarray | None = None,
    elbo_post_mean_std: np.ndarray | None = None,
    # elbo_post_mean_std_across_strata: np.ndarray | None = None,
    elbo_l_nuisance_per_head: np.ndarray | None = None,
    theta_mean_per_head: np.ndarray | None = None,
    raw_reward_standardize: bool = False,
    raw_standardize_scope: Literal["subset", "global"] | None = None,
) -> None:
    nz = int(meta["latent_dim"])
    layout = resolve_reward_map_layout(
        env_name=str(meta["env"]),
        plot_xy=plot_xy,
        obs_dim=int(obs1.shape[-1]),
    )
    n_active = min(nz, MAX_HEAD_PANELS)
    compact = int(n_panels) < MAX_HEAD_PANELS
    n_show = min(n_active, int(n_panels)) if compact else MAX_HEAD_PANELS
    scatter_order = scatter_head_order(
        kl_per_head=elbo_kl_per_head,
        n_active=n_active,
        n_show=n_show,
    )
    bar_order = elbo_bar_head_order(kl_per_head=elbo_kl_per_head)
    per_head_obs, per_head_r = flatten_obs_and_head_rewards(
        obs1=obs1,
        rewards=rewards,
        latent_dim=nz,
        scatter_t_idx=scatter_t_idx,
        n_panels=n_panels,
        head_order=scatter_order,
    )
    panel_titles = head_panel_titles(
        latent_dim=nz,
        n_panels=n_panels,
        kl_per_head=elbo_kl_per_head,
        score_panel_prefix=(
            head_order_score_panel_prefix
            if head_order_score_panel_prefix is not None
            else "KL"
        ),
        head_order=scatter_order,
    )
    highlight_panel_indices: list[int] | None = None
    if include_summed_head_panel:
        n_active = min(nz, MAX_HEAD_PANELS)
        if summed_rewards is None:
            if reward_mode != "raw":
                raise ValueError(
                    "summed_rewards is required for btl mode summed-head panel"
                )
            summed_rewards = rewards[..., :n_active].sum(axis=-1)
        highlight_panel_indices = [
            append_summed_head_reward_panel(
                obs1=obs1,
                rewards=rewards,
                per_head_obs=per_head_obs,
                per_head_r=per_head_r,
                panel_titles=panel_titles,
                summed_rewards=summed_rewards,
                scatter_t_idx=scatter_t_idx,
            )
        ]
    use_scatter_frame = scatter_t_idx is not None and rewards.ndim == 3
    if use_scatter_frame:
        obs_for_limits = scatter_obs_at_traj_frames(
            obs1=obs1, scatter_t_idx=scatter_t_idx
        )
    else:
        obs_for_limits = obs1.reshape(-1, obs1.shape[-1])
    xlim, ylim = observation_xy_limits(
        obs=obs_for_limits,
        x_idx=int(layout["x_idx"]),
        y_idx=int(layout["y_idx"]),
    )
    if reward_mode == "btl":
        vmin = 0.0 if reward_vmin is None else float(reward_vmin)
        vmax = 1.0 if reward_vmax is None else float(reward_vmax)
    else:
        vmin = reward_vmin
        vmax = reward_vmax
    ncols = min(HEAD_GRID_NCOLS, max(len(per_head_obs), 1))
    #* slightly larger than layout default so ``z_i (KL=…)`` stays readable
    title_fontsize = int(layout["title_fontsize"]) + 2
    fig = plot_observation_rewards_by_annotator_mode(
        per_head_obs,
        per_head_r,
        mode_titles=panel_titles,
        suptitle=format_reward_map_suptitle(
            meta=meta,
            reward_mode=reward_mode,
            ckpt_label=ckpt_label,
            split=split,
            btl_unit=btl_unit,
            infomec=infomec,
            pref_mode_name=pref_mode_name,
            n_queries=n_queries,
            recon_loss=recon_loss,
            raw_reward_standardize=raw_reward_standardize,
            raw_standardize_scope=raw_standardize_scope,
        ),
        scatter_s=float(layout["scatter_s"]),
        ncols=ncols,
        x_idx=int(layout["x_idx"]),
        y_idx=int(layout["y_idx"]),
        axis_xlabel=str(layout["axis_xlabel"]),
        axis_ylabel=str(layout["axis_ylabel"]),
        invert_yaxis=bool(layout.get("invert_yaxis", False)),
        xlim=xlim,
        ylim=ylim,
        title_fontsize=title_fontsize,
        reward_vmin=vmin,
        reward_vmax=vmax,
        highlight_panel_indices=highlight_panel_indices,
        elbo_kl_per_head=elbo_kl_per_head,
        head_order_score_ylabel=head_order_score_ylabel,
        head_order_score_title=head_order_score_title,
        head_order_score_panel_prefix=head_order_score_panel_prefix,
        qlae_bin_entropy_per_head=qlae_bin_entropy_per_head,
        elbo_post_mean=elbo_post_mean,
        elbo_post_mean_std=elbo_post_mean_std,
        # elbo_post_mean_std_across_strata=elbo_post_mean_std_across_strata,
        elbo_l_nuisance_per_head=elbo_l_nuisance_per_head,
        theta_mean_per_head=theta_mean_per_head,
        elbo_head_order=bar_order,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def reward_modes_to_run(*, reward_mode: RewardMapMode) -> list[Literal["raw", "btl"]]:
    if reward_mode == RewardMapMode.both:
        return ["raw", "btl"]
    return [reward_mode.value]  # type: ignore[list-item]


def main() -> None:
    args = tyro.cli(LatentGridRewardMapCLI)
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir is not a directory: {input_dir}")

    sweep_param_value, sweep_param_selection = resolve_sweep_param_value(
        input_dir=input_dir,
        sweep_param_value=args.sweep_param_value,
        latentgrid_metrics_dir=args.latentgrid_metrics_dir,
        recon_top_k=args.recon_top_k,
        recon_loss_threshold=args.recon_loss_threshold,
    )

    rows = filter_subruns_for_sweep_value(
        input_dir=input_dir,
        sweep_param_value=sweep_param_value,
        model_selection=args.model_selection,
        rtol=args.sweep_param_rtol,
        atol=args.sweep_param_atol,
    )
    if not rows:
        raise RuntimeError(f"No subruns with {sweep_param_value!r} under {input_dir}")

    decoder_type = str(rows[0]["decoder_type"])
    if not decoder_sums_latent_heads(decoder_type=decoder_type):
        raise RuntimeError(
            f"decoder_type={decoder_type!r} does not sum per-latent heads; "
            f"choose one of multi / cond / slot / odin."
        )
    for row in rows[1:]:
        if str(row["decoder_type"]) != decoder_type:
            raise ValueError(
                "Mixed decoder_type across matched subruns: "
                f"{decoder_type!r} vs {row['decoder_type']!r}"
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = rewardmap_run_dir_tag(
        ts=ts,
        sweep_param_name=str(rows[0]["sweep_param_name"]),
        sweep_param_value=float(sweep_param_value),
        btl_unit=args.btl_unit,
        plot_xy=args.plot_xy,
        btl_logit_scaling=args.btl_logit_scaling,
    )
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else input_dir / "evals_rewardmap" / run_tag
    )
    outdir.mkdir(parents=True, exist_ok=True)

    ref_run_dir = Path(rows[0]["run_dir"])
    (
        obs1,
        obs2,
        labels,
        encode_batch_size,
        _variant,
        dataset_path,
        query_source,
        device,
    ) = plt_infomec.prepare_preference_eval_arrays(
        ref_run_dir,
        split=args.split,
        batch_size=args.batch_size,
        set_length=args.set_length,
        dataset_override=None,
        device_str=args.device,
        viz_data_frac=args.viz_data_frac,
        viz_data_seed=args.viz_data_seed,
    )
    batch_size = int(args.batch_size or encode_batch_size)
    q_n = int(len(obs1))
    s_n = int(obs1.shape[1])
    t_n = int(obs1.shape[2])
    traj_mode = args.btl_unit == RewardMapBtlUnit.traj
    frame_mode = args.btl_unit == RewardMapBtlUnit.frame
    scatter_t_idx: np.ndarray | None = None
    if traj_mode or frame_mode:
        scatter_t_idx = resolve_btl_traj_scatter_indices(
            q_n=q_n,
            s_n=s_n,
            t_n=t_n,
            mode=args.btl_traj_scatter,
        )
    n_scatter = q_n * s_n if (traj_mode or frame_mode) else q_n * s_n * t_n
    scatter_frame_msg = (
        f", btl_traj_scatter={args.btl_traj_scatter.value}"
        if (traj_mode or frame_mode)
        else ""
    )
    print(
        f"Reward maps: {len(rows)} subruns | {rows[0]['sweep_param_label']}"
        f"={sweep_param_value:g} | dec={decoder_type} | "
        f"training dataset={dataset_path} ({q_n} queries, "
        f"{n_scatter} scatter pts, btl_unit={args.btl_unit.value}, "
        f"viz_data_frac={args.viz_data_frac:g}, plot_xy={args.plot_xy.value}, "
        f"btl_logit_scaling={args.btl_logit_scaling.value}, "
        f"raw_reward_standardize={args.raw_reward_standardize}"
        f"{scatter_frame_msg}) | device={device}"
    )

    pref_mode_labels: np.ndarray | None = None
    pref_mode_names: list[str] | None = None
    n_pref_modes = 0
    env_name = str(rows[0]["env"])
    #* sort-easy: always stratify reward maps by GT preference mode
    stratify_pref_mode = plt_infomec.is_ravens_sort_env(env_name)
    if stratify_pref_mode:
        pref_mode_labels = aligned_pref_mode_labels(
            dataset_path=dataset_path,
            split=args.split,
            n_queries=q_n,
            viz_data_frac=args.viz_data_frac,
            viz_data_seed=args.viz_data_seed,
        )
        raw_names = plt_infomec.source_mode_names_from_dataset(dataset_path=dataset_path)
        n_pref_modes = plt_infomec.resolve_n_annotator_modes(
            mode_labels=pref_mode_labels,
            mode_names=raw_names,
        )
        if n_pref_modes <= 0:
            raise RuntimeError(
                f"No preference modes in dataset {dataset_path} for sort-easy stratify"
            )
        pref_mode_names = raw_names
        if pref_mode_names is None:
            pref_mode_names = [
                plt_infomec.source_mode_display_name(m, mode_names=None)
                for m in range(n_pref_modes)
            ]
        print(
            f"Stratify pref mode: {n_pref_modes} modes, "
            f"comparison pool stays mixed ({q_n} subsampled queries total)"
        )

    comparison_pool: np.ndarray | None = None
    modes = reward_modes_to_run(reward_mode=args.reward_mode)
    infomec_root = resolve_latentgrid_infomec_root(
        input_dir=input_dir,
        metrics_dir=args.latentgrid_metrics_dir,
    )
    metrics_csv_path = resolve_latentgrid_metrics_csv(
        input_dir=input_dir,
        metrics_dir=args.latentgrid_metrics_dir,
    )
    recon_by_run_dir = load_latentgrid_recon_loss_by_run_dir(csv_path=metrics_csv_path)
    if "btl" in modes:
        if traj_mode:
            comparison_pool = plt_infomec.build_comparison_traj_pool(
                obs1=obs1,
                obs2=obs2,
                max_comp=args.prob_max_comp,
                seed=args.viz_data_seed,
            )
        else:
            comparison_pool = plt_infomec.build_comparison_obs_pool(
                obs1=obs1,
                obs2=obs2,
                max_comp=args.prob_max_comp,
            )

    manifest: dict[str, Any] = {
        "input_dir": str(input_dir),
        "outdir": str(outdir),
        "generated_at": ts,
        "sweep_param_value": float(sweep_param_value),
        "sweep_param_selection": sweep_param_selection,
        "sweep_param_name": rows[0]["sweep_param_name"],
        "sweep_param_label": rows[0]["sweep_param_label"],
        "decoder_type": decoder_type,
        "reward_mode": args.reward_mode.value,
        "split": args.split,
        "viz_data_frac": float(args.viz_data_frac),
        "viz_data_seed": int(args.viz_data_seed),
        "plot_xy": args.plot_xy.value,
        "btl_logit_scaling": args.btl_logit_scaling.value,
        "btl_unit": args.btl_unit.value,
        "btl_traj_scatter": (
            args.btl_traj_scatter.value if (traj_mode or frame_mode) else None
        ),
        "raw_reward_standardize": bool(args.raw_reward_standardize),
        "n_scatter_points": int(n_scatter),
        "stratify_pref_mode": bool(stratify_pref_mode),
        "n_pref_modes": int(n_pref_modes) if stratify_pref_mode else None,
        "dataset": str(dataset_path),
        "figures": [],
    }

    sweep_param_name = str(rows[0]["sweep_param_name"])
    sweep_param_label = str(rows[0]["sweep_param_label"])
    sweep_param_val = float(sweep_param_value)

    nz_bar = tqdm(rows, unit="subrun")
    for row in nz_bar:
        nz = int(row["latent_dim"])
        nz_bar.set_description(
            f"rewardmap {sweep_param_label}={sweep_param_val:g} nz={nz}"
        )
        run_dir = Path(row["run_dir"])
        infomec = load_infomec_metrics_for_run(
            infomec_root=infomec_root,
            run_dir=run_dir,
        )
        recon_loss = lookup_latentgrid_recon_loss(
            recon_by_run_dir=recon_by_run_dir,
            run_dir=run_dir,
        )
        ckpt_path, ckpt_label, selected_epoch, model_selection_tag = (
            plt_infomec.resolve_run_checkpoint(
                run_dir, model_selection=args.model_selection
            )
        )
        model = torch.load(ckpt_path, map_location=device, weights_only=False).to(device)
        model.eval()
        if not isinstance(model, (VAEModel, QLAE)):
            raise TypeError(
                f"Unsupported model type {type(model).__name__!r} in {run_dir}; "
                "expected VAEModel or QLAE."
            )

        #* full-split head-order stats (shared across every mode / aggregate PNG)
        elbo_stats: dict[str, np.ndarray] | None = None
        qlae_head_stats: dict[str, np.ndarray] | None = None
        if isinstance(model, VAEModel):
            elbo_stats = batched_per_head_elbo_stats(
                model=model,
                obs1=obs1,
                obs2=obs2,
                labels=labels,
                batch_size=batch_size,
                device=device,
                source=query_source,
            )
        elif isinstance(model, QLAE):
            qlae_head_stats = batched_per_head_zq_std(
                model=model,
                obs1=obs1,
                obs2=obs2,
                labels=labels,
                batch_size=batch_size,
                device=device,
                source=query_source,
            )

        full_thetas: np.ndarray | None = None
        multi_decoder = isinstance(model.decoder, MultiplicativeDecoder)
        if multi_decoder:
            full_thetas = batched_per_head_theta_on_obs1(
                decoder=model.decoder,
                obs1=obs1,
                batch_size=batch_size,
                device=device,
                btl_unit=args.btl_unit,
                scatter_t_idx=scatter_t_idx,
            )

        entry: dict[str, Any] = {
            "run_dir": str(run_dir),
            "comment": row["comment"],
            "latent_dim": int(row["latent_dim"]),
            "checkpoint": ckpt_label,
            "selected_epoch": selected_epoch,
            "model_selection": model_selection_tag,
            "pngs": {},
        }
        if elbo_stats is not None:
            entry["elbo_head_stats"] = {
                "kl_per_head": elbo_stats["kl_per_head"].tolist(),
                "prior_std": elbo_stats["prior_std"].tolist(),
                "post_mean": elbo_stats["post_mean"].tolist(),
                "post_mean_std": elbo_stats["post_mean_std"].tolist(),
            }
        if qlae_head_stats is not None:
            entry["qlae_head_stats"] = {
                "zq_std_per_head": qlae_head_stats["zq_std_per_head"].tolist(),
                "zq_mean_per_head": qlae_head_stats["zq_mean_per_head"].tolist(),
                "zq_bin_entropy_per_head": qlae_head_stats[
                    "zq_bin_entropy_per_head"
                ].tolist(),
            }
        if stratify_pref_mode:
            entry["stratified_pngs"] = {}
        if infomec is not None:
            entry["infomec"] = infomec

        path_metrics = infomec_path_metrics(
            infomec=infomec,
            recon_loss=recon_loss,
        )
        global_theta_mean: np.ndarray | None = None
        if multi_decoder:
            global_theta_mean = batched_mean_theta_per_head(
                decoder=model.decoder,
                obs1=obs1,
                batch_size=batch_size,
                device=device,
            )

        for mode in modes:
            rewards = batched_per_head_rewards_on_obs1(
                model=model,
                obs1=obs1,
                obs2=obs2,
                labels=labels,
                batch_size=batch_size,
                device=device,
                reward_mode=mode,
                btl_unit=args.btl_unit,
                btl_logit_scaling=args.btl_logit_scaling,
                comparison_obs=comparison_pool,
                prob_max_comp=args.prob_max_comp,
                prob_chunk_size=args.prob_chunk_size,
                prob_comp_chunk_size=args.prob_comp_chunk_size,
                source=query_source,
                scatter_t_idx=scatter_t_idx,
                precomputed_theta=full_thetas if mode == "raw" else None,
            )
            save_kw: dict[str, Any] = dict(
                meta=row,
                reward_mode=mode,
                ckpt_label=ckpt_label,
                split=args.split,
                dpi=args.dpi,
                plot_xy=args.plot_xy,
                btl_unit=args.btl_unit,
                infomec=infomec,
                recon_loss=recon_loss,
                n_panels=nz,
            )
            if elbo_stats is not None:
                save_kw["elbo_kl_per_head"] = elbo_stats["kl_per_head"]
                save_kw["elbo_post_mean"] = elbo_stats["post_mean"]
                save_kw["elbo_post_mean_std"] = elbo_stats["post_mean_std"]
            if qlae_head_stats is not None:
                save_kw["elbo_kl_per_head"] = qlae_head_stats["zq_std_per_head"]
                save_kw["head_order_score_ylabel"] = r"$\mathrm{std}_q(z)$"
                save_kw["head_order_score_title"] = r"per-head $\mathrm{std}_q(z_q)$"
                save_kw["head_order_score_panel_prefix"] = "σ_q"
                save_kw["qlae_bin_entropy_per_head"] = qlae_head_stats[
                    "zq_bin_entropy_per_head"
                ]
            if global_theta_mean is not None:
                save_kw["theta_mean_per_head"] = global_theta_mean

            if not stratify_pref_mode:
                png_path = rewardmap_out_png(
                    outdir=outdir,
                    sweep_param_name=sweep_param_name,
                    sweep_param_value=sweep_param_val,
                    latent_dim=nz,
                    reward_mode=mode,
                    **path_metrics,
                )
                save_per_head_reward_figure(
                    obs1=obs1,
                    rewards=rewards,
                    out_png=png_path,
                    scatter_t_idx=scatter_t_idx,
                    **save_kw,
                )
                entry["pngs"][mode] = str(png_path.relative_to(outdir))
                tqdm.write(f"{row['comment']}: saved {png_path.relative_to(outdir)}")
                continue

            #* sort-easy: panels = per-mode + aggregate (idx = n_pref_modes)
            assert pref_mode_labels is not None
            assert pref_mode_names is not None
            n_active = min(nz, MAX_HEAD_PANELS)
            global_raw_mean: float | None = None
            global_raw_std: float | None = None
            if mode == "raw":
                summed_rewards = rewards[..., :n_active].sum(axis=-1)
                if args.raw_reward_standardize:
                    global_raw_mean, global_raw_std = raw_reward_standardization_stats(
                        rewards=rewards,
                        summed_rewards=summed_rewards,
                        latent_dim=nz,
                    )
                    save_kw.pop("reward_vmin", None)
                    save_kw.pop("reward_vmax", None)
                else:
                    raw_vmin, raw_vmax = global_raw_reward_limits(
                        rewards=rewards,
                        latent_dim=nz,
                    )
                    save_kw["reward_vmin"] = raw_vmin
                    save_kw["reward_vmax"] = raw_vmax
            else:
                summed_rewards = batched_summed_reward_on_obs1(
                    model=model,
                    obs1=obs1,
                    obs2=obs2,
                    labels=labels,
                    batch_size=batch_size,
                    device=device,
                    reward_mode=mode,
                    btl_unit=args.btl_unit,
                    comparison_obs=comparison_pool,
                    prob_chunk_size=args.prob_chunk_size,
                    source=query_source,
                    scatter_t_idx=scatter_t_idx,
                )
                save_kw["reward_vmin"] = 0.0
                save_kw["reward_vmax"] = 1.0
            save_kw["include_summed_head_panel"] = True

            panels: list[
                tuple[
                    int,
                    str,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray | None,
                    np.ndarray,
                    np.ndarray | None,
                    np.ndarray,
                    int,
                ]
            ] = []
            for pref_mode in range(n_pref_modes):
                mask = np.asarray(pref_mode_labels).reshape(-1) == int(pref_mode)
                scatter_m = None if scatter_t_idx is None else scatter_t_idx[mask]
                source_m = None if query_source is None else query_source[mask]
                panels.append(
                    (
                        pref_mode,
                        str(pref_mode_names[pref_mode]),
                        obs1[mask],
                        obs2[mask],
                        labels[mask],
                        source_m,
                        rewards[mask],
                        scatter_m,
                        summed_rewards[mask],
                        int(mask.sum()),
                    )
                )
            panels.append(
                (
                    n_pref_modes,
                    "aggregate",
                    obs1,
                    obs2,
                    labels,
                    query_source,
                    rewards,
                    scatter_t_idx,
                    summed_rewards,
                    q_n,
                )
            )

            stratified: dict[str, Any] = {}
            # strata_post_means: list[np.ndarray] = []
            for (
                panel_idx,
                panel_name,
                obs_p,
                obs2_p,
                labels_p,
                source_p,
                rew_p,
                scatter_p,
                sum_p,
                n_q_p,
            ) in panels:
                stem = sort_easy_panel_stem(
                    latent_dim=nz,
                    panel_idx=panel_idx,
                    panel_name=panel_name,
                )
                png_path = sort_easy_nz_png(
                    outdir=outdir,
                    reward_mode=mode,
                    latent_dim=nz,
                    stem=stem,
                    **path_metrics,
                )
                rew_plot = rew_p
                sum_plot = sum_p
                panel_save_kw = dict(save_kw)
                raw_standardize_scope: Literal["subset", "global"] | None = None
                panel_elbo: dict[str, np.ndarray] | None = None
                if mode == "raw" and args.raw_reward_standardize:
                    assert global_raw_mean is not None
                    assert global_raw_std is not None
                    if panel_name == "aggregate":
                        mean = global_raw_mean
                        std = global_raw_std
                        raw_standardize_scope = "global"
                    else:
                        mean, std = raw_reward_standardization_stats(
                            rewards=rew_p,
                            summed_rewards=sum_p,
                            latent_dim=nz,
                        )
                        raw_standardize_scope = "subset"
                    rew_plot, sum_plot = standardize_raw_reward_arrays(
                        rewards=rew_p,
                        summed_rewards=sum_p,
                        mean=mean,
                        std=std,
                    )
                    panel_save_kw.pop("reward_vmin", None)
                    panel_save_kw.pop("reward_vmax", None)
                    panel_save_kw["raw_reward_standardize"] = True
                    panel_save_kw["raw_standardize_scope"] = raw_standardize_scope
                if elbo_stats is not None and isinstance(model, VAEModel) and n_q_p > 0:
                    panel_elbo = batched_per_head_elbo_stats(
                        model=model,
                        obs1=obs_p,
                        obs2=obs2_p,
                        labels=labels_p,
                        batch_size=batch_size,
                        device=device,
                        source=source_p,
                    )
                    panel_save_kw["elbo_post_mean"] = panel_elbo["post_mean"]
                    panel_save_kw["elbo_post_mean_std"] = panel_elbo["post_mean_std"]
                    panel_save_kw.pop("elbo_l_nuisance_per_head", None)
                    if panel_name == "aggregate":
                        panel_save_kw["elbo_l_nuisance_per_head"] = per_head_l_nuisance(
                            post_mean=panel_elbo["post_mean"],
                            post_mean_std=panel_elbo["post_mean_std"],
                        )
                    # if panel_name != "aggregate":
                    #     strata_post_means.append(panel_elbo["post_mean"])
                    # if panel_name == "aggregate":
                    #     across = std_post_mean_across_strata(
                    #         per_strata_post_mean=strata_post_means,
                    #     )
                    #     if across is not None:
                    #         panel_save_kw["elbo_post_mean_std_across_strata"] = across
                    #         panel_save_kw.pop("elbo_post_mean", None)
                    #         panel_save_kw.pop("elbo_post_mean_std", None)
                elif elbo_stats is not None:
                    panel_save_kw.pop("elbo_kl_per_head", None)
                    panel_save_kw.pop("elbo_post_mean", None)
                    panel_save_kw.pop("elbo_post_mean_std", None)
                    panel_save_kw.pop("elbo_l_nuisance_per_head", None)
                    # panel_save_kw.pop("elbo_post_mean_std_across_strata", None)
                if multi_decoder and n_q_p > 0:
                    panel_save_kw["theta_mean_per_head"] = batched_mean_theta_per_head(
                        decoder=model.decoder,
                        obs1=obs_p,
                        batch_size=batch_size,
                        device=device,
                    )
                elif multi_decoder:
                    panel_save_kw.pop("theta_mean_per_head", None)
                save_per_head_reward_figure(
                    obs1=obs_p,
                    rewards=rew_plot,
                    out_png=png_path,
                    scatter_t_idx=scatter_p,
                    pref_mode_name=panel_name,
                    n_queries=n_q_p,
                    summed_rewards=sum_plot,
                    **panel_save_kw,
                )
                rel = str(png_path.relative_to(outdir))
                stratified_entry: dict[str, Any] = {
                    "path": rel,
                    "pref_mode": int(panel_idx),
                    "pref_mode_name": panel_name,
                    "n_queries": int(n_q_p),
                }
                if mode == "raw" and args.raw_reward_standardize:
                    stratified_entry["raw_standardize_scope"] = raw_standardize_scope
                    stratified_entry["raw_standardize_mean"] = float(mean)
                    stratified_entry["raw_standardize_std"] = float(std)
                stratified[stem] = stratified_entry
                tqdm.write(f"{row['comment']}: saved {rel} (n_queries={n_q_p})")
            entry["stratified_pngs"][mode] = stratified
            agg_stem = sort_easy_panel_stem(
                latent_dim=nz,
                panel_idx=n_pref_modes,
                panel_name="aggregate",
            )
            entry["pngs"][mode] = stratified[agg_stem]["path"]

        manifest["figures"].append(entry)

    manifest_path = outdir / "rewardmap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
