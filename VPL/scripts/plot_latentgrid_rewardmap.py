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
from pref_learn.models.nets import decoder_sums_latent_heads  # noqa: E402
from pref_learn.models.qlae import QLAE  # noqa: E402
from pref_learn.models.vae import VAEModel  # noqa: E402
from pref_learn.utils.disentangle_sweep import (  # noqa: E402
    discover_sweep_subruns,
    sweep_from_variant,
)
from pref_learn.utils.plot_utils import (  # noqa: E402
    observation_xy_limits,
    plot_observation_rewards_by_annotator_mode,
)

MAX_HEAD_PANELS = 16
HEAD_GRID_NCOLS = 4

REWARD_MODE_OUTDIR: dict[str, str] = {
    "raw": "raw_rewards",
    "prob": "prob_rewards",
}

# sweep_param_name -> filename prefix (e.g. kl_weight -> beta)
SWEEP_PARAM_FILENAME_PREFIX: dict[str, str] = {
    "kl_weight": "beta",
    "weight_decay": "wd",
    "tc_weight": "tc",
}


def sweep_param_filename_prefix(*, sweep_param_name: str) -> str:
    return SWEEP_PARAM_FILENAME_PREFIX.get(str(sweep_param_name), str(sweep_param_name))


def rewardmap_run_dir_tag(
    *,
    ts: str,
    sweep_param_name: str,
    sweep_param_value: float,
) -> str:
    """``<timestamp>_<prefix>=<value>`` for ``evals_rewardmap/<tag>/``."""
    prefix = sweep_param_filename_prefix(sweep_param_name=sweep_param_name)
    return f"{ts}_{prefix}={sweep_param_value:g}"


def rewardmap_out_png(
    *,
    outdir: Path,
    sweep_param_name: str,
    sweep_param_value: float,
    latent_dim: int,
    reward_mode: Literal["raw", "prob"],
    infom: Optional[float] = None,
    infoc: Optional[float] = None,
) -> Path:
    """``evals_rewardmap/<ts>_<prefix>=<val>/<raw|prob>_rewards/{mode}_{prefix}={val}_nz={nz}[_infom=…][_infoc=…].png``."""
    mode_dir = outdir / REWARD_MODE_OUTDIR[reward_mode]
    prefix = sweep_param_filename_prefix(sweep_param_name=sweep_param_name)
    fname = f"{reward_mode}_{prefix}={sweep_param_value:g}_nz={latent_dim}"
    if infom is not None:
        fname += f"_infom={infom:.3g}"
    if infoc is not None:
        fname += f"_infoc={infoc:.3g}"
    fname += ".png"
    return mode_dir / fname


class RewardMapMode(str, Enum):
    raw = "raw"
    prob = "prob"
    both = "both"


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
    """Auto HP: among threshold-passing rows, keep up to this many lowest avg-``recon_loss`` HPs, then pick max InfoM."""

    recon_loss_threshold: Optional[float] = None
    """Auto HP: only consider HP rows with avg ``recon_loss`` at or below this value; default no threshold."""

    outdir: Optional[Path] = None
    """Default ``<input_dir>/evals_rewardmap/<timestamp>_<prefix>=<value>/``."""

    reward_mode: RewardMapMode = RewardMapMode.both
    """``raw``: decoder head output; ``prob``: BTL preference using one head only."""

    model_selection: plt_infomec.ModelSelection = plt_infomec.ModelSelection.final
    """Checkpoint: ``best_model.pt`` or final epoch checkpoint."""

    split: Literal["all", "train", "eval"] = "all"
    """Dataset split for reward maps."""

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


def select_sweep_param_value(
    *,
    rows: list[dict[str, str]],
    recon_top_k: int,
    recon_loss_threshold: Optional[float] = None,
) -> tuple[float, dict[str, Any]]:
    """Filter by avg-``recon_loss`` threshold, keep up to top-k, then pick highest max InfoM."""
    by_hp: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        hp = float(row["sweep_param_value"])
        recon = float(row["recon_loss"])
        infom = float(row["infom"])
        bucket = by_hp.setdefault(hp, {"recon": [], "infom": []})
        bucket["recon"].append(recon)
        bucket["infom"].append(infom)

    if not by_hp:
        raise RuntimeError("No sweep rows in latentgrid metrics CSV")

    row_stats = {
        hp: {
            "avg_recon_loss": float(mean(vals["recon"])),
            "max_infom": float(max(vals["infom"])),
        }
        for hp, vals in by_hp.items()
    }

    #* threshold filter (optional)
    filtered_stats = row_stats
    if recon_loss_threshold is not None:
        filtered_stats = {
            hp: stats
            for hp, stats in row_stats.items()
            if stats["avg_recon_loss"] <= float(recon_loss_threshold)
        }
        if not filtered_stats:
            best_hp, best_stats = min(
                row_stats.items(),
                key=lambda item: (item[1]["avg_recon_loss"], item[0]),
            )
            raise RuntimeError(
                f"No HP rows with avg_recon_loss <= {recon_loss_threshold:g}; "
                f"lowest is {best_hp:g} "
                f"(avg_recon_loss={best_stats['avg_recon_loss']:.6g})"
            )

    k = min(int(recon_top_k), len(filtered_stats))
    recon_candidates = sorted(
        filtered_stats.items(),
        key=lambda item: (item[1]["avg_recon_loss"], item[0]),
    )[:k]
    selected_hp, selected_stats = max(
        recon_candidates,
        key=lambda item: (item[1]["max_infom"], -item[1]["avg_recon_loss"]),
    )

    if recon_loss_threshold is not None:
        method = (
            f"avg_recon_loss<={recon_loss_threshold:g}, "
            f"lowest-{k}-of-{len(filtered_stats)} then max_infom"
        )
    else:
        method = f"lowest-{k}-avg_recon_loss then max_infom"

    selection_info: dict[str, Any] = {
        "method": method,
        "recon_top_k": k,
        "recon_loss_threshold": recon_loss_threshold,
        "n_threshold_passing": len(filtered_stats),
        "row_stats": {f"{hp:g}": stats for hp, stats in row_stats.items()},
        "threshold_passing": [
            {"sweep_param_value": hp, **stats}
            for hp, stats in sorted(
                filtered_stats.items(),
                key=lambda item: (item[1]["avg_recon_loss"], item[0]),
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
    print(
        f"Auto-selected {selection_info['method']}: "
        f"sweep_param_value={selected_hp:g} "
        f"(avg_recon_loss={selected['avg_recon_loss']:.6g}, "
        f"max_infom={selected['max_infom']:.4g}) "
        f"from {csv_path.parent.name}"
    )
    for cand in selection_info["recon_candidates"]:
        hp = float(cand["sweep_param_value"])
        print(
            f"  candidate {hp:g}: avg_recon_loss={cand['avg_recon_loss']:.6g}, "
            f"max_infom={cand['max_infom']:.4g}"
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
    return {
        "run_dir": run_dir.resolve(),
        "comment": str(variant.get("comment", run_dir.name)),
        "model_type": str(variant.get("model_type", "")),
        "decoder_type": str(variant.get("decoder_type", "mlp")).lower(),
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


def resolve_reward_map_layout(*, env_name: str) -> dict[str, Any]:
    if plt_infomec.is_ravens_sort_env(env_name):
        layout = plt_infomec.sort_learned_reward_map_layout(env_name=env_name, n_modes=1)
        return dict(
            x_idx=layout.x_idx,
            y_idx=layout.y_idx,
            axis_xlabel=layout.axis_xlabel,
            axis_ylabel=layout.axis_ylabel,
            scatter_s=layout.scatter_s,
            title_fontsize=layout.title_fontsize,
        )
    return dict(
        x_idx=0,
        y_idx=1,
        axis_xlabel="obs[0]",
        axis_ylabel="obs[1]",
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
            raise ValueError(
                "Model has encode_sources=True but source is None."
            )
        enc_kw["source"] = source
    if isinstance(model, QLAE):
        z_q, _ = model.encode(obs1, obs2, labels, **enc_kw)
        return z_q
    mean, _ = model.encode(obs1, obs2, labels, **enc_kw)
    return mean


def _probabilistic_reward_heads_chunk(
    *,
    model: torch.nn.Module,
    s_a: torch.Tensor,
    z: torch.Tensor,
    comparison_obs: torch.Tensor,
    scaling: float,
    comp_chunk_size: int,
) -> torch.Tensor:
    """``E_{s_B}[p(y=1 | s_A, s_B, z)]`` per head; one query chunk. ``(M, Z)``."""
    s_a = model._as_batl_obs(obs=s_a)  # (M, 1, T, O)
    s_b = model._as_batl_obs(obs=comparison_obs)  # (N, 1, T, O)
    m_n, _, t_n, _ = s_a.shape
    n_n = int(s_b.shape[0])
    z_dim = int(z.shape[-1])
    p_sum = torch.zeros((m_n, z_dim), device=s_a.device, dtype=s_a.dtype)

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
        p_pref = torch.sigmoid(r_a_j - r_b_j)  # (M, N_c, Z)
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
            )
        )
    return torch.cat(chunks, dim=0)  # (M, Z)


def batched_per_head_rewards_on_obs1(
    *,
    model: torch.nn.Module,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    reward_mode: Literal["raw", "prob"],
    comparison_obs: np.ndarray | None = None,
    prob_max_comp: int = 512,
    prob_chunk_size: int = 64,
    prob_comp_chunk_size: int = 64,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """``(Q, S, T, Z)`` per-head rewards on ``obs1`` with encoder latents per query."""
    q_n = int(len(obs1))
    s_n = int(obs1.shape[1])
    t_n = int(obs1.shape[2])
    o_dim = int(obs1.shape[-1])
    z_dim = int(model.latent_dim)
    n_heads = min(z_dim, MAX_HEAD_PANELS)
    rewards = np.full((q_n, s_n, t_n, n_heads), np.nan, dtype=np.float64)

    comp_t: torch.Tensor | None = None
    if reward_mode == "prob":
        assert comparison_obs is not None
        comp_t = torch.from_numpy(comparison_obs).float().to(device)

    scaling = float(getattr(model, "scaling", 1.0))

    for start in range(0, q_n, batch_size):
        end = min(start + batch_size, q_n)
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)  # (B, S, T, O)
        o2 = torch.from_numpy(obs2[start:end]).float().to(device)
        y = torch.from_numpy(labels[start:end]).float().to(device)
        b_n = end - start
        u: torch.Tensor | None = None
        if plt_infomec.model_encode_sources(model=model):
            if source is None:
                raise ValueError(
                    "Model has encode_sources=True but source is None."
                )
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
                r_heads = model.decoder(x, vector_rewards=True)
                rewards[start:end, :, :, :n_heads] = (
                    r_heads[..., :n_heads].detach().cpu().numpy()
                )
                continue

            assert comp_t is not None
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
            )  # (B*S*T, Z)
            rewards[start:end, :, :, :n_heads] = (
                r_flat[:, :n_heads].reshape(b_n, s_n, t_n, n_heads).detach().cpu().numpy()
            )
    return rewards


def flatten_obs_and_head_rewards(
    *,
    obs1: np.ndarray,
    rewards: np.ndarray,
    latent_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build 16 panel lists: shared obs, per-head scalar reward."""
    obs_flat = ei.rearrange(obs1, "q s t o -> (q s t) o")
    rw_flat = ei.rearrange(rewards, "q s t z -> (q s t) z")
    n_active = min(int(latent_dim), MAX_HEAD_PANELS)
    per_head_obs: list[np.ndarray] = []
    per_head_r: list[np.ndarray] = []
    for head_idx in range(MAX_HEAD_PANELS):
        if head_idx < n_active:
            per_head_obs.append(obs_flat)
            per_head_r.append(rw_flat[:, head_idx : head_idx + 1])
        else:
            per_head_obs.append(np.empty((0, obs_flat.shape[1]), dtype=np.float64))
            per_head_r.append(np.empty((0, 1), dtype=np.float64))
    return per_head_obs, per_head_r


def head_panel_titles(*, latent_dim: int) -> list[str]:
    n_active = min(int(latent_dim), MAX_HEAD_PANELS)
    titles: list[str] = []
    for head_idx in range(MAX_HEAD_PANELS):
        if head_idx < n_active:
            titles.append(f"$z_{head_idx}$")
        else:
            titles.append(f"$z_{head_idx}$ (inactive)")
    return titles


def format_infomec_suptitle_suffix(*, infomec: dict[str, float]) -> str:
    return (
        f"InfoM={infomec['infom']:.3g} "
        f"InfoE={infomec['infoe']:.3g} "
        f"InfoC={infomec['infoc']:.3g}"
    )


def format_reward_map_suptitle(
    *,
    meta: dict[str, Any],
    reward_mode: Literal["raw", "prob"],
    ckpt_label: str,
    split: str,
    infomec: dict[str, float] | None = None,
) -> str:
    param_label = meta["sweep_param_label"]
    param_val = meta["sweep_param_value"]
    lines = [
        f"Per-head reward ({reward_mode})",
        str(meta["env"]),
        f"{param_label}={param_val:g}, nz={meta['latent_dim']}, dec={meta['decoder_type']}",
        f"split={split}, {ckpt_label}",
    ]
    if infomec is not None:
        lines.append(format_infomec_suptitle_suffix(infomec=infomec))
    return "\n".join(lines)


def save_per_head_reward_figure(
    *,
    obs1: np.ndarray,
    rewards: np.ndarray,
    meta: dict[str, Any],
    reward_mode: Literal["raw", "prob"],
    ckpt_label: str,
    split: str,
    out_png: Path,
    dpi: int,
    infomec: dict[str, float] | None = None,
) -> None:
    layout = resolve_reward_map_layout(env_name=str(meta["env"]))
    per_head_obs, per_head_r = flatten_obs_and_head_rewards(
        obs1=obs1,
        rewards=rewards,
        latent_dim=int(meta["latent_dim"]),
    )
    xlim, ylim = observation_xy_limits(
        obs=obs1.reshape(-1, obs1.shape[-1]),
        x_idx=int(layout["x_idx"]),
        y_idx=int(layout["y_idx"]),
    )
    fig = plot_observation_rewards_by_annotator_mode(
        per_head_obs,
        per_head_r,
        mode_titles=head_panel_titles(latent_dim=int(meta["latent_dim"])),
        suptitle=format_reward_map_suptitle(
            meta=meta,
            reward_mode=reward_mode,
            ckpt_label=ckpt_label,
            split=split,
            infomec=infomec,
        ),
        scatter_s=float(layout["scatter_s"]),
        ncols=HEAD_GRID_NCOLS,
        x_idx=int(layout["x_idx"]),
        y_idx=int(layout["y_idx"]),
        axis_xlabel=str(layout["axis_xlabel"]),
        axis_ylabel=str(layout["axis_ylabel"]),
        xlim=xlim,
        ylim=ylim,
        title_fontsize=int(layout["title_fontsize"]),
        reward_vmin=0.0 if reward_mode == "prob" else None,
        reward_vmax=1.0 if reward_mode == "prob" else None,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def reward_modes_to_run(*, reward_mode: RewardMapMode) -> list[Literal["raw", "prob"]]:
    if reward_mode == RewardMapMode.both:
        return ["raw", "prob"]
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
        raise RuntimeError(
            f"No subruns with {sweep_param_value!r} under {input_dir}"
        )

    decoder_type = str(rows[0]["decoder_type"])
    if not decoder_sums_latent_heads(decoder_type=decoder_type):
        raise RuntimeError(
            f"decoder_type={decoder_type!r} does not sum per-latent heads; "
            f"choose one of linear / odinlinear / slotlinear / ellenlinear / offsetlinear."
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
        _sources,
        _source_names,
        _source_keys,
        encode_batch_size,
        _variant,
        dataset_path,
        query_source,
        device,
    ) = plt_infomec.prepare_infomec_eval_arrays(
        ref_run_dir,
        split=args.split,
        num_bins=10,
        batch_size=args.batch_size,
        set_length=args.set_length,
        dataset_override=None,
        device_str=args.device,
    )
    batch_size = int(args.batch_size or encode_batch_size)
    print(
        f"Reward maps: {len(rows)} subruns | {rows[0]['sweep_param_label']}"
        f"={sweep_param_value:g} | dec={decoder_type} | "
        f"dataset={dataset_path} ({len(obs1)} queries) | device={device}"
    )

    comparison_pool: np.ndarray | None = None
    modes = reward_modes_to_run(reward_mode=args.reward_mode)
    infomec_root = resolve_latentgrid_infomec_root(
        input_dir=input_dir,
        metrics_dir=args.latentgrid_metrics_dir,
    )
    if "prob" in modes:
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

        entry: dict[str, Any] = {
            "run_dir": str(run_dir),
            "comment": row["comment"],
            "latent_dim": int(row["latent_dim"]),
            "checkpoint": ckpt_label,
            "selected_epoch": selected_epoch,
            "model_selection": model_selection_tag,
            "pngs": {},
        }
        if infomec is not None:
            entry["infomec"] = infomec

        for mode in modes:
            rewards = batched_per_head_rewards_on_obs1(
                model=model,
                obs1=obs1,
                obs2=obs2,
                labels=labels,
                batch_size=batch_size,
                device=device,
                reward_mode=mode,
                comparison_obs=comparison_pool,
                prob_max_comp=args.prob_max_comp,
                prob_chunk_size=args.prob_chunk_size,
                prob_comp_chunk_size=args.prob_comp_chunk_size,
                source=query_source,
            )
            png_path = rewardmap_out_png(
                outdir=outdir,
                sweep_param_name=sweep_param_name,
                sweep_param_value=sweep_param_val,
                latent_dim=nz,
                reward_mode=mode,
                infom=float(infomec["infom"]),
                infoc=float(infomec["infoc"]),
            )
            save_per_head_reward_figure(
                obs1=obs1,
                rewards=rewards,
                meta=row,
                reward_mode=mode,
                ckpt_label=ckpt_label,
                split=args.split,
                out_png=png_path,
                dpi=args.dpi,
                infomec=infomec,
            )
            entry["pngs"][mode] = str(png_path.relative_to(outdir))
            tqdm.write(f"{row['comment']}: saved {png_path.relative_to(outdir)}")

        manifest["figures"].append(entry)

    manifest_path = outdir / "rewardmap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
