import inspect
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Quiet transitive TF / numba noise before UMAP or other deps import them.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import einops as ei
import matplotlib
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pref_learn.configs.data_configs import (
    infomec_dataset_path_for_variant,
    pref_dataset_path_for_variant,
)
from pref_learn.models.qlae import QLAE
from pref_learn.models.utils import PreferenceDataset
from pref_learn.models.vae import VAEModel
from pref_learn.utils.infomec import InfoMECResult, compute_infomec
from pref_learn.utils.plot_utils import plot_observation_rewards_by_annotator_mode
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


class InfoMECCurveColor(str, Enum):
    """Matplotlib line colors for InfoM / InfoE / InfoC (stable across flat series)."""

    infom = "tab:blue"
    infoe = "tab:orange"
    infoc = "tab:green"


class LatentTableMapEnvId(str, Enum):
    """``variant['env']`` values that enable the sort-easy object (x,y) table latent map in :func:`main`."""

    sort_easy_v0 = "sort-easy-v0"


def is_ravens_sort_env(env_name: str) -> bool:
    """True for Ravens sort-easy gym ids (``sort-easy-*``)."""
    return str(env_name).startswith("sort-easy")


_SORT_N_GOALS_RE = re.compile(r"sort-easy-(?P<n>\d+)goals")


@dataclass(frozen=True)
class SortLearnedRewardMapLayout:
    """Scatter-grid layout for Ravens sort learned-reward maps."""

    n_goals: Optional[int]
    ncols: int
    scatter_s: float
    x_idx: int = 1
    y_idx: int = 0
    axis_xlabel: str = "EE y"
    axis_ylabel: str = "EE x"
    invert_yaxis: bool = True
    title_fontsize: int = 10


def parse_sort_n_goals(env_name: str) -> Optional[int]:
    """Parse goal count from ids like ``sort-easy-2goals-v0`` / ``sort-easy-4goals-v0``."""
    match = _SORT_N_GOALS_RE.search(str(env_name))
    if match is None:
        return None
    return int(match.group("n"))


def sort_learned_reward_map_layout(
    *, env_name: str, n_modes: int
) -> SortLearnedRewardMapLayout:
    """
    Layout for learned-reward scatter grids on Ravens sort envs.

    * 2 goals / 2 pref modes → 1×2 grid
    * 4 goals / 8 pref modes → 2×4 grid (matches GT ``queries_relabelled_by_annot_mode.png``)
    """
    n_goals = parse_sort_n_goals(env_name=env_name)
    if n_modes <= 2:
        return SortLearnedRewardMapLayout(
            n_goals=n_goals,
            ncols=2,
            scatter_s=0.4,
            title_fontsize=10,
        )
    return SortLearnedRewardMapLayout(
        n_goals=n_goals,
        ncols=4,
        scatter_s=0.25,
        title_fontsize=9,
    )


def format_sort_learned_reward_map_suptitle(
    *,
    run_title: Optional[str],
    env_name: str,
    n_modes: int,
    layout: SortLearnedRewardMapLayout,
) -> str:
    goal_part = (
        f"{layout.n_goals} goals / {n_modes} pref modes"
        if layout.n_goals is not None
        else f"{n_modes} pref modes"
    )
    map_line = f"Learned reward vs EE (y,x): {goal_part} ({env_name})"
    if run_title:
        return f"{map_line}\n{run_title}"
    return map_line


def model_supports_obs_latent_reward_decode(model: Any) -> bool:
    """``decode(obs, z)`` reward maps: ``VAEModel``, ``QLAE``."""
    return isinstance(model, (VAEModel, QLAE))


def model_supports_probabilistic_reward(model: Any) -> bool:
    """Models with VPL Sec. 4.1 ``get_probabilistic_reward`` (e.g. VAE, QLAE)."""
    return callable(getattr(model, "get_probabilistic_reward", None))


class LatentClusterMethod(str, Enum):
    """2D latent embedding for mode-colored scatter plots."""

    none = "none"
    tsne = "tsne"
    umap = "umap"


class ModelSelection(str, Enum):
    """Which training checkpoint to load for InfoMEC eval plots."""

    best = "best"
    final = "final"


def latent_cluster_enabled(method: LatentClusterMethod) -> bool:
    """True when ``method`` requests a 2D latent embedding plot."""
    return method != LatentClusterMethod.none


@dataclass
class AnalysisCLI:
    """InfoMEC / NMI for one training directory (variant.json + checkpoint)."""

    input_dir: Path
    """Training run directory with variant.json and the checkpoint (see ckpt-name)."""

    ckpt_name: str = "best_model.pt"
    """Checkpoint file name inside ``input_dir`` when ``model_selection=best``."""

    model_selection: ModelSelection = ModelSelection.best
    """``best``: ``best_model.pt``; ``final``: latest ``checkpoints/model_<epoch>.pt``."""

    dataset: Optional[Path] = None
    """Override InfoMEC eval dataset pickle; default from variant infomec_dataset_path."""

    split: Literal["all", "train", "eval"] = "all"
    """Which part of the preference dataset to use."""

    batch_size: Optional[int] = None
    """Encoder batch size; default from variant.json."""

    set_length: Optional[int] = None
    """Truncate set dimension; default from variant.json set_size."""

    num_bins: int = 10
    """Histogram bins when discretizing continuous sources."""

    pref_diagonal: bool = False
    """Sort-4goals InfoMEC only: append ``pref_diagonal`` source (3 classes) from vert/hori."""

    device: str = "auto"
    """torch device name, or 'auto'."""

    latent_xy_table_grid: bool = True
    """If True, write latent-mean heatmaps over **object** table position (``O[3:5]``) when env is sort-easy-v0."""

    xy_grid_resolution: int = 48
    """Square grid size (Gx=Gy) for the sort-easy-v0 table map; no effect if env is not sort-easy-v0."""

    xy_grid_pad_rel: float = 0.02
    """Pad data-driven x/y bounds by this relative margin; no effect if env is not sort-easy-v0."""

    nmi_heatmap_active_dots: bool = True
    """If True, draw a small red dot per (source, latent) cell for active latents on the NMI heatmap. Tyro: ``--no-nmi-heatmap-active-dots`` turns this off."""

    latent_cluster_method: LatentClusterMethod = LatentClusterMethod.tsne
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


_MODEL_CKPT_RE = re.compile(r"^model_(\d+)\.pt$")

# Per-run: ``evals_individual/<timestamp>/`` (plot_infomec) or ``evals_subsweep/<timestamp>/`` (plot_infomec_sweep).
# Sweep aggregate: ``evals_sweep/<timestamp>_model=<best|final>/``.
BEST_CHECKPOINT_SELECTION = "best_eval_pref"
FINAL_CHECKPOINT_SELECTION = "final_epoch_checkpoint"
LATENT_PROJECTION_SCATTER_S = 2.0
LATENT_PROJECTION_SCATTER_ALPHA = 0.85


def learned_reward_sweep_copy_path(
    sweep_outdir: Path,
    *,
    comment: str,
    mode: Literal["raw", "prob"],
) -> Path:
    """Aggregate sweep copy under ``learned_rewards_mode_{prob|raw}/``."""
    stem = str(comment).strip() or "run"
    subdir = "learned_rewards_mode_prob" if mode == "prob" else "learned_rewards_mode_raw"
    return sweep_outdir / subdir / f"learned_reward_{stem}.png"


def eval_run_outdir(
    input_dir: Path,
    eval_timestamp: str,
    *,
    kind: Literal["individual", "subsweep"] = "individual",
) -> Path:
    """Per-run InfoMEC artifacts under ``evals_individual/`` or ``evals_subsweep/``."""
    subdir = "evals_individual" if kind == "individual" else "evals_subsweep"
    return input_dir / subdir / eval_timestamp


def beta_from_variant(variant: dict) -> float:
    """Backward-compatible alias for the swept disentanglement hyperparameter."""
    from pref_learn.utils.disentangle_sweep import sweep_from_variant

    return sweep_from_variant(variant=variant)[1]


def infomec_run_label(input_dir: Path, variant: dict) -> str:
    comment = str(variant.get("comment", "")).strip()
    if comment:
        return comment
    return input_dir.name


def format_checkpoint_choice_line(
    *,
    ckpt_name: str,
    best_epoch: Optional[int],
    best_selection: str = BEST_CHECKPOINT_SELECTION,
) -> str:
    """Epoch index only (for plot titles; ``ckpt_name`` / ``best_selection`` kept for callers)."""
    if best_epoch is not None:
        return f"epoch_{int(best_epoch)}"
    return ""


def format_latent_mask_counts_line(
    *,
    n_varies: int,
    n_inform: int,
    n_active: int,
    latent_dim: int,
) -> str:
    return (
        f"{{varies, inform, active}} / total: "
        f"{{{n_varies},{n_inform},{n_active}}} / {latent_dim}"
    )


def latent_mask_counts_from_result(result: InfoMECResult) -> Tuple[int, int, int, int]:
    latent_dim = int(len(result.active_latents))
    return (
        int(np.sum(result.varies_latents)),
        int(np.sum(result.inform_latents)),
        int(np.sum(result.active_latents)),
        latent_dim,
    )


def format_infomec_run_title(
    *,
    infom: float,
    infoe: float,
    infoc: float,
    n_varies: int,
    n_inform: int,
    n_active: int,
    latent_dim: int,
    run_label: str,
    ckpt_name: str = "best_model.pt",
    best_epoch: Optional[int] = None,
    best_selection: str = BEST_CHECKPOINT_SELECTION,
) -> str:
    """Run label, InfoM/E/C, latent mask counts, and checkpoint epoch (sweep + individual subplots)."""
    counts_line = format_latent_mask_counts_line(
        n_varies=n_varies,
        n_inform=n_inform,
        n_active=n_active,
        latent_dim=latent_dim,
    )
    title_lines = [
        run_label,
        f"InfoMEC = {infom:.3f} / {infoe:.3f} / {infoc:.3f}",
        counts_line,
    ]
    ckpt_line = format_checkpoint_choice_line(
        ckpt_name=ckpt_name,
        best_epoch=best_epoch,
        best_selection=best_selection,
    )
    if ckpt_line:
        title_lines.append(ckpt_line)
    return "\n".join(title_lines)


def gather_epoch_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    """``run_dir/checkpoints/model_<epoch>.pt``; sorted by epoch. Empty if ``checkpoints/`` is missing."""
    run_dir = run_dir.expanduser().resolve()
    ckpt_sub = run_dir / "checkpoints"
    if not ckpt_sub.is_dir():
        return []
    pairs: List[Tuple[int, Path]] = []
    for p in ckpt_sub.iterdir():
        if not p.is_file():
            continue
        m = _MODEL_CKPT_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def resolve_final_epoch_checkpoint(run_dir: Path) -> Optional[Tuple[int, Path]]:
    """Latest ``checkpoints/model_<epoch>.pt`` by epoch; None if ``checkpoints/`` is empty."""
    ckpts = gather_epoch_checkpoints(run_dir)
    if not ckpts:
        return None
    return ckpts[-1]


def run_dir_has_checkpoint(run_dir: Path, *, model_selection: ModelSelection) -> bool:
    run_dir = run_dir.expanduser().resolve()
    if not (run_dir / "variant.json").is_file():
        return False
    if model_selection == ModelSelection.best:
        return (run_dir / "best_model.pt").is_file()
    return resolve_final_epoch_checkpoint(run_dir) is not None


def model_selection_display_label(model_selection: ModelSelection) -> str:
    if model_selection == ModelSelection.best:
        return "best_model.pt"
    return "final checkpoint"


def resolve_run_checkpoint(
    run_dir: Path,
    *,
    model_selection: ModelSelection,
    ckpt_name: str = "best_model.pt",
) -> Tuple[Path, str, Optional[int], str]:
    """Return ``(path, title_label, epoch, selection_tag)`` for one training run."""
    run_dir = run_dir.expanduser().resolve()
    if model_selection == ModelSelection.best:
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint {ckpt_name!r} under run dir: {ckpt_path}"
            )
        epoch = resolve_best_checkpoint_epoch(run_dir, ckpt_name=ckpt_name)
        return ckpt_path, ckpt_name, epoch, BEST_CHECKPOINT_SELECTION

    final = resolve_final_epoch_checkpoint(run_dir)
    if final is None:
        raise FileNotFoundError(
            f"No checkpoints/model_<epoch>.pt under {run_dir / 'checkpoints'}"
        )
    epoch, ckpt_path = final
    ckpt_label = f"checkpoints/{ckpt_path.name}"
    return ckpt_path, ckpt_label, int(epoch), FINAL_CHECKPOINT_SELECTION


def _model_state_dict(model_or_state: Any) -> Dict[str, torch.Tensor]:
    if hasattr(model_or_state, "state_dict"):
        return model_or_state.state_dict()
    if isinstance(model_or_state, dict):
        return model_or_state
    raise TypeError(f"Expected torch Module or state_dict; got {type(model_or_state)!r}")


def _state_dicts_allclose(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> bool:
    if a.keys() != b.keys():
        return False
    for key in a:
        if not torch.allclose(a[key], b[key], rtol=0.0, atol=0.0):
            return False
    return True


def resolve_best_checkpoint_epoch(
    run_dir: Path, ckpt_name: str = "best_model.pt"
) -> Optional[int]:
    """Match ``run_dir/<ckpt_name>`` weights to ``checkpoints/model_<epoch>.pt``; None if unmatched."""
    run_dir = run_dir.expanduser().resolve()
    best_path = run_dir / ckpt_name
    if not best_path.is_file():
        return None
    best_sd = _model_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=False)
    )
    for epoch, ckpt_path in gather_epoch_checkpoints(run_dir):
        ckpt_sd = _model_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=False)
        )
        if _state_dicts_allclose(best_sd, ckpt_sd):
            return int(epoch)
    return None


def place_infomec_twin_legend_outside(
    ax_left: Axes,
    ax_right: Axes,
    *,
    legend_fontsize: int = 8,
) -> None:
    """Combined legend for InfoM/E/C + active counts, centered below the left axis (outside the data)."""
    h1, lab1 = ax_left.get_legend_handles_labels()
    h2, lab2 = ax_right.get_legend_handles_labels()
    if not h1 and not h2:
        return
    ax_left.legend(
        h1 + h2,
        lab1 + lab2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        fontsize=legend_fontsize,
        frameon=True,
    )


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
    selected_epoch: Optional[int] = None,
) -> None:
    """Draw InfoM/E/C + active latents (twin) on existing axes. Set ``show_legend=False`` when a figure-level legend is used (e.g. sweep grid)."""
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
        color=InfoMECCurveColor.infom,
    )
    ax_left.plot(
        epochs,
        infoe_a,
        label="InfoE",
        marker="o",
        markersize=3,
        color=InfoMECCurveColor.infoe,
    )
    ax_left.plot(
        epochs,
        infoc_a,
        label="InfoC",
        marker="o",
        markersize=3,
        color=InfoMECCurveColor.infoc,
    )
    ax_left.set_xlabel("epoch")
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

    if selected_epoch is not None:
        ep_arr = np.asarray(epochs, dtype=np.int64)
        matches = np.flatnonzero(ep_arr == int(selected_epoch))
        if matches.size:
            j = int(matches[0])
            eval_dot = dict(
                color="red",
                marker="o",
                markersize=5,
                linestyle="none",
                zorder=5,
            )
            ax_left.plot(int(selected_epoch), infom_a[j], **eval_dot)
            ax_left.plot(int(selected_epoch), infoe_a[j], **eval_dot)
            ax_left.plot(int(selected_epoch), infoc_a[j], **eval_dot)
            ax_right.plot(int(selected_epoch), n_active_norm[j], **eval_dot)

    if show_legend:
        place_infomec_twin_legend_outside(
            ax_left, ax_right, legend_fontsize=legend_fontsize
        )


def plot_infomec_epoch_trajectory(
    epochs: List[int],
    infom: np.ndarray,
    infoe: np.ndarray,
    infoc: np.ndarray,
    n_active: np.ndarray,
    latent_dim: int,
    outpath: Path,
    *,
    title: Optional[str] = None,
    selected_epoch: Optional[int] = None,
) -> None:
    """Line plot: x=epoch (linear only); left y = InfoM/E/C in [0,1]; right = active = n_active/L (ticks show counts)."""
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
        title=title,
        selected_epoch=selected_epoch,
    )
    fig.tight_layout(rect=[0, 0.1, 1, 0.99])
    plt.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0.12)
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
    source: np.ndarray | None = None,
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
        latents = batched_encode(
            model, obs1, obs2, labels, batch_size, device, source=source
        )
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
    *,
    title: Optional[str] = None,
    best_epoch: Optional[int] = None,
    best_selection: str = BEST_CHECKPOINT_SELECTION,
) -> None:
    """Write one epoch (linear x) curve and infomec_epoch_checkpoint_curve.json, if payload is set."""
    if payload is None:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    latent_dim = int(payload["latent_dim"])
    epochs = payload["epochs"]
    infom = np.asarray(payload["infom"], dtype=np.float64)
    infoe = np.asarray(payload["infoe"], dtype=np.float64)
    infoc = np.asarray(payload["infoc"], dtype=np.float64)
    n_active = np.asarray(payload["n_active_latents"], dtype=np.float64)
    plot_infomec_epoch_trajectory(
        epochs,
        infom,
        infoe,
        infoc,
        n_active,
        latent_dim,
        outdir / "infomec_over_epoch_checkpoints.png",
        title=title,
        selected_epoch=best_epoch,
    )

    payload_out = {
        **payload,
        "best_checkpoint_epoch": best_epoch,
        "best_checkpoint_selection": best_selection,
    }
    with open(outdir / "infomec_epoch_checkpoint_curve.json", "w", encoding="utf-8") as f:
        json.dump(payload_out, f, indent=2)


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
    source: np.ndarray | None = None,
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
        source=source,
    )
    save_infomec_epoch_series_artifacts(payload, outdir)


def save_eval_artifacts_after_encode(
    outdir: Path,
    input_dir: Path,
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
    *,
    run_title: Optional[str] = None,
    nmi_heatmap_active_dots: bool = True,
    latent_cluster_method: LatentClusterMethod = LatentClusterMethod.tsne,
    tsne_perplexity: Optional[float] = None,
    tsne_max_iter: Optional[int] = None,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    model: Optional[torch.nn.Module] = None,
    obs2: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    learned_reward_mode: Literal["raw", "prob"] = "prob",
    learned_reward_max_comp: int = 512,
    learned_reward_also_save_as: Optional[Path] = None,
    learned_reward_sweep_outdir: Optional[Path] = None,
    learned_reward_sweep_comment: Optional[str] = None,
    query_source: np.ndarray | None = None,
    pref_diagonal: bool = False,
) -> Dict[str, Any]:
    """Persist latents, NMI heatmap, latent 2D embedding, infomec_metrics.json, summary.txt under outdir.

    latents: (N, L) — N preference sets after split, L latent dim.
    sources: (N, S) — S binned/discrete source columns.
    obs1: (N, T, ...) or truncated set dim T per variant set_size.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    n_varies = int(np.sum(result.varies_latents))
    n_inform = int(np.sum(result.inform_latents))
    n_active_latents = int(np.sum(result.active_latents))
    latent_dim = int(latents.shape[1])

    np.save(outdir / "latents.npy", latents)
    np.save(outdir / "sources.npy", sources)
    np.save(outdir / "nmi.npy", result.nmi)

    latent_names = [f"z{j}" for j in range(latents.shape[1])]
    al_for_plot = result.active_latents if nmi_heatmap_active_dots else None
    plot_title = run_title or "InfoMEC normalized MI (sources x latents)"
    save_heatmap(
        result.nmi,
        source_names,
        latent_names,
        title=plot_title,
        outpath=str(outdir / "infomec_nmi_heatmap.png"),
        active_latents=al_for_plot,
        title_fontsize=7,
    )
    # save_heatmap(
    #     result.nmi,
    #     source_names,
    #     latent_names,
    #     title="InfoMEC normalized MI (sources x latents)",
    #     outpath=str(outdir / "infomec_nmi_heatmap.png"),
    #     active_latents=result.active_latents,
    # )

    mode_color = source_mode_labels_for_split(dataset_path=dataset_path, split=split)
    mode_names = source_mode_names_from_dataset(dataset_path=dataset_path)
    assert len(mode_color) == len(latents), (
        f"source_mode N={len(mode_color)} != latents N={len(latents)}"
    )
    projection_png = outdir / f"{latent_cluster_method.value}_by_source_mode.png"
    projection_info = maybe_save_latent_projection(
        latents,
        mode_color,
        str(projection_png),
        plot_title,
        method=latent_cluster_method,
        tsne_perplexity=tsne_perplexity,
        tsne_max_iter=tsne_max_iter,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        mode_names=mode_names,
        title_fontsize=7,
    )
    projection_metrics = latent_projection_metrics_from_info(
        projection_info=projection_info,
        method=latent_cluster_method,
        png_path=projection_png,
        mode_names=mode_names,
    )

    learned_reward_metrics: Optional[Dict[str, Any]] = None
    if (
        model is not None
        and obs2 is not None
        and labels is not None
        and device is not None
    ):
        reward_kw = dict(
            model=model,
            obs1=obs1,
            obs2=obs2,
            labels=labels,
            variant=variant,
            dataset_path=dataset_path,
            split=split,
            outdir=outdir,
            device=device,
            batch_size=batch_size,
            run_title=run_title,
            learned_reward_max_comp=learned_reward_max_comp,
            query_source=query_source,
        )
        if model_supports_probabilistic_reward(model=model):
            sweep_comment = str(learned_reward_sweep_comment or input_dir.name).strip()
            learned_reward_metrics = {}
            for mode in ("prob", "raw"):
                sweep_copy_path: Optional[Path] = None
                if learned_reward_sweep_outdir is not None:
                    sweep_copy_path = learned_reward_sweep_copy_path(
                        learned_reward_sweep_outdir,
                        comment=sweep_comment,
                        mode=mode,
                    )
                mode_metrics = maybe_save_learned_reward_by_annotator_mode(
                    **reward_kw,
                    learned_reward_mode=mode,
                    also_save_as=sweep_copy_path,
                )
                if mode_metrics is None:
                    continue
                learned_reward_metrics[f"learned_reward_by_annot_mode_{mode}_png"] = (
                    mode_metrics["learned_reward_by_annot_mode_png"]
                )
                if sweep_copy_path is not None:
                    learned_reward_metrics[f"learned_reward_sweep_{mode}_png"] = str(
                        sweep_copy_path
                    )
                if mode == "prob":
                    learned_reward_metrics.update(
                        {
                            k: v
                            for k, v in mode_metrics.items()
                            if k
                            not in {
                                "learned_reward_by_annot_mode_png",
                                "learned_reward_sweep_copy_png",
                                "learned_reward_mode",
                            }
                        }
                    )
            if not learned_reward_metrics:
                learned_reward_metrics = None
        else:
            learned_reward_metrics = maybe_save_learned_reward_by_annotator_mode(
                **reward_kw,
                learned_reward_mode=learned_reward_mode,
                also_save_as=learned_reward_also_save_as,
            )

    metrics = result.to_dict()
    metrics.update(
        {
            "input_dir": str(input_dir),
            "eval_dir": str(outdir),
            "eval_timestamp": eval_timestamp,
            "dataset_path": str(dataset_path),
            "env": variant.get("env"),
            "batch_size_used": batch_size,
            "set_length_used": int(variant["set_size"]),
            "source_keys": source_keys,
            "source_names": source_names,
            "pref_diagonal": pref_diagonal,
            "num_samples": int(len(obs1)),
            "num_sources": int(sources.shape[1]),
            "latent_dim": int(latents.shape[1]),
            "split": split,
            **projection_metrics,
        }
    )
    if learned_reward_metrics is not None:
        metrics.update(learned_reward_metrics)
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
        f.write(
            format_latent_mask_counts_line(
                n_varies=n_varies,
                n_inform=n_inform,
                n_active=n_active_latents,
                latent_dim=latent_dim,
            )
            + "\n"
        )
        if projection_info is not None:
            f.write(
                f"latent_projection_gt_modes_seen={projection_info['n_gt_modes_seen']} "
                f"{projection_info['gt_modes_seen']}\n"
            )

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


def resolve_infomec_dataset_path(
    run_dir: Path, variant: dict, override: Optional[Path]
) -> Path:
    """InfoMEC held-out pref set; train/val use ``variant['dataset_path']``."""
    dataset_rel = infomec_dataset_path_for_variant(variant=variant)
    return resolve_dataset_path(run_dir, dataset_rel, override)


def resolve_training_dataset_path(
    run_dir: Path, variant: dict, override: Optional[Path]
) -> Path:
    """Training preference pickle from ``variant['pref_dataset']``."""
    dataset_rel = pref_dataset_path_for_variant(variant=variant)
    return resolve_dataset_path(run_dir, dataset_rel, override)


def subsample_preference_queries(
    *,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    query_source: np.ndarray | None,
    frac: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Random query subsample; ``frac=1`` keeps all rows."""
    frac_f = float(frac)
    if not (0.0 < frac_f <= 1.0):
        raise ValueError(f"viz_data_frac must be in (0, 1]; got {frac_f!r}")
    n = int(len(obs1))
    if frac_f >= 1.0 or n <= 1:
        return obs1, obs2, labels, query_source
    k = max(1, int(round(frac_f * n)))
    rng = np.random.default_rng(seed=seed)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    src_out = None if query_source is None else np.asarray(query_source)[idx]
    return obs1[idx], obs2[idx], labels[idx], src_out


def prepare_preference_eval_arrays(
    run_dir: Path,
    *,
    split: str,
    batch_size: Optional[int],
    set_length: Optional[int],
    dataset_override: Optional[Path],
    device_str: str,
    viz_data_frac: float = 1.0,
    viz_data_seed: int = 0,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    dict,
    Path,
    np.ndarray | None,
    torch.device,
]:
    """Load **training** preference data for reward-map / viz (not InfoMEC held-out set)."""
    run_dir = run_dir.expanduser().resolve()
    variant_path = run_dir / "variant.json"
    if not variant_path.is_file():
        raise FileNotFoundError(f"Missing variant.json under run_dir: {variant_path}")
    with open(variant_path, encoding="utf-8") as vf:
        meta = json.load(vf)
    variant = meta["variant"]
    bs = int(variant["batch_size"]) if batch_size is None else int(batch_size)
    sl = int(variant["set_size"]) if set_length is None else int(set_length)
    dataset_path = resolve_training_dataset_path(run_dir, variant, dataset_override)
    with open(dataset_path, "rb") as f:
        batch = pickle.load(f)
    n = len(batch["observations"])
    idx = split_indices(n, split)
    obs1 = maybe_truncate(np.asarray(batch["observations"])[idx], sl)
    obs2 = maybe_truncate(np.asarray(batch["observations_2"])[idx], sl)
    labels = maybe_truncate(np.asarray(batch["labels"])[idx], sl)
    query_source = slice_query_source(batch=batch, idx=idx)
    obs1, obs2, labels, query_source = subsample_preference_queries(
        obs1=obs1,
        obs2=obs2,
        labels=labels,
        query_source=query_source,
        frac=viz_data_frac,
        seed=viz_data_seed,
    )
    device = resolve_eval_device(device_str)
    return (
        obs1,
        obs2,
        labels,
        bs,
        variant,
        dataset_path,
        query_source,
        device,
    )


def split_indices(n: int, split: str) -> np.ndarray:
    eval_size = int(0.1 * n)
    train_size = n - eval_size
    if split == "all":
        return np.arange(n)
    if split == "train":
        return np.arange(train_size)
    return np.arange(train_size, n)


def display_source_variable_name(name: str) -> str:
    """``pref_vertical`` -> ``vertical``; leave other names unchanged."""
    if name.startswith("pref_"):
        return name[len("pref_") :]
    return name


def source_variable_display_names_from_batch(batch: dict) -> Optional[List[str]]:
    raw = batch.get("source_names")
    if raw is None:
        return None
    return [display_source_variable_name(str(n)) for n in raw]


def source_mode_names_from_dataset(dataset_path: Path) -> Optional[List[str]]:
    """Mode index -> name list from relabelled dataset (``source_mode_names``), if present."""
    with open(dataset_path, "rb") as fp:
        batch = pickle.load(fp)
    raw = batch.get("source_mode_names")
    if raw is None:
        return None
    return [str(x) for x in raw]


def source_mode_display_name(mode: int, mode_names: Optional[List[str]]) -> str:
    if mode_names is not None and 0 <= mode < len(mode_names):
        return mode_names[mode]
    return f"mode {mode}"


def source_mode_labels_for_split(dataset_path: Path, split: str) -> np.ndarray:
    """Per-query preference annotator mode from relabelled dataset: ``source_mode`` (Q, 1) -> (N,)."""
    with open(dataset_path, "rb") as fp:
        batch = pickle.load(fp)
    if "source_mode" not in batch:
        raise KeyError(
            f"dataset missing 'source_mode' (create with store_source_mode=True): {dataset_path}"
        )
    idx = split_indices(len(batch["observations"]), split)
    sm = np.asarray(batch["source_mode"])[idx]  # (N, 1) or (N,)
    if sm.ndim == 2:
        sm = sm[:, 0]
    return np.asarray(sm).reshape(-1)


def maybe_truncate(arr: np.ndarray, set_length: int) -> np.ndarray:
    if set_length < 0 or arr.ndim < 2:
        return arr
    return arr[:, :set_length]


def infer_source_keys_from_batch(batch: dict) -> List[str]:
    available = set(batch.keys())
    for candidates in (["source_variable"], ["source_mode"], ["annotator_id"]):
        if candidates[0] in available:
            return candidates
    raise KeyError(
        "Could not infer source keys from dataset. Expected one of source_variable, source_mode, or annotator_id. "
        f"Available keys include: {sorted(batch.keys())[:40]}"
    )


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


PREF_DIAGONAL_SOURCE_NAME = "pref_diagonal"


def is_sort_four_goals_env(env_name: str) -> bool:
    """True for Ravens sort-easy-4goals gym ids."""
    return parse_sort_n_goals(env_name=env_name) == 4


def diagonality_from_vert_hori(*, vert_hori: np.ndarray) -> np.ndarray:
    """
    Derive ``pref_diagonal`` from ``(pref_vertical, pref_horizontal)``, each in ``{0,1,2}``.

    * 0: exactly one axis is 0 (axis-aligned modes).
    * 1: both equal and non-zero, i.e. ``[1,1]`` or ``[2,2]``.
    * 2: both non-zero and unequal, i.e. ``[1,2]`` or ``[2,1]``.
    """
    vh = np.asarray(vert_hori, dtype=np.int32)
    if vh.ndim != 2 or int(vh.shape[1]) != 2:
        raise ValueError(f"expected vert_hori (N, 2), got {vh.shape!r}")
    v = vh[:, 0]
    h = vh[:, 1]
    diag = np.full(v.shape[0], -1, dtype=np.int32)
    one_zero = (v == 0) ^ (h == 0)
    both_eq = (v == h) & (v != 0) & (h != 0)
    both_neq = (v != 0) & (h != 0) & (v != h)
    diag[one_zero] = 0
    diag[both_eq] = 1
    diag[both_neq] = 2
    if np.any(diag < 0):
        bad = np.where(diag < 0)[0][:5]
        raise ValueError(
            "diagonality_from_vert_hori: unmapped (pref_vertical, pref_horizontal) rows, "
            f"e.g. {vert_hori[bad].tolist()}"
        )
    return diag


def maybe_append_diagonality_source(
    *,
    sources: np.ndarray,
    source_names: List[str],
    pref_diagonal: bool,
    env_name: str,
) -> Tuple[np.ndarray, List[str]]:
    """Append ``pref_diagonal`` column for sort-4goals InfoMEC: ``(N, 2)`` -> ``(N, 3)``."""
    if not pref_diagonal:
        return sources, source_names
    if not is_sort_four_goals_env(env_name=env_name):
        raise ValueError(
            "pref_diagonal=True requires sort-easy-4goals env; "
            f"got env={env_name!r}"
        )
    src = np.asarray(sources)
    if src.ndim != 2 or int(src.shape[1]) < 2:
        raise ValueError(
            "pref_diagonal needs sources with >=2 columns "
            f"(pref_vertical, pref_horizontal); got shape {src.shape!r}"
        )
    vert_hori = src[:, :2]
    diag_col = diagonality_from_vert_hori(vert_hori=vert_hori)[:, None]  # (N, 1)
    names = list(source_names)
    if len(names) >= 2:
        names_out = [*names[:2], PREF_DIAGONAL_SOURCE_NAME, *names[2:]]
    else:
        names_out = ["pref_vertical", "pref_horizontal", PREF_DIAGONAL_SOURCE_NAME]
    sources_out = np.concatenate([vert_hori, diag_col, src[:, 2:]], axis=1)
    return sources_out, names_out


def model_encode_sources(model: torch.nn.Module) -> bool:
    return bool(getattr(model, "encode_sources", False))


def slice_query_source(batch: dict, idx: np.ndarray) -> np.ndarray | None:
    """``(N, n_sources)`` GT source indices for the split, or ``None`` if absent."""
    if "source_variable" not in batch:
        return None
    return np.asarray(batch["source_variable"])[idx]


def encode_source_kwargs(
    *,
    model: torch.nn.Module,
    source: np.ndarray | None,
    start: int,
    end: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not model_encode_sources(model=model):
        return {}
    if source is None:
        raise ValueError(
            "Model has encode_sources=True but source is None. "
            "Use a relabelled dataset with source_variable."
        )
    u = torch.from_numpy(np.asarray(source[start:end])).long().to(device)
    return {"source": u}


def encode_preference_latent_mean(
    model: torch.nn.Module,
    o1: torch.Tensor,
    o2: torch.Tensor,
    y: torch.Tensor,
    **enc_kw,
) -> torch.Tensor:
    """``(B, Z)`` posterior mean for VAE / QLAE."""
    if isinstance(model, QLAE):
        z_q, _ = model.encode(o1, o2, y, **enc_kw)
        return z_q
    mean, _ = model.encode(o1, o2, y, **enc_kw)
    return mean


def batched_encode(
    model,
    obs1,
    obs2,
    labels,
    batch_size,
    device,
    source=None,
):
    means = []
    for start in range(0, len(obs1), batch_size):
        end = min(start + batch_size, len(obs1))
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)
        o2 = torch.from_numpy(obs2[start:end]).float().to(device)
        y = torch.from_numpy(labels[start:end]).float().to(device)
        enc_kw = encode_source_kwargs(
            model=model,
            source=source,
            start=start,
            end=end,
            device=device,
        )
        with torch.no_grad():
            mean = encode_preference_latent_mean(model, o1, o2, y, **enc_kw)
        means.append(mean.cpu().numpy())
    return np.concatenate(means, axis=0)


def batched_learned_rewards_on_obs1(
    model: Union[VAEModel, QLAE],
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    *,
    comparison_obs: np.ndarray | None = None,
    reward_mode: Literal["raw", "prob"] = "prob",
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Encoder latent per query (VAE posterior mean; QLAE ``z_q``), then reward on ``obs1``.

    ``prob`` (default): VPL Sec. 4.1 expected BTL preference vs a comparison pool.
    ``raw``: decoder output ``r(s, z)``.
    """
    q_n = int(len(obs1))
    s_n = int(obs1.shape[1])
    t_n = int(obs1.shape[2])
    o_dim = int(obs1.shape[-1])
    rewards = np.zeros((q_n, s_n, t_n), dtype=np.float64)
    comp_t: torch.Tensor | None = None
    if reward_mode == "prob":
        assert comparison_obs is not None, "prob reward_mode requires comparison_obs pool"
        comp_t = torch.from_numpy(comparison_obs).float().to(device)  # (N, O)

    for start in range(0, q_n, batch_size):
        end = min(start + batch_size, q_n)
        o1 = torch.from_numpy(obs1[start:end]).float().to(device)  # (B, S, T, O)
        o2 = torch.from_numpy(obs2[start:end]).float().to(device)
        y = torch.from_numpy(labels[start:end]).float().to(device)
        b_n = end - start
        enc_kw = encode_source_kwargs(
            model=model, source=source, start=start, end=end, device=device
        )
        with torch.no_grad():
            mean = encode_preference_latent_mean(model, o1, o2, y, **enc_kw)
            if reward_mode == "prob":
                assert comp_t is not None
                o_flat = o1.reshape(b_n * s_n * t_n, o_dim)  # (B*S*T, O)
                z_flat = ei.repeat(mean, "B Z -> (B S T) Z", S=s_n, T=t_n)
                r_flat = model.get_probabilistic_reward(
                    s_a=o_flat,
                    z=z_flat,
                    comparison_obs=comp_t,
                    chunk_size=64,
                )
                r = r_flat.reshape(b_n, s_n, t_n)
            else:
                z = ei.repeat(mean, "B Z -> B S T Z", S=s_n, T=t_n)
                r = model.decode(o1, z).squeeze(-1)  # (B, S, T)
        rewards[start:end] = r.detach().cpu().numpy()
    return rewards


def build_comparison_obs_pool(
    obs1: np.ndarray,
    obs2: np.ndarray,
    *,
    max_comp: int = 2048,
    seed: int = 0,
) -> np.ndarray:
    """Sample comparison states ``s_B`` from preference dataset observations."""
    pool = np.concatenate(
        [
            obs1.reshape(-1, obs1.shape[-1]),
            obs2.reshape(-1, obs2.shape[-1]),
        ],
        axis=0,
    )
    if pool.shape[0] > max_comp:
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(pool.shape[0], size=max_comp, replace=False)
        pool = pool[idx]
    return pool


def build_comparison_traj_pool(
    obs1: np.ndarray,
    obs2: np.ndarray,
    *,
    max_comp: int = 2048,
    seed: int = 0,
) -> np.ndarray:
    """Sample comparison trajectories ``s_B`` with shape ``(N, T, O)`` from pref dataset."""
    t_n = int(obs1.shape[2])
    o_dim = int(obs1.shape[-1])
    segs1 = obs1.reshape(-1, t_n, o_dim)
    segs2 = obs2.reshape(-1, t_n, o_dim)
    pool = np.concatenate([segs1, segs2], axis=0)  # (N, T, O)
    if pool.shape[0] > max_comp:
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(pool.shape[0], size=max_comp, replace=False)
        pool = pool[idx]
    return pool


def gather_learned_rewards_by_annotator_mode(
    obs1: np.ndarray,
    rewards: np.ndarray,
    mode_labels: np.ndarray,
    *,
    n_modes: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Group flattened (obs, r) pairs by integer annotator mode index."""
    o_dim = int(obs1.shape[-1])
    per_mode_obs: List[np.ndarray] = []
    per_mode_r: List[np.ndarray] = []
    mode_labels = np.asarray(mode_labels).reshape(-1)
    for mode in range(n_modes):
        mask = mode_labels == mode
        if not np.any(mask):
            per_mode_obs.append(np.empty((0, o_dim), dtype=np.float64))
            per_mode_r.append(np.empty((0, 1), dtype=np.float64))
            continue
        ob = obs1[mask]  # (Q_m, S, T, O)
        rw = rewards[mask]  # (Q_m, S, T)
        per_mode_obs.append(ei.rearrange(ob, "q s t o -> (q s t) o"))
        per_mode_r.append(ei.rearrange(rw, "q s t -> (q s t) 1"))
    return per_mode_obs, per_mode_r


def resolve_n_annotator_modes(
    mode_labels: np.ndarray,
    mode_names: Optional[List[str]],
) -> int:
    if mode_names is not None:
        return len(mode_names)
    if mode_labels.size == 0:
        return 0
    return int(np.max(mode_labels)) + 1


def maybe_save_learned_reward_by_annotator_mode(
    *,
    model: Optional[torch.nn.Module],
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    variant: dict,
    dataset_path: Path,
    split: str,
    outdir: Path,
    device: torch.device,
    batch_size: int,
    run_title: Optional[str] = None,
    learned_reward_mode: Literal["raw", "prob"] = "prob",
    learned_reward_max_comp: int = 512,
    also_save_as: Optional[Path] = None,
    query_source: np.ndarray | None = None,
) -> Optional[Dict[str, Any]]:
    """Ravens sort env: scatter learned reward vs EE (x,y), one subplot per GT annotator mode.

    Supports ``sort-easy-2goals-v0`` (2 modes) and ``sort-easy-4goals-v0`` (8 modes).
    """
    env_name = str(variant.get("env", ""))
    if not is_ravens_sort_env(env_name):
        return None
    if model is None or not model_supports_obs_latent_reward_decode(model):
        return None

    mode_labels = source_mode_labels_for_split(dataset_path=dataset_path, split=split)
    mode_names = source_mode_names_from_dataset(dataset_path=dataset_path)
    n_modes = resolve_n_annotator_modes(mode_labels=mode_labels, mode_names=mode_names)
    if n_modes <= 0:
        return None
    layout = sort_learned_reward_map_layout(env_name=env_name, n_modes=n_modes)

    comparison_pool: np.ndarray | None = None
    if learned_reward_mode == "prob":
        comparison_pool = build_comparison_obs_pool(
            obs1=obs1,
            obs2=obs2,
            max_comp=learned_reward_max_comp,
        )

    rewards = batched_learned_rewards_on_obs1(
        model=model,
        obs1=obs1,
        obs2=obs2,
        labels=labels,
        batch_size=batch_size,
        device=device,
        comparison_obs=comparison_pool,
        reward_mode=learned_reward_mode,
        source=query_source,
    )
    per_mode_obs, per_mode_r = gather_learned_rewards_by_annotator_mode(
        obs1,
        rewards,
        mode_labels,
        n_modes=n_modes,
    )
    if mode_names is not None:
        mode_titles = [f"mode: {name}" for name in mode_names]
    else:
        mode_titles = [
            f"mode: {source_mode_display_name(m, mode_names=None)}"
            for m in range(n_modes)
        ]

    fig = plot_observation_rewards_by_annotator_mode(
        per_mode_obs,
        per_mode_r,
        mode_titles=mode_titles,
        suptitle=format_sort_learned_reward_map_suptitle(
            run_title=run_title,
            env_name=env_name,
            n_modes=n_modes,
            layout=layout,
        ),
        scatter_s=layout.scatter_s,
        ncols=layout.ncols,
        x_idx=layout.x_idx,
        y_idx=layout.y_idx,
        axis_xlabel=layout.axis_xlabel,
        axis_ylabel=layout.axis_ylabel,
        invert_yaxis=layout.invert_yaxis,
        title_fontsize=layout.title_fontsize,
        reward_vmin=0.0 if learned_reward_mode == "prob" else None,
        reward_vmax=1.0 if learned_reward_mode == "prob" else None,
    )
    png_path = outdir / f"learned_reward_by_annot_mode_{learned_reward_mode}.png"
    savefig_kw = dict(dpi=150, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png_path, **savefig_kw)
    sweep_copy_path: Optional[Path] = None
    if also_save_as is not None:
        also_save_as = Path(also_save_as)
        also_save_as.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(also_save_as, **savefig_kw)
        sweep_copy_path = also_save_as
    plt.close(fig)
    metrics_out: Dict[str, Any] = {
        "learned_reward_by_annot_mode_png": str(png_path),
        "learned_reward_by_annot_mode_env": env_name,
        "learned_reward_by_annot_mode_n_modes": int(n_modes),
        "learned_reward_by_annot_mode_n_goals": layout.n_goals,
        "learned_reward_mode": learned_reward_mode,
        "learned_reward_max_comp": (
            int(comparison_pool.shape[0]) if comparison_pool is not None else None
        ),
    }
    if sweep_copy_path is not None:
        metrics_out["learned_reward_sweep_copy_png"] = str(sweep_copy_path)
    return metrics_out


def preference_dataloader_for_split(
    dataset_path: Path,
    split: str,
    set_length: int,
    observation_dim: int,
    action_dim: int,
    batch_size: int,
) -> DataLoader:
    """Preference DataLoader over one train/eval/all split (shuffle=False, matches eval order)."""
    with open(dataset_path, "rb") as fp:
        batch = pickle.load(fp)

    batch["observations"] = batch["observations"][..., :observation_dim]
    batch["observations_2"] = batch["observations_2"][..., :observation_dim]
    assert batch["actions"].shape[-1] == action_dim
    if set_length < 0:
        set_length = batch["observations"].shape[1]

    idx = split_indices(len(batch["observations"]), split)
    sub = {
        "observations": batch["observations"][idx, :set_length],
        "observations_2": batch["observations_2"][idx, :set_length],
        "labels": batch["labels"][idx, :set_length],
    }
    query_source = slice_query_source(batch=batch, idx=idx)
    if query_source is not None:
        sub["source_variable"] = query_source
    dataset = PreferenceDataset(
        sub,
        encode_sources=query_source is not None,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def encode_latents_dataloader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Encoder posterior means for each preference row, in DataLoader order (train.py-style batches)."""
    model.eval()
    means: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            o1 = batch["observations"].to(device).float()  # (B, A, T, O)
            o2 = batch["observations_2"].to(device).float()
            y = batch["labels"].to(device).float()
            enc_kw: dict[str, torch.Tensor] = {}
            if model_encode_sources(model=model):
                enc_kw["source"] = batch["source_variable"].to(device).long()
            mean = encode_preference_latent_mean(model, o1, o2, y, **enc_kw)
            means.append(mean.cpu().numpy())
    return np.concatenate(means, axis=0)  # (N, Z)


def xy_table_extents_from_dataset_observations(
    obs: np.ndarray,
    frame_index: int = -1,
    pad_rel: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Min/max of table plane using **object** horizontal position, ``O[3:5]`` (SortEnv: obj x, obj y).

    obs: (N, A, T, O) — N preference rows, A annotations, T time, O state; require ``O >= 5``.
    frame_index: T index; -1 = last frame.
    pad_rel: expand each axis span by this fraction (e.g. 0.02) so the grid does not clip.

    Returns (lo, hi) with shape (2,) each: ``[obj_x_lo, obj_y_lo]`` and ``[obj_x_hi, obj_y_hi]`` (world).
    """
    obs = np.asarray(obs)
    if obs.ndim != 4 or obs.shape[-1] < 5:
        raise ValueError(
            f"expected obs (N, A, T, O) with O>=5 for O[3:5] object (x,y); got shape {obs.shape!r}"
        )
    t = (
        int(frame_index)
        if int(frame_index) >= 0
        else int(obs.shape[2] + int(frame_index))
    )
    corners = obs[:, :, t, 3:5]  # (N, A, 2) — object (x, y) world (SortEnv)
    # corners = obs[:, :, t, 0:2]  # (N, A, 2) — (ee x, y) if using gripper
    flat = corners.reshape(-1, 2)
    lo = np.min(flat, axis=0)
    hi = np.max(flat, axis=0)
    span = hi - lo
    span = np.where(span < 1e-12, 1.0, span)
    if pad_rel > 0.0:
        lo = lo - float(pad_rel) * span
        hi = hi + float(pad_rel) * span
    return lo, hi


def kl_diag_gaussian_to_standard_normal(
    mean: np.ndarray, log_var: np.ndarray
) -> np.ndarray:
    """
    KL( N(mean, exp(log_var)) || N(0, I) ) for matching shapes, elementwise (Kingma & Welling).

    mean, log_var: same shape, last dim = latent — e.g. (Gy, Gx, L).
    Returns same shape, each entry is the KL for that scalar marginal vs standard normal.
    """
    mean = np.asarray(mean, dtype=np.float64)
    log_var = np.asarray(log_var, dtype=np.float64)
    return 0.5 * (np.exp(log_var) + mean**2.0 - 1.0 - log_var)


def encode_posterior_mu_logvar_unperturbed(
    model: torch.nn.Module,
    obs1: np.ndarray,
    obs2: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    source: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    q(z|s1,s2,y) for a single batch row: encoder mean and log_var (diagonal), no table sweep.

    obs1, obs2: (1, A, T, O) — (B, A, T, O) with B=1.
    labels: (1, A) or (1, A, 1) matching the dataset layout.

    Returns
    -------
    mu : (L,) float64 — posterior mean per dim.
    log_var : (L,) float64 — log sigma^2; Var_q(z_k) = exp(log_var[k]).
    """
    o1 = np.asarray(obs1, dtype=np.float32)
    o2 = np.asarray(obs2, dtype=np.float32)
    y0 = np.asarray(labels, dtype=np.float32)
    a_max = int(o1.shape[1])
    if o1.shape[0] != 1 or o2.shape[0] != 1:
        raise ValueError("expected batch 1 for template encode")
    if int(y0.shape[1]) != a_max:
        raise ValueError(f"labels second dim must match A={a_max}; got {y0.shape!r}")
    y_3d = y0.reshape(1, a_max, -1)
    t1 = torch.from_numpy(o1).float().to(device)
    t2 = torch.from_numpy(o2).float().to(device)
    yv = torch.from_numpy(y_3d).float().to(device)
    enc_kw = encode_source_kwargs(
        model=model,
        source=source,
        start=0,
        end=1,
        device=device,
    )
    with torch.no_grad():
        m, lv = model.encode(t1, t2, yv, **enc_kw)
    return m.cpu().numpy().reshape(-1), lv.cpu().numpy().reshape(-1)


def encode_latent_means_on_xy_table_grid(
    model: torch.nn.Module,
    base_obs1: np.ndarray,
    base_obs2: np.ndarray,
    base_labels: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    device: torch.device,
    *,
    annotation_index: int = 0,
    time_index: int = -1,
    move: Literal["obs1", "obs2", "both"] = "obs1",
    max_batch: int = 1024,
    base_source: np.ndarray | None = None,
) -> np.ndarray:
    """
    VAE encoder posterior *mean* on a 2D grid of **object** table positions (SortEnv ``O[3:5]``).

    Subplot layout: **horizontal axis = object world y (``O[4]``)** = left vs. right goal;
    **vertical axis = object world x (``O[3]``)**.

    Starts from a single preference example (batch 1), overwrites only ``O[3]``, ``O[4]`` in
    the chosen obs slot(s) at (annotation_index, time_index) with a Cartesian grid, and
    encodes: (Gy, Gx) points, latent_dim L => means ``(Gy, Gx, L)``.

    Shapes: base_obs{1,2} (1, A, T, O) with O>=5. ``x_vals`` (Gx,) = 1D grid of **object y** (plot
    columns / horiz.); ``y_vals`` (Gy,) = 1D grid of **object x** (plot rows / vert.).
    time_index: -1 = last T (default).

    Returns
    -------
    means : (Gy, Gx, L) — row = object x, col = object y; imshow: x-axis = object y, y-axis = object x.
    """
    o1 = np.asarray(base_obs1, dtype=np.float32)
    o2 = np.asarray(base_obs2, dtype=np.float32)
    y0 = np.asarray(base_labels, dtype=np.float32)
    gx = len(np.asarray(x_vals).reshape(-1))
    gy = len(np.asarray(y_vals).reshape(-1))
    b_total = int(gx * gy)
    if o1.shape[0] != 1 or o2.shape[0] != 1:
        raise ValueError("base_obs1/base_obs2 must have batch 1 for the template")
    a_max = o1.shape[1]
    t_len = o1.shape[2]
    o_dim = int(o1.shape[-1])
    if o_dim < 5:
        raise ValueError(
            f"encode_latent_means_on_xy_table_grid needs O>=5 for object O[3:5]; got O={o_dim}"
        )
    if not (0 <= int(annotation_index) < a_max):
        raise ValueError(
            f"annotation_index={annotation_index} out of range for A={a_max}"
        )
    t = int(time_index) if int(time_index) >= 0 else int(t_len + int(time_index))
    if not (0 <= t < t_len):
        raise ValueError(f"time_index resolved to t={t} out of range for T={t_len}")

    o1b = np.broadcast_to(o1, (b_total, *o1.shape[1:])).copy()
    o2b = np.broadcast_to(o2, (b_total, *o2.shape[1:])).copy()
    if int(y0.shape[1]) != a_max:
        raise ValueError(f"base_labels second dim must match A={a_max}; got {y0.shape!r}")
    y_3d = y0.reshape(1, a_max, -1)  # (1, A, 1) — label per annotation
    yb = np.broadcast_to(y_3d, (b_total, a_max, y_3d.shape[2])).copy()

    # x_i, y_i: (Gy, Gx) — x_i = object y (horizontal in plot), y_i = object x (vertical)
    y_i, x_i = np.meshgrid(
        np.asarray(y_vals).reshape(-1), np.asarray(x_vals).reshape(-1), indexing="ij"
    )
    flat_obj_y = x_i.ravel()  # (B,) = object y (G_obj_x * G_obj_y) — B grid points
    flat_obj_x = y_i.ravel()  # (B,) = object x
    if move in ("obs1", "both"):
        o1b[:, annotation_index, t, 3] = flat_obj_x
        o1b[:, annotation_index, t, 4] = flat_obj_y
        # o1b[:, annotation_index, t, 0] = flat_grip_x
        # o1b[:, annotation_index, t, 1] = flat_grip_y
    if move in ("obs2", "both"):
        o2b[:, annotation_index, t, 3] = flat_obj_x
        o2b[:, annotation_index, t, 4] = flat_obj_y
        # o2b[:, annotation_index, t, 0] = flat_grip_x
        # o2b[:, annotation_index, t, 1] = flat_grip_y

    means_b: List[np.ndarray] = []
    o1b_t = o1b
    o2b_t = o2b
    yb_t = yb
    for start in range(0, b_total, int(max_batch)):
        end = min(start + int(max_batch), b_total)
        t1 = torch.from_numpy(o1b_t[start:end]).float().to(device)
        t2 = torch.from_numpy(o2b_t[start:end]).float().to(device)
        yv = torch.from_numpy(yb_t[start:end]).float().to(device)
        enc_kw = encode_source_kwargs(
            model=model,
            source=base_source,
            start=0,
            end=1,
            device=device,
        )
        if enc_kw:
            u = enc_kw["source"]  # (1, n_sources)
            enc_kw["source"] = u.expand(end - start, -1)
        with torch.no_grad():
            m = encode_preference_latent_mean(model, t1, t2, yv, **enc_kw)
        means_b.append(m.cpu().numpy())
    m_all = np.concatenate(means_b, axis=0)  # (B, L) — (Gx*Gy, L)
    l_m = int(m_all.shape[1])
    return m_all.reshape(gy, gx, l_m)


def save_latent_xy_grid_heatmaps(
    means: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    outpath: Union[str, Path],
    *,
    template_mean: Optional[np.ndarray] = None,
    template_log_var: Optional[np.ndarray] = None,
    title: str = "Encoder mean vs object (x, y) on table at (annotation, time) grid",
) -> None:
    """
    means: (Gy, Gx, L) from :func:`encode_latent_means_on_xy_table_grid` — row = object x, col = object y.

    template_mean, template_log_var: optional (L,) from
    :func:`encode_posterior_mu_logvar_unperturbed` on the *same* base row (unperturbed);
    if both set, subplot titles add sigma^2_k = exp(log_var_k) and KL( q_k || N(0,1) )
    for that data point (no average over the grid).
    x_vals: (Gx,) = object **y** (world) — imshow **horizontal**; y_vals: (Gy,) = object **x** (world) — **vertical**.
    """
    means = np.asarray(means)
    if means.ndim != 3:
        raise ValueError(f"expected means (Gy, Gx, L), got {means.shape!r}")
    gy, gx, l_dim = int(means.shape[0]), int(means.shape[1]), int(means.shape[2])
    if (
        len(np.asarray(x_vals).reshape(-1)) != gx
        or len(np.asarray(y_vals).reshape(-1)) != gy
    ):
        raise ValueError("x_vals / y_vals length must match Gx, Gy in means")
    use_subtitles = template_mean is not None and template_log_var is not None
    if use_subtitles:
        tm = np.asarray(template_mean, dtype=np.float64).reshape(-1)
        tlv = np.asarray(template_log_var, dtype=np.float64).reshape(-1)
        if tm.shape[0] != l_dim or tlv.shape[0] != l_dim:
            raise ValueError(
                f"template_mean and template_log_var must have length L={l_dim}; "
                f"got {tm.shape!r} and {tlv.shape!r}"
            )
        kl_1d = kl_diag_gaussian_to_standard_normal(tm, tlv)  # (L,)
    ncols = int(min(4, max(1, l_dim)))
    nrows = (l_dim + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows + 0.3), squeeze=False
    )
    xv = np.asarray(x_vals, dtype=np.float64).reshape(-1)
    yv = np.asarray(y_vals, dtype=np.float64).reshape(-1)
    extent = (float(xv[0]), float(xv[-1]), float(yv[0]), float(yv[-1]))
    for k in range(l_dim):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        im = ax.imshow(
            means[:, :, k],
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("object y (O[4]) — left vs right goal")
        ax.set_ylabel("object x (O[3])")
        if use_subtitles:
            sigma2_k = float(np.exp(tlv[k]))
            kl_k = float(kl_1d[k])
            ax.set_title(
                f"$z_{{{k}}}$ (encoder mean)\n"
                rf"$\sigma^2_k$={sigma2_k:.4g}  $\mathrm{{KL}}_k$={kl_k:.4g} (at template)",
                fontsize=8,
            )
        else:
            ax.set_title(f"$z_{{{k}}}$ (encoder mean)")
    for k in range(l_dim, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def _overlay_nmi_active_latent_dots(
    ax: Axes,
    mat: np.ndarray,
    active_latents: np.ndarray,
    *,
    color: str = "red",
    size_pts2: float = 5.0,
) -> None:
    """One marker per (source, latent) cell: row i = source, col j = latent. mat (S, L). active (L,)."""
    S, L = int(mat.shape[0]), int(mat.shape[1])
    act = np.asarray(active_latents, dtype=bool).reshape(-1)
    if act.size != L or not act.any():
        return
    # ii, jj: (S, L) — row and column index in imshow; cell centers (jj, ii) in data coords, origin=upper
    ii, jj = np.meshgrid(
        np.arange(S, dtype=np.float64), np.arange(L, dtype=np.float64), indexing="ij"
    )
    m = act[jj.astype(np.intp)]
    ax.scatter(
        jj[m],
        ii[m],
        c=color,
        s=size_pts2,
        marker="o",
        zorder=5,
        linewidths=0,
    )


def _nmi_heatmap_vlim(mat: np.ndarray) -> Tuple[float, float]:
    """NMI heatmaps: floor at 0, ceiling at observed max in this matrix."""
    vmax = float(np.max(mat)) if mat.size else 0.0
    return 0.0, max(vmax, 1e-12)


def save_heatmap(
    mat: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    title: str,
    outpath: str,
    active_latents: Optional[np.ndarray] = None,
    *,
    title_fontsize: int = 8,
):
    plt.figure(figsize=(max(6, len(col_names) * 0.35), max(3, len(row_names) * 0.55)))
    ax = plt.gca()
    vmin, vmax = _nmi_heatmap_vlim(mat)
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    # im = plt.imshow(mat, aspect="auto")
    if active_latents is not None:
        _overlay_nmi_active_latent_dots(ax, mat, active_latents)
    plt.colorbar(im)
    plt.xticks(np.arange(len(col_names)), col_names, rotation=90)
    plt.yticks(np.arange(len(row_names)), row_names)
    plt.title(title, fontsize=title_fontsize)
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
    active_latents: Optional[np.ndarray] = None,
) -> Any:
    """Draw NMI heatmap on an existing axes (for sweep grid); mirrors save_heatmap styling."""
    vmin, vmax = _nmi_heatmap_vlim(mat)
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    if active_latents is not None:
        _overlay_nmi_active_latent_dots(ax, mat, active_latents)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=6)
    ax.set_title(title, fontsize=title_fontsize)
    return im


def resolve_discrete_latents(model: Any, variant: Dict[str, Any]) -> bool:
    """Prefer checkpoint attribute; fall back to variant.json for old checkpoints."""
    if hasattr(model, "discrete_latents"):
        return bool(model.discrete_latents)
    return bool(variant.get("discrete_latents", False))


def resolve_eval_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def prepare_infomec_eval_arrays(
    run_dir: Path,
    *,
    split: str,
    num_bins: int,
    batch_size: Optional[int],
    set_length: Optional[int],
    dataset_override: Optional[Path],
    device_str: str,
    pref_diagonal: bool = False,
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
    np.ndarray | None,
    torch.device,
]:
    """Load InfoMEC eval dataset + build observation tensors and source matrix."""
    run_dir = run_dir.expanduser().resolve()
    variant_path = run_dir / "variant.json"
    if not variant_path.is_file():
        raise FileNotFoundError(f"Missing variant.json under run_dir: {variant_path}")
    with open(variant_path, encoding="utf-8") as vf:
        meta = json.load(vf)
    variant = meta["variant"]
    bs = int(variant["batch_size"]) if batch_size is None else int(batch_size)
    sl = int(variant["set_size"]) if set_length is None else int(set_length)
    dataset_path = resolve_infomec_dataset_path(run_dir, variant, dataset_override)
    with open(dataset_path, "rb") as f:
        batch = pickle.load(f)
    n = len(batch["observations"])
    idx = split_indices(n, split)
    obs1 = maybe_truncate(np.asarray(batch["observations"])[idx], sl)
    obs2 = maybe_truncate(np.asarray(batch["observations_2"])[idx], sl)
    labels = maybe_truncate(np.asarray(batch["labels"])[idx], sl)
    device = resolve_eval_device(device_str)
    sk = infer_source_keys_from_batch(batch)
    dataset_var_names = source_variable_display_names_from_batch(batch)
    source_cols = []
    source_names: List[str] = []
    var_name_idx = 0
    for key in sk:
        cols, names = infer_source_columns(np.asarray(batch[key])[idx], key, num_bins)
        if key == "source_variable" and dataset_var_names is not None:
            ncols = int(cols.shape[1])
            if var_name_idx + ncols <= len(dataset_var_names):
                names = dataset_var_names[var_name_idx : var_name_idx + ncols]
                var_name_idx += ncols
        source_cols.append(cols)
        source_names.extend(names)
    sources = np.concatenate(source_cols, axis=1)
    sources, source_names = maybe_append_diagonality_source(
        sources=sources,
        source_names=source_names,
        pref_diagonal=pref_diagonal,
        env_name=str(variant.get("env", "")),
    )
    query_source = slice_query_source(batch=batch, idx=idx)
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
        query_source,
        device,
    )


def resolve_tsne_perplexity(n: int, perplexity: Optional[float] = None) -> float:
    if perplexity is None:
        perp = float(min(30.0, max(2.0, n // 3)))
    else:
        perp = float(perplexity)
    return min(perp, float(max(2, n - 1)))


def resolve_umap_n_neighbors(n: int, n_neighbors: Optional[int] = None) -> int:
    if n_neighbors is None:
        return max(2, min(15, n - 1))
    return max(2, min(int(n_neighbors), n - 1))


def projection_display_name(method: LatentClusterMethod) -> str:
    if method == LatentClusterMethod.umap:
        return "UMAP"
    return "t-SNE"


def projection_axis_labels(method: LatentClusterMethod) -> Tuple[str, str]:
    name = projection_display_name(method=method)
    return f"{name} 1", f"{name} 2"


def _tsne_iteration_kwarg_name() -> str:
    """``max_iter`` (sklearn >= 1.2) or legacy ``n_iter``."""
    params = inspect.signature(TSNE.__init__).parameters
    if "max_iter" in params:
        return "max_iter"
    if "n_iter" in params:
        return "n_iter"
    raise RuntimeError("TSNE has neither max_iter nor n_iter")


_TSNE_ITER_KWARG = _tsne_iteration_kwarg_name()


def fit_tsne_projection(
    latents: np.ndarray,
    *,
    perplexity: Optional[float] = None,
    max_iter: Optional[int] = None,
    random_state: int = 0,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Return (proj, params) with shape (N, 2), or None if t-SNE is not feasible."""
    if latents.ndim != 2 or latents.shape[0] < 3:
        return None
    n = int(latents.shape[0])
    perp = resolve_tsne_perplexity(n=n, perplexity=perplexity)
    max_iter_used = 1000 if max_iter is None else int(max_iter)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        **{_TSNE_ITER_KWARG: max_iter_used},
    )
    proj = tsne.fit_transform(latents)  # (N, 2)
    return proj, {"method": "tsne", "perplexity": perp, "max_iter": max_iter_used}


def fit_umap_projection(
    latents: np.ndarray,
    *,
    n_neighbors: Optional[int] = None,
    min_dist: Optional[float] = None,
    random_state: int = 0,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Return (proj, params) with shape (N, 2), or None if UMAP is not feasible."""
    if latents.ndim != 2 or latents.shape[0] < 3:
        return None
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "Install umap-learn for --latent-cluster-method umap "
            "(e.g. pip install umap-learn)"
        ) from e
    n = int(latents.shape[0])
    n_neighbors_used = resolve_umap_n_neighbors(n=n, n_neighbors=n_neighbors)
    min_dist_used = 0.1 if min_dist is None else float(min_dist)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_used,
        min_dist=min_dist_used,
        random_state=random_state,
        n_jobs=1,
    )
    proj = reducer.fit_transform(latents)  # (N, 2)
    return proj, {
        "method": "umap",
        "n_neighbors": n_neighbors_used,
        "min_dist": min_dist_used,
    }


def fit_latent_projection(
    latents: np.ndarray,
    method: LatentClusterMethod,
    *,
    tsne_perplexity: Optional[float] = None,
    tsne_max_iter: Optional[int] = None,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    random_state: int = 0,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    if method == LatentClusterMethod.umap:
        return fit_umap_projection(
            latents,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
    return fit_tsne_projection(
        latents,
        perplexity=tsne_perplexity,
        max_iter=tsne_max_iter,
        random_state=random_state,
    )


def latent_projection_metrics_from_info(
    *,
    projection_info: Optional[Dict[str, Any]],
    method: LatentClusterMethod,
    png_path: Path,
    mode_names: Optional[List[str]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "latent_cluster_method": method.value,
        "latent_projection_png": str(png_path) if projection_info is not None else None,
        "latent_projection_colored_by": "source_mode",
        "latent_projection_source_mode_names": mode_names,
    }
    if projection_info is None:
        out.update(
            {
                "tsne_perplexity_used": None,
                "tsne_max_iter_used": None,
                "tsne_png": None,
                "umap_n_neighbors_used": None,
                "umap_min_dist_used": None,
            }
        )
        return out
    out.update(
        {
            "latent_projection_n_gt_modes_seen": projection_info["n_gt_modes_seen"],
            "latent_projection_gt_modes_seen": projection_info["gt_modes_seen"],
            "latent_projection_gt_mode_names": projection_info["gt_mode_names"],
            "latent_projection_gt_mode_counts": projection_info["gt_mode_counts"],
        }
    )
    if method == LatentClusterMethod.tsne:
        out.update(
            {
                "tsne_perplexity_used": projection_info.get("perplexity"),
                "tsne_max_iter_used": projection_info.get("max_iter"),
                "tsne_png": str(png_path),
                "umap_n_neighbors_used": None,
                "umap_min_dist_used": None,
            }
        )
    else:
        out.update(
            {
                "tsne_perplexity_used": None,
                "tsne_max_iter_used": None,
                "tsne_png": None,
                "umap_n_neighbors_used": projection_info.get("n_neighbors"),
                "umap_min_dist_used": projection_info.get("min_dist"),
            }
        )
    return out


def projection_mode_legend_ncol(n_modes: int) -> int:
    return min(8, max(1, n_modes))


def projection_mode_display_names(
    modes: List[int], mode_names: Optional[List[str]]
) -> Dict[int, str]:
    return {
        int(m): source_mode_display_name(int(m), mode_names=mode_names) for m in modes
    }


def plot_latent_projection_scatter_on_ax(
    ax: Axes,
    proj: np.ndarray,
    y: np.ndarray,
    *,
    method: LatentClusterMethod,
    mode_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Scatter 2D latent embedding on ``ax``, colored by integer ``source_mode`` labels."""
    y_int = np.asarray(y).astype(int).reshape(-1)
    uniq_modes, mode_counts = np.unique(y_int, return_counts=True)
    gt_modes_seen = [int(m) for m in uniq_modes]
    gt_mode_names = projection_mode_display_names(gt_modes_seen, mode_names=mode_names)
    gt_mode_counts = {int(m): int(c) for m, c in zip(uniq_modes, mode_counts)}

    cmap = plt.get_cmap("tab10")
    for mode in sorted(gt_modes_seen):
        mask = y_int == mode
        color = cmap(mode % 10)
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            c=[color],
            s=LATENT_PROJECTION_SCATTER_S,
            alpha=LATENT_PROJECTION_SCATTER_ALPHA,
            linewidths=0,
            edgecolors="none",
        )
    xlabel, ylabel = projection_axis_labels(method=method)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    return {
        "gt_modes_seen": gt_modes_seen,
        "gt_mode_names": gt_mode_names,
        "gt_mode_counts": gt_mode_counts,
        "n_gt_modes_seen": int(len(gt_modes_seen)),
        "n_samples": int(len(y_int)),
    }


def add_projection_mode_legend_below(
    fig: plt.Figure,
    modes: List[int],
    *,
    mode_names: Optional[List[str]] = None,
    title: str = "source mode",
    fontsize: int = 8,
    bbox_y: float = 0.0,
) -> None:
    """Shared figure legend matching individual latent projection plots."""
    cmap = plt.get_cmap("tab10")
    display = projection_mode_display_names(modes, mode_names=mode_names)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4,
            markerfacecolor=cmap(int(mode) % 10),
            markeredgecolor=cmap(int(mode) % 10),
            label=display[int(mode)],
        )
        for mode in sorted(int(m) for m in modes)
    ]
    n_modes = len(handles)
    ncol = projection_mode_legend_ncol(n_modes)
    fig.legend(
        handles=handles,
        title=title,
        fontsize=fontsize,
        loc="lower center",
        bbox_to_anchor=(0.5, bbox_y),
        ncol=ncol,
        frameon=True,
    )


def save_aggregate_latent_projection_sweep_grid(
    runs: List[Dict[str, Any]],
    outpath: Path,
    mode_color: np.ndarray,
    *,
    method: LatentClusterMethod,
    mode_names: Optional[List[str]] = None,
    tsne_perplexity: Optional[float] = None,
    tsne_max_iter: Optional[int] = None,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    title_fontsize: int = 7,
    random_state: int = 0,
    ckpt_label: str = "best_model.pt",
) -> None:
    """One latent embedding subplot per sweep run; shared source-mode legend below the grid."""
    n = len(runs)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    col_w_in, row_h_in = 3.4, 3.2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(col_w_in * ncols, row_h_in * nrows),
        squeeze=False,
    )

    all_modes_seen: set[int] = set()
    last_i = 0
    display = projection_display_name(method=method)
    for i, run in enumerate(runs):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        latents = np.asarray(run["latents"])
        fit = fit_latent_projection(
            latents,
            method=method,
            tsne_perplexity=tsne_perplexity,
            tsne_max_iter=tsne_max_iter,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            random_state=random_state,
        )
        if fit is None:
            ax.set_title(
                run.get("run_title", run.get("name", "")), fontsize=title_fontsize
            )
            ax.text(
                0.5,
                0.5,
                f"{display} skipped\n(N < 3)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            proj, _params = fit
            info = plot_latent_projection_scatter_on_ax(
                ax,
                proj,
                mode_color,
                method=method,
                mode_names=mode_names,
            )
            all_modes_seen.update(int(m) for m in info["gt_modes_seen"])
            ax.set_title(
                str(run.get("run_title", run.get("name", ""))),
                fontsize=title_fontsize,
            )
        last_i = i

    for j in range(last_i + 1, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    legend_modes = (
        list(range(len(mode_names))) if mode_names is not None else sorted(all_modes_seen)
    )
    n_legend = len(legend_modes)
    ncol_leg = projection_mode_legend_ncol(n_legend)
    nrow_leg = (n_legend + ncol_leg - 1) // ncol_leg if n_legend else 1
    add_projection_mode_legend_below(
        fig,
        legend_modes,
        mode_names=mode_names,
        bbox_y=0.0,
    )
    fig.suptitle(f"{display} by source mode ({ckpt_label})", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.02 + 0.05 * nrow_leg, 1, 0.99])
    fig.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def maybe_save_latent_projection(
    latents: np.ndarray,
    y: np.ndarray,
    outpath: str,
    title: str,
    *,
    method: LatentClusterMethod,
    tsne_perplexity: Optional[float] = None,
    tsne_max_iter: Optional[int] = None,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    random_state: int = 0,
    mode_names: Optional[List[str]] = None,
    title_fontsize: int = 8,
) -> Optional[Dict[str, Any]]:
    """2D latent embedding; one point per preference row, colored by ``y`` (GT ``source_mode``)."""
    if not latent_cluster_enabled(method=method):
        return None
    fit = fit_latent_projection(
        latents,
        method=method,
        tsne_perplexity=tsne_perplexity,
        tsne_max_iter=tsne_max_iter,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        random_state=random_state,
    )
    if fit is None:
        return None
    proj, params = fit

    fig, ax = plt.subplots(figsize=(6, 5))
    info = plot_latent_projection_scatter_on_ax(
        ax, proj, y, method=method, mode_names=mode_names
    )
    ax.set_title(title, fontsize=title_fontsize)
    n_gt_modes_seen = int(info["n_gt_modes_seen"])
    ncol = projection_mode_legend_ncol(n_gt_modes_seen)
    nrow = (n_gt_modes_seen + ncol - 1) // ncol
    add_projection_mode_legend_below(
        fig,
        info["gt_modes_seen"],
        mode_names=mode_names,
        bbox_y=-0.14 - 0.05 * (nrow - 1),
    )
    fig.tight_layout(rect=[0, 0.02 + 0.05 * nrow, 1, 1])
    fig.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return {**params, **info}


def maybe_save_tsne(
    latents: np.ndarray,
    y: np.ndarray,
    outpath: str,
    title: str,
    *,
    perplexity: Optional[float] = None,
    max_iter: Optional[int] = None,
    random_state: int = 0,
    enabled: bool = True,
    mode_names: Optional[List[str]] = None,
    title_fontsize: int = 8,
) -> Optional[Dict[str, Any]]:
    """Backward-compatible t-SNE wrapper around :func:`maybe_save_latent_projection`."""
    return maybe_save_latent_projection(
        latents,
        y,
        outpath,
        title,
        method=LatentClusterMethod.tsne,
        tsne_perplexity=perplexity,
        tsne_max_iter=max_iter,
        random_state=random_state,
        enabled=enabled,
        mode_names=mode_names,
        title_fontsize=title_fontsize,
    )


def save_aggregate_tsne_sweep_grid(
    runs: List[Dict[str, Any]],
    outpath: Path,
    mode_color: np.ndarray,
    *,
    mode_names: Optional[List[str]] = None,
    perplexity: Optional[float] = None,
    max_iter: Optional[int] = None,
    title_fontsize: int = 7,
    random_state: int = 0,
) -> None:
    """Backward-compatible t-SNE wrapper around :func:`save_aggregate_latent_projection_sweep_grid`."""
    save_aggregate_latent_projection_sweep_grid(
        runs,
        outpath,
        mode_color,
        method=LatentClusterMethod.tsne,
        mode_names=mode_names,
        tsne_perplexity=perplexity,
        tsne_max_iter=max_iter,
        title_fontsize=title_fontsize,
        random_state=random_state,
    )


def main(args: AnalysisCLI) -> None:
    input_dir = args.input_dir.expanduser().resolve()
    variant_path = input_dir / "variant.json"
    ckpt_path, ckpt_label, selected_epoch, model_selection_tag = resolve_run_checkpoint(
        input_dir,
        model_selection=args.model_selection,
        ckpt_name=args.ckpt_name,
    )

    if not variant_path.is_file():
        raise FileNotFoundError(f"Missing variant.json under --input-dir: {variant_path}")

    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = eval_run_outdir(input_dir=input_dir, eval_timestamp=eval_timestamp)
    outdir.mkdir(parents=True, exist_ok=True)

    (
        obs1,  # (N, A, T, O)
        obs2,
        labels,  # (N, A, T)
        sources,  # (N, S), where S is num source variables
        source_names,
        source_keys,
        batch_size,
        variant,
        dataset_path,
        query_source,
        device,
    ) = prepare_infomec_eval_arrays(
        input_dir,
        split=args.split,
        num_bins=args.num_bins,
        batch_size=args.batch_size,
        set_length=args.set_length,
        dataset_override=args.dataset,
        device_str=args.device,
        pref_diagonal=args.pref_diagonal,
    )
    print(
        f"Loaded eval split={args.split!r} pref_dataset="
        f"{variant['pref_dataset']!r} from {dataset_path} ({len(obs1)} queries)"
        f"(device={device})"
    )

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint under --input-dir: {ckpt_path}")

    model = torch.load(ckpt_path, map_location=device, weights_only=False).to(device)
    model.eval()
    discrete_latents = resolve_discrete_latents(model=model, variant=variant)

    epoch_payload = collect_epoch_checkpoint_infomec_series(
        input_dir,
        obs1,
        obs2,
        labels,
        sources,
        batch_size,
        device,
        discrete_latents,
        source=query_source,
    )

    latent_xy_table_grid_png: Optional[Path] = None
    env_name = str(variant.get("env", ""))
    if (
        args.latent_xy_table_grid
        and env_name == LatentTableMapEnvId.sort_easy_v0
        and len(obs1) > 0
    ):
        lo, hi = xy_table_extents_from_dataset_observations(
            obs1, frame_index=-1, pad_rel=float(args.xy_grid_pad_rel)
        )
        g = int(args.xy_grid_resolution)
        # lo[0:2], hi[0:2] = (obj_x, obj_y) world; horiz. subplot = object y, vert. = object x
        obj_x_grid = np.linspace(float(lo[0]), float(hi[0]), g)  # (G,) = O[3]
        obj_y_grid = np.linspace(float(lo[1]), float(hi[1]), g)  # (G,) = O[4] — goal axis
        tpl_mu, tpl_logvar = encode_posterior_mu_logvar_unperturbed(
            model,
            obs1[0:1],
            obs2[0:1],
            labels[0:1],
            device,
            source=(None if query_source is None else query_source[0:1]),
        )  # (L,), (L,)
        means = encode_latent_means_on_xy_table_grid(
            model,
            obs1[0:1],
            obs2[0:1],
            labels[0:1],
            obj_y_grid,  # x_vals in encoder = Gx, object y (columns)
            obj_x_grid,  # y_vals = Gy, object x (rows)
            device,
            annotation_index=0,
            time_index=-1,
            move="obs1",
            base_source=(None if query_source is None else query_source[0:1]),
        )  # (G, G, L)
        latent_xy_table_grid_png = outdir / "latent_mean_vs_xy_table_grid.png"
        save_latent_xy_grid_heatmaps(
            means,
            obj_y_grid,
            obj_x_grid,
            latent_xy_table_grid_png,
            template_mean=tpl_mu,
            template_log_var=tpl_logvar,
            title=f"{LatentTableMapEnvId.sort_easy_v0.value}: encoder mean vs object (x,y) on table — template=first {args.split!r} row",
        )

    set_length_used = (
        int(variant["set_size"]) if args.set_length is None else int(args.set_length)
    )
    with open(dataset_path, "rb") as _fp:
        _raw_batch = pickle.load(_fp)
    action_dim = int(_raw_batch["actions"].shape[-1])
    observation_dim = int(obs1.shape[-1])
    pref_loader = preference_dataloader_for_split(
        dataset_path=dataset_path,
        split=args.split,
        set_length=set_length_used,
        observation_dim=observation_dim,
        action_dim=action_dim,
        batch_size=batch_size,
    )
    latents = encode_latents_dataloader(model=model, loader=pref_loader, device=device)
    assert latents.shape[0] == len(obs1), (
        f"DataLoader latents N={latents.shape[0]} != split obs1 N={len(obs1)}"
    )

    result = compute_infomec(
        sources=sources, latents=latents, discrete_latents=discrete_latents
    )
    n_varies, n_inform, n_active_latents, latent_dim_counts = (
        latent_mask_counts_from_result(result)
    )
    assert latent_dim_counts == int(latents.shape[1])
    run_title = format_infomec_run_title(
        infom=float(result.infom),
        infoe=float(result.infoe),
        infoc=float(result.infoc),
        n_varies=n_varies,
        n_inform=n_inform,
        n_active=n_active_latents,
        latent_dim=int(latents.shape[1]),
        run_label=infomec_run_label(input_dir, variant),
        ckpt_name=ckpt_label,
        best_epoch=selected_epoch,
        best_selection=model_selection_tag,
    )

    metrics = save_eval_artifacts_after_encode(
        outdir,
        input_dir,
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
        query_source=query_source,
        pref_diagonal=args.pref_diagonal,
    )
    if epoch_payload is not None:
        save_infomec_epoch_series_artifacts(
            epoch_payload,
            outdir,
            title=run_title,
            best_epoch=selected_epoch,
            best_selection=model_selection_tag,
        )
    if latent_xy_table_grid_png is not None:
        metrics = {
            **metrics,
            "latent_xy_table_grid_png": str(latent_xy_table_grid_png),
            "latent_xy_table_grid_env": env_name,
        }
    print(
        f"Wrote {outdir}  \n"
        f"InfoM={metrics['infom']:.3f} InfoE={metrics['infoe']:.3f} "
        f"InfoC={metrics['infoc']:.3f}"
    )


if __name__ == "__main__":
    main(tyro.cli(AnalysisCLI))
