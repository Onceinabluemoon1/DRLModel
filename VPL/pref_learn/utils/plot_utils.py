import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pref_learn.models.utils import get_all_posterior, get_biased
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_reward(
    ax,
    obs,
    r,
    no_norm=False,
    *,
    x_idx: int = 0,
    y_idx: int = 1,
    reward_size_threshold: float | None = None,
    s_below: float = 6.0,
    s_above: float = 32.0,
    scatter_s: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | matplotlib.colors.Colormap | None = None,
    colorbar: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    invert_yaxis: bool = False,
):
    # if not no_norm:
    #     r = (r - r.min()) / (r.max() - r.min())
    #     r = np.exp(r / 0.1)
    #     r = (r - r.min()) / (r.max() - r.min())
    r_plot = np.asarray(r).reshape(-1)
    kwargs = {}
    if scatter_s is not None:
        kwargs["s"] = scatter_s
    elif reward_size_threshold is not None:
        kwargs["s"] = np.where(r_plot < reward_size_threshold, s_below, s_above)
    lim_kw = {}
    if vmin is not None:
        lim_kw["vmin"] = vmin
    if vmax is not None:
        lim_kw["vmax"] = vmax
    if cmap is not None:
        lim_kw["cmap"] = cmap
    sc = ax.scatter(obs[:, x_idx], obs[:, y_idx], c=r_plot, **lim_kw, **kwargs)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if invert_yaxis:
        ax.invert_yaxis()
    if colorbar:
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("r(s)")
    return sc


def observation_xy_limits(
    obs: np.ndarray,
    *,
    x_idx: int,
    y_idx: int,
    margin_frac: float = 0.02,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits for scatter planes; ``margin_frac`` is relative padding per span."""
    ob = np.asarray(obs, dtype=np.float64)
    if ob.size == 0:
        return (0.0, 1.0), (0.0, 1.0)
    x = ob[:, int(x_idx)]
    y = ob[:, int(y_idx)]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_pad = margin_frac * max(x_max - x_min, 1e-6)
    y_pad = margin_frac * max(y_max - y_min, 1e-6)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def latent_index_latex(head_idx: int) -> str:
    """Matplotlib-safe latent index label, e.g. ``$z_{13}$`` (not ``$z_13$``)."""
    return rf"$z_{{{int(head_idx)}}}$"


def _draw_per_head_score_bars(
    ax,
    *,
    score_per_head: np.ndarray,
    ylabel: str,
    title: str,
    title_fontsize: int,
    head_order: list[int] | None = None,
) -> None:
    score = np.asarray(score_per_head, dtype=np.float64).reshape(-1)
    order = list(range(score.shape[0])) if head_order is None else list(head_order)
    order_arr = np.asarray(order, dtype=np.int64)
    score_ord = score[order_arr]
    x = np.arange(len(order))
    ax.bar(x, score_ord, color="steelblue", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([latent_index_latex(j) for j in order], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)


def _draw_per_head_kl_bars(
    ax,
    *,
    kl_per_head: np.ndarray,
    title_fontsize: int,
    head_order: list[int] | None = None,
) -> None:
    _draw_per_head_score_bars(
        ax,
        score_per_head=kl_per_head,
        ylabel=r"$\mathrm{KL}$",
        title=r"per-head $\mathrm{KL}(q\|p)$",
        title_fontsize=title_fontsize,
        head_order=head_order,
    )
    # if kl_per_head_strata is not None:
    #     kl_strata = np.asarray(kl_per_head_strata, dtype=np.float64).reshape(-1)
    #     ...


# def _draw_per_head_std_bars(
#     ax,
#     *,
#     prior_std: np.ndarray,
#     post_std: np.ndarray,
#     title_fontsize: int,
#     head_order: list[int] | None = None,
# ) -> None:
#     prior = np.asarray(prior_std, dtype=np.float64).reshape(-1)
#     post = np.asarray(post_std, dtype=np.float64).reshape(-1)
#     assert prior.shape == post.shape
#     order = list(range(prior.shape[0])) if head_order is None else list(head_order)
#     order_arr = np.asarray(order, dtype=np.int64)
#     prior_ord = prior[order_arr]
#     post_ord = post[order_arr]
#     x = np.arange(len(order))
#     w = 0.35
#     ax.bar(x - w / 2, prior_ord, width=w, label="prior", color="coral")
#     ax.bar(x + w / 2, post_ord, width=w, label="posterior", color="seagreen")
#     ax.set_xticks(x)
#     ax.set_xticklabels([latent_index_latex(j) for j in order], fontsize=8)
#     ax.set_ylabel(r"$\sigma$", fontsize=8)
#     ax.set_title(r"per-head $\sigma$ (prior | posterior)", fontsize=title_fontsize)
#     ymax = float(max(float(np.max(prior)), float(np.max(post)), 1e-6)) * 1.15
#     ax.set_ylim(0.0, ymax)
#     ax.tick_params(axis="y", labelsize=8)
#     ax.legend(fontsize=7, loc="best")
#     ax.grid(axis="y", alpha=0.25)


def _draw_per_head_mean_std(
    ax,
    *,
    post_mean: np.ndarray,
    post_mean_std: np.ndarray,
    title_fontsize: int,
    head_order: list[int] | None = None,
) -> None:
    mu = np.asarray(post_mean, dtype=np.float64).reshape(-1)
    mu_std = np.asarray(post_mean_std, dtype=np.float64).reshape(-1)
    assert mu.shape == mu_std.shape
    order = list(range(mu.shape[0])) if head_order is None else list(head_order)
    order_arr = np.asarray(order, dtype=np.int64)
    mu_ord = mu[order_arr]
    mu_std_ord = mu_std[order_arr]
    x = np.arange(len(order))
    ax.errorbar(
        x,
        mu_ord,
        yerr=mu_std_ord,
        fmt="o",
        color="seagreen",
        ecolor="seagreen",
        elinewidth=1.2,
        capsize=3,
        markersize=5,
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([latent_index_latex(j) for j in order], fontsize=8)
    ax.set_ylabel(r"$\bar{\mu} \pm \mathrm{std}(\mu)$", fontsize=8)
    ax.set_title(r"per-head $\bar{\mu} \pm \mathrm{std}_q(\mu)$", fontsize=title_fontsize)
    y_abs = float(np.max(np.abs(mu_ord) + mu_std_ord)) if mu_ord.size > 0 else 1.0
    y_pad = max(y_abs * 0.15, 1e-3)
    ax.set_ylim(-y_abs - y_pad, y_abs + y_pad)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)


# def _draw_per_head_post_mean_std_across_strata_bars(
#     ax,
#     *,
#     post_mean_std_across_strata: np.ndarray,
#     title_fontsize: int,
#     head_order: list[int] | None = None,
# ) -> None:
#     sig = np.asarray(post_mean_std_across_strata, dtype=np.float64).reshape(-1)
#     order = list(range(sig.shape[0])) if head_order is None else list(head_order)
#     order_arr = np.asarray(order, dtype=np.int64)
#     sig_ord = sig[order_arr]
#     x = np.arange(len(order))
#     ax.bar(x, sig_ord, color="seagreen", width=0.7)
#     ax.set_xticks(x)
#     ax.set_xticklabels([latent_index_latex(j) for j in order], fontsize=8)
#     ax.set_ylabel(r"$\mathrm{std}_{\mathrm{strata}}(\bar{\mu})$", fontsize=8)
#     ax.set_title(
#         r"per-head $\mathrm{std}_{\mathrm{strata}}(\bar{\mu})$",
#         fontsize=title_fontsize,
#     )
#     ymax = float(np.nanmax(sig_ord)) if sig_ord.size > 0 else 1.0
#     ax.set_ylim(0.0, max(ymax * 1.15, 1e-6))
#     ax.tick_params(axis="y", labelsize=8)
#     ax.grid(axis="y", alpha=0.25)


def per_head_l_nuisance(
    *,
    post_mean: np.ndarray,
    post_mean_std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    r"""Per-head ``(\bar{\mu}_j)^2 / (\mathrm{Var}(\mu_j) + \varepsilon)``; ``(Z,)``."""
    mu = np.asarray(post_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(post_mean_std, dtype=np.float64).reshape(-1)
    assert mu.shape == std.shape
    var = std * std
    return (mu * mu) / (var + float(eps))


def _draw_per_head_l_nuisance_bars(
    ax,
    *,
    l_nuisance_per_head: np.ndarray,
    title_fontsize: int,
    head_order: list[int] | None = None,
) -> None:
    ln = np.asarray(l_nuisance_per_head, dtype=np.float64).reshape(-1)
    order = list(range(ln.shape[0])) if head_order is None else list(head_order)
    order_arr = np.asarray(order, dtype=np.int64)
    ln_ord = ln[order_arr]
    x = np.arange(len(order))
    ax.bar(x, ln_ord, color="darkorange", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([latent_index_latex(j) for j in order], fontsize=8)
    ax.set_ylabel(r"$L_{\mathrm{nuis},j}$", fontsize=8)
    ax.set_title(
        r"per-head $L_{\mathrm{nuis},j}=\bar{\mu}_j^2/(\mathrm{Var}(\mu_j)+\varepsilon)$",
        fontsize=title_fontsize,
    )
    ymax = float(np.nanmax(ln_ord)) if ln_ord.size > 0 else 1.0
    ax.set_ylim(0.0, max(ymax * 1.15, 1.0))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)


def _draw_per_head_theta_bars(
    ax,
    *,
    theta_mean_per_head: np.ndarray,
    title_fontsize: int,
    head_order: list[int] | None = None,
) -> None:
    theta = np.asarray(theta_mean_per_head, dtype=np.float64).reshape(-1)
    order = list(range(theta.shape[0])) if head_order is None else list(head_order)
    order_arr = np.asarray(order, dtype=np.int64)
    theta_ord = theta[order_arr]
    x = np.arange(len(order))
    ax.bar(x, theta_ord, color="mediumpurple", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([latent_index_latex(j) for j in order], fontsize=8)
    ax.set_ylabel(r"$\mathbb{E}[\theta]$", fontsize=8)
    ax.set_title(r"per-head mean $\theta(s)$", fontsize=title_fontsize)
    ymax = float(np.nanmax(theta_ord)) if theta_ord.size > 0 else 1.0
    ax.set_ylim(0.0, max(ymax * 1.15, 1e-6))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)


def plot_observation_rewards_by_annotator_mode(
    per_mode_obs: list,
    per_mode_r: list,
    *,
    mode_titles: list[str],
    suptitle: str | None = None,
    cmap: str = "viridis",
    reward_size_threshold: float = 0.5,
    scatter_s: float | None = None,
    s_below: float = 6.0,
    s_above: float = 32.0,
    reward_vmin: float | None = None,
    reward_vmax: float | None = None,
    ncols: int | None = None,
    x_idx: int = 1,
    y_idx: int = 0,
    axis_xlabel: str = "EE y",
    axis_ylabel: str = "EE x",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    invert_yaxis: bool = True,
    title_fontsize: int = 10,
    highlight_panel_indices: list[int] | None = None,
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
    elbo_head_order: list[int] | None = None,
    colorbar_label: str = "r(s)",
):
    """
    One scatter panel per annotator mode; each uses only (obs, r) from that mode.

    Args:
        per_mode_obs: length M; each entry (N_m, O) or empty (0, O).
        per_mode_r: length M; each (N_m,) or (N_m, 1) aligned with ``per_mode_obs``.
        mode_titles: length M subplot titles (e.g. ANNOTATOR_SPECS ``name``).
        elbo_kl_per_head: optional ``(Z,)`` global head-order score (VAE: KL;
            QLAE: ``std_q(z_q)``). Bar heights + scatter panel order.
        head_order_score_ylabel / head_order_score_title: bar panel labels when
            not VAE KL (defaults: KL ylabel/title).
        head_order_score_panel_prefix: scatter subtitle prefix, e.g. ``"KL"`` or
            ``"σ_q"`` → ``$z_j$ (σ_q=…)``.
        qlae_bin_entropy_per_head: optional ``(Z,)`` per-head FSQ bin entropy
            (QLAE code-utilization panel after the head-order score bar).
        elbo_post_mean / elbo_post_mean_std: optional ``(Z,)`` encoder-μ mean and
            cross-query std of μ over the panel's queries. elbo_l_nuisance_per_head:
            optional ``(Z,)`` aggregate-only nuisance score from cross-modal pooled μ
            stats; when set, adds an ``L_nuisance`` bar panel between μ and θ.
            theta_mean_per_head: optional ``(Z,)`` mean ``softplus(w(h))`` (multi only).
            When set with ``elbo_kl_per_head``, adds KL, μ, optional ``L_nuisance``, and θ
            bar panels after the aggregate (highlight) panel when that row has enough free
            cells; else they start a new row.
        elbo_head_order: optional head indices (descending global KL) for bar-panel x order.
    """
    n_modes = len(per_mode_obs)
    assert len(per_mode_r) == n_modes == len(mode_titles), (
        f"len mismatches: obs {n_modes}, r {len(per_mode_r)}, titles {len(mode_titles)}"
    )
    all_r = np.concatenate(
        [np.asarray(per_mode_r[m]).reshape(-1) for m in range(n_modes) if np.size(per_mode_r[m]) > 0],
        axis=0,
    )
    if all_r.size > 0:
        if reward_vmin is not None and reward_vmax is not None:
            vmin = float(reward_vmin)
            vmax = float(reward_vmax)
        else:
            vmin = float(np.min(all_r))
            vmax = float(np.max(all_r))
    else:
        vmin, vmax = 0.0, 1.0

    ncols = min(4, n_modes) if ncols is None else min(int(ncols), max(n_modes, 1))
    map_nrows = int(np.ceil(n_modes / ncols))
    show_head_score = elbo_kl_per_head is not None
    show_theta = theta_mean_per_head is not None
    has_entropy_panel = qlae_bin_entropy_per_head is not None
    has_mu_panel = (
        elbo_post_mean is not None
        and elbo_post_mean_std is not None
    )
    has_l_nuisance_panel = elbo_l_nuisance_per_head is not None
    score_slot: int | None = None
    entropy_slot: int | None = None
    mean_std_slot: int | None = None
    l_nuisance_slot: int | None = None
    theta_slot: int | None = None
    if show_head_score:
        panel_kinds: list[str] = ["score"]
        if has_entropy_panel:
            panel_kinds.append("entropy")
        if has_mu_panel:
            panel_kinds.append("mu")
        if has_l_nuisance_panel:
            panel_kinds.append("l_nuisance")
        if show_theta:
            panel_kinds.append("theta")
        n_diag_panels = len(panel_kinds)
        #* glue score / optional entropy|μ / θ after aggregate when row has room
        agg_idx = (
            int(highlight_panel_indices[0])
            if highlight_panel_indices
            else None
        )
        free_after_agg = (
            ncols - (agg_idx % ncols) - 1 if agg_idx is not None else 0
        )
        if agg_idx is not None and free_after_agg >= n_diag_panels:
            nrows = map_nrows
            first_slot = agg_idx + 1
        else:
            nrows = map_nrows + 1
            first_slot = map_nrows * ncols
        #* diagnostic panels may span multiple cols (e.g. nz=2 → 4 panels, ncols=3)
        nrows = max(
            nrows,
            int(np.ceil((first_slot + n_diag_panels) / ncols)),
        )
        for i, kind in enumerate(panel_kinds):
            slot = first_slot + i
            if kind == "score":
                score_slot = slot
            elif kind == "entropy":
                entropy_slot = slot
            elif kind == "mu":
                mean_std_slot = slot
            elif kind == "l_nuisance":
                l_nuisance_slot = slot
            elif kind == "theta":
                theta_slot = slot
    else:
        nrows = map_nrows
    n_suptitle_lines = (suptitle.count("\n") + 1) if suptitle else 0
    # * Last column is only for the shared colorbar so it does not overlap subplot data.
    fig_w = 4.2 * ncols + 1.25
    plot_h = 4.1 * nrows
    #* fixed-inch title band (+ clearance for first-row ax titles); grow fig so plots keep size
    if suptitle:
        title_h_in = 0.22 + n_suptitle_lines * 0.165 + 0.32
        fig_h = plot_h + title_h_in
        top = 1.0 - title_h_in / fig_h
        suptitle_y = 1.0 - (0.06 / fig_h)
    else:
        fig_h = plot_h
        top = 0.92
        suptitle_y = None
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.10],
        wspace=0.32,
        hspace=0.38,
    )
    gs.update(
        left=0.07,
        right=0.91,
        bottom=0.08,
        top=top,
    )
    flat_axes: list = []
    for row in range(nrows):
        for col in range(ncols):
            flat_axes.append(fig.add_subplot(gs[row, col]))
    cax = fig.add_subplot(gs[:, ncols])

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
    )
    mappable.set_array([])

    for m in range(n_modes):
        ax = flat_axes[m]
        ob = np.asarray(per_mode_obs[m])
        rw = np.asarray(per_mode_r[m])
        ax.set_title(mode_titles[m], fontsize=title_fontsize)
        if ob.size == 0 or rw.size == 0:
            ax.text(
                0.5,
                0.5,
                "no samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if invert_yaxis:
                ax.invert_yaxis()
            if highlight_panel_indices is not None and m in highlight_panel_indices:
                for spine in ax.spines.values():
                    spine.set_edgecolor("blue")
                    spine.set_linewidth(2.5)
            continue
        plot_reward(
            ax,
            ob,
            rw,
            no_norm=False,
            x_idx=x_idx,
            y_idx=y_idx,
            reward_size_threshold=reward_size_threshold,
            scatter_s=scatter_s,
            s_below=s_below,
            s_above=s_above,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar=False,
            xlabel=axis_xlabel,
            ylabel=axis_ylabel,
            xlim=xlim,
            ylim=ylim,
            invert_yaxis=invert_yaxis,
        )
        if highlight_panel_indices is not None and m in highlight_panel_indices:
            for spine in ax.spines.values():
                spine.set_edgecolor("blue")
                spine.set_linewidth(2.5)

    keep_slots = {score_slot, entropy_slot, mean_std_slot, l_nuisance_slot, theta_slot}
    for k in range(n_modes, len(flat_axes)):
        if k in keep_slots:
            continue
        flat_axes[k].set_visible(False)

    if show_head_score:
        assert score_slot is not None
        score_ylabel = (
            head_order_score_ylabel
            if head_order_score_ylabel is not None
            else r"$\mathrm{KL}$"
        )
        score_title = (
            head_order_score_title
            if head_order_score_title is not None
            else r"per-head $\mathrm{KL}(q\|p)$"
        )
        _draw_per_head_score_bars(
            flat_axes[score_slot],
            score_per_head=elbo_kl_per_head,
            ylabel=score_ylabel,
            title=score_title,
            title_fontsize=title_fontsize,
            head_order=elbo_head_order,
        )
        if has_entropy_panel:
            assert entropy_slot is not None
            _draw_per_head_score_bars(
                flat_axes[entropy_slot],
                score_per_head=qlae_bin_entropy_per_head,
                ylabel=r"$H$",
                title=r"per-head FSQ bin entropy",
                title_fontsize=title_fontsize,
                head_order=elbo_head_order,
            )
        if has_mu_panel:
            assert mean_std_slot is not None
            assert elbo_post_mean is not None and elbo_post_mean_std is not None
            _draw_per_head_mean_std(
                flat_axes[mean_std_slot],
                post_mean=elbo_post_mean,
                post_mean_std=elbo_post_mean_std,
                title_fontsize=title_fontsize,
                head_order=elbo_head_order,
            )
        if has_l_nuisance_panel:
            assert l_nuisance_slot is not None and elbo_l_nuisance_per_head is not None
            _draw_per_head_l_nuisance_bars(
                flat_axes[l_nuisance_slot],
                l_nuisance_per_head=elbo_l_nuisance_per_head,
                title_fontsize=title_fontsize,
                head_order=elbo_head_order,
            )
        if show_theta:
            assert theta_slot is not None and theta_mean_per_head is not None
            _draw_per_head_theta_bars(
                flat_axes[theta_slot],
                theta_mean_per_head=theta_mean_per_head,
                title_fontsize=title_fontsize,
                head_order=elbo_head_order,
            )

    # * Shared color scale across modes (matplotlib needs a nonempty array on the mappable).
    any_data = all_r.size > 0
    if any_data:
        mappable.set_array(np.linspace(vmin, vmax, 256))
        cb = fig.colorbar(mappable, cax=cax, label=colorbar_label)
        cb.ax.tick_params(labelsize=8)
    else:
        cax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=suptitle_y, va="top")
    plt.close(fig)
    return fig


def plot_observations(observations, base_path):
    if observations.shape[1] > 2:
        observations = PCA(n_components=2).fit_transform(observations)
    plt.figure()
    plt.scatter(observations[:, 0], observations[:, 1], s=1)
    plt.savefig(base_path.replace("queries", "observation_plot"))


def plot_goals(env, ax, target_p, scale):
    if hasattr(env, "plot_goals"):
        env.plot_goals(ax, scale)
    else:
        ax.scatter(x=target_p[0], y=target_p[1], color="red", s=100)


def plot_observation_rewards(
    obs,
    r,
    no_norm=False,
    *,
    suptitle: str | None = None,
    x_idx: int = 0,
    y_idx: int = 1,
    axis_xlabel: str = "EE x",
    axis_ylabel: str = "EE y",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    fig, ax = plt.subplots()
    plot_reward(
        ax,
        obs,
        r,
        no_norm=no_norm,
        reward_size_threshold=0.5,
        x_idx=x_idx,
        y_idx=y_idx,
        xlabel=axis_xlabel,
        ylabel=axis_ylabel,
        xlim=xlim,
        ylim=ylim,
    )
    if suptitle:
        fig.subplots_adjust(top=0.78)
        fig.suptitle(suptitle, fontsize=11, y=0.94)
    plt.close(fig)
    return fig


def plot_mlp(env, reward_model):
    obs = env.get_obs_grid()
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)
    r = reward_model.get_reward(obs).detach().cpu().numpy()

    fig, ax = plt.subplots()
    plot_reward(ax, obs_copy, r)
    env.plot_goals(ax)
    plt.title("reward model")

    plot_dict = dict(reward_plot=wandb.Image(fig))
    plt.close(fig)
    if hasattr(env, "plot_gt"):
        plot_dict["gt"] = env.plot_gt(wandb_log=True)
    return plot_dict


def plot_classifier(env, reward_model, dataset):
    obs = env.get_obs_grid()
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)

    comparison_obs = dataset.get_mode_data(batch_size=100)["observations"]
    comparison_obs = comparison_obs.reshape(-1, comparison_obs.shape[-1])
    comparison_obs = (
        torch.from_numpy(comparison_obs)
        .float()
        .to(next(reward_model.parameters()).device)
    )

    r = reward_model.get_reward(obs, comparison_obs).detach().cpu().numpy()

    fig, ax = plt.subplots()
    plot_reward(ax, obs_copy, r, no_norm=True)
    env.plot_goals(ax)
    plt.title("reward model")

    plot_dict = dict(reward_plot=wandb.Image(fig))
    plt.close(fig)
    if hasattr(env, "plot_gt"):
        plot_dict["gt"] = env.plot_gt(wandb_log=True)
    return plot_dict


def plot_mlp_samples(env, reward_model, samples=4):
    obs = env.get_obs_grid()
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)

    plot_dict = plot_mlp(env, reward_model)
    fig, axs = plt.subplots(samples // 2, 2, figsize=(20, 16))
    for i, ax in enumerate(axs.flatten()):
        plot_reward(ax, obs_copy, reward_model.sample_reward(obs).detach().cpu().numpy())
        ax.set_title(f"reward model sample: {i}")

    plot_dict["reward_samples"] = wandb.Image(fig)
    plt.close(fig)

    r = reward_model.get_variance(obs).detach().cpu().numpy()
    fig, ax = plt.subplots()
    # r = reward_model.sample_reward(obs).detach().cpu().numpy()
    plot_reward(ax, obs_copy, r)
    env.plot_goals(ax)
    plt.title("reward model variance")
    plot_dict["reward_variance"] = wandb.Image(fig)
    plt.close(fig)
    return plot_dict


def plot_prior(reward_model):
    with torch.no_grad():
        fig = plt.figure()
        x = np.linspace(-10, 10, 1000)
        for i in range(reward_model.latent_dim):
            y = stats.norm(
                reward_model.prior_mean[i].cpu(),
                np.exp(reward_model.prior_logvar[i].cpu() * 0.5),
            ).pdf(x)
            plt.plot(x, y)
        plt.title("Learned prior")
    plt.close(fig)
    return dict(prior_plot=wandb.Image(fig))


def plot_latents(env, reward_model, dataset):
    if reward_model.flow_prior:
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        ax1 = axs[0]
        ax2 = axs[1]
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        ax1 = axs

    modes_n = env.get_num_modes()
    latents = get_all_posterior(env, reward_model, dataset, 128)
    for mode_n in range(modes_n):
        z = latents[mode_n]
        # X_embedded = PCA(n_components=2).fit_transform(z)
        X_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random", perplexity=10
        ).fit_transform(z)
        ax1.scatter(X_embedded[:, 0], X_embedded[:, 1], c=f"C{mode_n}")

        if reward_model.flow_prior:
            transformed_z = (
                reward_model.flow(
                    torch.from_numpy(z).float().to(next(reward_model.parameters()).device)
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            X_embedded = PCA(n_components=2).fit_transform(transformed_z)
            ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], c=f"C{mode_n}")
    ax1.set_title("Latent embeddings")
    if reward_model.flow_prior:
        ax2.set_title("Transformed latent embeddings")
    plt.close(fig)
    return dict(latent_plot=wandb.Image(fig))


def plot_z(obs, env, reward_model, latents, comp_obs=None):
    assert latents.shape[0] == env.get_num_modes()
    assert latents.shape[2] == reward_model.latent_dim
    num_samples = latents.shape[1]
    modes_n = env.get_num_modes()
    fig, axs = plt.subplots(
        modes_n, num_samples, figsize=(10 * num_samples, 8 * env.get_num_modes())
    )
    obs_copy = np.copy(obs)
    obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)

    axs = axs.flatten()
    for mode_n in range(modes_n):
        for i in range(num_samples):
            ax = axs[mode_n * num_samples + i]
            z = latents[None, mode_n, i]
            z = np.repeat(z, obs.shape[0], axis=0)
            z = torch.from_numpy(z).float().to(next(reward_model.parameters()).device)
            if comp_obs is not None:
                r = reward_model.decode(obs, comp_obs, z).detach().cpu().numpy()
                plot_reward(ax, obs_copy, r, no_norm=True)
            else:
                r = (
                    reward_model.decode(obs, z).detach().cpu().numpy()
                )  # .reshape((NX, NY))
                plot_reward(ax, obs_copy, r)
            ax.set_title(f"Mode {mode_n}")
            env.plot_goals(ax)

    plt.tight_layout()
    plt.close(fig)
    return wandb.Image(fig)


def plot_vae(env, reward_model, dataset, classifier=False, num_samples=4):
    obs = env.get_obs_grid()
    comp_obs = None
    if classifier:
        comp_obs = dataset.get_mode_data(batch_size=50)[0]["observations"]
        comp_obs = comp_obs.reshape(-1, obs.shape[-1])
        comp_obs = (
            torch.from_numpy(comp_obs).float().to(next(reward_model.parameters()).device)
        )
    plot_dict = plot_prior(reward_model)
    plot_dict.update(plot_latents(env, reward_model, dataset))
    n = env.get_num_modes()
    prior_latents = (
        reward_model.sample_prior(size=num_samples * n)
        .view(n, num_samples, reward_model.latent_dim)
        .detach()
        .cpu()
        .numpy()
    )

    posterior_latents = get_all_posterior(env, reward_model, dataset, num_samples)
    biased_latents = get_biased(env, reward_model)

    reward_model.update_posteriors(posterior_latents, biased_latents)

    plot_dict["prior"] = plot_z(obs, env, reward_model, prior_latents, comp_obs=comp_obs)
    plot_dict["posterior"] = plot_z(
        obs, env, reward_model, posterior_latents, comp_obs=comp_obs
    )
    plot_dict["biased"] = plot_z(
        obs, env, reward_model, biased_latents, comp_obs=comp_obs
    )
    if hasattr(env, "plot_gt"):
        plot_dict["gt"] = env.plot_gt(wandb_log=True)
    return plot_dict


def update_posterior(env, reward_model, dataset):
    biased_latents = get_biased(env, reward_model, dataset)
    reward_model.update_posteriors(None, biased_latents)
