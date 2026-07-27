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
):
    """
    One scatter panel per annotator mode; each uses only (obs, r) from that mode.

    Args:
        per_mode_obs: length M; each entry (N_m, O) or empty (0, O).
        per_mode_r: length M; each (N_m,) or (N_m, 1) aligned with ``per_mode_obs``.
        mode_titles: length M subplot titles (e.g. ANNOTATOR_SPECS ``name``).
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

    ncols = min(4, n_modes) if ncols is None else min(int(ncols), n_modes)
    nrows = int(np.ceil(n_modes / ncols))
    n_suptitle_lines = (suptitle.count("\n") + 1) if suptitle else 0
    # * Last column is only for the shared colorbar so it does not overlap subplot data.
    fig_w = 4.2 * ncols + 1.25
    suptitle_line_frac = 0.038  # figure-height fraction per suptitle line (~fontsize 9)
    suptitle_top_margin = 0.020
    if suptitle:
        top = 1.0 - suptitle_top_margin - n_suptitle_lines * suptitle_line_frac
        top = max(top, 0.52 + 0.04 * max(nrows - 1, 0))
        suptitle_y = 1.0 - suptitle_top_margin
    else:
        top = 0.92
        suptitle_y = None
    fig_h = 4.1 * nrows
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

    for k in range(n_modes, len(flat_axes)):
        flat_axes[k].set_visible(False)

    # * Shared color scale across modes (matplotlib needs a nonempty array on the mappable).
    any_data = all_r.size > 0
    if any_data:
        mappable.set_array(np.linspace(vmin, vmax, 256))
        cb = fig.colorbar(mappable, cax=cax, label="r(s)")
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
