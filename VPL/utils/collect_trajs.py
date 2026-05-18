# coding=utf-8
# Copyright 2023 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data collection script."""

import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass

import gym
import jaxrl_m.envs
import numpy as np
import tyro
from ravens.dataset import Dataset
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)


@dataclass(frozen=True)
class CollectTrajsConfig:
    """Oracle rollout collection for Ravens / Gym Sort."""

    """Run root; ``raw_rollouts/*.pkl``, ``rollout_visualization.png``; HDF5 via ``demos2dataset``."""
    data_dir: str = "."

    """Enable PyBullet GUI (EGL off) when True."""
    disp: bool = False

    """``gym.make`` env id (e.g. ``sort-easy-v0``)."""
    env_name: str = "sort-easy-v0"

    """Stop after this many trajectories are saved successfully."""
    n: int = 1000

    """Ravens oracle steps per motion segment (``task.oracle``)."""
    steps_per_seg: int = 3

    """When False with multi-goal envs, always use goal index ``0``."""
    multimodal: bool = True

    """When True, cycle goals ``0,1,...`` each attempt; else sample with ``mode_rng``."""
    alternate_goal_modes: bool = False

    """Only persist rollouts that pass sparse task success when True."""
    success_only: bool = True

    """How many random saved paths go into ``rollout_visualization.png``; ``0`` skips plots and videos."""
    plot_random_trajs: int = 10

    """FPS for ``raw_rollouts_videos/traj_*.mp4`` (same rollouts as the visualization panel)."""
    sample_video_fps: int = 24


def traj_sparse_success(traj: dict) -> bool:
    """Success iff sparse_reward from SortEnv.step_oracle info was 1 at some timestep."""
    sparse_steps = np.array(
        [float(step_info["sparse_reward"]) for step_info in traj["info"]],
        dtype=np.float64,
    )
    return bool(np.any(sparse_steps >= 0.5))


def rollout_trajectory(
    env,
    agent,
    max_steps: int,
    *,
    video_frames: list[np.ndarray] | None = None,
):
    obs = env.reset()
    init_pose = obs
    done = False
    traj = defaultdict(list)

    def grab_frame() -> None:
        if video_frames is None:
            return
        frame = np.asarray(env.render(mode="rgb_array"))
        video_frames.append(frame)

    grab_frame()

    for i in range(max_steps):
        action = agent.act(obs, {})

        pos = action["move_cmd"][0]
        rot = action["move_cmd"][1]
        noise = np.random.normal(0, 0.01, 3)
        pos = pos + noise
        action["move_cmd"] = (pos, rot)
        next_obs, reward, done, info = env.step_oracle(action)
        grab_frame()

        traj["obs"].append(obs)
        traj["action_dict"].append(action)
        traj["reward"].append(reward)
        traj["info"].append(info)
        traj["next_obs"].append(next_obs)
        traj["done"].append(done)
        obs = next_obs

        if done:
            # ! only works when oracle uses max_steps as done signal
            action = agent.act(obs, {})
            next_obs, reward, done, info = env.step_oracle(action)
            grab_frame()

            traj["obs"].append(obs)
            traj["action_dict"].append(action)
            traj["reward"].append(reward)
            traj["info"].append(info)
            traj["next_obs"].append(next_obs)
            traj["done"].extend([False, True])
            break

    traj["obs"] = np.stack(traj["obs"])
    traj["next_obs"] = np.stack(traj["next_obs"])
    traj["action_dict"] = traj["action_dict"]
    traj["reward"] = np.array(traj["reward"])
    traj["info"] = traj["info"]
    traj["done"] = np.array(traj["done"])

    return traj, init_pose


def save_rollout_sample_video(
    *,
    frames: list[np.ndarray],
    save_path: str,
    fps: int,
) -> None:
    """Write one MP4 from Ravens ``rgb_array`` frames (needs ``imageio-ffmpeg``)."""
    if not frames:
        return
    import imageio.v2 as imageio

    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    imageio.mimsave(
        uri=save_path,
        ims=frames,
        fps=fps,
        macro_block_size=1,
        codec="libx264",
        quality=8,
    )


def convert_traj(env, traj):
    ## quat = obs[9:13]
    # delta_pos = traj["next_obs"][:, 0:3] - traj["obs"][:, 0:3]
    ## delta_pos = np.clip(delta_pos, -1, 1)
    # suction = np.array([step["suction_cmd"] for step in traj["action_dict"]])[:, None]
    ## import pdb; pdb.set_trace()
    # action = np.concatenate([delta_pos, suction], axis=1)
    # assert len(action) == len(traj["obs"])
    # traj["action"] = action
    pose = np.stack([np.array(step["move_cmd"][0]) for step in traj["action_dict"]])
    suction = np.array([step["suction_cmd"] for step in traj["action_dict"]])[:, None]
    action = np.concatenate([pose, suction], axis=1)
    assert len(action) == len(traj["obs"])
    traj["action"] = action
    return traj


def test_traj(env, traj, init_pose):
    env.unwrapped.task.debug = True
    env.unwrapped.task.initial_pose = (init_pose[7:10], init_pose[10:14])
    obs = env.reset()
    done = False
    for i in range(len(traj["action"])):
        action = traj["action"][i]
        next_obs, reward, done, info = env.step(action)
        # if done:
        #     break
    env.unwrapped.task.debug = False


def plot_sample_box_trajectories(
    samples: list[tuple[int, dict]],
    *,
    save_path: str,
) -> None:
    """Single PNG: grid of workspace box paths (y horizontal, x vertical)."""
    import math

    import matplotlib.pyplot as plt

    if not samples:
        return
    samples = sorted(samples, key=lambda item: item[0])
    k = len(samples)
    ncols = min(3, k)
    nrows = math.ceil(k / ncols)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.8 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    marker_s = 28
    marker_lw = 0.35
    for i, (episode_idx, traj) in enumerate(samples):
        row, col = divmod(i, ncols)
        ax = axs[row][col]
        wx = traj["obs"][:, 3]
        wy = traj["obs"][:, 4]
        ax.plot(wy, wx, color="C0", linewidth=1.2)
        scatter_ini_kw = {"label": "initial"} if i == 0 else {}
        scatter_fin_kw = {"label": "final"} if i == 0 else {}
        ax.scatter(
            wy[0],
            wx[0],
            color="tab:green",
            s=marker_s,
            zorder=6,
            edgecolors="black",
            linewidths=marker_lw,
            **scatter_ini_kw,
        )
        ax.scatter(
            wy[-1],
            wx[-1],
            color="tab:red",
            s=marker_s,
            zorder=6,
            edgecolors="black",
            linewidths=marker_lw,
            **scatter_fin_kw,
        )
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("y (m)")
        ax.set_ylabel("x (m)")
        ax.set_title(f"traj_{episode_idx}")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    for j in range(k, nrows * ncols):
        axs.flat[j].set_visible(False)
    fig.suptitle("Sample box workspace trajectories (top = +x, right = +y)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(fname=save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dump_traj(save_dir, traj, episode):
    with open(os.path.join(save_dir, f"traj_{episode}.pkl"), "wb") as f:
        pickle.dump(traj, f)


def main(cfg: CollectTrajsConfig) -> None:
    env = gym.make(cfg.env_name, disp=cfg.disp, debug=True)
    task = env.unwrapped.task
    agent = task.oracle(env.unwrapped.env, steps_per_seg=cfg.steps_per_seg)

    seed = 0  # dataset.max_seed
    max_steps = task.max_steps * (cfg.steps_per_seg * agent.num_poses)
    episode = 0
    failed_rollouts = 0
    os.makedirs(cfg.data_dir, exist_ok=True)
    raw_rollouts_dir = os.path.join(cfg.data_dir, "raw_rollouts")
    os.makedirs(raw_rollouts_dir, exist_ok=True)
    k_plot = min(cfg.plot_random_trajs, cfg.n) if cfg.n > 0 else 0
    plot_episodes = (
        set(np.random.choice(cfg.n, size=k_plot, replace=False).tolist())
        if k_plot > 0
        else set()
    )
    videos_dir = os.path.join(cfg.data_dir, "raw_rollouts_videos")
    if k_plot > 0:
        os.makedirs(videos_dir, exist_ok=True)
    plot_samples: list[tuple[int, dict]] = []
    mode_rng = np.random.default_rng()
    collection_attempt = -1
    saved_mode_counts: Counter[int] = Counter()
    saved_traj_lengths: list[int] = []
    with tqdm(total=cfg.n, desc="collect_trajs") as pbar:
        while episode < cfg.n:
            collection_attempt += 1
            n_modes = len(env.goal_poses)
            if n_modes <= 1:
                env.set_mode(mode=0)
            elif cfg.multimodal:
                if cfg.alternate_goal_modes:
                    env.set_mode(mode=collection_attempt % n_modes)
                else:
                    mode = int(mode_rng.integers(n_modes))
                    env.set_mode(mode=mode)
            else:
                env.set_mode(mode=0)
            np.random.seed(seed)
            agent.reset()
            record_visual = k_plot > 0 and episode in plot_episodes
            video_frames: list[np.ndarray] | None = [] if record_visual else None
            try:
                traj, init_pose = rollout_trajectory(
                    env,
                    agent,
                    max_steps,
                    video_frames=video_frames,
                )
                traj = convert_traj(env, traj)

                # np.random.seed(seed)
                # test_traj(env, traj, init_pose)
            except Exception:
                failed_rollouts += 1
                seed += 2
                continue
            if cfg.success_only and not traj_sparse_success(traj=traj):
                failed_rollouts += 1
                seed += 2
                continue
            if episode in plot_episodes:
                plot_samples.append((episode, traj))
                if video_frames is not None:
                    save_rollout_sample_video(
                        frames=video_frames,
                        save_path=os.path.join(videos_dir, f"traj_{episode}.mp4"),
                        fps=cfg.sample_video_fps,
                    )
            saved_mode_counts[env.unwrapped.mode] += 1
            saved_traj_lengths.append(int(traj["reward"].shape[0]))
            dump_traj(raw_rollouts_dir, traj, episode)
            episode += 1
            seed += 2
            pbar.update(n=1)
    if plot_samples:
        plot_sample_box_trajectories(
            samples=plot_samples,
            save_path=os.path.join(cfg.data_dir, "rollout_visualization.png"),
        )
    saved = episode
    total_attempts = saved + failed_rollouts
    mode_total = sum(saved_mode_counts.values())
    mode_pct_lines: list[str] = []
    if mode_total > 0:
        mode_pct_lines = [
            f"  mode {m}: {saved_mode_counts[m]} ({100.0 * saved_mode_counts[m] / mode_total:.1f}%)"
            for m in sorted(saved_mode_counts)
        ]
    modes_block = (
        "Modes among saved trajectories:\n" + "\n".join(mode_pct_lines) + "\n"
        if mode_pct_lines
        else ""
    )
    if saved_traj_lengths:
        tl = np.asarray(saved_traj_lengths, dtype=np.float64)
        traj_len_line = (
            f"Trajectory length (saved, steps): "
            f"min={tl.min():.0f}, max={tl.max():.0f}, mean={tl.mean():.3f}\n"
        )
    else:
        traj_len_line = ""
    print(
        f"Total rollout attempts: {total_attempts}.\n"
        f"Succeeded (saved): {saved}.\n"
        f"Failed (not saved): {failed_rollouts}.\n"
        f"Save target: {cfg.n}.\n"
        f"Data directory: {cfg.data_dir}.\n"
        f"{modes_block}"
        f"{traj_len_line}"
    )


if __name__ == "__main__":
    main(tyro.cli(CollectTrajsConfig))
