import os
import pickle
from datetime import datetime
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import absl.app
import absl.flags
import gym
import jaxrl_m.envs  # noqa: F401 — register gym ids (sort-easy-v0, ...)
import numpy as np
from einops import rearrange
from jaxrl_m.envs.sort_easy import SortEnv, annotator_from_env
from jaxrl_m.learners.d4rl_utils import get_dataset
from tqdm import tqdm

from pref_learn.utils.data_utils import (
    get_labels,
    sample_pref_from_dataset,
    sample_pref_from_env,
)
from pref_learn.utils.plot_utils import (
    plot_observation_rewards,
    plot_observation_rewards_by_annotator_mode,
)
from pref_learn.utils.utils import define_flags_with_default, set_random_seed

FLAGS_DEF = define_flags_with_default(
    env="sort-easy-4goals-v0",
    sample_from_env=False,
    pref_data_dir="./pref_datasets",  # todo not used
    data_seed=42,
    num_query=100,  # Q
    segment_len=1,  # T
    set_size=32,  # S
    relabel=False,
    dataset_path="",  # when relabel=True: pickled pairwise queries path (not HDF5)
    offline_dataset_h5="",  # D4RL HDF5 for get_dataset (e.g. SortEnv) qlearning style; empty uses env default
    ensure_unique_pairs=True,
    ensure_unique_segments_within_set=True,
    max_sampling_attempts_per_pair=1000,
    store_source_mode=True,
    debug=False,
)


def _gym_make_kw_from_offline_h5(*, offline_h5: str) -> dict:
    """If ``offline_h5`` is set, kwargs for ``gym.make(..., dataset_path=...)``."""
    if not offline_h5.strip():
        return {}
    h5_path = os.path.abspath(os.path.expanduser(offline_h5.strip()))
    assert os.path.isfile(h5_path), f"offline_dataset_h5 not found: {h5_path}"
    return {"dataset_path": h5_path}


def main(_):
    """
    First transition samples datasets: N = num_samples
        traj_dataset:
            observations: (N', O) -> (N, T, O)
            actions: (N, A)
            rewards: (N,)
            masks: (N,)
            dones_float: (N,)
            next_observations: (N, O)
            traj_done: (N,)
            next_rewards: (N,)

    Next, preference dataset: Q = num_query, S = set_size (per user)
        dataset:
            labels: (Q, S, 1)

            observations: (Q, S, T, O)
            next_observations: (Q, S, T, O)
            actions: (Q, S, T, A)

            observations_2: (Q, S, T, O)
            next_observations_2: (Q, S, T, O)
            actions_2: (Q, S, T, A)

            # metadata, probably not needed
            timestep_1: (Q, S, T)
            timestep_2: (Q, S, T)
            start_indices: (Q, S)
            start_indices_2: (Q, S)
            time_indices: (Q, S)
            time_indices_2: (Q, S)
            traj_indices: (Q, S)
            traj_indices_2: (Q, S)

    """
    FLAGS = absl.flags.FLAGS
    set_random_seed(FLAGS.data_seed)
    # pref_base_path = os.path.join(FLAGS.pref_data_dir, FLAGS.env)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pref_dir = (
        Path(FLAGS.offline_dataset_h5).parent
        / f"pref_datasets_{ts}_Q={FLAGS.num_query}_S={FLAGS.set_size}_T={FLAGS.segment_len}"
    )
    os.makedirs(pref_dir, exist_ok=True)
    pref_fp = pref_dir / "queries.pkl"

    # * Step 1: Load trajectory dataset and get random pairwise query sets
    print("\n========== Loading trajectories and sampling pairwise query sets ==========")
    gym_env = gym.make(
        FLAGS.env,
        **_gym_make_kw_from_offline_h5(offline_h5=FLAGS.offline_dataset_h5),
    )
    gym_env.reset()
    raw_env = gym_env.unwrapped
    if isinstance(raw_env, SortEnv) and len(raw_env.goal_poses) in (2, 4):
        annot = annotator_from_env(env=raw_env)
    else:
        annot = gym_env

    if FLAGS.relabel:
        # todo: update this to use the new query path
        assert os.path.exists(FLAGS.dataset_path)
        query_path = FLAGS.dataset_path
        dataset = pickle.load(open(FLAGS.dataset_path, "rb"))
    else:
        if FLAGS.sample_from_env:
            dataset, query_path = sample_pref_from_env(
                gym_env,
                n_query=FLAGS.num_query,
                set_sz=FLAGS.set_size,
                segment_len=FLAGS.segment_len,
                query_path=pref_fp,
            )
        else:
            traj_dataset, _ = get_dataset(gym_env)  # qlearning style dataset
            dataset, query_path = sample_pref_from_dataset(
                env=gym_env,  # not used?
                dataset=traj_dataset,
                num_query=FLAGS.num_query,  # Q
                seg_len=FLAGS.segment_len,  # T
                set_sz=FLAGS.set_size,  # S
                query_path=pref_fp,
                ensure_unique_pairs=FLAGS.ensure_unique_pairs,
                ensure_unique_segments_within_set=FLAGS.ensure_unique_segments_within_set,
                max_sampling_attempts_per_pair=FLAGS.max_sampling_attempts_per_pair,
            )
        print("Saved queries at:", query_path)

    # * Step 2: compute pref label for each query set, according to pref behavior modes
    print("\n========== Computing pref labels for each query set ==========")
    print(f"Query set obs shape: {dataset['observations'].shape}")  # (Q, S, T, O)
    n_qsets = len(dataset["observations"])

    obs = []  # List[(S, T, O) * Q]
    rewards = []  # List[(S, T) * Q]
    labels = []  # List[(S, T) * Q]
    source_vars = np.zeros((n_qsets, annot.n_sources), dtype=np.int32)  # (Q, n_sources)
    source_modes = np.zeros((n_qsets, 1), dtype=np.int32)  # (Q, 1)

    n_modes = getattr(annot, "n_pref_modes", 0)
    per_mode_obs: list[list] = [[] for _ in range(n_modes)]
    per_mode_rew: list[list] = [[] for _ in range(n_modes)]

    for i in tqdm(range(n_qsets), desc="relabel queries"):  # Q
        # sample pref behavior mode + compute label based on gt return per segment
        obs1, obs2 = dataset["observations"][i], dataset["observations_2"][i]  # (S, T, O)
        # mode = int(gym_env.sample_mode())  # 0 or 1
        # seg_return1 = gym_env.compute_reward(obs1, mode)  # (S, T, O) -> (S, T)
        # seg_return2 = gym_env.compute_reward(obs2, mode)  # (S, T, O) -> (S, T)
        # labels = get_labels(seg_return1, seg_return2)  # (S, T) -> (S, 1)

        mode = int(annot.sample_pref_mode())
        seg_return1 = annot.compute_reward(obs1, mode)  # (S, T, O) -> (S, T)
        seg_return2 = annot.compute_reward(obs2, mode)  # (S, T, O) -> (S, T)
        labels = get_labels(seg_return1, seg_return2)  # (S, T) -> (S, 1)
        src_var = annot.pref_spec_for_mode(mode, as_array=True)  # (n_sources,)

        dataset["labels"][i] = labels
        source_vars[i] = src_var  # (n_sources,)
        source_modes[i] = mode  # (1,)

        obs.append(obs1)
        rewards.append(seg_return1)
        if n_modes > 0:
            per_mode_obs[mode].append(obs1)
            per_mode_rew[mode].append(seg_return1)

    if FLAGS.store_source_mode:
        # Query-set level source variables: these line up 1:1 with latent vectors.
        dataset["source_variable"] = source_vars  # (Q, n_sources)
        dataset["source_mode"] = source_modes  # (Q, 1)
        dataset["source_names"] = annot.source_specs["source_names"]  # [str]
        dataset["source_dims"] = annot.source_specs["source_dims"]  # [int]
        dataset["source_types"] = annot.source_specs["source_types"]  # [str]
        dataset["source_mode_names"] = [str(s["name"]) for s in annot.ANNOTATOR_SPECS]

    # pref dataset assumed to have no labels.
    relabelled_path = str(query_path).replace("queries", "queries_relabelled")
    if FLAGS.debug:
        relabelled_path = str(query_path).replace("queries", "queries_relabelled_debug")
    with open(relabelled_path, "wb") as f:
        pickle.dump(dataset, f)
    print("Saved relabelled queries at:", relabelled_path)
    print("Modes during pref gathering (sampled index):")
    uniq_modes, mode_counts = np.unique(source_modes.squeeze(), return_counts=True)
    n_tot = len(source_modes)
    for m, c in zip(uniq_modes, mode_counts):
        pct = 100.0 * float(c) / float(n_tot)
        print(f"  pref mode {int(m):>4}: count {int(c):>6} ({pct:5.1f}%)")
    print(f"  total:       {n_tot:>6}")

    # * Observation–reward scatter: EE (x,y) axes match SortEnv observations only.
    if isinstance(raw_env, SortEnv):
        # plot all modes pooled
        qst_line = f"(Q={FLAGS.num_query}, S={FLAGS.set_size}, T={FLAGS.segment_len})"
        obs_plot = rearrange(np.stack(obs, axis=0), "Q S T O -> (Q S T) O")
        reward_plot = rearrange(np.stack(rewards, axis=0), "Q S T -> (Q S T) 1")
        fig = plot_observation_rewards(
            obs_plot,
            reward_plot,
            suptitle=("Annotator reward vs EE (x,y), all modes pooled\n" + qst_line),
        )
        png_path = f"{relabelled_path.replace('pkl', 'png')}"
        fig.savefig(png_path)

        # plot by annotator mode
        if n_modes > 0 and hasattr(annot, "ANNOTATOR_SPECS"):
            O_dim = int(obs[0].shape[-1])
            mode_titles = [str(spec["name"]) for spec in annot.ANNOTATOR_SPECS]
            assert len(mode_titles) == n_modes, (
                f"ANNOTATOR_SPECS length {len(mode_titles)} vs n_pref_modes {n_modes}"
            )
            per_mode_obs_arr: list[np.ndarray] = []
            per_mode_r_arr: list[np.ndarray] = []
            for m in range(n_modes):
                if not per_mode_obs[m]:
                    per_mode_obs_arr.append(np.empty((0, O_dim), dtype=np.float64))
                    per_mode_r_arr.append(np.empty((0, 1), dtype=np.float64))
                else:
                    per_mode_obs_arr.append(
                        rearrange(
                            np.stack(per_mode_obs[m], axis=0), "q s t o -> (q s t) o"
                        )
                    )
                    per_mode_r_arr.append(
                        rearrange(np.stack(per_mode_rew[m], axis=0), "q s t -> (q s t) 1")
                    )
            fig_modes = plot_observation_rewards_by_annotator_mode(
                per_mode_obs_arr,
                per_mode_r_arr,
                mode_titles=mode_titles,
                suptitle=(
                    "Annotator reward vs EE (x,y): one subplot per sampled "
                    "(observation,reward) by annotator mode\n" + qst_line
                ),
            )
            p_png = Path(png_path)
            fig_modes.savefig(
                p_png.with_name(f"{p_png.stem}_by_annot_mode{p_png.suffix}")
            )


if __name__ == "__main__":
    absl.app.run(main)
