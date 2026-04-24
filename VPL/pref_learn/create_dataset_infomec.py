import os
import pickle
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import absl.app
import absl.flags
import gym
import jaxrl_m.envs
import numpy as np
from jaxrl_m.learners.d4rl_utils import get_dataset
from tqdm import tqdm

from pref_learn.utils.data_utils import (
    get_labels,
    get_queries_from_multi,
    sample_from_env,
)
from pref_learn.utils.plot_utils import plot_observation_rewards
from pref_learn.utils.utils import define_flags_with_default, set_random_seed

FLAGS_DEF = define_flags_with_default(
    env="maze2d-twogoals-multimodal-v0",
    sample_from_env=False,
    data_dir="./pref_datasets",
    data_seed=42,
    num_query=100,
    query_len=1,
    set_len=32,
    relabel=False,
    dataset_path="",
    ensure_unique_pairs=True,
    ensure_unique_segments_within_set=True,
    max_sampling_attempts_per_pair=1000,
    store_source_mode=True,
)


def _as_int_column(x):
    x = np.asarray(x)
    return x.astype(np.int32).reshape(len(x), -1)


def _as_float_column(x):
    x = np.asarray(x)
    return x.astype(np.float32).reshape(len(x), -1)


def main(_):
    FLAGS = absl.flags.FLAGS
    set_random_seed(FLAGS.data_seed)
    base_path = os.path.join(FLAGS.data_dir, FLAGS.env)
    os.makedirs(base_path, exist_ok=True)

    gym_env = gym.make(FLAGS.env)
    gym_env.reset()

    if FLAGS.relabel:
        assert os.path.exists(FLAGS.dataset_path)
        query_path = FLAGS.dataset_path
        dataset = pickle.load(open(FLAGS.dataset_path, "rb"))
    else:
        if FLAGS.sample_from_env:
            dataset, query_path = sample_from_env(
                gym_env, FLAGS.num_query, FLAGS.set_len, FLAGS.query_len, base_path
            )
        else:
            dataset, _ = get_dataset(gym_env)
            dataset, query_path = get_queries_from_multi(
                gym_env,
                dataset,
                FLAGS.num_query,
                FLAGS.query_len,
                FLAGS.set_len,
                base_path,
                ensure_unique_pairs=FLAGS.ensure_unique_pairs,
                ensure_unique_segments_within_set=FLAGS.ensure_unique_segments_within_set,
                max_sampling_attempts_per_pair=FLAGS.max_sampling_attempts_per_pair,
            )
        print("Saved queries at:", query_path)

    num_sets = len(dataset["observations"])
    set_len = dataset["observations"].shape[1]

    source_mode = np.zeros((num_sets, 1), dtype=np.int32)
    source_mode_per_pair = np.zeros((num_sets, set_len, 1), dtype=np.int32)
    annotator_id = np.zeros((num_sets, 1), dtype=np.int32)
    preferred_side = np.zeros((num_sets, set_len, 1), dtype=np.int32)
    reward_sum_1 = np.zeros((num_sets, set_len, 1), dtype=np.float32)
    reward_sum_2 = np.zeros((num_sets, set_len, 1), dtype=np.float32)
    reward_margin = np.zeros((num_sets, set_len, 1), dtype=np.float32)

    obs = []
    rewards = []
    modes = []

    for i in tqdm(range(num_sets), desc="relabel queries"):
        mode = int(gym_env.sample_mode())
        modes.append(mode)

        seg_reward_1 = gym_env.compute_reward(dataset["observations"][i], mode)
        seg_reward_2 = gym_env.compute_reward(dataset["observations_2"][i], mode)
        labels = get_labels(seg_reward_1, seg_reward_2)
        dataset["labels"][i] = labels

        seg_reward_sum_1 = np.sum(seg_reward_1, axis=-1, keepdims=True).astype(np.float32)
        seg_reward_sum_2 = np.sum(seg_reward_2, axis=-1, keepdims=True).astype(np.float32)
        seg_reward_margin = seg_reward_sum_1 - seg_reward_sum_2

        source_mode[i, 0] = mode
        source_mode_per_pair[i, :, 0] = mode
        annotator_id[i, 0] = mode
        preferred_side[i] = labels.astype(np.int32)
        reward_sum_1[i] = seg_reward_sum_1
        reward_sum_2[i] = seg_reward_sum_2
        reward_margin[i] = seg_reward_margin

        obs.append(dataset["observations"][i])
        rewards.append(seg_reward_1)

    if FLAGS.store_source_mode:
        # Query-set level source variables: these line up 1:1 with latent vectors.
        dataset["source_mode"] = source_mode
        dataset["annotator_id"] = annotator_id
        dataset["source_variable"] = source_mode.copy()

        # Pair-level metadata: useful for auditing or future finer-grained analyses.
        dataset["source_mode_per_pair"] = source_mode_per_pair
        dataset["preferred_side"] = preferred_side
        dataset["left_is_preferred"] = preferred_side.copy()
        dataset["reward_sum_1"] = reward_sum_1
        dataset["reward_sum_2"] = reward_sum_2
        dataset["reward_margin"] = reward_margin

        # Lightweight schema hints for downstream analysis scripts.
        dataset["source_variable_name"] = "source_mode"
        dataset["source_variable_type"] = "discrete"
        dataset["source_variable_dim"] = np.array([1], dtype=np.int32)
        try:
            dataset["source_classes"] = np.arange(gym_env.get_num_modes(), dtype=np.int32)
        except Exception:
            dataset["source_classes"] = np.unique(source_mode[:, 0]).astype(np.int32)

    relabelled_path = str(query_path).replace("queries", "relabelled_queries")
    with open(relabelled_path, "wb") as f:
        pickle.dump(dataset, f)
    print("Saved relabelled queries at:", relabelled_path)
    print("Average mode during relabelling:", sum(modes) / len(modes))

    obs = np.concatenate(obs, axis=0).reshape(-1, obs[0].shape[-1])
    rewards = np.array(rewards).reshape(-1, 1)
    fig = plot_observation_rewards(obs, rewards)
    fig.savefig("relabelled_queries.png")


if __name__ == "__main__":
    absl.app.run(main)
