import os
import pickle

import absl.app
import absl.flags
import gym
import numpy as np
from tqdm import tqdm

import jaxrl_m.envs
from jaxrl_m.learners.d4rl_utils import get_dataset
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

    modes = []
    obs = []
    rewards = []
    preferred_side = []
    for i in tqdm(range(len(dataset["observations"])), desc="relabel queries"):
        mode = int(gym_env.sample_mode())
        modes.append(mode)
        seg_reward_1 = gym_env.compute_reward(dataset["observations"][i], mode)
        seg_reward_2 = gym_env.compute_reward(dataset["observations_2"][i], mode)
        labels = get_labels(seg_reward_1, seg_reward_2)
        dataset["labels"][i] = labels

        obs.append(dataset["observations"][i])
        rewards.append(seg_reward_1)
        preferred_side.append(labels.astype(np.int32).reshape(-1))

    if FLAGS.store_source_mode:
        source_mode = np.asarray(modes, dtype=np.int32).reshape(-1, 1)
        preferred_side = np.asarray(preferred_side, dtype=np.int32).reshape(len(preferred_side), -1, 1)

        dataset["source_mode"] = source_mode
        dataset["annotator_id"] = source_mode.copy()
        dataset["source_variable"] = source_mode.copy()
        dataset["preferred_side"] = preferred_side
        dataset["left_is_preferred"] = preferred_side.copy()

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
