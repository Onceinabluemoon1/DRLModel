import pickle
from typing import Optional, Tuple

import numpy as np
from jaxrl_m.learners.d4rl_utils import new_get_trj_idx
from tqdm import tqdm

SegmentKey = Tuple[int, int, int]
PairKey = Tuple[SegmentKey, SegmentKey]


def _segment_key(trj_idx: int, time_idx: int, len_query: int) -> SegmentKey:
    return int(trj_idx), int(time_idx), int(len_query)


def _pair_key(seg_a: SegmentKey, seg_b: SegmentKey) -> PairKey:
    # Order invariant so swapping left/right does not create a "new" query pair.
    return tuple(sorted((seg_a, seg_b)))


def _sample_segment(
    dataset,
    trj_idx_list: np.ndarray,
    trj_len_list: np.ndarray,
    valid_trj_indices: np.ndarray,
    seg_len: int,
    skip_flag: int,
    first_reward_sum: Optional[float],
):
    trj_idx = int(np.random.choice(valid_trj_indices))
    len_trj = int(trj_len_list[trj_idx])
    if len_trj < seg_len:
        return None

    time_idx = int(np.random.choice(len_trj - seg_len + 1))
    start_idx = int(trj_idx_list[trj_idx][0] + time_idx)
    end_idx = start_idx + seg_len
    assert end_idx <= trj_idx_list[trj_idx][1] + 1

    reward_seq = dataset["rewards"][start_idx:end_idx]
    reward_sum = float(np.sum(reward_seq))

    if first_reward_sum is not None:
        if skip_flag == 1 and first_reward_sum == reward_sum:
            return None
        if skip_flag == 2 and first_reward_sum == reward_sum:
            return None
        if skip_flag == 3 and first_reward_sum == reward_sum:
            return None

    obs_seq = dataset["observations"][start_idx:end_idx]
    next_obs_seq = dataset["next_observations"][start_idx:end_idx]
    act_seq = dataset["actions"][start_idx:end_idx]
    timestep_seq = np.arange(1, seg_len + 1)

    return {
        "trj_idx": trj_idx,
        "time_idx": time_idx,
        "start_idx": start_idx,
        "reward_seq": reward_seq,
        "obs_seq": obs_seq,
        "next_obs_seq": next_obs_seq,
        "act_seq": act_seq,
        "timestep_seq": timestep_seq,
        "seg_key": _segment_key(trj_idx, time_idx, seg_len),
        "reward_sum": reward_sum,
    }


def sample_pref_from_dataset(
    env,
    dataset: dict,
    num_query: int,  # Q
    seg_len: int,  # T
    set_sz: int,  # S
    # data_dir=None,
    query_path: str,
    skip_flag=1,
    ensure_unique_pairs: bool = True,
    ensure_unique_segments_within_set: bool = True,
    max_sampling_attempts_per_pair: int = 1000,
):
    """Sample query pairs and group them into sets.

    This version fixes the main issue in the original code:
      * no repeated unordered query pairs across the full dataset when
        ``ensure_unique_pairs`` is True
      * no repeated segment within a set when
        ``ensure_unique_segments_within_set`` is True
      * stores trajectory/time metadata needed for auditing and downstream analysis

    Args:
        dataset: dict
            observations: (N, O) -> (Q, S, T, O)
            actions: (N, A)
            rewards: (N,)
            masks: (N,)
            dones_float: (N,)
            next_observations: (N, O)
            traj_done: (N,)
            next_rewards: (N,)

    """
    _num_query = num_query
    total_pairs = num_query * set_sz

    # os.makedirs(data_dir, exist_ok=True)
    trj_idx_list = np.array(new_get_trj_idx(dataset))  # (Ntrj, 2), start_idx, end_idx
    labeler_info = np.zeros(len(trj_idx_list) - 1)  # (Ntrj - 1,), zeros -> unfiltered
    valid_trj_indices = np.arange(len(trj_idx_list) - 1)[np.logical_not(labeler_info)]

    trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1  # (Ntrj,), traj len
    assert max(trj_len_list) >= seg_len

    total_reward_seq_1 = np.zeros((total_pairs, seg_len))
    total_reward_seq_2 = np.zeros((total_pairs, seg_len))

    ob_dim = dataset["observations"].shape[-1]
    total_obs_seq_1 = np.zeros((total_pairs, seg_len, ob_dim))
    total_obs_seq_2 = np.zeros((total_pairs, seg_len, ob_dim))
    total_next_obs_seq_1 = np.zeros((total_pairs, seg_len, ob_dim))
    total_next_obs_seq_2 = np.zeros((total_pairs, seg_len, ob_dim))

    act_dim = dataset["actions"].shape[-1]
    total_act_seq_1 = np.zeros((total_pairs, seg_len, act_dim))
    total_act_seq_2 = np.zeros((total_pairs, seg_len, act_dim))

    total_timestep_1 = np.zeros((total_pairs, seg_len), dtype=np.int32)
    total_timestep_2 = np.zeros((total_pairs, seg_len), dtype=np.int32)

    start_indices_1 = np.zeros(total_pairs, dtype=np.int32)
    start_indices_2 = np.zeros(total_pairs, dtype=np.int32)
    time_indices_1 = np.zeros(total_pairs, dtype=np.int32)
    time_indices_2 = np.zeros(total_pairs, dtype=np.int32)
    traj_indices_1 = np.zeros(total_pairs, dtype=np.int32)
    traj_indices_2 = np.zeros(total_pairs, dtype=np.int32)

    # query_path = os.path.join(data_dir, f"queries_num{_num_query}_q{seg_len}_s{set_sz}")

    used_pair_keys = set()

    for set_idx in tqdm(range(_num_query), desc="get query sets"):
        set_segment_keys = set()
        for within_set_idx in range(set_sz):
            pair_slot = set_idx * set_sz + within_set_idx
            first_segment = None

            for _ in range(max_sampling_attempts_per_pair):
                candidate_first = _sample_segment(
                    dataset,
                    trj_idx_list,
                    trj_len_list,
                    valid_trj_indices,
                    seg_len,
                    skip_flag=0,
                    first_reward_sum=None,
                )  # dict:
                if candidate_first is None:
                    continue
                if (
                    ensure_unique_segments_within_set
                    and candidate_first["seg_key"] in set_segment_keys
                ):
                    continue
                first_segment = candidate_first
                break

            if first_segment is None:
                raise RuntimeError(
                    "Failed to sample the first segment for a query pair. "
                    "Try reducing set_len / num_query or disabling "
                    "ensure_unique_segments_within_set."
                )

            second_segment = None
            for _ in range(max_sampling_attempts_per_pair):
                candidate_second = _sample_segment(
                    dataset,
                    trj_idx_list,
                    trj_len_list,
                    valid_trj_indices,
                    seg_len,
                    skip_flag=skip_flag,
                    first_reward_sum=first_segment["reward_sum"],
                )
                if candidate_second is None:
                    continue
                if candidate_second["seg_key"] == first_segment["seg_key"]:
                    continue
                if (
                    ensure_unique_segments_within_set
                    and candidate_second["seg_key"] in set_segment_keys
                ):
                    continue
                pair_key = _pair_key(
                    first_segment["seg_key"], candidate_second["seg_key"]
                )
                if ensure_unique_pairs and pair_key in used_pair_keys:
                    continue
                second_segment = candidate_second
                break

            if second_segment is None:
                raise RuntimeError(
                    "Failed to sample a unique second segment for a query pair. "
                    "Try reducing num_query / set_len, relaxing uniqueness, or "
                    "increasing max_sampling_attempts_per_pair."
                )

            pair_key = _pair_key(first_segment["seg_key"], second_segment["seg_key"])
            used_pair_keys.add(pair_key)
            if ensure_unique_segments_within_set:
                set_segment_keys.add(first_segment["seg_key"])
                set_segment_keys.add(second_segment["seg_key"])

            start_indices_1[pair_slot] = first_segment["start_idx"]
            start_indices_2[pair_slot] = second_segment["start_idx"]
            time_indices_1[pair_slot] = first_segment["time_idx"]
            time_indices_2[pair_slot] = second_segment["time_idx"]
            traj_indices_1[pair_slot] = first_segment["trj_idx"]
            traj_indices_2[pair_slot] = second_segment["trj_idx"]

            total_reward_seq_1[pair_slot] = first_segment["reward_seq"]
            total_reward_seq_2[pair_slot] = second_segment["reward_seq"]
            total_obs_seq_1[pair_slot] = first_segment["obs_seq"]
            total_obs_seq_2[pair_slot] = second_segment["obs_seq"]
            total_next_obs_seq_1[pair_slot] = first_segment["next_obs_seq"]
            total_next_obs_seq_2[pair_slot] = second_segment["next_obs_seq"]
            total_act_seq_1[pair_slot] = first_segment["act_seq"]
            total_act_seq_2[pair_slot] = second_segment["act_seq"]
            total_timestep_1[pair_slot] = first_segment["timestep_seq"]
            total_timestep_2[pair_slot] = second_segment["timestep_seq"]

    seg_reward_1 = total_reward_seq_1.copy()
    seg_reward_2 = total_reward_seq_2.copy()

    batch = {
        "labels": get_labels(seg_reward_1, seg_reward_2).reshape(_num_query, set_sz, 1),
        "observations": total_obs_seq_1.reshape(_num_query, set_sz, seg_len, ob_dim),
        "next_observations": total_next_obs_seq_1.reshape(
            _num_query, set_sz, seg_len, ob_dim
        ),
        "actions": total_act_seq_1.reshape(_num_query, set_sz, seg_len, act_dim),
        "observations_2": total_obs_seq_2.reshape(_num_query, set_sz, seg_len, ob_dim),
        "next_observations_2": total_next_obs_seq_2.reshape(
            _num_query, set_sz, seg_len, ob_dim
        ),
        "actions_2": total_act_seq_2.reshape(_num_query, set_sz, seg_len, act_dim),
        # metadata, probably not needed
        "timestep_1": total_timestep_1.reshape(_num_query, set_sz, seg_len),
        "timestep_2": total_timestep_2.reshape(_num_query, set_sz, seg_len),
        "start_indices": start_indices_1.reshape(_num_query, set_sz),
        "start_indices_2": start_indices_2.reshape(_num_query, set_sz),
        "time_indices": time_indices_1.reshape(_num_query, set_sz),
        "time_indices_2": time_indices_2.reshape(_num_query, set_sz),
        "traj_indices": traj_indices_1.reshape(_num_query, set_sz),
        "traj_indices_2": traj_indices_2.reshape(_num_query, set_sz),
    }

    with open(query_path, "wb") as fp:
        pickle.dump(batch, fp)

    return batch, query_path


def get_labels(seg_reward_1, seg_reward_2):
    """
    Args:
        seg_reward_1: (set_size, T)
        seg_reward_2: (set_size, T)
    Returns:
        binary_label: (set_size, 1)
    """
    seg_return1 = np.sum(seg_reward_1, axis=-1)  # (S,T) -> (S,)
    seg_return2 = np.sum(seg_reward_2, axis=-1)  # (S,T) -> (S,)
    binary_label = (seg_return1 > seg_return2).reshape(-1, 1).astype(np.float32)
    return binary_label  # (S, 1)
