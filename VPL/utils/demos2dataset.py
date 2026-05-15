import pathlib
import pickle
from dataclasses import dataclass

import h5py
import numpy as np
import tyro


def get_data(path: str, *, max_horizon: int) -> tuple:
    traj_paths = sorted(pathlib.Path(path).rglob("traj*.pkl"))

    rewards = []
    infos = []
    observations = []
    dones = []
    traj_len = []
    init_poses = []
    last_poses = []
    actions = []
    sl = slice(None, max_horizon)
    for traj_path in traj_paths:
        with open(traj_path, "rb") as f:
            traj = pickle.load(f)
        rewards.append(traj["reward"][sl])
        infos.append(traj["info"][sl])
        observations.append(traj["obs"][sl])
        dones.append(traj["done"][sl])
        traj_len.append(len(traj["reward"][sl]))
        init_poses.append(traj["obs"][0])
        last_poses.append(traj["obs"][-1])
        actions.append(traj["action"][sl])

    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    observations = np.concatenate(observations, axis=0)
    dones = np.concatenate(dones, axis=0)
    traj_len = np.array(traj_len)
    init_poses = np.stack(init_poses, axis=0)
    last_poses = np.stack(last_poses, axis=0)
    return (
        actions,
        rewards,
        observations,
        dones,
        traj_len,
        init_poses,
        last_poses,
        infos,
    )


def reset_data():
    return {
        "observations": [],
        "actions": [],
        "terminals": [],
        "rewards": [],
        "suctions": [],
        "init_poses": [],
    }


def add_data(s, a, r, done, init_poses, _infos):
    # print(s.shape, a.shape, r.shape, done.shape)
    data = reset_data()
    data["observations"] = s
    data["rewards"] = r
    data["terminals"] = done
    data["init_poses"] = init_poses
    data["action_max"] = np.max(a, axis=0)
    data["action_min"] = np.min(a, axis=0)
    normalised_action = (a - data["action_min"]) / (
        data["action_max"] - data["action_min"]
    )
    normalised_action = normalised_action * 2 - 1
    data["actions"] = normalised_action
    return data


def npify(data):
    for k in data:
        if k == "infos":
            continue
        if k == "terminals":
            dtype = np.bool_
        else:
            dtype = np.float32
        data[k] = np.array(data[k], dtype=dtype)


@dataclass(frozen=True)
class DemosToH5Config:
    """Pack traj_*.pkl under rollouts_dir into a D4RL-style gzip HDF5 dataset."""

    rollouts_dir: str  # e.g. …/raw_rollouts_<ts>/raw_rollouts
    output_h5: str
    max_horizon: int = (
        25  # timesteps kept per trajectory (matches legacy demos2dataset [:25])
    )


def main(cfg: DemosToH5Config) -> None:
    (
        actions,
        rewards,
        observations,
        dones,
        _traj_len,
        init_poses,
        _last_poses,
        infos,
    ) = get_data(cfg.rollouts_dir, max_horizon=cfg.max_horizon)
    print(f"{actions.shape=}, {rewards.shape=}, {observations.shape=}, {dones.shape=}")
    data = add_data(observations, actions, rewards, dones, init_poses, infos)
    out_path = pathlib.Path(cfg.output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg.output_h5, "w") as dataset:
        npify(data)
        for k in data:
            dataset.create_dataset(k, data=data[k], compression="gzip")
    infos_path = out_path.with_name("infos.pkl")
    with open(infos_path, "wb") as f:
        pickle.dump(infos, f)
    print(f"Wrote {cfg.output_h5} and {infos_path}")


if __name__ == "__main__":
    main(tyro.cli(DemosToH5Config))
