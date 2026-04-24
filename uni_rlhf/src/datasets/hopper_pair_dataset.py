from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class HopperPairSample:
    x1_obs: torch.Tensor
    x1_act: torch.Tensor
    x2_obs: torch.Tensor
    x2_act: torch.Tensor
    y: torch.Tensor
    x1_return: torch.Tensor
    x2_return: torch.Tensor
    index_1: torch.Tensor
    index_2: torch.Tensor


class HopperPairDataset(Dataset):
    """
    Pairwise preference dataset built from:
      - D4RL Hopper trajectories
      - UniRLHF pairwise label files

    Expected .npz keys:
      - x1_obs: (N, T, obs_dim)
      - x1_act: (N, T, act_dim)
      - x2_obs: (N, T, obs_dim)
      - x2_act: (N, T, act_dim)
      - y:      (N,)
      - x1_return: (N,)
      - x2_return: (N,)
      - indices_1: (N,)
      - indices_2: (N,)
      - seg_len: scalar
    """

    def __init__(
        self,
        npz_path: str | Path,
        normalize_obs: bool = False,
        normalize_act: bool = False,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        act_mean: Optional[np.ndarray] = None,
        act_std: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.npz_path = Path(npz_path)
        self.dtype = dtype

        if not self.npz_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.npz_path}")

        data = np.load(self.npz_path)

        required_keys = [
            "x1_obs",
            "x1_act",
            "x2_obs",
            "x2_act",
            "y",
            "x1_return",
            "x2_return",
            "indices_1",
            "indices_2",
        ]
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise KeyError(f"Missing required keys in {self.npz_path}: {missing}")

        self.x1_obs = data["x1_obs"].astype(np.float32)
        self.x1_act = data["x1_act"].astype(np.float32)
        self.x2_obs = data["x2_obs"].astype(np.float32)
        self.x2_act = data["x2_act"].astype(np.float32)

        self.y = data["y"].astype(np.int64)
        self.x1_return = data["x1_return"].astype(np.float32)
        self.x2_return = data["x2_return"].astype(np.float32)
        self.indices_1 = data["indices_1"].astype(np.int64)
        self.indices_2 = data["indices_2"].astype(np.int64)

        self.seg_len = int(data["seg_len"]) if "seg_len" in data else self.x1_obs.shape[1]

        self._validate_shapes()

        self.obs_dim = self.x1_obs.shape[-1]
        self.act_dim = self.x1_act.shape[-1]
        self.num_examples = len(self.y)

        self.normalize_obs = normalize_obs
        self.normalize_act = normalize_act

        if self.normalize_obs:
            if obs_mean is None or obs_std is None:
                obs_mean, obs_std = self._compute_obs_stats()
            self.obs_mean = obs_mean.astype(np.float32)
            self.obs_std = np.clip(obs_std.astype(np.float32), 1e-6, None)
        else:
            self.obs_mean = None
            self.obs_std = None

        if self.normalize_act:
            if act_mean is None or act_std is None:
                act_mean, act_std = self._compute_act_stats()
            self.act_mean = act_mean.astype(np.float32)
            self.act_std = np.clip(act_std.astype(np.float32), 1e-6, None)
        else:
            self.act_mean = None
            self.act_std = None

    def _validate_shapes(self) -> None:
        n = len(self.y)

        expected = {
            "x1_obs": self.x1_obs.shape[0],
            "x1_act": self.x1_act.shape[0],
            "x2_obs": self.x2_obs.shape[0],
            "x2_act": self.x2_act.shape[0],
            "x1_return": self.x1_return.shape[0],
            "x2_return": self.x2_return.shape[0],
            "indices_1": self.indices_1.shape[0],
            "indices_2": self.indices_2.shape[0],
        }
        bad = {k: v for k, v in expected.items() if v != n}
        if bad:
            raise ValueError(
                f"Dataset arrays have inconsistent lengths relative to y={n}: {bad}"
            )

        if self.x1_obs.shape != self.x2_obs.shape:
            raise ValueError(
                f"x1_obs and x2_obs shapes differ: {self.x1_obs.shape} vs {self.x2_obs.shape}"
            )
        if self.x1_act.shape != self.x2_act.shape:
            raise ValueError(
                f"x1_act and x2_act shapes differ: {self.x1_act.shape} vs {self.x2_act.shape}"
            )

        if self.x1_obs.ndim != 3:
            raise ValueError(
                f"x1_obs must have shape (N, T, obs_dim), got {self.x1_obs.shape}"
            )
        if self.x1_act.ndim != 3:
            raise ValueError(
                f"x1_act must have shape (N, T, act_dim), got {self.x1_act.shape}"
            )

        if self.x1_obs.shape[1] != self.x1_act.shape[1]:
            raise ValueError(
                f"Observation and action segment lengths differ: "
                f"{self.x1_obs.shape[1]} vs {self.x1_act.shape[1]}"
            )

    def _compute_obs_stats(self) -> tuple[np.ndarray, np.ndarray]:
        obs_all = np.concatenate([self.x1_obs, self.x2_obs], axis=0)  # (2N, T, D)
        mean = obs_all.mean(axis=(0, 1))
        std = obs_all.std(axis=(0, 1))
        return mean, std

    def _compute_act_stats(self) -> tuple[np.ndarray, np.ndarray]:
        act_all = np.concatenate([self.x1_act, self.x2_act], axis=0)  # (2N, T, D)
        mean = act_all.mean(axis=(0, 1))
        std = act_all.std(axis=(0, 1))
        return mean, std

    def _normalize_obs(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize_obs:
            return x
        return (x - self.obs_mean) / self.obs_std

    def _normalize_act(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize_act:
            return x
        return (x - self.act_mean) / self.act_std

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x1_obs = self._normalize_obs(self.x1_obs[idx])
        x1_act = self._normalize_act(self.x1_act[idx])
        x2_obs = self._normalize_obs(self.x2_obs[idx])
        x2_act = self._normalize_act(self.x2_act[idx])

        sample = {
            "x1_obs": torch.tensor(x1_obs, dtype=self.dtype),
            "x1_act": torch.tensor(x1_act, dtype=self.dtype),
            "x2_obs": torch.tensor(x2_obs, dtype=self.dtype),
            "x2_act": torch.tensor(x2_act, dtype=self.dtype),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "x1_return": torch.tensor(self.x1_return[idx], dtype=self.dtype),
            "x2_return": torch.tensor(self.x2_return[idx], dtype=self.dtype),
            "index_1": torch.tensor(self.indices_1[idx], dtype=torch.long),
            "index_2": torch.tensor(self.indices_2[idx], dtype=torch.long),
        }
        return sample

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "num_examples": self.num_examples,
            "seg_len": self.seg_len,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "label_counts": {int(v): int((self.y == v).sum()) for v in np.unique(self.y)},
            "x1_return_mean": float(self.x1_return.mean()),
            "x2_return_mean": float(self.x2_return.mean()),
        }

        if self.normalize_obs:
            stats["obs_mean_shape"] = tuple(self.obs_mean.shape)
            stats["obs_std_shape"] = tuple(self.obs_std.shape)

        if self.normalize_act:
            stats["act_mean_shape"] = tuple(self.act_mean.shape)
            stats["act_std_shape"] = tuple(self.act_std.shape)

        return stats


def hopper_pair_collate_fn(
    batch: list[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate function for fixed-length Hopper segments.

    Since all segments are already fixed length (T=200), this is just stacking.
    """
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.stack([item[k] for item in batch], dim=0)
    return out
