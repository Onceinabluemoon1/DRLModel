import math
import pickle
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from pref_learn.utils.data_utils import get_labels
from torch.utils.data import DataLoader, Dataset


def logvar_to_std(logvar: torch.Tensor) -> torch.Tensor:
    return torch.exp(0.5 * logvar)


def _merge_infomec_fields(
    *,
    batch: dict,
    infomec_path: str | Path | None,
) -> tuple[np.ndarray, list[int]]:
    """
    Ensure ``batch`` has InfoMEC annotator metadata (merge from ``infomec_path`` if needed).

    Returns:
        source_variable: ``(Q, n_sources)`` int array — discrete GT source index per query
            set (one row per preference query, shared across all A pairs in that set).
            Example (Sort 4-goals): ``n_sources=2``, entries in ``{0,1,2}`` per axis
            (no pref / top-or-left / bottom-or-right).
        source_dims: ``[n_classes_0, n_classes_1, ...]`` — one-hot width per source axis.
            Example: ``[3, 3]`` for two 3-way discrete prefs.
    """
    infomec_keys = (
        "source_variable",
        "source_dims",
        "source_names",
        "source_types",
        "source_mode",
        "source_mode_names",
    )
    if "source_variable" not in batch:
        if infomec_path is None or not str(infomec_path).strip():
            raise ValueError(
                "encode_sources=True but dataset has no 'source_variable'. "
                "Create with store_source_mode=True or pass --infomec_dataset_path."
            )
        with open(infomec_path, "rb") as fp:
            infomec_batch = pickle.load(fp)
        for key in infomec_keys:
            if key in infomec_batch:
                batch[key] = infomec_batch[key]
    if "source_variable" not in batch:
        raise ValueError(
            "encode_sources=True but no 'source_variable' in dataset or infomec path."
        )
    source_dims_raw = batch.get("source_dims")
    if source_dims_raw is None:
        n_sources = int(batch["source_variable"].shape[-1])
        source_dims = [int(np.max(batch["source_variable"])) + 1] * n_sources
    else:
        source_dims = [int(d) for d in source_dims_raw]
    return np.asarray(batch["source_variable"]), source_dims


def get_datasets(
    query_path: str | Path,
    observation_dim: int,
    action_dim: int,
    batch_size: int,
    set_size: int = -1,  # -1 means use all queries in the query set
    *,
    encode_sources: bool = False,
    infomec_path: str | Path | None = None,
):
    with open(query_path, "rb") as fp:
        batch = pickle.load(fp)

    batch["observations"] = batch["observations"][..., :observation_dim]
    batch["observations_2"] = batch["observations_2"][..., :observation_dim]
    assert batch["actions"].shape[-1] == action_dim
    if set_size < 0:
        set_size = batch["observations"].shape[1]

    source_variable = None
    source_dims: list[int] | None = None
    if encode_sources:
        source_variable, source_dims = _merge_infomec_fields(
            batch=batch,
            infomec_path=infomec_path,
        )

    eval_data_size = int(0.1 * len(batch["observations"]))
    train_data_size = len(batch["observations"]) - eval_data_size

    train_batch = {
        "observations": batch["observations"][:train_data_size, :set_size],
        "actions": batch["actions"][:train_data_size, :set_size],
        "observations_2": batch["observations_2"][:train_data_size, :set_size],
        "actions_2": batch["actions_2"][:train_data_size, :set_size],
        "labels": batch["labels"][:train_data_size, :set_size],
    }
    eval_batch = {
        "observations": batch["observations"][train_data_size:, :set_size],
        "actions": batch["actions"][train_data_size:, :set_size],
        "observations_2": batch["observations_2"][train_data_size:, :set_size],
        "actions_2": batch["actions_2"][train_data_size:, :set_size],
        "labels": batch["labels"][train_data_size:, :set_size],
    }
    if encode_sources:
        assert source_variable is not None
        train_batch["source_variable"] = source_variable[:train_data_size]
        eval_batch["source_variable"] = source_variable[train_data_size:]

    train_dataset = PreferenceDataset(
        train_batch, encode_sources=encode_sources, source_dims=source_dims
    )
    eval_dataset = PreferenceDataset(
        eval_batch, encode_sources=encode_sources, source_dims=source_dims
    )
    kwargs = {"num_workers": 1, "pin_memory": True}
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    _, _, segment_len, obs_dim = batch["observations"].shape  # (Q, S, T, O)
    return (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        set_size,
        segment_len,
        obs_dim,
        source_dims,
    )


class PreferenceDataset(Dataset):
    def __init__(
        self,
        pref_dataset,
        *,
        encode_sources: bool = False,
        source_dims: list[int] | None = None,
    ):
        self.pref_dataset = pref_dataset
        self.encode_sources = bool(encode_sources)
        self.source_dims = source_dims

    def __len__(self):
        return len(self.pref_dataset["observations"])

    def __getitem__(self, idx):
        observations = self.pref_dataset["observations"][idx]
        observations_2 = self.pref_dataset["observations_2"][idx]
        labels = self.pref_dataset["labels"][idx]
        out = dict(
            observations=observations,
            observations_2=observations_2,
            labels=labels,
        )
        if self.encode_sources:
            out["source_variable"] = self.pref_dataset["source_variable"][idx]
        return out

    def get_mode_data(self, batch_size):
        batch_size = min(batch_size, len(self))
        idxs = np.random.choice(range(len(self)), size=batch_size, replace=False)
        out = dict(
            observations=self.pref_dataset["observations"][idxs],
            observations_2=self.pref_dataset["observations_2"][idxs],
        )
        if self.encode_sources:
            out["source_variable"] = self.pref_dataset["source_variable"][idxs]
        return out, batch_size


class Annealer:
    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_latent(batch, env, reward_model, mode, num_samples):
    # obs_dim = env.reward_observation_space.shape[0]
    set_size = batch["observations"].shape[1]
    obs1 = batch["observations"]  # (B, S, T, O)
    obs2 = batch["observations_2"]  # (B, S, T, O)
    obs1_flat = rearrange(obs1, "b s t o -> (b s) t o")  # (B*S, T, O)
    obs2_flat = rearrange(obs2, "b s t o -> (b s) t o")  # (B*S, T, O)
    rewards1 = env.compute_reward(obs1_flat, mode)  # ((B*S), T, ...)
    rewards2 = env.compute_reward(obs2_flat, mode)

    rewards1 = rearrange(rewards1, "(b s) t ... -> b s t ...", b=num_samples, s=set_size)
    rewards2 = rearrange(rewards2, "(b s) t ... -> b s t ...", b=num_samples, s=set_size)

    labels = get_labels(rewards1, rewards2)
    # todo: why wasn't this needed before?
    labels = rearrange(labels, "(b s) ... -> b s ...", b=num_samples, s=set_size)
    device = next(reward_model.parameters()).device
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    source = None
    if "source_variable" in batch:
        source = torch.from_numpy(batch["source_variable"]).long().to(device)
    elif getattr(reward_model, "encode_sources", False):
        raise ValueError("encode_sources=True but source is None in batch")
    with torch.no_grad():
        mean, _ = reward_model.encode(
            obs1,
            obs2,
            labels,
            source=source,
        )
    return mean.cpu().numpy()


def get_posterior(env, reward_model, dataset, mode, num_samples):
    batch, num_samples = dataset.get_mode_data(num_samples)
    return get_latent(batch, env, reward_model, mode, num_samples)


def get_all_posterior(env, reward_model, dataset, num_samples):
    means = []
    for mode in range(env.get_num_modes()):
        means.append(get_posterior(env, reward_model, dataset, mode, num_samples))
    return np.stack(means, axis=0)


def get_biased(env, reward_model, dataset=None):
    means = []
    if dataset:
        batch, _ = dataset.get_mode_data(1)
    else:
        obs1, obs2 = env.get_biased_data(reward_model.annotation_size)
        batch = dict(
            observations=obs1[None, :, None],
            observations_2=obs2[None, :, None],
        )
    # import pdb; pdb.set_trace()
    for mode in range(env.get_num_modes()):
        means.append(get_latent(batch, env, reward_model, mode, 1))
    return np.stack(means, axis=0)
