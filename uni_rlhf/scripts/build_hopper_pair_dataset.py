from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


def load_one_pickle(label_dir: Path, pattern: str):
    matches = list(label_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 match for pattern {pattern!r}, "
            f"got {len(matches)}: {[str(m) for m in matches]}"
        )
    with open(matches[0], "rb") as f:
        return pickle.load(f)


def crosses_episode(
    start: int,
    seg_len: int,
    terminals: np.ndarray | None,
    timeouts: np.ndarray | None,
) -> bool:
    end = start + seg_len - 1
    if terminals is not None and terminals[start:end].any():
        return True
    if timeouts is not None and timeouts[start:end].any():
        return True
    return False


def infer_label_mapping(
    rewards: np.ndarray,
    indices_1: np.ndarray,
    indices_2: np.ndarray,
    labels: np.ndarray,
    keep_mask: np.ndarray,
    seg_len: int,
) -> None:
    print("\n=== REWARD-SUM VS LABEL SANITY ===")
    unique_labels = np.unique(labels)
    for target_label in unique_labels:
        diffs = []
        for a, b, y, keep in zip(indices_1, indices_2, labels, keep_mask):
            if not keep or y != target_label:
                continue
            a = int(a)
            b = int(b)
            r1 = float(np.sum(rewards[a : a + seg_len]))
            r2 = float(np.sum(rewards[b : b + seg_len]))
            diffs.append(r1 - r2)

        diffs = np.asarray(diffs, dtype=np.float32)
        if len(diffs) == 0:
            print(f"label {int(target_label)}: no kept examples")
            continue

        print(
            f"label {int(target_label)}: "
            f"count={len(diffs)}, "
            f"mean_r1_minus_r2={diffs.mean():.4f}, "
            f"frac_r1_gt_r2={(diffs > 0).mean():.4f}, "
            f"frac_r1_lt_r2={(diffs < 0).mean():.4f}"
        )


def build_pair_dataset(
    hdf5_path: Path,
    label_dir: Path,
    output_path: Path,
    seg_len: int = 200,
    drop_ties: bool = True,
    drop_boundary_crossers: bool = True,
) -> None:
    indices_1 = np.asarray(load_one_pickle(label_dir, "*indices_1*.pkl"))
    indices_2 = np.asarray(load_one_pickle(label_dir, "*indices_2*.pkl"))
    labels = np.asarray(load_one_pickle(label_dir, "*human_label*.pkl"))

    if not (len(indices_1) == len(indices_2) == len(labels)):
        raise RuntimeError(
            "Mismatched lengths: "
            f"indices_1={len(indices_1)}, "
            f"indices_2={len(indices_2)}, "
            f"labels={len(labels)}"
        )

    with h5py.File(hdf5_path, "r") as f:
        observations = f["observations"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]
        terminals = f["terminals"][:] if "terminals" in f else None
        timeouts = f["timeouts"][:] if "timeouts" in f else None

    num_transitions = len(observations)

    print("=== BASIC INFO ===")
    print("hdf5_path:", hdf5_path)
    print("label_dir:", label_dir)
    print("output_path:", output_path)
    print("num transitions:", num_transitions)
    print("num comparisons:", len(labels))
    print("obs shape:", observations.shape)
    print("act shape:", actions.shape)
    print(
        "label counts:",
        {int(v): int((labels == v).sum()) for v in np.unique(labels)},
    )

    invalid_mask = np.zeros(len(labels), dtype=bool)
    boundary_mask = np.zeros(len(labels), dtype=bool)

    for i, (a, b) in enumerate(zip(indices_1, indices_2)):
        a = int(a)
        b = int(b)

        invalid = (
            a < 0
            or b < 0
            or a + seg_len > num_transitions
            or b + seg_len > num_transitions
        )
        invalid_mask[i] = invalid

        if invalid:
            continue

        if drop_boundary_crossers:
            cross1 = crosses_episode(a, seg_len, terminals, timeouts)
            cross2 = crosses_episode(b, seg_len, terminals, timeouts)
            boundary_mask[i] = cross1 or cross2

    keep_mask = ~invalid_mask
    if drop_boundary_crossers:
        keep_mask &= ~boundary_mask
    if drop_ties:
        keep_mask &= labels != -1

    print("\n=== FILTERING ===")
    print("invalid comparisons removed:", int(invalid_mask.sum()))
    print("boundary-crossing comparisons removed:", int(boundary_mask.sum()))
    print("tie/ambiguous comparisons removed:", int((labels == -1).sum()) if drop_ties else 0)
    print("comparisons kept:", int(keep_mask.sum()))

    infer_label_mapping(
        rewards=rewards,
        indices_1=indices_1,
        indices_2=indices_2,
        labels=labels,
        keep_mask=~invalid_mask & (~boundary_mask if drop_boundary_crossers else True),
        seg_len=seg_len,
    )

    indices_1 = indices_1[keep_mask].astype(np.int64)
    indices_2 = indices_2[keep_mask].astype(np.int64)
    labels = labels[keep_mask].astype(np.int64)

    # Expected mapping from your sanity checks:
    # 1 -> first segment preferred
    # 0 -> second segment preferred
    # -1 -> dropped if drop_ties=True
    if np.any((labels != 0) & (labels != 1)):
        raise RuntimeError(
            f"After filtering, expected only binary labels {{0,1}}, got {np.unique(labels)}"
        )

    x1_obs = np.stack(
        [observations[i : i + seg_len] for i in indices_1],
        axis=0,
    ).astype(np.float32)
    x1_act = np.stack(
        [actions[i : i + seg_len] for i in indices_1],
        axis=0,
    ).astype(np.float32)

    x2_obs = np.stack(
        [observations[i : i + seg_len] for i in indices_2],
        axis=0,
    ).astype(np.float32)
    x2_act = np.stack(
        [actions[i : i + seg_len] for i in indices_2],
        axis=0,
    ).astype(np.float32)

    # Also save simple trajectory-level reward sums for debugging / analysis.
    x1_return = np.asarray(
        [np.sum(rewards[i : i + seg_len]) for i in indices_1],
        dtype=np.float32,
    )
    x2_return = np.asarray(
        [np.sum(rewards[i : i + seg_len]) for i in indices_2],
        dtype=np.float32,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x1_obs=x1_obs,
        x1_act=x1_act,
        x2_obs=x2_obs,
        x2_act=x2_act,
        y=labels,
        x1_return=x1_return,
        x2_return=x2_return,
        indices_1=indices_1,
        indices_2=indices_2,
        seg_len=np.asarray(seg_len, dtype=np.int64),
    )

    print("\n=== SAVED DATASET ===")
    print("saved to:", output_path)
    print("x1_obs:", x1_obs.shape, x1_obs.dtype)
    print("x1_act:", x1_act.shape, x1_act.dtype)
    print("x2_obs:", x2_obs.shape, x2_obs.dtype)
    print("x2_act:", x2_act.shape, x2_act.dtype)
    print("y:", labels.shape, labels.dtype)
    print(
        "binary label counts:",
        {int(v): int((labels == v).sum()) for v in np.unique(labels)},
    )

    print("\n=== FIRST 5 EXAMPLES ===")
    for i in range(min(5, len(labels))):
        print(
            {
                "i": i,
                "start1": int(indices_1[i]),
                "start2": int(indices_2[i]),
                "label": int(labels[i]),
                "x1_return": round(float(x1_return[i]), 3),
                "x2_return": round(float(x2_return[i]), 3),
                "return_diff": round(float(x1_return[i] - x2_return[i]), 3),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cleaned Hopper pairwise preference dataset from D4RL HDF5 + UniRLHF label PKLs."
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=Path("/home/tini/.d4rl/datasets/hopper_medium-v2.hdf5"),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path(
            "/mnt/c/Users/tini/Documents/code/ML/Disentangled_Reward_Learning/uni_rlhf/data/hopper-medium-v2_human_labels"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "/mnt/c/Users/tini/Documents/code/ML/Disentangled_Reward_Learning/uni_rlhf/data/processed/hopper_medium_v2_pairwise_clean.npz"
        ),
    )
    parser.add_argument("--seg-len", type=int, default=200)
    parser.add_argument(
        "--keep-ties",
        action="store_true",
        help="Keep label -1 examples instead of dropping them.",
    )
    parser.add_argument(
        "--keep-boundary-crossers",
        action="store_true",
        help="Keep comparisons whose 200-step windows cross episode boundaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pair_dataset(
        hdf5_path=args.hdf5_path,
        label_dir=args.label_dir,
        output_path=args.output_path,
        seg_len=args.seg_len,
        drop_ties=not args.keep_ties,
        drop_boundary_crossers=not args.keep_boundary_crossers,
    )


if __name__ == "__main__":
    main()