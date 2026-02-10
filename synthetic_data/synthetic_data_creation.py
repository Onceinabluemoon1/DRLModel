import argparse
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import scipy as sp
import tyro

from utils.array import _save_array
from utils.functions import bradley_terry_model, get_legendre_functions
from utils.sampling import sample_annotator_coefficients


@dataclass
class Args:
    """
    Sample from a uniform-component GMM (defaults: K=5, D_h=7).
    The number of components K is the number of annotators.
    The dimension D_h is the number of independent functions that the annotators care about.
    """

    n_samples: int = 100  # number of items
    n_feats: int = 1  # dimension of the input space
    seed: Optional[int] = 0  # random seed


def main() -> None:
    args = tyro.cli(Args)
    rng = (
        np.random.default_rng(args.seed)
        if args.seed is not None
        else np.random.default_rng()
    )
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)
    print(f"n={args.n_samples}, n_feats={args.n_feats}, seed={args.seed}")

    # *sample items from Uniform [-1, 1] -> (n, n_feats)
    x_samples = rng.uniform(-1, 1, size=(args.n_samples, args.n_feats))
    print(f"x_samples: {x_samples.shape}")

    # *sample pairs of items -> (nC2, 2, n_feats)
    x_pairs = list(combinations(x_samples, r=2))
    x_pairs = np.stack([np.stack(pair) for pair in x_pairs])
    print(f"x_pairs: {x_pairs.shape}")

    # * for each pair preference label: sample legendre coeff from GMM
    # 1) sample an annotator (1 of K components) -> (nC2,)
    # 2) sample legendre coeffs for that annotator (Dh-dim Normal) -> (nC2, D_h)
    annotator_coeffs, annotator_indices, info = sample_annotator_coefficients(
        n_samples=len(x_pairs), rng=rng, seed=args.seed
    )
    print(f"K={info['K']}, Dh={info['Dh']}")

    # * for each pair: build reward function from coeffs sampled from GMM
    # reward function: (..., n_feats) -> R, i.e. f: R^n_feats -> R
    reward_fns = get_legendre_functions(annotator_coeffs, args.n_feats, test=False)

    # evaluate rewards of each pair, get pref labels: (nC2, 2) -> (nC2,)
    rewards = np.stack([rew_fn(x_pair) for rew_fn, x_pair in zip(reward_fns, x_pairs)])
    prefs = bradley_terry_model(rewards, rng, beta=1.0)
    print(f"prefs: {prefs.shape}")

    # Default outputs to data/
    out_path = os.path.join(data_dir, "samples.npy")
    comps_path = os.path.join(data_dir, "components.npy")
    x_sample_pairs_path = os.path.join(data_dir, "x_sample_pairs.npy")
    preference_array_path = os.path.join(data_dir, "preference_array.npy")

    _save_array(annotator_coeffs, out_path)
    _save_array(annotator_indices, comps_path)
    _save_array(x_pairs, x_sample_pairs_path)
    _save_array(prefs, preference_array_path)
    print(
        f"Saved data to {out_path}, {comps_path}, {x_sample_pairs_path}, {preference_array_path}"
    )


if __name__ == "__main__":
    main()
