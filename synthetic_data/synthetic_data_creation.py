import argparse

import numpy as np
import os
from itertools import combinations

from utils.array import _save_array
from utils.sampling import sample_annotator_function_coefficients, sample_random_x_values
from utils.functions import get_legendre_functions

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a uniform-component GMM (defaults: K=5, D_h=7). \
        The number of components K is the number of annotators. \
        The dimension D_h is the number of independent functions that the annotators care about."
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to draw (default: 1000).",
    )
    parser.add_argument(
        "--x-dim",
        type=int,
        default=2,
        help="Dimensionality of input x values (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)

    # sample x values from a gaussian distribution with mean 0 and standard deviation 1
    # this is the input to the function
    x_samples = sample_random_x_values(
        n_samples=args.n_samples,
        x_dim=args.x_dim,
        seed=args.seed,
    )

    # x_sample pairs are the pairs of x_samples
    x_sample_pairs = list(combinations(x_samples, 2))
    # Convert to a well-shaped ndarray: (n_pairs, 2, x_dim) or (n_pairs, 2) if x_dim==1
    try:
        x_sample_pairs_arr = np.stack([np.stack(pair) for pair in x_sample_pairs])
    except Exception:
        # Fallback to array conversion if stack fails for any reason
        x_sample_pairs_arr = np.asarray(x_sample_pairs)

    # the number of annotations required is n choose 2, where n is the number of x_samples
    num_annotations = int(args.n_samples * (args.n_samples - 1) / 2)

    # sample the function coefficients for each annotator
    theta_samples, annotator_indices = sample_annotator_function_coefficients(
        n_samples=num_annotations,
        seed=args.seed,
    )

    # based on the theta samples, use legendre polynomials to create lambda functions for each annotation
    reward_functions = get_legendre_functions(theta_samples, args.x_dim, test=False)

    # evaluate the reward functions at the x_sample pairs
    reward_function_values = [f(x_pair) for f, x_pair in zip(reward_functions, x_sample_pairs_arr)]

    # create a preference array from the reward function values
    preference_array = np.array([np.argmax(reward_function_pair) for reward_function_pair in reward_function_values])

    # Default outputs to data/
    out_path = os.path.join(data_dir, "samples.npy")
    comps_path = os.path.join(data_dir, "components.npy")
    x_sample_pairs_path = os.path.join(data_dir, "x_sample_pairs.npy")
    preference_array_path = os.path.join(data_dir, "preference_array.npy")

    _save_array(theta_samples, out_path)
    _save_array(annotator_indices, comps_path)
    _save_array(x_sample_pairs_arr, x_sample_pairs_path)
    _save_array(preference_array, preference_array_path)
    print(f"Saved data to {out_path}, {comps_path}, {x_sample_pairs_path}, {preference_array_path}")

if __name__ == "__main__":
    main()

