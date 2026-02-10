import os
from typing import Optional, Tuple, Union

import numpy as np

from utils.array import _load_array
from utils.viz import _run_tsne_and_save


def sample_random_x_values(
    n_samples: int,
    n_feats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample x values from a Uniform distribution on the unit cube [-1, 1]^x_dim.
    Returns shape (n_samples, x_dim).
    """
    return rng.uniform(-1, 1, size=(n_samples, n_feats))


def sample_annotator_coefficients(
    n_samples: int,
    rng: np.random.Generator,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from a (potentially) N-dimensional Gaussian Mixture Model with UNIFORM component selection.

    - Let K = number of annotators, D = dimensionality.
    - The uniform component selection implies that we randomly choose an annotator to sample from.
    - Each annotator k is a D-dimensional Normal(mean_k, diag(std_k^2)) when stds are scalar per annotator.

    Returns:
        theta_samples: shape (n_samples, D)
        annotator_indices: shape (n_samples,), the chosen annotator per sample
    """
    mean_std_data_dir = "synthetic_data/"
    data_dir = "data/"
    means_file = os.path.join(mean_std_data_dir, "means.csv")
    stds_file = os.path.join(mean_std_data_dir, "stds.csv")

    # Load means and stds if provided, infer and validate K and D
    means = _load_array(means_file)  # (K, D)
    assert means.ndim == 2, f"Error: {means_file} must be a 2D array of shape (K,D)."
    K, D = means.shape

    stds = _load_array(stds_file)
    assert stds.ndim == 2, f"Error: {stds_file} must be a 2D array of shape (K,D)."
    assert stds.shape == (K, D), f"Error: {stds_file} must have shape (K,D)."

    # (n_samples, D) and (n_samples,)
    theta_samples, annotator_indices = sample_gmm(
        n_samples=n_samples,
        means=means,
        stds=stds,
        rng=rng,
    )

    # t-SNE visualization (saved to data/)
    tsne_png = os.path.join(data_dir, "function_viz.png")
    # sample 1000 points for t-SNE visualization
    tsne_sample_size = min(1000, n_samples)
    tsne_sample_indices = rng.choice(n_samples, size=tsne_sample_size, replace=False)

    tsne_title = (
        "t-SNE of GMM Samples (colored by Annotator) \n"
        + "The number of annotators is K = "
        + str(K)
        + "\n"
        + " and the dimension of the latent space is D_h = "
        + str(D)
        + "\n"
        + "The number of annotations performed = "
        + str(n_samples)
    )

    _run_tsne_and_save(
        samples=theta_samples[tsne_sample_indices],
        annotator_indices=annotator_indices[tsne_sample_indices],
        out_png_path=tsne_png,
        seed=seed,
        title=tsne_title,
    )
    print(f"t-SNE plot saved to: {tsne_png}")

    return theta_samples, annotator_indices


def sample_gmm(
    n_samples: int,
    means: np.ndarray,
    stds: Union[float, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from a (potentially) N-dimensional Gaussian Mixture Model with UNIFORM component selection.

    - Let K = number of components, D = dimensionality.
    - Each component k is a D-dimensional Normal(mean_k, diag(std_k^2)) when stds are scalar per component.
    - The component (mode) is chosen uniformly among the components.

    Returns:
        samples: shape (n_samples, D)
        component_indices: shape (n_samples,), the chosen component per sample
    """
    means_arr = np.asarray(means, dtype=float)
    assert means_arr.ndim == 2, "Error: means must be a 2D array of shape (K,D)."
    K, D = means_arr.shape

    # Normalize stds into shape (K, D) for broadcasted sampling
    if np.isscalar(stds):
        stds_arr = np.full((K, D), float(stds), dtype=float)
    else:
        stds_arr_raw = np.asarray(stds, dtype=float)
        if stds_arr_raw.ndim == 1:
            if stds_arr_raw.shape[0] != K:
                raise ValueError("stds length must match number of components K.")
            stds_arr = np.repeat(stds_arr_raw[:, None], D, axis=1)
        elif stds_arr_raw.ndim == 2:
            if stds_arr_raw.shape != (K, D):
                raise ValueError("stds must be shape (K,) or (K, D) or a scalar.")
            stds_arr = stds_arr_raw
        else:
            raise ValueError("stds must be scalar, shape (K,), or shape (K, D).")

    assert np.all(stds_arr > 0), "All stds must be positive."

    # Uniformly sample an annotator for each annotation
    component_indices = rng.integers(0, K, size=n_samples)  # (n_samples,)

    samples = np.empty((n_samples, D), dtype=float)
    for k in range(K):
        mask = component_indices == k
        count = int(mask.sum())
        if count == 0:
            continue
        samples[mask] = rng.normal(loc=means_arr[k], scale=stds_arr[k], size=(count, D))

    return samples, component_indices
