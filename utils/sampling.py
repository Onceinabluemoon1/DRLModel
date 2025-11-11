import numpy as np
import os
from typing import Optional, Tuple, Union
from utils.array import _load_array
from utils.viz import _run_tsne_and_save





def sample_random_x_values(
    n_samples: int,
    x_dim: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample x values from a Uniform distribution on the unit cube [-1, 1]^x_dim.
    Returns shape (n_samples, x_dim) when x_dim > 1, else (n_samples,).
    """
    rng = np.random.default_rng(seed)
    if x_dim <= 1:
        return rng.uniform(-1, 1, size=n_samples)
    return rng.uniform(-1, 1, size=(n_samples, x_dim))


def sample_annotator_function_coefficients(
    n_samples: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from a (potentially) N-dimensional Gaussian Mixture Model with UNIFORM component selection.

    - Let K = number of annotators, D = dimensionality.
    - The uniform component selection implies that we randomly choose an annotator to sample from.
    - Each annotator k is a D-dimensional Normal(mean_k, diag(std_k^2)) when stds are scalar per annotator.
    - The annotator is chosen uniformly among the annotators.

    Returns:
        theta_samples: shape (n_samples, D)
        annotator_indices: shape (n_samples,), the chosen annotator per sample
    """
    mean_std_data_dir = "synthetic_data/"
    data_dir = "data/"
    means_file = os.path.join(mean_std_data_dir, "means.csv")
    stds_file = os.path.join(mean_std_data_dir, "stds.csv")
    # Infer K (num components) and D (dimension) from provided files when possible
    means = None
    stds_loaded: Optional[np.ndarray] = None
    K: Optional[int] = None
    D: Optional[int] = None

    # Load means if provided and infer K, D
    means = _load_array(means_file)
    if means.ndim != 2:
        raise SystemExit("Error: --means-file must be a 2D array of shape (K,D).")
    K, D = int(means.shape[0]), int(means.shape[1])

    # Load stds if provided and infer/validate K, D
    stds_loaded = _load_array(stds_file)
    if stds_loaded.ndim == 2:
        k2, d2 = int(stds_loaded.shape[0]), int(stds_loaded.shape[1])
        if k2 != K or d2 != D:
            raise SystemExit(
                f"Error: --stds-file shape ({k2},{d2}) does not match means shape ({K},{D})."
            )
    else:
        raise SystemExit("Error: --stds-file must be a 2D array of shape (K,D).")

    theta_samples, annotator_indices = sample_gmm(
        n_samples=n_samples,
        means=means,
        stds=stds_loaded,
        seed=seed,
    )

    # t-SNE visualization (saved to data/)
    tsne_png = os.path.join(data_dir, "function_viz.png")
    # sample 1000 points for t-SNE visualization
    tsne_sample_size = min(1000, n_samples)
    tsne_sample_indices = np.random.choice(n_samples, size=tsne_sample_size, replace=False)


    tsne_title = "t-SNE of GMM Samples (colored by Annotator) \n" + \
        "The number of annotators is K = " + str(K) + "\n" + \
        " and the dimension of the latent space is D_h = " + str(D) + "\n" + \
        "The number of annotations performed = " + str(n_samples)

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
    seed: Optional[int] = None,
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
    rng = np.random.default_rng(seed)

    means_arr = np.asarray(means, dtype=float)
    if means_arr.ndim != 2:
        raise ValueError("means must be of shape (K, D).")
    num_components, dim = means_arr.shape

    # Normalize stds into shape (K, D) for broadcasted sampling
    if np.isscalar(stds):
        stds_arr = np.full((num_components, dim), float(stds), dtype=float)
    else:
        stds_arr_raw = np.asarray(stds, dtype=float)
        if stds_arr_raw.ndim == 1:
            if stds_arr_raw.shape[0] != num_components:
                raise ValueError("stds length must match number of components K.")
            stds_arr = np.repeat(stds_arr_raw[:, None], dim, axis=1)
        elif stds_arr_raw.ndim == 2:
            if stds_arr_raw.shape != (num_components, dim):
                raise ValueError("stds must be shape (K,) or (K, D) or a scalar.")
            stds_arr = stds_arr_raw
        else:
            raise ValueError("stds must be scalar, shape (K,), or shape (K, D).")

    if np.any(stds_arr <= 0):
        raise ValueError("All stds must be positive.")

    # Uniformly choose which component to sample from for each point
    component_indices = rng.integers(0, num_components, size=n_samples)

    samples = np.empty((n_samples, dim), dtype=float)
    for k in range(num_components):
        mask = component_indices == k
        count = int(mask.sum())
        if count == 0:
            continue
        samples[mask] = rng.normal(
            loc=means_arr[k], scale=stds_arr[k], size=(count, dim)
        )

    return samples, component_indices