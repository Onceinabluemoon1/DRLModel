
# Sythetic Data

We generate synthetic pairwise preference data from a mixture of annotators. Each annotator is modeled by a latent linear function in a polynomial orthogonal basis. We:
- Sample input features `x` (points in [-1, 1]^M)
- Create all pairs of those points
- Sample annotator-specific function coefficients from a Gaussian mixture (uniform over annotators)
- Build multivariate polynomial functions (Legendre basis) from those coefficients
- Evaluate each function on each pair and record the preference (which item has larger value)

Run synthetic data generation:
```bash
python -m synthetic_data.synthetic_data_creation -n 10 --x-dim 3 --seed 42
```
- `-n`: number of trajectory features (x-samples). The script will form n choose 2 pairs.
- `--x-dim`: input dimensionality M for x; samples are drawn uniformly from [-1, 1]^M.
- `--seed`: optional RNG seed.

Outputs (written to `data/`):
- `samples.npy`: theta_samples of shape (nC2, D_h) where nC2 = n(n-1)/2
- `components.npy`: annotator indices of shape (nC2,)
- `x_sample_pairs.npy`: array of nC2 pairs of x (shape depends on `--x-dim`)
- `preference_array.npy`: array of length nC2 with argmax preferences in each pair
- `function_viz.png`: t-SNE visualization of theta_samples colored by annotator


## Annotator mixture (GMM) configuration

Annotator means and standard deviations live in:
- `synthetic_data/means.csv`  → shape (K, D_h)
- `synthetic_data/stds.csv`   → shape (K, D_h)

Where:
- `K` is the number of annotators (mixture components)
- `D_h` is the number of basis coefficients per annotator

You can edit these files to change the annotator population. The generator infers `K` and `D_h` from these files.


## Function family

Functions are built using multivariate Legendre polynomials:
- We take the first `D_h` multivariate Legendre basis functions (by total degree)
- Each annotator’s coefficient vector `theta` (length `D_h`) defines a function on R^M
- Inputs `x` are in [-1, 1]^M (sampled uniformly)

Internally implemented in `utils/functions.py`:
- `build_legendre_sum(coeffs, d)` builds a function f: R^d → R from coefficients
- `get_legendre_functions(theta_samples, x_dim, test=False)` returns one function per row of `theta_samples`


## End-to-end pipeline (what the script does)

1) Sample `n` inputs `x` in [-1, 1]^M
2) Form all pairs of inputs (nC2 pairs)
3) Sample `theta_samples` and `annotator_indices` using a uniform-component GMM with means/stds from `synthetic_data/`
4) Build one function per sampled theta (Legendre basis)
5) Evaluate both elements of each pair and set preference = argmax
6) Save all outputs to `data/` and write a t-SNE plot of thetas to `data/function_viz.png`


## Example

Generate 20 inputs in 5D and save outputs:
```bash
python -m synthetic_data.synthetic_data_creation -n 20 --x-dim 5 --seed 0
```
After running, inspect:
```python
import numpy as np
thetas = np.load("data/samples.npy")             # (nC2, D_h)
annotators = np.load("data/components.npy")       # (nC2,)
pairs = np.load("data/x_sample_pairs.npy", allow_pickle=True)
prefs = np.load("data/preference_array.npy")      # (nC2,)
```


## Notes
- Edit `synthetic_data/means.csv` and `synthetic_data/stds.csv` to control annotator behavior; shapes must match (K, D_h).
- Inputs are sampled uniformly in [-1, 1]^M to align with orthogonality properties of Legendre polynomials.
- The t-SNE plot is optional; if scikit-learn/matplotlib aren’t available, the script will skip plotting.
