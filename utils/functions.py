import numpy as np
import scipy as sp


def _compositions(total, parts):
    """Yield all tuples of nonnegative ints of length `parts` summing to `total`."""
    if parts == 1:
        yield (total,)
    else:
        for i in range(total + 1):
            for rest in _compositions(total - i, parts - 1):
                yield (i,) + rest


def _first_N_multi_indices(d, N):
    """First N multi-indices in R^d, ordered by total degree then lexicographic."""
    out = []
    deg = 0
    while len(out) < N:
        for alpha in _compositions(deg, d):
            out.append(alpha)
            if len(out) == N:
                break
        deg += 1
    return out


def build_legendre_sum(coeffs, d):
    """
    Returns a function f(x) where x is R^d (assumed each coord in [-1, 1]),
    and f(x) = sum_{k=0}^{N-1} coeffs[k] * prod_j P_{alpha_k[j]}(x_j),
    using the first N multivariate Legendre basis functions by total degree.
    Args:
        coeffs: (D_h,), the coefficients for the legendre basis functions
            represents the coeff of a single annotator's preference
        d: number of dimensions for item to have preference over
    Returns:
        function
    """
    coeffs = np.asarray(coeffs, dtype=float)  # (D_h,)
    assert coeffs.ndim == 1 and coeffs.size > 0
    assert d >= 1, "Error: Dimension d must be >= 1."
    multi_indices = _first_N_multi_indices(d, coeffs.size)  #

    def f(x):
        """

        Args:
            x: (..., d), the input to the function
        Returns:
            total: (...,) the output of the function
        """

        x = np.asarray(x, dtype=float)
        if x.shape[-1] != d:
            raise ValueError(f"x must have last dimension {d}, got {x.shape}.")
        # Evaluate sum_k c_k * ∏_j P_{alpha_k[j]}(x_j)
        total = 0.0
        for c, alpha in zip(coeffs, multi_indices):
            # product of univariate Legendre polynomials across dimensions
            term = 1.0
            for j, n in enumerate(alpha):
                if n == 0:
                    continue  # P0(x)=1
                term *= sp.special.eval_legendre(n, x[..., j])
            total = total + c * term
        return total

    return f


def test_legendre_sum():
    """
    Test the Legendre sum function.
    This tests the orthogonality and norms of the Legendre sum functions.

    The test is performed by sampling random points from the unit cube and evaluating the functions at those points.
    The inner product of the functions is then estimated by the empirical Gram matrix.
    The theoretical Gram matrix is then computed and compared to the empirical Gram matrix.
    """
    # 10-dimensional input, 7 coefficients
    coeffs1 = np.array([1, 0, 0, 0, 0, 0, 0])  # constant
    coeffs2 = np.array([0, 1, 0, 0, 0, 0, 0])  # P1 on x1
    coeffs3 = np.array([0, 0, 1, 0, 0, 0, 0])  # P1 on x2
    coeffs4 = np.array([0, 0, 0, 1, 0, 0, 0])  # P1 on x3
    coeffs5 = np.array([0, 0, 0, 0, 1, 0, 0])  # P1 on x4
    coeffs6 = np.array([0, 0, 0, 0, 0, 1, 0])  # P1 on x5
    coeffs7 = np.array([0, 0, 0, 0, 0, 0, 1])  # P1 on x6 (7th basis)

    f1 = build_legendre_sum(coeffs1, d=10)
    f2 = build_legendre_sum(coeffs2, d=10)
    f3 = build_legendre_sum(coeffs3, d=10)
    f4 = build_legendre_sum(coeffs4, d=10)
    f5 = build_legendre_sum(coeffs5, d=10)
    f6 = build_legendre_sum(coeffs6, d=10)
    f7 = build_legendre_sum(coeffs7, d=10)

    # Monte Carlo estimate of inner products
    rng = np.random.default_rng(0)
    M = 200000  # big sample to reduce MC noise
    X = rng.uniform(-1, 1, size=(M, 10))

    Y = np.vstack([f1(X), f2(X), f3(X), f4(X), f5(X), f6(X), f7(X)])  # shape (7, M)

    # empirical Gram matrix G_ij = (1/M) * sum Y_i * Y_j
    G = (Y @ Y.T) / M

    # Theoretical Gram (probability measure on [-1,1]^10)
    theo = np.diag([1.0] + [1 / 3] * 6)

    # check orthogonality (off-diagonals ~ 0) with a reasonable tolerance
    offdiag_err = np.abs(G - np.diag(np.diag(G)))
    assert np.all(offdiag_err < 5e-3), (G, "off-diagonals not close to 0")

    # check norms vs theory
    assert np.allclose(G, theo, rtol=0, atol=5e-3), (G, "diagonals off")

    print("Legendre sum test passed")


def get_legendre_functions(coeffs, x_dim, test=False):
    """
    Get the functions for each annotator.
    Args:
        coeffs: (nC2, D_h)
        x_dim: dimension of the input space
        test: whether to test the Legendre sum function
    Returns:
        list of functions
    """
    if test:
        test_legendre_sum()

    return [build_legendre_sum(coeff, d=x_dim) for coeff in coeffs]


def bradley_terry_model(rewards, rng, beta=1.0):
    """
    computes Bernoulli likelihood of the preference relation item 2 > item 1,
    given their rewards.

    p(t2 > t1) = exp(beta * r2) / (exp(beta * r1) + exp(beta * r2))
    assumes rewards[0] = item 1 reward, rewards[1] = item 2 reward

    Args:
        rewards: (nC2, 2), the rewards for each pair
    Returns:
        labesl: (nC2,), the labels for each pair
    """
    assert rewards.ndim == 2 and rewards.shape[1] == 2
    logits = beta * rewards
    probs = np.exp(sp.special.log_softmax(logits, axis=1))
    labels = np.stack([rng.choice(2, p=prob) for prob in probs])
    return labels
