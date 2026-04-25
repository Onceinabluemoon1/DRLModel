from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn import feature_selection, linear_model, metrics, preprocessing


@dataclass
class InfoMECResult:
    infom: float
    infoc: float
    infoe: float
    nmi: np.ndarray
    active_latents: np.ndarray
    source_explicitness: np.ndarray
    source_entropies: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "infom": float(self.infom),
            "infoc": float(self.infoc),
            "infoe": float(self.infoe),
            "active_latents": self.active_latents.astype(bool).tolist(),
            "source_explicitness": self.source_explicitness.astype(float).tolist(),
            "source_entropies": self.source_entropies.astype(float).tolist(),
        }


def process_sources(sources: np.ndarray):
    """
    Process discrete sources into integer labels between 0 and n_classes-1

    Args:
        sources: np.ndarray, shape (B, S); assumed to be discrete.

    Returns:
        np.ndarray, shape (B, S). Processed sources.
    """
    processed_sources = []
    for i in range(sources.shape[1]):
        processed_sources.append(
            preprocessing.LabelEncoder().fit_transform(sources[:, i])  # (B,)
        )
    return np.stack(processed_sources, axis=1)  # (B, S)


def compute_nmi(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool):
    """

    Args:
        sources: Array, shape (B, S). B = num_samples, S = num_sources.
        latents: Array, shape (B, L). B = num_samples, L = num_latents.
        discrete_latents: bool. If True, latents are discrete.

    Outputs:
        Array, shape (S, L). Normalized mutual information between sources and latents.
    """
    # * processing sources (B, S)
    processed_sources = process_sources(sources)  # (B, S)

    # * processing latents (B, L)
    processed_latents = []
    if discrete_latents:
        for l in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.LabelEncoder().fit_transform(latents[:, l])
            )  # (B,)
        processed_latents = np.stack(processed_latents, axis=1)  # (B, L)
    else:
        for l in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.StandardScaler().fit_transform(latents[:, [l]])
            )  # (B, 1)
        processed_latents = np.concatenate(processed_latents, axis=1)  # (B, L)

    # * compute mutual information (S, L)
    S, L = processed_sources.shape[1], processed_latents.shape[1]
    ret = np.empty(shape=(S, L))  # I(s_i; z_j)
    entropy_per_source = np.empty(shape=(S,))  # H(s_i) (S,)
    for s in range(S):
        for l in range(L):
            if discrete_latents:
                ret[s, l] = metrics.mutual_info_score(
                    processed_sources[:, s], processed_latents[:, l]
                )
            else:
                ret[s, l] = feature_selection.mutual_info_classif(
                    processed_latents[:, [l]],
                    processed_sources[:, s],
                    discrete_features=False,
                    n_neighbors=10,
                )

        # normalize source row of MI by its entropy: I(si; z_j) / H(s_i) for all j
        entropy_per_source[s] = metrics.mutual_info_score(
            processed_sources[:, s], processed_sources[:, s]
        )  # H(s_i)
        ret[s, :] /= entropy_per_source[s]
    return ret, entropy_per_source


def compute_infoe(sources, latents, discrete_latents: bool):
    """
    Args:
        sources: Array, shape (B, S).
        latents: Array, shape (B, L).
        discrete_latents: bool. If True, latents are discrete.

    Outputs:
        Array, shape (S,). Normalized predictive information per source.
    """

    def _logistic_regression(X: np.ndarray, y: np.ndarray) -> float:
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2
        assert y.ndim == 1
        # assert X.dtype in [np.float32, np.float64]
        assert y.dtype in [np.int32, np.int64]

        model = linear_model.LogisticRegression(
            penalty=None,
            dual=False,
            tol=1e-4,
            fit_intercept=True,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=100,
            # multi_class="multinomial",
            n_jobs=-1,
        )

        model.fit(X, y)
        y_pred = model.predict_proba(X)
        return metrics.log_loss(y, y_pred)

    # * process sources (B, S)
    processed_sources = process_sources(sources)

    # * process latents (B, L)
    if discrete_latents:
        processed_latents = preprocessing.OneHotEncoder().fit_transform(latents)
    else:
        processed_latents = preprocessing.StandardScaler().fit_transform(latents)

    S = processed_sources.shape[1]
    pred_nmi_per_source = np.empty(shape=(S,))  # I_pred(z -> s_i) (S,)
    for s in range(S):
        # I_pred(z -> s_i) = H_pred(s_i) - H_pred(s_i | z)
        conditional_entropy = _logistic_regression(
            processed_latents, processed_sources[:, s]
        )
        null = np.zeros_like(latents)
        marginal_entropy = _logistic_regression(null, processed_sources[:, s])
        predictive_nmi = (marginal_entropy - conditional_entropy) / marginal_entropy
        pred_nmi_per_source[s] = predictive_nmi

    return np.mean(pred_nmi_per_source), pred_nmi_per_source


def compute_infomec(sources, latents, discrete_latents):
    """
    :param sources: (num_samples, num_sources)
    :param latents: (num_samples, num_latents)
    :param discrete_latents: bool
    :return: dict
    """
    nmi, entropy_per_source = compute_nmi(sources, latents, discrete_latents)  # (S, L)

    # * generate active latents mask:
    latent_ranges = np.max(latents, axis=0) - np.min(latents, axis=0)  # (L,)
    if discrete_latents:
        active_latents = latent_ranges > 0
    else:
        active_latents = latent_ranges > np.max(latent_ranges) / 20

    # * prune out inactive latents
    S = sources.shape[1]
    L = latents.shape[1]
    L_active = np.sum(active_latents)
    pruned_nmi = nmi[:, active_latents]  # (S, L') where L' <= L

    # * compute InfoM (column-sparsity) and InfoC (row-sparsity)
    # 1) sparsity: compute ratio of max MI to sum MI for each latent (source)
    # 2) average sparsity over all latents (sources)
    # 3) normalize to be in range [1/S, 1] ([1/L', 1])
    latent_ratio = np.max(pruned_nmi, axis=0) / np.sum(pruned_nmi, axis=0)  # (L',)
    infom = np.mean(latent_ratio)
    if S > 1:
        infom = (infom - 1 / S) / (1 - 1 / S)
    source_ratio = np.max(pruned_nmi, axis=1) / np.sum(pruned_nmi, axis=1)  # (S,)
    infoc = np.mean(source_ratio)
    if L_active > 1:
        infoc = (infoc - 1 / L_active) / (1 - 1 / L_active)

    infoe, source_explicitness = compute_infoe(sources, latents, discrete_latents)

    return InfoMECResult(
        infom=infom,
        infoe=infoe,
        infoc=infoc,
        nmi=nmi,
        active_latents=active_latents,
        source_explicitness=source_explicitness,
        source_entropies=entropy_per_source,
    )


if __name__ == "__main__":
    B = 4000  # num_samples
    S = 1  # num_sources
    L = S * 2  # num_latents
    K = 10  # num discrete classes per dim
    seed = 0
    latent_method = "copy"

    def generate_latents(sources: np.ndarray, method="copy"):
        """
        Returns:
            latents: np.ndarray, shape (B, L)
            is_discrete: bool
        """

        if method == "copy":
            # good MEC, but watch out for B >> K realm
            latents = sources.copy()
            return latents, True
        elif method == "repeat":
            # good M, E, but bad C,
            assert L % S == 0
            latents = []
            for _ in range(L // S):
                latents.append(sources)
            latents = np.concatenate(latents, axis=1)  # (B, L)
            return latents, True
        elif method == "perturb":
            latents = sources.copy() + rng.normal(size=(B, S)) * 5
            return latents, False
        elif method == "random":
            # bad M, C, but good E
            latents = rng.integers(low=0, high=K, size=(B, L))
            return latents, True

        else:
            raise ValueError(f"Invalid method: {method}")

    print("\n=== Generating sources and latents ===")
    rng = np.random.default_rng(seed)
    sources = rng.integers(low=0, high=K, size=(B, S)).astype(np.float32)  # (B, S)
    latents, is_discrete = generate_latents(sources, method=latent_method)
    L = latents.shape[1]
    print(f"B={B}, S={S}, L={L}, K={K}, seed={seed}")
    print("sources.shape:", sources.shape)
    print("latents.shape:", latents.shape)

    # * compute InfoMEC
    print("\n=== Computing InfoMEC ===")
    result = compute_infomec(sources, latents, is_discrete)
    print(f"(M)odularity: {float(result.infom):.3f}")
    print(f"(E)xplicitness: {float(result.infoe):.3f}")
    print(f"(C)ompactness: {float(result.infoc):.3f}")
    print(f"active_latents: {result.active_latents.sum()} / {latents.shape[1]}")
    print("\n Source-level metrics:")
    print(f"source_entropies: {result.source_entropies}")
    print(f"linear_probe_accuracy_per_source: {result.source_explicitness}")
