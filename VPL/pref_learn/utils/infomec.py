import warnings
from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn import feature_selection, linear_model, metrics, preprocessing
from sklearn.exceptions import ConvergenceWarning


@dataclass
class InfoMECResult:
    infom: float
    infoc: float
    infoe: float
    max_nmi: float
    nmi: np.ndarray
    varies_latents: np.ndarray
    inform_latents: np.ndarray
    active_latents: np.ndarray
    source_explicitness: np.ndarray
    source_entropies: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "infom": float(self.infom),
            "infoc": float(self.infoc),
            "infoe": float(self.infoe),
            "max_nmi": float(self.max_nmi),
            "varies_latents": self.varies_latents.astype(bool).tolist(),
            "inform_latents": self.inform_latents.astype(bool).tolist(),
            "active_latents": self.active_latents.astype(bool).tolist(),
            "source_explicitness": self.source_explicitness.astype(float).tolist(),
            "source_entropies": self.source_entropies.astype(float).tolist(),
        }


def process_sources(sources: np.ndarray):
    """
    Process discrete sources into integer labels between 0 and n_classes-1

    Args:
        sources: np.ndarray, shape (B, ns); assumed to be discrete.

    Returns:
        np.ndarray, shape (B, ns). Processed sources.
    """
    processed_sources = []
    ns = sources.shape[1]
    for s in range(ns):  # ns
        processed_sources.append(
            preprocessing.LabelEncoder().fit_transform(sources[:, s])  # (B,)
        )
    return np.stack(processed_sources, axis=1)  # (B, ns)


def compute_nmi(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool):
    """

    Args:
        sources: Array, shape (B, ns). B = num_samples, ns = num_sources.
        latents: Array, shape (B, nz). B = num_samples, nz = num_latents.
        discrete_latents: bool. If True, latents are discrete.

    Outputs:
        Array, shape (ns, nz). Normalized mutual information between sources and latents.
    """
    # * processing sources (B, ns)
    processed_sources = process_sources(sources)  # (B, ns)

    # * processing latents (B, nz)
    processed_latents = []
    if discrete_latents:
        for z in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.LabelEncoder().fit_transform(latents[:, z])
            )  # (B,)
        processed_latents = np.stack(processed_latents, axis=1)  # (B, nz)
    else:
        for z in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.StandardScaler().fit_transform(latents[:, [z]])
            )  # (B, 1)
        processed_latents = np.concatenate(processed_latents, axis=1)  # (B, nz)

    # * compute mutual information (ns, nz)
    ns, nz = processed_sources.shape[1], processed_latents.shape[1]
    out = np.empty(shape=(ns, nz))  # I(s_i; z_j)
    source_entropies = np.empty(shape=(ns,))  # H(s_i) (ns,)
    for s in range(ns):
        for z in range(nz):
            if discrete_latents:
                out[s, z] = metrics.mutual_info_score(
                    processed_sources[:, s],  # (B,)
                    processed_latents[:, z],  # (B,)
                )
            else:
                out[s, z] = feature_selection.mutual_info_classif(
                    processed_latents[:, [z]],  # (B, 1)
                    processed_sources[:, s],  # (B,)
                    discrete_features=False,
                    n_neighbors=10,
                )[0]

        # normalize source row of MI by its entropy: I(si; z_j) / H(s_i) for all j
        source_entropies[s] = metrics.mutual_info_score(
            processed_sources[:, s], processed_sources[:, s]
        )  # H(s_i)
        out[s, :] /= source_entropies[s]
    return out, source_entropies


def compute_infoe(sources, latents, discrete_latents: bool):
    """
    Args:
        sources: Array, shape (B, ns).
        latents: Array, shape (B, nz).
        discrete_latents: bool. If True, latents are discrete.

    Outputs:
        Array, shape (ns,). Normalized predictive information per source.
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
            warm_start=False,
            # multi_class="multinomial",
            n_jobs=1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X, y)
        y_pred = model.predict_proba(X)
        return metrics.log_loss(y, y_pred)

    # * process sources (B, ns)
    processed_sources = process_sources(sources)

    # * process latents (B, nz)
    if discrete_latents:
        processed_latents = preprocessing.OneHotEncoder().fit_transform(latents)
    else:
        processed_latents = preprocessing.StandardScaler().fit_transform(latents)

    ns = processed_sources.shape[1]
    pred_nmi_per_source = np.empty(shape=(ns,))  # I_pred(z -> s_i) (ns,)
    for s in range(ns):
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
    :param sources: (N, n_sources)
    :param latents: (N, n_latents)
    :param discrete_latents: bool
    :return: dict
    """
    nmi, source_entropies = compute_nmi(sources, latents, discrete_latents)  # (ns, nz)
    ns, nz = sources.shape[1], latents.shape[1]

    # * generate active latents mask for InfoM / InfoC
    # varies: latent dimension actually varies (not posterior-collapsed)
    # inform: latent carries non-zero  NMI about at least one source
    latent_ranges = np.max(latents, axis=0) - np.min(latents, axis=0)  # (L,)
    if discrete_latents:
        mask_varies = latent_ranges > 0
    else:
        mask_varies = latent_ranges > np.max(latent_ranges) / 20

    mask_inform = np.max(nmi, axis=0) > 0  # (L,)
    mask_active = mask_varies & mask_inform  # (L,)

    # * prune out inactive latents
    n_active = np.sum(mask_active)
    pruned_nmi = nmi[:, mask_active]  # (S, L') where L' <= L

    # * compute InfoM (column-sparsity) and InfoC (row-sparsity)
    if n_active > 0:
        # InfoM: average column sparsity
        latent_max = np.max(pruned_nmi, axis=0)  # (L',)
        latent_sum = np.sum(pruned_nmi, axis=0)  # (L',)
        latent_ratio = np.divide(
            latent_max,  # (L',)
            latent_sum,  # (L',)
            out=np.zeros_like(latent_sum, dtype=float),
            where=latent_sum > 0,
        )
        infom = np.mean(latent_ratio)
        if ns > 1:
            infom = (infom - 1 / ns) / (1 - 1 / ns)

        # InfoC: average row sparsity
        source_max = np.max(pruned_nmi, axis=1)  # (S,)
        source_sum = np.sum(pruned_nmi, axis=1)  # (S,)
        source_ratio = np.divide(
            source_max,  # (S,)
            source_sum,  # (S,)
            out=np.zeros_like(source_sum, dtype=float),
            where=source_sum > 0,
        )
        infoc = np.mean(source_ratio)
        if n_active > 1:
            infoc = (infoc - 1 / n_active) / (1 - 1 / n_active)
    else:
        infom = 0.0
        infoc = 0.0

    infoe, source_explicitness = compute_infoe(sources, latents, discrete_latents)

    return InfoMECResult(
        infom=infom,
        infoe=infoe,
        infoc=infoc,
        max_nmi=np.max(nmi),
        nmi=nmi,
        varies_latents=mask_varies,
        inform_latents=mask_inform,
        active_latents=mask_active,
        source_explicitness=source_explicitness,
        source_entropies=source_entropies,
    )


if __name__ == "__main__":
    B = 4000  # num_samples
    ns = 1  # num_sources
    nz = ns * 2  # num_latents
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
            assert nz % ns == 0
            latents = []
            for _ in range(nz // ns):
                latents.append(sources)
            latents = np.concatenate(latents, axis=1)  # (B, L)
            return latents, True
        elif method == "perturb":
            latents = sources.copy() + rng.normal(size=(B, ns)) * 5
            return latents, False
        elif method == "random":
            # bad M, C, but good E
            latents = rng.integers(low=0, high=K, size=(B, nz))
            return latents, True

        else:
            raise ValueError(f"Invalid method: {method}")

    print("\n=== Generating sources and latents ===")
    rng = np.random.default_rng(seed)
    sources = rng.integers(low=0, high=K, size=(B, ns)).astype(np.float32)  # (B, S)
    latents, is_discrete = generate_latents(sources, method=latent_method)
    nz = latents.shape[1]
    print(f"B={B}, S={ns}, L={nz}, K={K}, seed={seed}")
    print("sources.shape:", sources.shape)
    print("latents.shape:", latents.shape)

    # * compute InfoMEC
    print("\n=== Computing InfoMEC ===")
    result = compute_infomec(sources, latents, is_discrete)
    print(f"(M)odularity: {float(result.infom):.3f}")
    print(f"(E)xplicitness: {float(result.infoe):.3f}")
    print(f"(C)ompactness: {float(result.infoc):.3f}")
    n_varies = int(np.sum(result.varies_latents))
    n_inform = int(np.sum(result.inform_latents))
    n_active = int(np.sum(result.active_latents))
    print(
        f"{{varies, inform, active}} / total: "
        f"{{{n_varies},{n_inform},{n_active}}} / {latents.shape[1]}"
    )
    print("\n Source-level metrics:")
    print(f"source_entropies: {result.source_entropies}")
    print(f"linear_probe_accuracy_per_source: {result.source_explicitness}")
