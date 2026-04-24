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
    linear_probe_accuracy_per_source: np.ndarray
    source_entropies: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "infom": float(self.infom),
            "infoc": float(self.infoc),
            "infoe": float(self.infoe),
            "active_latents": self.active_latents.astype(bool).tolist(),
            "linear_probe_accuracy_per_source": self.linear_probe_accuracy_per_source.astype(
                float
            ).tolist(),
            "source_entropies": self.source_entropies.astype(float).tolist(),
        }


def process_sources(sources):
    processed_sources = []
    for i in range(sources.shape[1]):
        processed_sources.append(
            preprocessing.LabelEncoder().fit_transform(sources[:, i])
        )
    return np.stack(processed_sources, axis=1)


def _process_latents_for_nmi(latents: np.ndarray, discrete_latents: bool) -> np.ndarray:
    processed_latents = []
    if discrete_latents:
        for j in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.LabelEncoder().fit_transform(latents[:, j])
            )
        processed_latents = np.stack(processed_latents, axis=1)
    else:
        for j in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.StandardScaler().fit_transform(latents[:, j][:, None])
            )
        processed_latents = np.concatenate(processed_latents, axis=1)
    return processed_latents


def _process_latents_for_infoe(latents: np.ndarray, discrete_latents: bool):
    if discrete_latents:
        return preprocessing.OneHotEncoder().fit_transform(latents)
    return preprocessing.StandardScaler().fit_transform(latents)


def _column_entropies(processed_sources: np.ndarray) -> np.ndarray:
    entropies = np.empty(processed_sources.shape[1], dtype=np.float64)
    for i in range(processed_sources.shape[1]):
        entropies[i] = float(
            metrics.mutual_info_score(processed_sources[:, i], processed_sources[:, i])
        )
    return entropies


def compute_nmi(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False):
    processed_sources = process_sources(sources)
    processed_latents = _process_latents_for_nmi(
        latents, discrete_latents=discrete_latents
    )

    ret = np.empty(
        shape=(processed_sources.shape[1], processed_latents.shape[1]),
        dtype=np.float64,
    )
    entropies = _column_entropies(processed_sources)

    for i in range(processed_sources.shape[1]):
        for j in range(processed_latents.shape[1]):
            if discrete_latents:
                ret[i, j] = float(
                    metrics.mutual_info_score(
                        processed_sources[:, i], processed_latents[:, j]
                    )
                )
            else:
                mi = feature_selection.mutual_info_classif(
                    processed_latents[:, j][:, None],
                    processed_sources[:, i],
                    discrete_features=False,
                    n_neighbors=10,
                )
                ret[i, j] = float(np.asarray(mi).reshape(-1)[0])

        entropy = entropies[i]
        if entropy > 0:
            ret[i, :] /= entropy
        else:
            ret[i, :] = 0.0

    return ret, entropies


def logistic_regression(X, y):
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64]
    assert y.dtype in [np.int32, np.int64]

    model = linear_model.LogisticRegression(
        penalty=None,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=100,
        multi_class="multinomial",
        n_jobs=-1,
    )
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    return metrics.log_loss(y, y_pred)


def compute_infoe(
    sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False
):
    normalized_predictive_information_per_source = []
    linear_probe_accuracy_per_source = []

    processed_sources = process_sources(sources)
    processed_latents = _process_latents_for_infoe(
        latents, discrete_latents=discrete_latents
    )

    for i in range(processed_sources.shape[1]):
        src = processed_sources[:, i].astype(np.int64)

        predictive_conditional_entropy = logistic_regression(processed_latents, src)
        null = np.zeros_like(latents, dtype=latents.dtype)
        marginal_source_entropy = logistic_regression(null, src)

        normalized_predictive_information_per_source.append(
            (marginal_source_entropy - predictive_conditional_entropy)
            / marginal_source_entropy
        )

        probe = linear_model.LogisticRegression(
            penalty=None,
            dual=False,
            tol=1e-4,
            fit_intercept=True,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=100,
            multi_class="multinomial",
            n_jobs=-1,
        )
        probe.fit(processed_latents, src)
        linear_probe_accuracy_per_source.append(
            float(probe.score(processed_latents, src))
        )

    return (
        float(np.mean(normalized_predictive_information_per_source)),
        np.asarray(linear_probe_accuracy_per_source),
    )


def compute_infomec(
    sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False
) -> InfoMECResult:
    """
    Official InfoMEC math, except:
      - when num_sources == 1, remove the modularity normalization term
        instead of dividing by zero.
    """
    nmi, entropies = compute_nmi(sources, latents, discrete_latents)

    latent_ranges = np.max(latents, axis=0) - np.min(latents, axis=0)
    if discrete_latents:
        active_latents = latent_ranges > 0
    else:
        active_latents = latent_ranges > np.max(latent_ranges) / 20

    pruned_nmi = nmi[:, active_latents]
    num_sources = sources.shape[1]
    num_active_latents = int(np.sum(active_latents))

    # InfoM
    ratios_m = np.max(pruned_nmi, axis=0) / np.sum(pruned_nmi, axis=0)
    if num_sources == 1:
        infom = float(np.mean(ratios_m))
    else:
        infom = float((np.mean(ratios_m) - 1 / num_sources) / (1 - 1 / num_sources))

    # InfoC
    ratios_c = np.max(pruned_nmi, axis=1) / np.sum(pruned_nmi, axis=1)
    infoc = float(
        (np.mean(ratios_c) - 1 / num_active_latents) / (1 - 1 / num_active_latents)
    )

    infoe, probe_acc = compute_infoe(sources, latents, discrete_latents)

    return InfoMECResult(
        infom=infom,
        infoe=infoe,
        infoc=infoc,
        nmi=nmi,
        active_latents=active_latents,
        linear_probe_accuracy_per_source=probe_acc,
        source_entropies=entropies,
    )
