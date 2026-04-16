"""InfoMEC utilities.

Core metric computations mirror the official implementation from
`latent_quantization/disentangle/metrics/infomec.py`, while preserving a
repo-friendly dataclass return type and a few extra diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import inspect

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
            "linear_probe_accuracy_per_source": self.linear_probe_accuracy_per_source.astype(float).tolist(),
            "source_entropies": self.source_entropies.astype(float).tolist(),
        }


def _make_logistic_regression(max_iter: int = 100) -> linear_model.LogisticRegression:
    params = inspect.signature(linear_model.LogisticRegression).parameters
    # The official InfoMEC code uses an effectively unregularized multinomial
    # logistic regression. Newer sklearn versions removed `multi_class` and
    # deprecate the old `penalty=None`/`n_jobs` combination, so we keep the
    # official settings when available and otherwise fall back to a large-C
    # lbfgs model that behaves similarly in practice.
    if "multi_class" in params:
        return linear_model.LogisticRegression(
            penalty=None,
            dual=False,
            tol=1e-4,
            fit_intercept=True,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=max_iter,
            multi_class="multinomial",
            n_jobs=-1,
        )
    return linear_model.LogisticRegression(
        C=1e12,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=max_iter,
    )


def process_sources(sources: np.ndarray) -> np.ndarray:
    processed_sources = []
    for i in range(sources.shape[1]):
        processed_sources.append(preprocessing.LabelEncoder().fit_transform(sources[:, i]))
    return np.stack(processed_sources, axis=1).astype(np.int64)


def _process_latents_for_nmi(latents: np.ndarray, discrete_latents: bool) -> np.ndarray:
    processed_latents = []
    if discrete_latents:
        for j in range(latents.shape[1]):
            processed_latents.append(preprocessing.LabelEncoder().fit_transform(latents[:, j]))
        processed_latents = np.stack(processed_latents, axis=1)
    else:
        for j in range(latents.shape[1]):
            processed_latents.append(preprocessing.StandardScaler().fit_transform(latents[:, j][:, None]))
        processed_latents = np.concatenate(processed_latents, axis=1)
    return processed_latents


def _process_latents_for_infoe(latents: np.ndarray, discrete_latents: bool):
    if discrete_latents:
        return preprocessing.OneHotEncoder().fit_transform(latents)
    return preprocessing.StandardScaler().fit_transform(latents)


def _column_entropies(processed_sources: np.ndarray) -> np.ndarray:
    entropies = np.empty(processed_sources.shape[1], dtype=np.float64)
    for i in range(processed_sources.shape[1]):
        entropies[i] = float(metrics.mutual_info_score(processed_sources[:, i], processed_sources[:, i]))
    return entropies


def compute_nmi(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    processed_sources = process_sources(sources)
    processed_latents = _process_latents_for_nmi(latents, discrete_latents=discrete_latents)

    ret = np.empty(shape=(processed_sources.shape[1], processed_latents.shape[1]), dtype=np.float64)
    entropies = _column_entropies(processed_sources)

    for i in range(processed_sources.shape[1]):
        for j in range(processed_latents.shape[1]):
            if discrete_latents:
                ret[i, j] = float(metrics.mutual_info_score(processed_sources[:, i], processed_latents[:, j]))
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

    model = _make_logistic_regression(max_iter=100)

    model.fit(X, y)
    y_pred = model.predict_proba(X)
    return float(metrics.log_loss(y, y_pred))


def compute_infoe(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False) -> Tuple[float, np.ndarray]:
    normalized_predictive_information_per_source: List[float] = []
    linear_probe_accuracy_per_source: List[float] = []
    processed_sources = process_sources(sources)

    processed_latents = _process_latents_for_infoe(latents, discrete_latents=discrete_latents)

    for i in range(processed_sources.shape[1]):
        src = processed_sources[:, i].astype(np.int64)
        predictive_conditional_entropy = logistic_regression(processed_latents, src)
        null = np.zeros_like(latents, dtype=latents.dtype)
        marginal_source_entropy = logistic_regression(null, src)

        if marginal_source_entropy > 0:
            normalized_predictive_information_per_source.append(
                (marginal_source_entropy - predictive_conditional_entropy) / marginal_source_entropy
            )
        else:
            normalized_predictive_information_per_source.append(0.0)

        probe = _make_logistic_regression(max_iter=100)
        probe.fit(processed_latents, src)
        linear_probe_accuracy_per_source.append(float(probe.score(processed_latents, src)))

    return float(np.mean(normalized_predictive_information_per_source)), np.asarray(linear_probe_accuracy_per_source)


def compute_infomec(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False) -> InfoMECResult:
    """
    :param sources: (num_samples, num_sources)
    :param latents: (num_samples, num_latents)
    :param discrete_latents: bool
    :return: InfoMECResult
    """
    nmi, entropies = compute_nmi(sources, latents, discrete_latents)

    latent_ranges = np.max(latents, axis=0) - np.min(latents, axis=0)
    if discrete_latents:
        active_latents = latent_ranges > 0
    else:
        max_range = float(np.max(latent_ranges)) if latent_ranges.size else 0.0
        active_latents = latent_ranges > (max_range / 20.0 if max_range > 0 else 0.0)

    if not np.any(active_latents):
        active_latents = np.ones(latents.shape[1], dtype=bool)

    num_sources = sources.shape[1]
    num_active_latents = int(np.sum(active_latents))
    pruned_nmi = nmi[:, active_latents]

    if num_sources <= 1:
        infom = 1.0
    else:
        denom = np.sum(pruned_nmi, axis=0)
        ratios = np.divide(np.max(pruned_nmi, axis=0), denom, out=np.zeros_like(denom), where=denom > 0)
        infom = float((np.mean(ratios) - 1 / num_sources) / (1 - 1 / num_sources))

    if num_active_latents <= 1:
        infoc = 1.0
    else:
        denom = np.sum(pruned_nmi, axis=1)
        ratios = np.divide(np.max(pruned_nmi, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
        infoc = float((np.mean(ratios) - 1 / num_active_latents) / (1 - 1 / num_active_latents))

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
