"""Standalone InfoMEC utilities adapted for this repo.

This module follows the same basic setup described in the InfoMEC paper/code:
- compute normalized mutual information between each source and each latent dim
- compute InfoM / InfoC from sparsity of the resulting matrix
- compute InfoE with logistic regression for discrete sources
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def process_sources(sources: np.ndarray) -> np.ndarray:
    processed_sources = []
    for i in range(sources.shape[1]):
        processed_sources.append(preprocessing.LabelEncoder().fit_transform(sources[:, i]))
    return np.stack(processed_sources, axis=1).astype(np.int64)


def _standardize_latents(latents: np.ndarray) -> np.ndarray:
    parts = []
    for j in range(latents.shape[1]):
        parts.append(preprocessing.StandardScaler().fit_transform(latents[:, j][:, None]))
    return np.concatenate(parts, axis=1).astype(np.float64)


def compute_nmi(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    processed_sources = process_sources(sources)
    if discrete_latents:
        processed_latents = []
        for j in range(latents.shape[1]):
            processed_latents.append(preprocessing.LabelEncoder().fit_transform(latents[:, j]))
        processed_latents = np.stack(processed_latents, axis=1)
    else:
        processed_latents = _standardize_latents(latents)

    ret = np.empty((processed_sources.shape[1], processed_latents.shape[1]), dtype=np.float64)
    entropies = np.empty(processed_sources.shape[1], dtype=np.float64)

    for i in range(processed_sources.shape[1]):
        src = processed_sources[:, i]
        entropy = metrics.mutual_info_score(src, src)
        entropies[i] = entropy
        if entropy <= 1e-12:
            ret[i, :] = 0.0
            continue

        for j in range(processed_latents.shape[1]):
            if discrete_latents:
                mi = metrics.mutual_info_score(src, processed_latents[:, j])
            else:
                mi = feature_selection.mutual_info_classif(
                    processed_latents[:, j][:, None],
                    src,
                    discrete_features=False,
                    n_neighbors=min(10, max(3, len(src) - 1)),
                    random_state=0,
                )[0]
            ret[i, j] = float(mi) / float(entropy)
    return ret, entropies


def _logistic_regression_log_loss(X: np.ndarray, y: np.ndarray) -> float:
    model = linear_model.LogisticRegression(
        penalty=None,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=0,
    )
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    return float(metrics.log_loss(y, y_pred))


def compute_infoe(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False) -> Tuple[float, np.ndarray]:
    processed_sources = process_sources(sources)
    if discrete_latents:
        processed_latents = preprocessing.OneHotEncoder().fit_transform(latents)
        if hasattr(processed_latents, "toarray"):
            processed_latents = processed_latents.toarray()
    else:
        processed_latents = preprocessing.StandardScaler().fit_transform(latents)

    normalized_predictive_information_per_source: List[float] = []
    linear_probe_accuracy_per_source: List[float] = []

    for i in range(processed_sources.shape[1]):
        src = processed_sources[:, i].astype(np.int64)
        predictive_conditional_entropy = _logistic_regression_log_loss(processed_latents.astype(np.float64), src)

        null = np.zeros((latents.shape[0], 1), dtype=np.float64)
        marginal_source_entropy = _logistic_regression_log_loss(null, src)
        if marginal_source_entropy <= 1e-12:
            normalized_predictive_information_per_source.append(0.0)
        else:
            normalized_predictive_information_per_source.append(
                (marginal_source_entropy - predictive_conditional_entropy) / marginal_source_entropy
            )

        probe = linear_model.LogisticRegression(
            penalty=None,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            multi_class="multinomial",
            n_jobs=-1,
            random_state=0,
        )
        probe.fit(processed_latents.astype(np.float64), src)
        linear_probe_accuracy_per_source.append(float(probe.score(processed_latents.astype(np.float64), src)))

    return float(np.mean(normalized_predictive_information_per_source)), np.asarray(linear_probe_accuracy_per_source)


def compute_infomec(sources: np.ndarray, latents: np.ndarray, discrete_latents: bool = False) -> InfoMECResult:
    nmi, entropies = compute_nmi(sources, latents, discrete_latents=discrete_latents)

    latent_ranges = np.max(latents, axis=0) - np.min(latents, axis=0)
    if discrete_latents:
        active_latents = latent_ranges > 0
    else:
        max_range = float(np.max(latent_ranges)) if latent_ranges.size else 0.0
        thresh = max_range / 20.0 if max_range > 0 else 0.0
        active_latents = latent_ranges > thresh

    if not np.any(active_latents):
        active_latents = np.ones(latents.shape[1], dtype=bool)

    pruned_nmi = nmi[:, active_latents]
    num_sources = sources.shape[1]
    num_active_latents = int(np.sum(active_latents))

    if num_sources <= 1:
        infom = 1.0
    else:
        denom = np.sum(pruned_nmi, axis=0)
        ratios = np.divide(
            np.max(pruned_nmi, axis=0),
            denom,
            out=np.zeros_like(denom),
            where=denom > 1e-12,
        )
        infom = float((np.mean(ratios) - 1.0 / num_sources) / (1.0 - 1.0 / num_sources))

    if num_active_latents <= 1:
        infoc = 1.0
    else:
        denom = np.sum(pruned_nmi, axis=1)
        ratios = np.divide(
            np.max(pruned_nmi, axis=1),
            denom,
            out=np.zeros_like(denom),
            where=denom > 1e-12,
        )
        infoc = float((np.mean(ratios) - 1.0 / num_active_latents) / (1.0 - 1.0 / num_active_latents))

    infoe, probe_acc = compute_infoe(sources, latents, discrete_latents=discrete_latents)
    return InfoMECResult(
        infom=infom,
        infoc=infoc,
        infoe=infoe,
        nmi=nmi,
        active_latents=active_latents,
        linear_probe_accuracy_per_source=probe_acc,
        source_entropies=entropies,
    )
