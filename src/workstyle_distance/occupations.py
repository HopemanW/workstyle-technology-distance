from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .embedding import TextEmbedder, cosine_distance, cosine_similarity_matrix, row_normalize


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(values, dtype=float) / temperature
    z -= np.max(z)
    e = np.exp(np.clip(z, -700, 700))
    return e / e.sum()


def predict_technology_occupation_weights(
    technology_docs: Sequence[str],
    occupation_task_corpus: Mapping[str, str] | pd.Series,
    *,
    mode: str = "tfidf",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int | None = 40,
    temperature: float = 0.08,
) -> pd.Series:
    if isinstance(occupation_task_corpus, pd.Series):
        occupations = occupation_task_corpus.index.astype(str).tolist()
        corpora = occupation_task_corpus.astype(str).tolist()
    else:
        occupations = [str(k) for k in occupation_task_corpus]
        corpora = [str(v) for v in occupation_task_corpus.values()]
    if not occupations or not technology_docs:
        raise ValueError("Technology documents and occupation corpus must be non-empty")

    embedder = TextEmbedder(mode=mode, model_name=model_name)
    occ_emb, tech_emb = embedder.encode_groups([corpora, list(technology_docs)])
    tech_vector = row_normalize(tech_emb.mean(axis=0))
    scores = cosine_similarity_matrix(occ_emb, tech_vector[None, :]).ravel()
    order = np.argsort(scores)[::-1]
    if top_k is not None:
        order = order[: min(top_k, len(order))]
    weights = _softmax(scores[order], temperature)
    return pd.Series(weights, index=np.asarray(occupations)[order], name="technology_weight").sort_values(ascending=False)


def _writable_float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return an owned, writable float array.

    Pandas 3 may expose read-only NumPy views from ``Series.to_numpy``.  The
    measurement code normalizes vectors, so taking an explicit owned copy keeps
    behavior stable across supported pandas/Python versions.
    """
    return np.array(values, dtype=float, copy=True)


def weighted_workstyle_vector(weights: pd.Series, workstyle_matrix: pd.DataFrame) -> np.ndarray:
    common = workstyle_matrix.index.astype(str).intersection(weights.index.astype(str))
    if len(common) == 0:
        raise ValueError("No common occupations between weights and O*NET workstyle matrix")
    w = _writable_float_array(weights.reindex(common).fillna(0.0))
    total = w.sum()
    if total <= 0:
        raise ValueError("Occupation weights must have positive mass")
    w = w / total
    return w @ workstyle_matrix.reindex(common).to_numpy(dtype=float, copy=True)


def workstyle_vector_distance(source_weights: pd.Series, target_weights: pd.Series, workstyle_matrix: pd.DataFrame) -> float:
    return cosine_distance(
        weighted_workstyle_vector(source_weights, workstyle_matrix),
        weighted_workstyle_vector(target_weights, workstyle_matrix),
    )


def sinkhorn_organizational_distance(
    source_weights: pd.Series,
    target_weights: pd.Series,
    workstyle_matrix: pd.DataFrame,
    *,
    regularization: float = 0.08,
    max_iter: int = 2000,
    tolerance: float = 1e-10,
) -> float:
    """Entropic optimal-transport cost with O*NET workstyle distance as the cost matrix."""
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    available = set(workstyle_matrix.index.astype(str))
    occupations = [o for o in sorted(set(source_weights.index.astype(str)) | set(target_weights.index.astype(str))) if o in available]
    if not occupations:
        raise ValueError("No occupations overlap the workstyle matrix")
    a = _writable_float_array(source_weights.reindex(occupations).fillna(0.0))
    b = _writable_float_array(target_weights.reindex(occupations).fillna(0.0))
    a_total = a.sum()
    b_total = b.sum()
    if a_total <= 0 or b_total <= 0:
        raise ValueError("Source and target weights must have positive mass")
    a = a / a_total
    b = b / b_total
    styles = row_normalize(workstyle_matrix.reindex(occupations).to_numpy(dtype=float, copy=True))
    cost = np.clip((1.0 - styles @ styles.T) / 2.0, 0.0, 1.0)
    kernel = np.maximum(np.exp(-cost / regularization), 1e-300)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max_iter):
        old_u = u.copy()
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
        if np.max(np.abs(u - old_u)) < tolerance:
            break
    plan = (u[:, None] * kernel) * v[None, :]
    return float(np.sum(plan * cost))


@dataclass(frozen=True)
class TechnologyMeasurement:
    workstyle_cosine_distance: float
    organizational_transport_distance: float
    technology_occupation_weights: pd.Series
    current_workstyle_vector: np.ndarray
    technology_workstyle_vector: np.ndarray


def measure_technology_distance(
    legacy_occupation_weights: pd.Series,
    technology_docs: Sequence[str],
    occupation_task_corpus: Mapping[str, str] | pd.Series,
    workstyle_matrix: pd.DataFrame,
    *,
    mode: str = "tfidf",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int | None = 40,
    temperature: float = 0.08,
    regularization: float = 0.08,
) -> TechnologyMeasurement:
    target = predict_technology_occupation_weights(
        technology_docs,
        occupation_task_corpus,
        mode=mode,
        model_name=model_name,
        top_k=top_k,
        temperature=temperature,
    )
    current = weighted_workstyle_vector(legacy_occupation_weights, workstyle_matrix)
    future = weighted_workstyle_vector(target, workstyle_matrix)
    return TechnologyMeasurement(
        workstyle_cosine_distance=cosine_distance(current, future),
        organizational_transport_distance=sinkhorn_organizational_distance(
            legacy_occupation_weights,
            target,
            workstyle_matrix,
            regularization=regularization,
        ),
        technology_occupation_weights=target,
        current_workstyle_vector=current,
        technology_workstyle_vector=future,
    )
