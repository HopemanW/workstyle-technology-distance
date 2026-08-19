from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TextEmbedder:
    """Joint text encoder used by the technology/occupation mapping layer."""

    mode: Literal["tfidf", "transformer"] = "tfidf"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def encode(self, documents: Iterable[str]) -> np.ndarray:
        docs = ["" if d is None else str(d) for d in documents]
        if not docs:
            return np.empty((0, 0), dtype=float)
        if self.mode == "transformer":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Install with pip install -e '.[transformer]'") from exc
            model = SentenceTransformer(self.model_name)
            return np.asarray(model.encode(docs, normalize_embeddings=True), dtype=float)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        return vec.fit_transform(docs).toarray().astype(float)

    def encode_pair(self, legacy_docs: Iterable[str], technology_docs: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        legacy = ["" if d is None else str(d) for d in legacy_docs]
        tech = ["" if d is None else str(d) for d in technology_docs]
        encoded = self.encode(legacy + tech)
        return encoded[: len(legacy)], encoded[len(legacy) :]

    def encode_groups(self, groups: Sequence[Iterable[str]]) -> list[np.ndarray]:
        materialized = [["" if d is None else str(d) for d in g] for g in groups]
        lengths = [len(g) for g in materialized]
        encoded = self.encode([d for group in materialized for d in group])
        out: list[np.ndarray] = []
        start = 0
        for n in lengths:
            out.append(encoded[start : start + n])
            start += n
        return out


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x if n == 0 else x / n
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(row_normalize(a) @ row_normalize(b).T, -1.0, 1.0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = row_normalize(np.asarray(a, dtype=float).ravel())
    b = row_normalize(np.asarray(b, dtype=float).ravel())
    if not a.size or not b.size:
        raise ValueError("Vectors must be non-empty")
    return float((1.0 - float(np.clip(a @ b, -1.0, 1.0))) / 2.0)


def workstyle_technology_distance(
    legacy_docs: Iterable[str],
    technology_docs: Iterable[str],
    *,
    mode: str = "tfidf",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """Cosine distance between average legacy and technology text embeddings."""
    embedder = TextEmbedder(mode=mode, model_name=model_name)
    a, b = embedder.encode_pair(legacy_docs, technology_docs)
    return cosine_distance(a.mean(axis=0), b.mean(axis=0))
