from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class TextEmbedder:
    mode: Literal["tfidf","transformer"] = "tfidf"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def encode_pair(self, legacy_docs: Iterable[str], technology_docs: Iterable[str]):
        legacy=list(legacy_docs); tech=list(technology_docs)
        if self.mode=="transformer":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Install with pip install -e '.[transformer]'") from exc
            model=SentenceTransformer(self.model_name)
            A=model.encode(legacy, normalize_embeddings=True)
            B=model.encode(tech, normalize_embeddings=True)
            return np.asarray(A,float), np.asarray(B,float)
        vec=TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
        M=vec.fit_transform(legacy+tech).toarray()
        return M[:len(legacy)], M[len(legacy):]

def _normalize(v: np.ndarray) -> np.ndarray:
    n=np.linalg.norm(v)
    return v if n==0 else v/n

def workstyle_technology_distance(
    legacy_docs: Iterable[str],
    technology_docs: Iterable[str],
    *,
    mode: str = "tfidf",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """Cosine distance between average legacy and technology workstyle embeddings."""
    emb=TextEmbedder(mode=mode, model_name=model_name)
    A,B=emb.encode_pair(legacy_docs,technology_docs)
    a=_normalize(A.mean(axis=0)); b=_normalize(B.mean(axis=0))
    similarity=float(np.clip(a@b,-1,1))
    return float((1-similarity)/2)  # map cosine distance to [0,1]
