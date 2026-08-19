from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

PATENTSVIEW_BASE = "https://search.patentsview.org/api/v1"


def build_publication_query(*, text_terms: Sequence[str] | None = None, year_from: int | None = None, year_to: int | None = None) -> Mapping[str, Any]:
    criteria: list[Mapping[str, Any]] = []
    if text_terms:
        criteria.append({"_text_any": {"publication_abstract": list(text_terms)}})
    if year_from is not None:
        criteria.append({"_gte": {"publication_year": int(year_from)}})
    if year_to is not None:
        criteria.append({"_lte": {"publication_year": int(year_to)}})
    if not criteria:
        return {"_gte": {"publication_year": 0}}
    return criteria[0] if len(criteria) == 1 else {"_and": criteria}


@dataclass
class PatentsViewClient:
    """Client for pre-grant title/abstract text from the current PatentSearch API."""

    api_key: str | None = None
    base_url: str = PATENTSVIEW_BASE
    timeout: int = 60

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("PATENTSVIEW_API_KEY")

    def search_publications(
        self,
        *,
        text_terms: Sequence[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("Set PATENTSVIEW_API_KEY or pass api_key=...")
        query = build_publication_query(text_terms=text_terms, year_from=year_from, year_to=year_to)
        fields = ["document_number", "publication_title", "publication_abstract", "publication_year"]
        params = {
            "q": json.dumps(query, separators=(",", ":")),
            "f": json.dumps(fields),
            "o": json.dumps({"size": min(max(int(limit), 1), 1000)}),
        }
        request = Request(
            f"{self.base_url}/publication/?{urlencode(params)}",
            headers={"X-Api-Key": self.api_key, "User-Agent": "workstyle-technology-distance/0.2 research"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return pd.DataFrame(payload.get("publications", []))


def publication_documents(df: pd.DataFrame) -> list[str]:
    title = df.get("publication_title", pd.Series("", index=df.index)).fillna("").astype(str)
    abstract = df.get("publication_abstract", pd.Series("", index=df.index)).fillna("").astype(str)
    return (title + ". " + abstract).str.strip().tolist()
