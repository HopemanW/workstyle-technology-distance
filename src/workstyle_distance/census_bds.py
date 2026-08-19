from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

BDS_BASE = "https://api.census.gov/data/timeseries/bds"


@dataclass
class CensusBDSClient:
    """Client for firm-age × industry outcomes from Census Business Dynamics Statistics."""

    api_key: str | None = None
    base_url: str = BDS_BASE
    timeout: int = 60

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("CENSUS_API_KEY")

    def query(
        self,
        *,
        variables: Sequence[str] = ("YEAR", "NAICS", "NAICS_LABEL", "FAGE", "FAGE_LABEL", "EMP", "NET_JOB_CREATION_RATE"),
        year: int | str = "*",
        naics: str | None = None,
        firm_age: str | None = None,
        geography: str = "us:1",
        extra_predicates: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("Set CENSUS_API_KEY or pass api_key=...")
        params: list[tuple[str, str]] = [
            ("get", ",".join(variables)),
            ("for", geography),
            ("YEAR", str(year)),
            ("key", self.api_key),
        ]
        if naics is not None:
            params.append(("NAICS", str(naics)))
        if firm_age is not None:
            params.append(("FAGE", str(firm_age)))
        for key, value in (extra_predicates or {}).items():
            params.append((str(key), str(value)))
        with urlopen(f"{self.base_url}?{urlencode(params)}", timeout=self.timeout) as response:
            rows = json.loads(response.read().decode("utf-8"))
        if not rows:
            return pd.DataFrame(columns=list(variables))
        return pd.DataFrame(rows[1:], columns=rows[0])
