from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from .occupations import TechnologyMeasurement, measure_technology_distance


def build_industry_technology_measurement(
    *,
    industry_code: str,
    employment_weights: pd.DataFrame,
    technology_docs: Sequence[str],
    occupation_task_corpus: Mapping[str, str] | pd.Series,
    workstyle_matrix: pd.DataFrame,
    mode: str = "tfidf",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int | None = 40,
) -> TechnologyMeasurement:
    sub = employment_weights.loc[employment_weights["industry_code"].astype(str) == str(industry_code)]
    if sub.empty:
        raise ValueError(f"No employment weights for industry_code={industry_code}")
    legacy = sub.set_index("occupation_code")["employment_share"]
    legacy.index = legacy.index.astype(str)
    return measure_technology_distance(
        legacy,
        technology_docs,
        occupation_task_corpus,
        workstyle_matrix,
        mode=mode,
        model_name=model_name,
        top_k=top_k,
    )


def measurement_summary(industry_code: str, measurement: TechnologyMeasurement) -> pd.DataFrame:
    return pd.DataFrame([{
        "industry_code": str(industry_code),
        "wtd_cosine": measurement.workstyle_cosine_distance,
        "wtd_optimal_transport": measurement.organizational_transport_distance,
        "top_technology_occupation": measurement.technology_occupation_weights.index[0],
        "top_technology_occupation_weight": measurement.technology_occupation_weights.iloc[0],
    }])
