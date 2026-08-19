"""Small offline example of the real-data architecture.

Replace the toy frames with official O*NET/BLS inputs; use PatentsViewClient for
technology text and CensusBDSClient for firm-age outcomes once API keys are set.
"""
from __future__ import annotations

import pandas as pd

from workstyle_distance import (
    build_industry_technology_measurement,
    measurement_summary,
    normalize_employment_weights,
)

workstyle = pd.DataFrame(
    [[2.0, -1.0, -0.5], [0.0, 1.5, 1.0], [-1.0, 2.0, 2.5]],
    index=["17-2112.00", "15-1252.00", "19-1029.00"],
    columns=["Attention to Detail", "Innovation", "Adaptability"],
)
tasks = pd.Series({
    "17-2112.00": "design manufacturing processes quality control reliability",
    "15-1252.00": "develop machine learning software conduct rapid experimentation",
    "19-1029.00": "conduct scientific research analyze experimental biological data",
})
employment = normalize_employment_weights(
    pd.DataFrame({
        "industry": ["334", "334", "334"],
        "occupation": ["17-2112.00", "15-1252.00", "19-1029.00"],
        "employment": [700, 200, 100],
    }),
    group_col="industry",
    occupation_col="occupation",
    employment_col="employment",
)
technology = [
    "machine learning systems for automated experimentation and continuous software deployment",
    "adaptive models that learn from data and support rapid product iteration",
]
measurement = build_industry_technology_measurement(
    industry_code="334",
    employment_weights=employment,
    technology_docs=technology,
    occupation_task_corpus=tasks,
    workstyle_matrix=workstyle,
    top_k=3,
)
print(measurement_summary("334", measurement).to_string(index=False))
print("\nTechnology-implied occupations:")
print(measurement.technology_occupation_weights)
