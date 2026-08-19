from io import StringIO

import numpy as np
import pandas as pd

from workstyle_distance import (
    build_occupation_task_corpus,
    build_occupation_workstyle_matrix,
    build_publication_query,
    load_onet_task_statements,
    load_onet_work_styles,
    measure_technology_distance,
    normalize_employment_weights,
    sinkhorn_organizational_distance,
)


def test_onet_parsers_and_matrix():
    styles = load_onet_work_styles(StringIO(
        "O*NET-SOC Code,Title,Element ID,Element Name,Scale ID,Scale Name,Data Value\n"
        "11-0001.00,A,1,Innovation,WI,Impact,2.0\n"
        "11-0001.00,A,2,Detail,WI,Impact,-1.0\n"
        "11-0002.00,B,1,Innovation,WI,Impact,-1.0\n"
        "11-0002.00,B,2,Detail,WI,Impact,2.0\n"
    ))
    matrix = build_occupation_workstyle_matrix(styles, standardize=False)
    assert matrix.shape == (2, 2)

    tasks = load_onet_task_statements(StringIO(
        "O*NET-SOC Code,Title,Task\n"
        "11-0001.00,A,develop software models\n"
        "11-0001.00,A,test new algorithms\n"
        "11-0002.00,B,inspect routine production\n"
    ))
    corpus = build_occupation_task_corpus(tasks)
    assert "algorithms" in corpus.loc["11-0001.00"]


def test_transport_zero_when_distributions_equal():
    matrix = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["a", "b"])
    weights = pd.Series([0.7, 0.3], index=["a", "b"])
    d = sinkhorn_organizational_distance(weights, weights, matrix, regularization=0.02)
    assert d < 0.02


def test_full_measurement_prefers_semantically_matching_occupation():
    matrix = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["old", "new"])
    tasks = pd.Series({"old": "routine production inspection", "new": "machine learning software experimentation"})
    legacy = pd.Series({"old": 0.9, "new": 0.1})
    result = measure_technology_distance(
        legacy,
        ["machine learning software experimentation"],
        tasks,
        matrix,
        top_k=2,
        temperature=0.05,
    )
    assert result.technology_occupation_weights.index[0] == "new"
    assert result.workstyle_cosine_distance > 0
    assert result.organizational_transport_distance > 0


def test_employment_normalization_and_patents_query():
    weights = normalize_employment_weights(
        pd.DataFrame({"i": [1, 1], "o": ["a", "b"], "e": [75, 25]}),
        group_col="i",
        occupation_col="o",
        employment_col="e",
    )
    assert np.isclose(weights["employment_share"].sum(), 1.0)
    q = build_publication_query(text_terms=["machine learning"], year_from=2020, year_to=2025)
    assert "_and" in q
