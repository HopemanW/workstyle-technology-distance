from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Mapping
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ONET_RELEASE = "30.3"
ONET_BASE = f"https://www.onetcenter.org/dl_files/database/db_{ONET_RELEASE.replace('.', '_')}_csv"
ONET_WORK_STYLES_URL = f"{ONET_BASE}/work_styles.csv"
ONET_TASK_STATEMENTS_URL = f"{ONET_BASE}/task_statements.csv"
ONET_WORK_STYLE_ACTIVITY_URL = f"{ONET_BASE}/work_styles_to_work_activities.csv"
BLS_INDUSTRY_OCCUPATION_MATRIX_URL = "https://www.bls.gov/emp/ind-occ-matrix/occupation.xlsx"


def _fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "workstyle-technology-distance/0.2 research"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def download_official_file(url: str, destination: str | Path, timeout: int = 60) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_fetch_bytes(url, timeout=timeout))
    return destination


def _read_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    if isinstance(source, (str, Path)) and str(source).startswith(("http://", "https://")):
        return pd.read_csv(BytesIO(_fetch_bytes(str(source))))
    return pd.read_csv(source)


def _slug(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("*", "").split())


def _normalized_columns(df: pd.DataFrame) -> Mapping[str, str]:
    return {_slug(c): c for c in df.columns}


def _require_column(df: pd.DataFrame, *candidates: str) -> str:
    columns = _normalized_columns(df)
    for candidate in candidates:
        key = _slug(candidate)
        if key in columns:
            return columns[key]
    raise KeyError(f"Could not find any of {candidates}; columns={list(df.columns)}")


def load_onet_work_styles(source: str | Path | BinaryIO = ONET_WORK_STYLES_URL) -> pd.DataFrame:
    raw = _read_csv(source)
    code = _require_column(raw, "O*NET-SOC Code", "onetsoc_code")
    title = _require_column(raw, "Title")
    element_id = _require_column(raw, "Element ID", "element_id")
    element_name = _require_column(raw, "Element Name", "element_name")
    scale_id = _require_column(raw, "Scale ID", "scale_id")
    value = _require_column(raw, "Data Value", "data_value")
    out = raw[[code, title, element_id, element_name, scale_id, value]].copy()
    out.columns = ["onet_soc_code", "occupation_title", "element_id", "work_style", "scale_id", "value"]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["onet_soc_code", "work_style", "value"])


def load_onet_task_statements(source: str | Path | BinaryIO = ONET_TASK_STATEMENTS_URL) -> pd.DataFrame:
    raw = _read_csv(source)
    code = _require_column(raw, "O*NET-SOC Code", "onetsoc_code")
    title = _require_column(raw, "Title")
    task = _require_column(raw, "Task", "Task Statement", "task")
    out = raw[[code, title, task]].copy()
    out.columns = ["onet_soc_code", "occupation_title", "task"]
    out["task"] = out["task"].fillna("").astype(str)
    return out[out["task"].str.len() > 0]


def build_occupation_workstyle_matrix(work_styles: pd.DataFrame, *, scale_id: str = "WI", standardize: bool = True) -> pd.DataFrame:
    data = work_styles.loc[work_styles["scale_id"].astype(str).str.upper() == scale_id.upper()].copy()
    if data.empty:
        raise ValueError(f"No O*NET work-style rows for scale_id={scale_id!r}")
    matrix = data.pivot_table(index="onet_soc_code", columns="work_style", values="value", aggfunc="mean").sort_index()
    if standardize:
        mu = matrix.mean(axis=0)
        sigma = matrix.std(axis=0).replace(0, 1.0)
        matrix = (matrix - mu) / sigma
    return matrix.fillna(0.0)


def build_occupation_task_corpus(tasks: pd.DataFrame) -> pd.Series:
    return tasks.groupby("onet_soc_code", sort=True)["task"].apply(
        lambda s: ". ".join(dict.fromkeys(x.strip() for x in s if x.strip()))
    ).rename("task_corpus")


def normalize_employment_weights(df: pd.DataFrame, *, group_col: str, occupation_col: str, employment_col: str) -> pd.DataFrame:
    out = df[[group_col, occupation_col, employment_col]].copy()
    out.columns = ["industry_code", "occupation_code", "employment"]
    out["employment"] = pd.to_numeric(out["employment"], errors="coerce").fillna(0.0)
    out = out[out["employment"] > 0].copy()
    totals = out.groupby("industry_code")["employment"].transform("sum")
    out["employment_share"] = np.divide(out["employment"], totals, out=np.zeros(len(out)), where=totals > 0)
    return out
