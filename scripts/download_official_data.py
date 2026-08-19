from __future__ import annotations

import argparse
from pathlib import Path

from workstyle_distance.official_data import (
    BLS_INDUSTRY_OCCUPATION_MATRIX_URL,
    ONET_TASK_STATEMENTS_URL,
    ONET_WORK_STYLES_URL,
    ONET_WORK_STYLE_ACTIVITY_URL,
    download_official_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw", help="Local raw-data directory")
    args = parser.parse_args()
    out = Path(args.output)
    files = {
        "onet_work_styles.csv": ONET_WORK_STYLES_URL,
        "onet_task_statements.csv": ONET_TASK_STATEMENTS_URL,
        "onet_work_styles_to_work_activities.csv": ONET_WORK_STYLE_ACTIVITY_URL,
        "bls_industry_occupation_matrix.xlsx": BLS_INDUSTRY_OCCUPATION_MATRIX_URL,
    }
    for name, url in files.items():
        path = download_official_file(url, out / name)
        print(f"Downloaded {url} -> {path}")


if __name__ == "__main__":
    main()
