# Official Data Sources

This project is designed around public, reproducible source data. Raw downloads and API keys should remain local and are excluded by `.gitignore`.

## O*NET 30.3

Resource Center: https://www.onetcenter.org/database.html

Used objects:

- Work Styles: `https://www.onetcenter.org/dl_files/database/db_30_3_csv/work_styles.csv`
- Task Statements: `https://www.onetcenter.org/dl_files/database/db_30_3_csv/task_statements.csv`
- Work Styles to Work Activities: `https://www.onetcenter.org/dl_files/database/db_30_3_csv/work_styles_to_work_activities.csv`

The Work Styles table contains O*NET-SOC occupation × work-style ratings. The default empirical matrix uses the `WI` Work Style Impact scale.

## U.S. Bureau of Labor Statistics

2024–2034 National Employment Matrix, industry-occupation data:

- https://www.bls.gov/emp/tables/industry-occupation-matrix-industry.htm
- workbook: `https://www.bls.gov/emp/ind-occ-matrix/occupation.xlsx`

The project deliberately uses an explicit column-mapping normalizer because spreadsheet labels/layout can change between BLS releases.

## PatentsView PatentSearch API

Documentation: https://search.patentsview.org/docs/docs/Search%20API/SearchAPIReference/

The current API requires an `X-Api-Key`. This project uses the pre-grant publication endpoint because it exposes publication title and abstract fields suitable for technology embeddings.

Set locally:

```bash
export PATENTSVIEW_API_KEY='...'
```

## Census Business Dynamics Statistics

API documentation: https://www.census.gov/programs-surveys/bds/data.API.html

API base: `https://api.census.gov/data/timeseries/bds`

The BDS API provides firm-age categories and outcomes including employment, job creation/destruction, and net job creation rates. Current public API documentation covers 1978–2023 and requires a Census API key.

Set locally:

```bash
export CENSUS_API_KEY='...'
```

## Reproducibility rule

Pin the source vintage used in an empirical result. Do not silently update O*NET/BLS/BDS vintages between estimation runs. A real paper should record source release, retrieval date, crosswalk version, embedding model, and model hash/configuration.
