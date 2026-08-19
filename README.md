# Workstyle–Technology Distance

An economics + machine-learning research project for measuring **organizational distance** between an incumbent firm's existing way of working and the workstyles required by a new technology.

The project is inspired by the organizational-friction mechanism in Zhiguo He, Nicolas Crouzet, Victor Lyonnet, and Yueran Ma's *Why Don't Old Firms Do New Things?* It is an independent methodological extension, not a replication and not affiliated with the original authors.

## Research object

A technology can be technically available to every firm while remaining costly to implement inside an incumbent whose occupations, routines, and problem-solving styles are far from those required by the technology.

This repository constructs **Workstyle–Technology Distance (WTD)** in two complementary ways:

1. **Workstyle cosine distance** between the incumbent's employment-weighted O*NET workstyle vector and the technology-implied workstyle vector.
2. **Organizational optimal-transport distance**: the minimum workstyle-adjustment cost required to move the incumbent occupation distribution toward the technology-implied occupation distribution.

The technology-implied occupation distribution is estimated by embedding patent/technology text and O*NET occupation-task corpora in a common semantic space.

## Aggressive public-data pipeline

```text
O*NET 30.3 Tasks ---------------------> occupation task corpus
O*NET 30.3 Work Styles --------------> occupation workstyle vectors
BLS 2024–34 Industry-Occupation -----> incumbent occupation weights
PatentsView publication abstracts ---> technology embeddings
                     \                 /
                      semantic matching
                             ↓
               technology-implied occupations
                             ↓
        cosine WTD + optimal-transport WTD
                             ↓
Census BDS firm-age × industry outcomes
                             ↓
             age × WTD empirical tests
```

The current official data interfaces are encoded in the package so the empirical pipeline is reproducible and easy to update when new releases appear.

## Core economic prediction

```text
Growth = ... + beta_1 * TechnologyExposure
             + beta_2 * FirmAge
             + beta_3 * WTD
             + beta_4 * FirmAge * WTD + error
```

with `beta_4 < 0`: older firms should perform relatively worse when implementation requires a larger organizational change.

## What is implemented

- offline TF-IDF embedding baseline;
- optional Sentence Transformer embeddings;
- O*NET 30.3 Work Styles and Task Statement loaders;
- occupation-level task corpora and workstyle matrices;
- generic BLS industry × occupation employment-share normalizer;
- patent/publication text client for PatentsView PatentSearch API;
- Census BDS client for firm-age × industry outcomes;
- semantic technology→occupation prediction;
- employment-weighted workstyle cosine distance;
- entropic optimal-transport organizational distance;
- synthetic DGP and age × distance benchmark;
- random-forest prediction layer;
- GitHub Actions tests on Python 3.10 and 3.12.

## Quick start

```bash
pip install -e '.[dev]'
python examples/run_demo.py
python examples/run_empirical_scaffold.py
pytest
```

Transformer mode:

```bash
pip install -e '.[transformer]'
python examples/run_demo.py --transformer
```

Download current O*NET/BLS public files locally:

```bash
python scripts/download_official_data.py
```

Raw data are ignored by Git and should stay local.

## API keys for live public APIs

PatentsView's current PatentSearch API requires `X-Api-Key`; Census BDS also requires a Census API key. Set:

```bash
export PATENTSVIEW_API_KEY='...'
export CENSUS_API_KEY='...'
```

The package then exposes `PatentsViewClient` and `CensusBDSClient`. No keys are stored in the repository.

## Next empirical milestone

Create an industry × technology × year panel in which:

- BLS supplies pre-technology occupational weights;
- O*NET supplies occupation workstyles and task text;
- PatentsView supplies technology text;
- WTD is computed before observing the outcome;
- Census BDS supplies firm-age-specific employment/job-creation outcomes.

The key empirical comparison is whether the old-vs-young performance gap is systematically more negative for high-WTD technologies **holding technology quantity/intensity separate**.

See `docs/RESEARCH_DESIGN.md`.
