# Workstyle–Technology Distance

A new economics + Transformer methodology project for measuring **organizational distance** between an incumbent firm's existing way of working and the workstyles required by a new technology.

The project is inspired by the research question: *when do old firms struggle to implement new technologies because the new technology requires different organizational workstyles?* It is an independent extension, not a replication and not affiliated with the original authors.

## New methodological contribution

Instead of relying only on occupation-level hand-coded mappings, this project represents:

1. a firm's legacy task/workstyle corpus; and
2. a new technology's required task/workstyle corpus

as semantic vectors using a Sentence Transformer (optional), then constructs a **Workstyle–Technology Distance (WTD)**.

```text
legacy occupations/tasks ----> embeddings ----\
                                        distance ---> WTD
new technology tasks --------> embeddings ----/
```

The key economic prediction is

```text
Growth = ... + beta_1 * TechnologyExposure
             + beta_2 * FirmAge
             + beta_3 * WTD
             + beta_4 * FirmAge * WTD + error
```

with `beta_4 < 0`: older firms should perform relatively worse when implementation requires a larger change in organizational workstyle.

## Why this is distinct

This project is about organizational economics, creative destruction, and technology adoption. It has no connection to syndicated loans or credit-market data.

## ML components

- offline TF-IDF baseline for full reproducibility;
- optional Sentence Transformer embeddings;
- cosine and optimal-transport-style distance measures;
- synthetic technology-adoption DGP with known age × distance friction;
- random-forest / causal-forest heterogeneity layer;
- a public-data roadmap using O*NET tasks/workstyles, BLS industry-occupation employment, and patent or technology text.

## Quick start

```bash
pip install -e .
python examples/run_demo.py
pytest
```

Transformer mode:

```bash
pip install -e '.[transformer]'
python examples/run_demo.py --transformer
```

## Intended empirical project

A scalable implementation would:

1. construct an industry-year legacy workstyle vector from current occupation shares;
2. use technology/patent text to predict which occupations are complemented or displaced;
3. infer the future workstyle vector implied by the technology;
4. define WTD as the semantic/occupational shift from current to predicted workstyle;
5. test whether age-related incumbent underperformance is concentrated in high-WTD technologies.

See `docs/RESEARCH_DESIGN.md`.
