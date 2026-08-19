# Research Design

## Economic idea

A technology may be technically available to every firm but organizationally costly to implement inside an incumbent whose existing workforce, routines, and problem-solving styles are far from those required by the new technology.

The key object here is **Workstyle–Technology Distance (WTD)**, a semantic and occupational measure of that mismatch.

## Measurement architecture

### 1. Current organizational workstyle

Use industry × occupation employment shares and O*NET workstyle/task descriptions to create an employment-weighted representation of the incumbent production organization.

### 2. Technology-implied workstyle

Map patent, product, AI, software, or other technology text into occupation/task space. A simple version uses text similarity; a richer version estimates whether each occupation is complemented or substituted by the technology.

### 3. Distance

Compare current and technology-implied workstyle vectors. Candidate metrics:

- cosine distance;
- Wasserstein/optimal-transport distance across occupations;
- Mahalanobis distance using historical covariance of workstyles;
- learned distance from a Siamese/contrastive Transformer.

## Empirical prediction

Estimate whether age-related underperformance is more negative when WTD is high, controlling separately for technology quantity/intensity. The quantity of technology and the organizational distance of technology are different economic objects.

## Public-data path

- O*NET: occupation tasks, workstyles, skills;
- BLS Occupational Employment and Wage Statistics: industry × occupation employment;
- Census BDS: firm-age-group employment growth at industry level;
- public patent text / technology descriptions where licensing permits.

## ML extensions

- Sentence Transformer embeddings of tasks and patent text;
- contrastive learning using known occupation–technology complement/substitute pairs;
- causal forest for age × WTD heterogeneity after a valid empirical design;
- SHAP/feature attribution to identify which workstyle dimensions drive organizational conflict.
