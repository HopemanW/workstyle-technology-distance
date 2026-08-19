# Research Design: Workstyle–Technology Distance

## 1. Economic mechanism

A new technology can create an implementation problem rather than a knowledge problem. Incumbents have accumulated occupations, routines, communication patterns, and decision rules. If a technology requires a very different mix of workstyles, the organization must reallocate labor and change how problems are solved. Those changes can be especially costly for older firms.

The project separates **technology intensity** from **organizational distance**. A sector can receive many patents but still face low reorganization cost if the technologies complement the existing occupational/workstyle structure.

## 2. Current organization

For industry `j` at baseline year `t`, let `s_jot` be the employment share of occupation `o`. Use the BLS National Employment Matrix or a compatible industry×occupation source.

O*NET provides each occupation's workstyle vector `w_o`. The incumbent workstyle is

`W_jt = sum_o s_jot w_o`.

The repository's `normalize_employment_weights()` and O*NET loaders create these objects.

## 3. Technology-implied organization

Construct one text corpus per occupation from O*NET task statements. Embed these occupation corpora and technology text (for example PatentsView publication titles/abstracts) in the same vector space.

Let `sim(k,o)` be semantic similarity between technology `k` and occupation `o`. Convert similarities to a probability distribution using a temperature-controlled softmax:

`q_ko = softmax(sim(k,o) / tau)`.

This is the technology-implied occupation distribution.

A stronger future version would train the occupation mapping on known complement/substitute labels rather than relying only on unsupervised similarity.

## 4. Two WTD measures

### Workstyle cosine WTD

`W_k = sum_o q_ko w_o`

and

`WTD_cos(j,k) = cosine_distance(W_j, W_k)`.

### Organizational transport WTD

Define the cost of reallocating organizational mass from occupation `o` to `o'` by the cosine distance between their O*NET workstyle vectors. Then compute an entropic optimal-transport approximation:

`WTD_OT(j,k) = min_pi sum_{o,o'} pi_oo' c(w_o,w_o')`

subject to the incumbent and technology-implied occupation marginals.

This is economically attractive because it measures the minimum workstyle adjustment required to transform the incumbent organization into the technology-implied one.

## 5. Outcomes

A public first-pass outcome panel can use Census Business Dynamics Statistics. BDS provides firm-age categories and measures including employment, job creation/destruction, and net job creation rates. Interactions are limited by the public tabulations, so the final panel must respect the BDS crossing structure.

The reduced-form target is conceptually

`Outcome_jakt = FE + beta * Old_a × WTD_jkt + controls + error`,

where `Old_a` identifies older firm-age bins and technology intensity is controlled separately.

## 6. Identification discipline

WTD is a measurement contribution; it does not automatically identify a causal effect. A causal design should define technology arrival/exposure before using ML for heterogeneity. Candidate designs include:

- technology shocks concentrated in pre-existing patent classes;
- exposure based on lagged industry occupational structure;
- shift-share designs using national technology growth × predetermined local/industry exposure;
- event studies around externally timed technology releases or regulatory changes.

Avoid using post-outcome occupation shares to construct baseline WTD.

## 7. Official public sources encoded in the repo

- O*NET 30.3 Work Styles, Task Statements, and Work-Style/Activity linkages.
- BLS 2024–2034 National Employment Matrix, industry–occupation tables.
- PatentsView PatentSearch API publication title/abstract text; API key required.
- Census BDS time-series API; API key required; current public API documentation covers 1978–2023.

## 8. Model-validation agenda

Compare alternative text representations while holding the economic target fixed:

1. TF-IDF baseline.
2. Generic Sentence Transformer.
3. Finance/economics-domain encoder.
4. Contrastively trained occupation–technology encoder.

Validation should emphasize economic signal, not just text similarity: stability across releases, out-of-sample prediction of occupational shifts, and whether high-WTD technologies predict larger age gradients in adoption/growth.
