# Research materials and reproduction guide

Research materials: [repository](https://github.com/DS-Gao/ev-charging-service-dynamics), [version 1.0.0](https://github.com/DS-Gao/ev-charging-service-dynamics/releases/tag/v1.0.0).

This guide describes the research-materials delivery for the manuscript on
city-level electric-vehicle charging facilities, recorded activity, and
operational policy. It separates a quick path for checking the supplied
figures and tables from a longer path that reconstructs analytical inputs and
estimates. Formal definitions, sample rules, estimands, and inference remain
in Supplementary Information (SI) Sections S1--S8.

## Start with the documents

The compact document delivery contains the following files under
`manuscript/`:

- `main_en.pdf`: English main article;
- `supplement_en.pdf`: formal Supplementary Information; and
- `reader_zh.pdf`: Chinese reader.

The same documents and their source files are included in the public research
repository. A source archive is provided as a convenience for typesetting;
the repository tree, rather than the archive alone, is the reproducibility
entry point.

## Public repository layout

The public repository uses relative paths and keeps the scientific materials
browsable:

```
metadata/          source, variable, sample, policy, and figure--table maps
data/processed/    permitted processed analytical inputs
data/results/      saved estimates, covariance, support, and memberships
data/figure_source/plot values used by the manuscript and SI figures
src/               preparation, descriptive, model, and figure modules
scripts/           documented preparation, estimation, plotting, and document entry points
manuscript/        LaTeX, bibliography, figures, and document compiler
docs/              data-access and reproduction instructions
```

The compact delivery may keep some of these files at its root for convenient
reading. In either form, all paths in executable examples are relative to
the selected delivery root; no private workspace path is required.

## Materials and access

Charging observations are from version 2 of the anonymized operator-network
replication archive, DOI
[10.5281/zenodo.15564121](https://doi.org/10.5281/zenodo.15564121). In that
archive, `Data/ev_air.csv` supplies city-week records, `Total` is the recorded
activity field, and `cumulative_plugs` is the supplied facility indicator.
`ChinaCity.geojson` supplies display geometry. The published article and its
later archive are cited as provenance references; the city-week input used by
this study is the version 2 release.

Policy evidence is linked to the registered multi-level energy-transition
corpus, DOI
[10.6084/m9.figshare.32182515.v1](https://doi.org/10.6084/m9.figshare.32182515.v1).
Some original policy texts require PKULaw or another source-specific access
route. NASA POWER daily fields and the compact subnational OxCGRT
`StringencyIndex_Average` field are supporting sources described in the SI;
their raw downloads are not silently represented as unrestricted repository
files. See `docs/DATA_ACCESS.md` and `metadata/policy_sources.csv` for source
locators, access conditions, and licensing notes.

The repository distinguishes original sources from derived files. It does
not claim that all raw operator records, subscription-only policy texts, or
third-party downloads are freely redistributable. Original DOI and official
source citations remain part of the manuscript and are not replaced by the
research-materials repository citation.

## Two reproduction paths

### A. Check the saved results and figures

Use the saved result tables and figure-source values under `data/results/` and
`data/figure_source/` together with `metadata/figure_table_map.csv`.
This path regenerates manuscript tables, figure panels, and the three
documents from supplied values. It is intended for rapid checking of the
reported numbers and does not imply that the full raw-data pipeline is being
re-run.

The compact delivery keeps the same material in `figure_source_data/` and
`figures/`; the package-ready vector figures are under
`manuscript/figures/`. Exact-parameter values and saved covariance/support
tables should be preferred when checking rounded prose, table cells, or
plots.

### B. Reconstruct inputs and estimates

Use `metadata/analysis_code_index.csv` as the ordered entrypoint index. Each
row identifies an entrypoint, required inputs, outputs, preconditions, and
the interpretation of that stage. The corresponding modules are under
`src/` and the runnable wrappers under `scripts/`. Preparation and model
commands must be run with the documented inputs and environment; missing
third-party files should be obtained from their cited source and placed at
the documented relative path.

The index records the original estimation code and the saved result path. It
does not infer precision from PDF text, change support rules, or substitute a
new model specification. A successful result-only replay, an estimation
replay from supplied processed inputs, and a rebuild from every original
source are different capabilities; the repository states which one each
entrypoint supports.

## Formal SI map

- **S1 — Data sources and construction:** source releases, linkage, fields,
  dates, geometry, weather, response coverage, and analytical supports.
- **S2 — Descriptive objects and comparisons:** observation states, windows,
  growth regimes, transitions, contributions, composition, and spatial checks.
- **S3 — Policy coding and timing:** functional coding, source records,
  governance dates, annual assignment, and quarterly intervals.
- **S4 — Annual models and inference:** annual supports, conditional means,
  parameterization, elasticity, covariance, degrees of freedom, and curvature.
- **S5 — Quarterly facility--activity dynamics:** quarterly support, lags,
  endpoint elasticities, covariance intervals, diagnostics, and the paired
  response-index comparison.
- **S6 — Annual policy associations:** complete policy, construction-support,
  response-index, weather, covariance, and support information.
- **S7 — Governance comparisons:** facility context, lead diagnostics,
  windows, common-city contrasts, time/composition terms, and validity rules.
- **S8 — City trajectories and observation coverage:** activity, facility,
  intensity, and observation-state trajectories for the city records.

## Typesetting and environment

To typeset from the repository, enter `manuscript/` and run:

```powershell
python compile_manuscript.py
```

The compiler requires Python's standard library, XeLaTeX, BibTeX, Times New
Roman, Arial, and SimSun. It first compiles the English main article and SI
with their bibliography passes, performs the additional XeLaTeX passes that
resolve cross-document labels, and then compiles the Chinese reader. A clean
copy needs no old auxiliary files or absolute workspace path.

The document compiler only typesets supplied sources. It does not download
the charging archive, retrieve subscription-only policy texts, fit models, or
rebuild the city-week panel. For analytical reproduction, follow the
entrypoints and preconditions in `metadata/analysis_code_index.csv` and the
source-access instructions in `docs/`.

The recorded statistical environment is Python 3.12.0 with NumPy 2.4.2,
pandas 3.0.0, SciPy 1.15.3, statsmodels 0.14.6, and matplotlib 3.10.8.
These packages are not required by the typesetting compiler itself.
