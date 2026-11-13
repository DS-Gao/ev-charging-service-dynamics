# Reproducing the research materials

Run all commands below from the repository root. They use repository-relative
paths and do not require a historical analysis directory or manuscript source
tree.

## Environment

The recorded statistical environment used Python 3.12 with NumPy 2.4.2,
pandas 3.0.0, SciPy 1.15.3, statsmodels 0.14.6, matplotlib 3.10.8,
GeoPandas 1.1.3, Shapely 2.1.2, Pyogrio 0.12.1, and PyProj 3.7.2. The
requirements file pins these versions; a clean virtual environment is
recommended.

```powershell
python -m pip install -r requirements.txt
```

## Route A: saved-result reproduction (no model fitting)

This route redraws the 18 main and supplementary scientific figures from
committed numeric tables and the bundled source geometry. It never runs an
estimator, bootstrap, covariance fit, or raw-input preparation step.

```powershell
python scripts/reproduce_saved_results.py
```

The generated PDF, SVG, PNG, preview, and derived-source files are written to
the Git-ignored `data/results/recomputed_figures/` directory. A saved redraw is
not a model re-estimation.

## Route B: estimator replay and saved figures

```powershell
python scripts/run_quick_reproduction.py
```

The wrapper runs five independent steps:

1. `src/models/fit_annual_models.py` reads the shipped annual metrics and
   weather table, applies the prespecified positive-support and common-sample
   rules, and fits the explicit linear, restricted-cubic, and Poisson models.
2. `src/models/fit_annual_zero.py` fits the prespecified positive-support and
   zero-inclusive annual PPML specifications, retaining the sample audit and
   city-cluster score/Hessian covariance.
3. `src/models/fit_policy_association.py` reads the 325-row processed policy
   input and applies the declared estimator to outcomes `N`, `A`, and `I` under
   the base and COVID-adjusted specifications.
4. `src/descriptive/windows.py` recomputes the predefined common-city windows
   and facility-scale groups from the annual panel.
5. `scripts/quick_figures.py` consumes the committed numeric figure-source
   tables and redraws the scientific figure set without fitting a model.

The descriptive diagnostics are an additional audit entrypoint; they are not
part of the five-step quick wrapper. They recompute recorded states,
transitions, and city contributions, then perform read-only reconciliation
checks against the shipped descriptive references:

```powershell
python -X utf8 src/descriptive/diagnostics.py --check-reference
```

The figure-only component can also be run separately:

```powershell
python scripts/quick_figures.py
```

For a fresh raw-data preparation, download the Zenodo version-2 `ev_air.csv`
and pass it together with the supplied city-label crosswalk. The command
recomputes annual and quarterly `Q`, `E`, `W`, `A`, `N`, and `I` panels and
writes a machine-readable QC report:

```powershell
$rawEvAir = Join-Path (Get-Location) 'downloads\ev_air.csv'
python scripts/prepare_charging_panels.py `
  --input $rawEvAir `
  --crosswalk data/processed/city_crosswalk_v4.csv `
  --output-dir data/processed/raw_rebuilt
```

The raw route enforces the source-date calendar (53 dates in 2017 and 2023,
52 otherwise), keeps compatible `Total == 0` rows, and performs no weather,
policy, or imputation step. The bundled processed panels remain the audit
reference; the download is not included in this repository.

The weather and pandemic-control preparation command consumes local source
files and cached daily API responses. It performs no download. Obtain the
pinned OxCGRT compact subnational file and NASA POWER daily responses using
the source details in `docs/DATA_ACCESS.md` and the command's `--help` contract:

```powershell
python scripts/prepare_context_controls.py `
  --coordinates data/processed/context_city_coordinates.csv `
  --weather-json-dir downloads/nasa_power `
  --ev-air downloads/ev_air.csv `
  --crosswalk data/processed/context_city_crosswalk.csv `
  --oxcgrt downloads/OxCGRT_compact_subnational_v1.csv `
  --output-dir data/results/recomputed_context
```

The weather route retains the original valid-day thresholds and complete-year
requirements for totals and degree days. The COVID route averages over the
operator's compatible source dates, retains compatible zero-activity weeks,
labels pre-2020 values as structural zeros, and leaves post-coverage dates
unknown. The bundled processed controls are comparison references.

The policy preparation command starts with the retained coded clauses,
first-record registry, coded construction documents, and independently
recorded local-corpus coverage. It derives preceding-year-end R/S indicators
and the preceding-calendar-year B document count, preserving unknown coverage
as missing. It never reads or classifies policy full text.

```powershell
python scripts/prepare_policy_exposure.py --check-reference
```

The command writes its derived exposure and comparison tables under
`data/results/recomputed_policy/`. The additional R/S document counts describe
the retained clause table only; the fixed first-record registry supplies the
historical timing used by the models.

Generated files are placed under `data/results/recomputed_*`. They are
deliberately separate from `data/results/reference/`, which contains the saved
study artifacts. Estimator outputs can be compared to the corresponding
reference CSVs with any CSV reader.

The quarterly dynamic and governance estimators are separate entrypoints:

```powershell
python src/models/fit_quarterly.py
python src/models/fit_governance.py
python src/models/fit_governance.py --bootstrap
```

Their packaged inputs, output directories, and capability limits are recorded
in `metadata/analysis_code_index.csv`. The 999-draw governance bootstrap is
optional and is not silently invoked by the short annual/zero-inclusive
annual/policy quick command. The repository retains the effective draw-level
support, covariance, sample, and failure-handling references needed to audit
the reported uncertainty; recomputation writes fresh output below
`data/results/recomputed_governance/`.

## What can and cannot be rerun

The annual and policy estimators are independent numerical reruns from the
included processed inputs. They preserve the recorded support, design, model
formula, and covariance construction. The complete saved quarterly and
governance coefficient and covariance tables remain available for audit, while
fresh-fit commands use only the packaged processed inputs. No command
reconstructs restricted policy text; raw charging preparation requires the
original source archive supplied by the user.

The public preparation routes are independent of any historical
analysis-output tree. They read only documented source files, cached API
responses, or bundled coded inputs; no credentials or browser data are read.

## Result and figure map

`metadata/figure_table_map.csv` records 58 main and supplementary figure/table
entries from the final compiled labels. Each row points directly to its numeric reference or figure-source
inputs and names the code used for a saved numeric read or redraw. It contains
no manuscript-source or PDF path. `metadata/sample_definitions.csv` maps the
model and descriptive samples to their members and rules. Full covariance
tables remain with their matching coefficient tables under
`data/results/reference/annual/`, `policy/`, `quarterly/`, and `governance/`.

For a numeric table, read the path in the `numeric_sources` column. To rebuild a
figure, run the `reproduction` command in that row or use
`scripts/reproduce_saved_results.py` for the complete 18-figure route. Reading
saved values and redrawing figures do not claim to re-estimate a model.

## Validation

```powershell
python scripts/validate_package.py
```

The validator checks required paths, CSV readability, sample dimensions,
reference-model row counts, the absence of absolute local paths in shipped
metadata/input files, and the absence of obvious credential/browser artifacts.
It ignores Git-ignored recomputation output and virtual environments and does
not perform a new scientific audit or assert that every source archive is
currently reachable.

## Complete-compatible-week comparison

The descriptive support comparison is generated with:

```powershell
python scripts/compare_compatible_weeks.py
```

The script reads the repository annual panel and optional source-date support, reports the original 225-city comparison and the 167-city subset with W = 52 in 2022 and W = 53 in 2023, and writes a compact CSV below `data/results/recomputed_descriptive/`. Network intensity is `sum(A) / sum(N)`, not an unweighted city mean. This is an observation-support comparison, not a causal or nationwide estimate.
