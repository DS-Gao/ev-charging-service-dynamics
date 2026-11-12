# Reproducing the research materials

All commands below are run from the repository root. They use only relative
paths inside the repository.

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

This route redraws all main and supplementary figures from committed numeric
tables and the bundled source geometry. It never runs an estimator, bootstrap,
covariance fit, or raw-input preparation step.

```powershell
python scripts/reproduce_saved_results.py
```

To compile the optional manuscript source after the figure redraw, use:

```powershell
python scripts/reproduce_saved_results.py --compile
```

The `--compile` option requires XeLaTeX and BibTeX. It is independent of the
statistical model commands.

## Route B: estimator replay and saved figures

```powershell
python scripts/run_quick_reproduction.py
```

The script runs five independent steps:

1. `src/models/fit_annual_models.py` reads the shipped annual metrics and
   weather table, applies the prespecified positive-support and common-sample
   rules, and fits the explicit linear, restricted-cubic, and Poisson models.
2. `src/models/fit_annual_zero.py` fits the prespecified positive-support and
   zero-inclusive annual PPML specifications, retaining the sample audit and
   city-cluster score/Hessian covariance.
3. `src/models/fit_policy_association.py` reads the 325-row processed policy
   input and applies the original QR OLS / province-clustered CR1 estimator to
   outcomes `N`, `A`, and `I` under the base and COVID-adjusted specifications.
4. `src/descriptive/windows.py` recomputes the predefined common-city windows
   and facility-scale groups from the annual panel.
5. `scripts/quick_figures.py` consumes the committed numeric figure-source
   tables and redraws the main and supplementary figure set without fitting a
   model. This includes the representative growth chart and does not fit a
   model.

The descriptive diagnostics are an additional audit entrypoint; they are not
part of the five-step quick wrapper. They recompute recorded states,
transitions, and city contributions, then perform read-only reconciliation
checks against the shipped descriptive references:

```powershell
python -X utf8 src/descriptive/diagnostics.py --check-reference
```

The figure-only component can also be run separately after installing the same
dependencies:

```powershell
python scripts/quick_figures.py
```

For a fresh raw-data preparation, download the Zenodo version-2 `ev_air.csv`
and pass it together with the supplied city-label crosswalk. The command
recomputes the annual and quarterly `Q`, `E`, `W`, `A`, `N`, and `I` panels and
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
the source details in `docs/DATA_ACCESS.md` and the command's `--help` contract.
The bundled context crosswalk uses the canonical provinces of the fixed
annual analysis panel; the original charging-label crosswalk is kept separately.
The coordinate file must identify `city_key`, `latitude`, `longitude`, and
`coordinate_status`; API JSON filenames must retain the documented MERRA-2
coordinates and request dates.

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
first-record registry, coded construction documents, and independently recorded
local-corpus coverage. It derives preceding-year-end R/S indicators and the
preceding-calendar-year B document count, preserving unknown coverage as
missing. It never reads or classifies policy full text.

```powershell
python scripts/prepare_policy_exposure.py --check-reference
```

The command writes `policy_exposure_annual.csv`, document timing, and a
reference comparison under `data/results/recomputed_policy/`. Its additional
R/S document counts describe the retained clause table only; the fixed
first-record registry supplies the historical timing used by the models.

Generated files are placed under `data/results/recomputed_*`. They are
deliberately separate from `data/results/reference/`, which contains the saved
study artifacts. The estimator outputs can be compared to the corresponding
reference CSVs with any CSV reader.

The quarterly dynamic and governance estimators are separate entrypoints in
`src/models/fit_quarterly.py` and `src/models/fit_governance.py`. Their
packaged inputs, output directories, and capability limits are recorded in
`metadata/analysis_code_index.csv`; they are not silently invoked by the short
annual/zero-inclusive annual/policy quick command.

## What can and cannot be rerun

The annual and policy estimators are independent numerical reruns from the
included processed inputs. They preserve the recorded support, design, model
formula, and covariance construction. The complete saved quarterly and
governance results and covariances remain available for audit, while their
fresh-fit commands use only the packaged processed inputs. No command
reconstructs restricted policy text; raw charging preparation requires the
original source archive supplied by the user.

The public preparation routes are independent of any historical
analysis-output tree. They read only documented source files, cached API
responses, or bundled coded inputs; no credentials or browser data are read.

## Result and figure map

`metadata/figure_table_map.csv` maps each main and supplementary figure to its
numeric source. `metadata/sample_definitions.csv` maps the model and
descriptive samples to their members and rules. The full covariance tables are
kept with their matching coefficient tables under
`data/results/reference/annual/`, `policy/`, and `quarterly/`.

## Validation

```powershell
python scripts/validate_package.py
```

The validator checks required paths, CSV readability, sample dimensions,
reference-model row counts, the absence of absolute local paths in shipped
metadata/input files, and the absence of obvious credential/browser artifacts.
It does not perform a new scientific audit or assert that every source archive
is currently reachable.
