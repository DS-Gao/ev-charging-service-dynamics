# EV charging service dynamics: research materials

This repository contains the code, processed inputs, saved estimation results,
figure-source values, metadata, and reproducibility notes for the study of
charging activity, facility exposure, service intensity, observation coverage,
and policy associations across Chinese cities. The manuscript files and PDFs
are distributed alongside this research-materials tree by the authors; the
analysis package remains usable without downloading the original policy
documents.

The package has two complementary uses:

1. **Read and audit the delivered results.** Complete coefficient tables,
   covariance tables, sample members, descriptive memberships, and exact figure
   source values are under `data/results/reference/` and `data/figure_source/`.
   The source-keyed policy table is `metadata/policy_sources.csv`.
2. **Run either of two explicit reproduction routes.** Route A redraws all
   saved figures without fitting a model. Route B replays the independent
   annual, zero-inclusive annual, policy, and descriptive estimators and then
   redraws the saved figures. Both routes use repository-relative paths and do
   not read a historical analysis directory.

## Quick start

Use Python 3.12 and install the pinned family of scientific dependencies:

```powershell
python -m pip install -r requirements.txt

# Route A: saved-result figures only; no model fit or data preparation
python scripts/reproduce_saved_results.py

# Route B: independent estimator replay plus saved-result figures
python scripts/run_quick_reproduction.py

python scripts/validate_package.py
```

Route A can optionally compile the manuscript source after rendering:

```powershell
python scripts/reproduce_saved_results.py --compile
```

The `--compile` option requires XeLaTeX and BibTeX; it remains separate from
all model and raw-input commands.

The quick command writes regenerated artifacts to:

```text
data/results/recomputed_annual/
data/results/recomputed_annual_zero/
data/results/recomputed_policy/
data/results/recomputed_descriptive/
data/results/recomputed_figures/
```

The annual model rerun reproduces the 464-row positive-support candidate, the
445-row/104-city common model sample, and the four recorded spline knots. The
zero-inclusive annual PPML rerun reports 523 rows in 112 cities after its
all-zero-city and singleton-city rules, with its full sample audit and
city-cluster covariance under `data/results/recomputed_annual_zero/`. The
policy rerun implements the original QR OLS estimator with province-clustered
CR1 covariance and the same six outcome/specification combinations from the
325-row/94-city processed policy input. The descriptive rerun uses
positive `Q`, `E`, and `W` in every required year and keeps unknown or
no-compatible-week states distinct from source-coded zeros.

To run one component directly:

```powershell
python src/models/fit_annual_models.py
python src/models/fit_annual_zero.py
python src/models/fit_policy_association.py --force-fit
python src/descriptive/windows.py
python -X utf8 src/descriptive/diagnostics.py --check-reference
python src/figures/plot_saved_results.py
python scripts/quick_figures.py
python scripts/reproduce_saved_results.py
```

## Repository map

| Path | Contents |
| --- | --- |
| `data/processed/` | Redistributable annual/quarterly panels, city-label crosswalk and source geometry, weather table, and policy-model input. |
| `data/results/reference/` | Saved annual, policy, quarterly, and descriptive results, including covariances and sample members. |
| `data/figure_source/` | Numeric values used to draw the main and supplementary figures. |
| `data/results/recomputed_*` | Outputs created by the quick reproduction commands. |
| `metadata/` | Source, variable, sample, figure/table, policy-provision, and entrypoint dictionaries. |
| `metadata/analysis_code_index.csv` | Public entrypoint, input, output, and capability index. |
| `src/models/` | Relative-path annual, zero-inclusive annual, policy, quarterly, and governance estimators. |
| `src/descriptive/` | Prespecified common-city windows plus recorded state, transition, and contribution diagnostics. |
| `src/figures/` | Plotting from shipped numeric figure-source values; no model fit. |
| `src/preparation/` | Charging aggregation and source-control preparation from documented inputs. |
| `scripts/` | Data preparation, quick reproduction, and package validation entrypoints. |
| `docs/` | Data-access limits and detailed reproduction instructions. |

## Definitions used by the analysis

`Q` is the sum of recorded activity over compatible source dates; `E` is the
sum of the supplied facility indicator over the same dates; and `W` is the
number of compatible weeks. The derived measures are `A = Q/W`, `N = E/W`, and
`I = Q/E`. Thus `N` is an average facility exposure, not a count of facilities
newly built during the period, and `I` is an activity-per-facility-exposure
index rather than engineering utilization.

The annual positive-support candidate has 464 city-years in 123 cities. The
common annual fixed-effect model uses 445 rows in 104 cities. The policy
association input has 325 rows in 94 cities and contains the already-checked
COVID-compatible exposure flag. The adjacent positive-support growth windows
contain 39, 55, 77, and 93 cities for 2019–2020 through 2022–2023,
respectively. Full definitions and membership files are in
`metadata/sample_definitions.csv` and `data/results/reference/descriptive/`.

## Data and code access boundaries

The charging archive, policy-corpus metadata, and weather/API provenance are
listed in `metadata/data_sources.csv` and `THIRD_PARTY_SOURCES.md`. The package
does not include downloaded policy full text, subscription-only PKULaw records,
raw operator records, raw NASA API JSON, or a mirror of the raw OxCGRT
download. The processed inputs and original result tables are sufficient for
the quick path and for auditing the reported numbers. Rebuilding every source
table from first-party raw downloads requires access and terms described in
`docs/DATA_ACCESS.md`; this repository does not claim that those restricted
source routes are fully automated.

The raw charging preparation route is available through
`scripts/prepare_charging_panels.py`; it requires a directly obtained Zenodo
version-2 `ev_air.csv` and the bundled `data/processed/city_crosswalk_v4.csv`.
The route aggregates compatible source dates. Additional documented commands
prepare weather and pandemic controls from source caches and policy exposures
from the included coded metadata; see `docs/REPRODUCING.md`.

The quarterly dynamic and governance estimators have their own public input
tables and output directories. They are listed in
`metadata/analysis_code_index.csv`; the annual policy and nonlinear commands
above are the shortest representative reruns. Saved reference covariance and
membership files remain available when a user chooses only the plotting path.

The Zenodo DOI for the charging archive and the Figshare DOI for the policy
corpus are original-source citations, not substitutes for this repository
record. The GitHub URL is a citation for the research software and materials;
it is not a new DOI or a claim that the manuscript has been published.

See `docs/REPRODUCING.md` for the result-to-figure map, validation tolerances,
and the distinction between saved-result redraws and estimator reruns.

## Manuscript and release

The versioned research-materials release is
[v1.0.0](https://github.com/DS-Gao/ev-charging-service-dynamics/releases/tag/v1.0.0).
The release assets provide the compiled [English main manuscript
PDF](https://github.com/DS-Gao/ev-charging-service-dynamics/releases/download/v1.0.0/main_en.pdf),
[English supplement PDF](https://github.com/DS-Gao/ev-charging-service-dynamics/releases/download/v1.0.0/supplement_en.pdf),
[Chinese reader PDF](https://github.com/DS-Gao/ev-charging-service-dynamics/releases/download/v1.0.0/reader_zh.pdf),
and [manuscript source archive](https://github.com/DS-Gao/ev-charging-service-dynamics/releases/download/v1.0.0/manuscript_source_v1.0.0.zip).

## Citation and licenses

Cite this release using [CITATION.cff](CITATION.cff) or
[CITATION.bib](CITATION.bib), which identify version 1.0.0 and the release
URL. Original code in `src/` and `scripts/` is provided under the [MIT
License](LICENSE). Original manuscript text, research documentation, figures,
and statistical summaries are provided under [CC BY 4.0](LICENSES/CC-BY-4.0.txt).
Third-party data, source geometry, excerpts, and derived source content retain
their own terms; see [THIRD_PARTY_SOURCES.md](THIRD_PARTY_SOURCES.md) and the
[license-scope note](LICENSES/README.md) before redistributing them.
