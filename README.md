# EV charging service dynamics: replication materials

This repository provides the processed inputs, analysis code, saved numeric
results, figure-source tables, sample/source metadata, and instructions needed
to read or reproduce the reported charging-service analyses across Chinese
cities. Manuscript source, article PDFs, and author records are delivered
separately and are not part of this public tree. The package does not require
downloading restricted policy full text.

## Two public routes

Install the pinned scientific environment, then choose the route that matches
your purpose:

```powershell
python -m pip install -r requirements.txt

# Fast route: redraw 18 figures from saved numeric values; no model fitting
python scripts/reproduce_saved_results.py

# Replay route: run the annual, zero-inclusive annual, policy, descriptive,
# and saved-figure commands
python scripts/run_quick_reproduction.py
```

Both routes use repository-relative paths. The fast route only reads committed
numeric references and figure sources; it does not re-estimate a model. The
replay route runs the public estimator entrypoints before redrawing the saved
figures. Quarterly and governance estimation are separate, longer-running
commands documented in `docs/REPRODUCING.md`; the governance `--bootstrap`
option is not part of the default quick route.

Generated artifacts are written below Git-ignored paths:

```text
data/results/recomputed_annual/
data/results/recomputed_annual_zero/
data/results/recomputed_policy/
data/results/recomputed_descriptive/
data/results/recomputed_quarterly/
data/results/recomputed_governance/
data/results/recomputed_figures/
```

Run the package check from the repository root:

```powershell
python scripts/validate_package.py
```

## Where to find inputs and results

| Path | Contents |
| --- | --- |
| `data/processed/` | Annual and quarterly panels, weather/pandemic controls, city crosswalks and geometry, policy-model input, and governance event registry. |
| `data/results/reference/` | Saved coefficients, full covariances, sample/member records, bootstrap support, and other numeric references. |
| `data/figure_source/` | Numeric source values consumed by the scientific figure renderer. |
| `src/` | Preparation, descriptive, estimator, and figure-generation modules. |
| `scripts/` | Public preparation, replay, figure, sanitization, and validation entrypoints. |
| `metadata/` | Source/provenance, variable, sample, policy, code-entrypoint, and figure/table indexes. |
| `docs/REPRODUCING.md` | Commands, input contracts, saved-result versus refit boundaries, and numeric reading/rebuild guidance. |
| `docs/DATA_ACCESS.md` | First-party source links, access conditions, and redistribution limits. |

`metadata/figure_table_map.csv` records 58 reported figure/table entries from the final compiled labels (7 main tables, 9 main figures, 33 supplementary tables, and 9 supplementary figures). Each
row points directly to its numeric reference or figure-source inputs and names
the public redraw/read route; it contains no manuscript-source or PDF path.
`metadata/sample_definitions.csv`, `metadata/variable_dictionary.csv`, and
the source/provenance metadata are the authoritative definitions rather than
duplicate README summaries. To read a table, open the path listed in the map's
`numeric_sources` column. To redraw figures, run the listed command or the fast
route above.

The primary derived measures are defined from the processed fields as
`A = Q/W`, `N = E/W`, and `I = Q/E`; observation and eligibility states are
documented in the variable dictionary and sample metadata. The saved tables
preserve the reported coefficients, covariance elements, effective bootstrap
support, and sample-membership evidence. Reading those tables is not a claim
of fresh estimation.

## Source and access boundaries

Source identifiers, transformations, and terms are listed in
`metadata/data_sources.csv`, `metadata/data_provenance.csv`, and
`THIRD_PARTY_SOURCES.md`. The public package includes processed inputs and
policy metadata/coding, but not raw operator records, raw API JSON, a mirror of
the OxCGRT download, or subscription-only policy full text. The optional raw
charging, context-control, and policy-preparation commands and their input
contracts are documented in `docs/REPRODUCING.md`; access and redistribution
conditions are documented once in `docs/DATA_ACCESS.md`.

## Versioned release and citation

Research-materials release [v1.2.3](https://github.com/DS-Gao/ev-charging-service-dynamics/releases/tag/v1.2.3) retains the fixed v1.2.0 baseline and updates the public field-scale note, broad policy-responsibility dictionary, affected Figure 4 and Figure 8 renderer, and the 58-row final figure/table map. The release contains replication code, processed inputs, saved numeric results, and metadata only.

Cite the release with [CITATION.cff](CITATION.cff). Original code in `src/` and
`scripts/` is under the [MIT License](LICENSE); original repository
documentation, numeric figure sources, metadata, and saved statistical
summaries are under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Third-party data,
geometry, excerpts, and derived source content retain their own terms; see
`THIRD_PARTY_SOURCES.md`, `metadata/data_sources.csv`, and
`LICENSES/README.md` before redistribution.

## Additional public materials

The repository includes `docs/FIELD_SOURCES.md` and `docs/POLICY_METHODS.md` for released-field scope and the two-path policy construction. `scripts/compare_compatible_weeks.py` generates the saved 225-city reference and the 167-city complete-compatible-week comparison from repository inputs; it does not refit a model or infer source anonymization properties.

The final-label redraw command is `python scripts/quick_figures.py --released-scale-labels`; it uses saved values without fitting models.
