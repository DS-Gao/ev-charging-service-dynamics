# Third-party materials and rights

This inventory covers the files and fields actually included in the research
materials tree. The repository's MIT code license does not override a
third-party source's rights. Source DOI records and the research repository
URL must be cited separately.

| Material used | Source and version | Rights / access | Included here |
| --- | --- | --- | --- |
| City-week charging and facility fields (`Q`, `E`, `W`, `A`, `N`, `I`) | [Zenodo 15564121](https://doi.org/10.5281/zenodo.15564121), record published 2025-05-31; version 2 replication archive | The Zenodo record API reports an open record with license id `cc-by-4.0`. Attribute the archive and its authors. The archive is a processed/anonymized release, not a blanket grant for underlying operator records. | Selected processed annual and policy-model inputs, saved results, and source-keyed summaries |
| Policy corpus metadata and derived codes | [Figshare 32182515 v1](https://doi.org/10.6084/m9.figshare.32182515.v1) | The source repository README states CC BY 4.0 for its code/data. Full policy text accessed through PKULaw may require an institutional or individual subscription; its availability is not inferred from the repository license. | Policy metadata, coded summaries, and `metadata/policy_sources.csv`; no downloaded policy full text |
| COVID policy-response control | [OxCGRT compact subnational data](https://github.com/OxCGRT/covid-policy-dataset/tree/e7f66ee39654293b5c068efd2f195bd591dc27f6), commit `e7f66ee39654293b5c068efd2f195bd591dc27f6` | The source README states CC BY 4.0 and supplies the recommended academic attribution. | Its processed control appears in `data/processed/policy_model_input.csv`; the raw download is not mirrored |
| Annual weather controls | [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), API v2.10.1, MERRA-2 grid; see the [NASA POWER referencing guide](https://power.larc.nasa.gov/docs/referencing/) | NASA POWER source terms and attribution apply. The recorded API version and query description are in `metadata/data_provenance.csv`; the public provenance does not record a separate retrieval date, so a fresh request should record its own access date. NASA POWER requests notification for redistribution. This inventory does not assert a new license where the source record does not state one. | Processed city-year weather table and provenance; raw API JSON is not included |
| Derived city-label crosswalk | Zenodo charging archive, version 2; version 4 linkage snapshot | Use under the archive's terms. The included file contains labels and linkage status only; it is not a geometry redistribution. One `needs_review` linkage status is retained. | `data/processed/city_crosswalk_v4.csv` |
| Administrative geometry used for maps and linkage | Geometry supplied with the Zenodo charging archive, version 2 | Covered by the source archive's open CC BY 4.0 record terms as verified in the Zenodo metadata; cite the archive. No independent geometry license is asserted by this package. | `data/processed/city_geometry.geojson`; the file is the source geometry copy used by the map renderer and city linkage |
| Scientific Python libraries | NumPy, pandas, SciPy, statsmodels, matplotlib | Each dependency retains its own license; see its distribution metadata. | Runtime dependencies only; no vendored packages |

## What is not packaged

The package intentionally omits subscription-only PKULaw text, downloaded
original policy documents, raw operator records, raw NASA JSON responses, raw
OxCGRT downloads, PDFs downloaded from external publishers, browser/session
files, environment directories, credentials, and local review records. The
processed inputs and saved numerical outputs are sufficient for the stated
quick reproduction and audit paths. Users who rebuild from first-party source
downloads must obtain those sources directly and follow their access terms.

## Attribution practice

When reusing a figure, table, or processed input, cite both the relevant source
DOI or source repository listed above and the research-materials release. Do
not describe the GitHub repository as the source of the charging archive,
policy corpus, OxCGRT data, or NASA weather data.
