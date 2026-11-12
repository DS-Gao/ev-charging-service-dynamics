# Data access and redistribution limits

The package separates three levels of reproducibility.

## 1. Immediate public path

The repository contains the processed annual and quarterly service panels,
the city-label crosswalk used by the raw preparation command, the city-year
weather table, annual panel, and the column-reduced policy-association input
used by the independent commands. It also contains the complete saved result
tables needed to audit the reported coefficients, covariances, memberships,
and figure values. These files do not contain the original policy text.

## 2. First-party inputs available from their sources

The charging archive is [Zenodo 15564121](https://doi.org/10.5281/zenodo.15564121).
Its record metadata identifies the record as open and CC BY 4.0. The archive
contains processed city-week fields and geometry used for the study. Obtain
the archive directly, retain its attribution, and check its current record
terms before redistributing additional files.

The compact crosswalk in `data/processed/city_crosswalk_v4.csv` contains the
source-city label, public `city_key`, display name, province label, and the
recorded linkage-status flag. It is a derived label crosswalk rather than a
redistributed geometry file; the one `needs_review` status is retained so that
users do not mistake the linkage for a perfect boundary match.

The source geometry copy used by the map renderer is bundled as
`data/processed/city_geometry.geojson`. It is redistributed under the Zenodo
archive's recorded terms and is not assigned a separate package license.

The policy source is the registered corpus at
[Figshare 32182515 v1](https://doi.org/10.6084/m9.figshare.32182515.v1). The
source README states CC BY 4.0 for the repository's code/data, while access to
original PKULaw policy text can require an institutional or individual
subscription. The public package therefore provides policy metadata, source
keys, coded conditions, and derived exposures, but not a copy of the
subscription text.

The Oxford COVID-19 Government Response Tracker input is pinned to commit
`e7f66ee39654293b5c068efd2f195bd591dc27f6`, whose source README states CC BY
4.0. The source URL and attribution are recorded in
`metadata/data_sources.csv`; users can download the compact file from that
commit if they need to reconstruct the processed control.

Weather values were obtained through the NASA POWER Daily API using the
MERRA-2 grid and UTC calendar. The repository includes the processed
city-year table and request/provenance description, not raw API JSON. Use the
[NASA POWER API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
and the [NASA POWER referencing guide](https://power.larc.nasa.gov/docs/referencing/)
for the required POWER project/data citation and current source terms when
making a fresh retrieval. The recorded source is API v2.10.1; the query
description is in `metadata/data_provenance.csv`. The public provenance does
not record a separate API retrieval date, so users making a fresh request
should record their own access date. NASA POWER requests that redistribution
be notified; this package does not assign a license beyond the source's own
terms.

## 3. Rebuilding beyond this package

Rebuilding the complete historical source chain requires the source archives,
the city crosswalk/geometry, the policy corpus metadata, the pinned OxCGRT
file, and NASA POWER requests. Some original policy texts and source-specific
records are access-controlled. The present package does not promise that all
of those acquisition steps can be completed without subscriptions or source
account access. It does promise the relative-path estimator and descriptive
commands documented in `docs/REPRODUCING.md`.

## Fixed preparation metadata

`policy_coding_b_documents.csv` contains 1,213 coded document identifiers,
dates, city-scope flags and B/O functions; it contains no policy excerpts.
`policy_coding_coverage.csv` separately records coverage for all 334 cities,
so absence of a charging-specific document does not determine corpus coverage.
The R/S first-record registry is a fixed coded input; the preparation route
derives exposure from its recorded timing rather than repeating text coding.

`context_city_coordinates.csv` retains the 334 original derived city points
and their coordinate status. `context_city_crosswalk.csv` combines source-city
labels with the canonical province assignments used in the fixed annual model
panel. Use this context crosswalk for province-date pandemic matching. The
older charging-label crosswalk is retained for raw charging reconstruction;
its province labels are not the pandemic-control join key. These files are
derived metadata from the same cited source geometry and analysis registry.
