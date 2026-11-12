"""Reconstruct the bounded annual policy-exposure preparation route.

This module is intentionally a preparation layer, not a policy classifier or a
model.  It consumes three public, coded inputs:

* the 466 retained R/S clause-function rows in
  ``data/figure_source/fig8_responsibility_clauses_v20.csv``;
* the dated R/S/G first-record registry in
  ``data/processed/governance_input_event_registry.csv``; and
* the skinny B/O document coding in
  ``data/processed/policy_coding_b_documents.csv`` and the separately recorded
  local-corpus coverage in ``data/processed/policy_coding_coverage.csv``.

No policy full text is read.  The R/S exposure rule follows the V8/V9
implementation: for a known first record, exposure in year ``t`` is
``1[t > first_record_year]``; an unknown local corpus or uncertain timing is
missing, while an observed corpus with no qualifying record is zero.  Counts
are document counts (unique ``gid``), not clause counts.

The B timing rule follows V71's original preparation function.  When both
effective and publication years are known, ``later_year`` is their maximum;
when only effective year is known, effective year is used; when effective year
is unknown, timing remains missing.  A B document is assigned to its
``later_year`` and ``lag1_B_count_later`` is the count assigned to the
preceding calendar year.  This is a count of documents, not a cumulative
stock.  City scope and missing-coverage states are retained.

The public B input contains only coded identifiers, dates, scope flags and
function flags; it does not contain policy text.  It is therefore suitable for
reproducing the original date/count transformation without automatic
recoding of new documents.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLAUSE_PATH = REPOSITORY_ROOT / "data" / "figure_source" / "fig8_responsibility_clauses_v20.csv"
REGISTRY_PATH = REPOSITORY_ROOT / "data" / "processed" / "governance_input_event_registry.csv"
B_DOCUMENT_PATH = REPOSITORY_ROOT / "data" / "processed" / "policy_coding_b_documents.csv"
COVERAGE_PATH = REPOSITORY_ROOT / "data" / "processed" / "policy_coding_coverage.csv"
PANEL_PATH = REPOSITORY_ROOT / "data" / "processed" / "annual_panel_v12.csv"
MODEL_INPUT_PATH = REPOSITORY_ROOT / "data" / "processed" / "policy_model_input.csv"

WINDOW_START = 2016
WINDOW_END = 2023


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False).fillna("")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _year(value: Any) -> float:
    text = _text(value)
    if not text:
        return float("nan")
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer() and 1900 <= int(number) <= 2100:
            return float(int(number))
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?<!\d)(?:19|20|21)\d{2}(?!\d)", text)
    return float(match.group(0)) if match else float("nan")


def _flag(value: Any) -> bool:
    return _text(value).lower() in {"1", "1.0", "true", "yes", "y"}


def _number(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def derive_later_years(documents: pd.DataFrame) -> pd.DataFrame:
    """Apply the original effective/publication ``later_year`` rule."""

    required = {
        "gid",
        "scope_city_key_v7",
        "city_scope_eligible_v7",
        "effective_year",
        "publication_year",
        "publication_date",
        "B",
    }
    missing = sorted(required.difference(documents.columns))
    if missing:
        raise ValueError(f"policy_coding_b_documents.csv missing columns: {missing}")

    d = documents.copy()
    if d["gid"].duplicated().any():
        raise ValueError("policy_coding_b_documents.csv must contain one row per gid")
    d["city_key"] = d["scope_city_key_v7"].astype(str).str.strip()
    d["effective_year_coded"] = d["effective_year"].map(_year)
    publication_year = d["publication_year"].map(_year)
    publication_from_date = d["publication_date"].map(_year)
    d["publication_year_coded"] = publication_year.where(publication_year.notna(), publication_from_date)
    d["later_year"] = np.where(
        d["effective_year_coded"].notna() & d["publication_year_coded"].notna(),
        np.maximum(d["effective_year_coded"], d["publication_year_coded"]),
        np.where(d["effective_year_coded"].notna(), d["effective_year_coded"], np.nan),
    )
    d["timing_basis_later"] = np.select(
        [
            d["effective_year_coded"].notna() & d["publication_year_coded"].notna(),
            d["effective_year_coded"].notna(),
        ],
        ["later_max_effective_publication_year", "later_fallback_effective_publication_missing"],
        default="later_unavailable_effective_year_missing",
    )
    d["scope_city_eligible"] = d["city_scope_eligible_v7"].map(_flag)
    d["window_exposure_eligible_later"] = (
        d["scope_city_eligible"] & d["later_year"].notna() & d["later_year"].le(WINDOW_END)
    ).astype(int)
    d["B_coded"] = d["B"].map(_flag).astype(int)
    for column in ("B1", "B2", "B3", "O", "O1", "O2", "O3"):
        if column in d:
            d[f"{column}_coded"] = d[column].map(_flag).astype(int)
    return d


def _panel_skeleton(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"city_key", "city", "province", "year"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"annual_panel_v12.csv missing columns: {missing}")
    p = panel[["city_key", "city", "province", "year"]].copy()
    p["city_key"] = p["city_key"].astype(str).str.strip()
    p["year"] = p["year"].map(_year)
    p = p[p["year"].between(WINDOW_START, WINDOW_END)].copy()
    p["year"] = p["year"].astype(int)
    if p.duplicated(["city_key", "year"]).any() or len(p) != 334 * 8:
        raise ValueError(f"Expected the fixed 334 x 8 annual skeleton; got {len(p)} rows")
    return p.sort_values(["city_key", "year"], kind="stable").reset_index(drop=True)


def _registry_lookup(registry: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    required = {"city_key", "policy_type", "first_record_year", "first_record_timing_status", "policy_coverage_status", "first_gid"}
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(f"governance_input_event_registry.csv missing columns: {missing}")
    r = registry[registry["policy_type"].isin(["R", "S"])].copy()
    if r.duplicated(["city_key", "policy_type"]).any():
        raise ValueError("R/S registry must contain one row per city and policy type")
    return {
        (str(row.city_key), str(row.policy_type)): {
            "first_record_year": _year(row.first_record_year),
            "timing_status": _text(row.first_record_timing_status),
            "coverage_status": _text(row.policy_coverage_status),
            "first_gid": _text(row.first_gid),
        }
        for row in r.itertuples(index=False)
    }


def derive_rs_exposure(clauses: pd.DataFrame, registry: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Derive R/S first-year exposure and annual document counts."""

    required = {"city_key", "gid", "later_year", "R", "S"}
    missing = sorted(required.difference(clauses.columns))
    if missing:
        raise ValueError(f"fig8_responsibility_clauses_v20.csv missing columns: {missing}")
    c = clauses.copy()
    c["city_key"] = c["city_key"].astype(str).str.strip()
    c["year"] = c["later_year"].map(_year)
    c["R_flag"] = c["R"].map(_flag)
    c["S_flag"] = c["S"].map(_flag)
    # The public clause table is already the retained 466-row, study-window
    # coding table.  Counts are document counts, hence gid deduplication.
    count_rows: list[pd.DataFrame] = []
    for typ, flag in (("R", "R_flag"), ("S", "S_flag")):
        one = c[c[flag] & c["year"].between(WINDOW_START, WINDOW_END)]
        counts = one.groupby(["city_key", "year"], as_index=False)["gid"].nunique().rename(columns={"gid": f"{typ}_count_v9"})
        count_rows.append(counts)
    counts = count_rows[0].merge(count_rows[1], on=["city_key", "year"], how="outer").fillna(0)
    lookup_counts = {(str(row.city_key), int(row.year)): (int(row.R_count_v9), int(row.S_count_v9)) for row in counts.itertuples(index=False)}
    lookup = _registry_lookup(registry)
    rows: list[dict[str, Any]] = []
    for base in panel.itertuples(index=False):
        values: dict[str, Any] = {"city_key": base.city_key, "city": base.city, "province": base.province, "year": int(base.year)}
        statuses: list[str] = []
        for typ in ("R", "S"):
            info = lookup.get((str(base.city_key), typ), {})
            first_year = info.get("first_record_year", float("nan"))
            timing = info.get("timing_status", "")
            coverage = info.get("coverage_status", "")
            unknown = not info or coverage == "unknown_city_coverage" or timing in {"unknown_city_coverage", "time_uncertain"}
            if unknown:
                prev = np.nan
                status = f"{typ}_unknown_coverage_or_timing"
            elif np.isfinite(first_year):
                prev = int(int(base.year) > int(first_year))
                status = f"{typ}_known_first_{int(first_year)}"
            else:
                prev = 0
                status = f"{typ}_no_qualifying_record"
            current_count, _ = lookup_counts.get((str(base.city_key), int(base.year)), (0, 0))
            if typ == "S":
                _, current_count = lookup_counts.get((str(base.city_key), int(base.year)), (0, 0))
            previous_count, _ = lookup_counts.get((str(base.city_key), int(base.year) - 1), (0, 0))
            if typ == "S":
                _, previous_count = lookup_counts.get((str(base.city_key), int(base.year) - 1), (0, 0))
            if coverage == "unknown_city_coverage":
                current_count = np.nan
                previous_count = np.nan
            values[f"{typ}_prev_observed_v9"] = prev
            values[f"first_{typ}_record_year_v9"] = int(first_year) if np.isfinite(first_year) else ""
            values[f"{typ}_retained_clause_document_count"] = current_count
            values[f"lag1_{typ}_retained_clause_document_count"] = previous_count
            values[f"{typ}_timing_status_v9"] = timing
            values[f"{typ}_coverage_status_v9"] = coverage
            values[f"{typ}_first_gid_v9"] = info.get("first_gid", "")
            statuses.append(status)
        if any(x.endswith("unknown_coverage_or_timing") for x in statuses):
            values["requirement_observation_status_v9"] = "unknown_city_coverage"
        elif all(x.endswith("no_qualifying_record") for x in statuses):
            values["requirement_observation_status_v9"] = "no_qualifying_record_in_corpus"
        else:
            values["requirement_observation_status_v9"] = "observed_or_no_record"
        rows.append(values)
    return pd.DataFrame(rows)


def derive_b_exposure(documents: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive later-date B counts and the one-year lag from coded documents."""

    d = derive_later_years(documents)
    coverage_records = read_csv(COVERAGE_PATH)
    coverage = set(coverage_records.loc[coverage_records.local_policy_observed_v7.map(_flag), "city_key"])
    eligible = d.loc[d["window_exposure_eligible_later"].eq(1) & d["B_coded"].eq(1)].copy()
    eligible["year"] = eligible["later_year"].astype(int)
    counts = eligible.groupby(["city_key", "year"], as_index=False).agg(B_count_later=("B_coded", "sum"))
    count_lookup = {(str(row.city_key), int(row.year)): int(row.B_count_later) for row in counts.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for base in panel.itertuples(index=False):
        city, year = str(base.city_key), int(base.year)
        observed = city in coverage
        current = count_lookup.get((city, year), 0) if observed else np.nan
        previous = count_lookup.get((city, year - 1), 0) if observed else np.nan
        rows.append(
            {
                "city_key": city,
                "year": year,
                "B_count_later": current,
                "lag1_B_count_later": previous,
                "B_coverage_status_later": "observed_local_coverage" if observed else "unknown_city_coverage",
                "B_assignment_rule": "later=max(effective_year,publication_year); effective fallback; unknown effective remains missing",
                "B_lag_rule": "count of B documents assigned to year-1; not cumulative",
            }
        )
    document_dates = d[
        [
            "gid",
            "city_key",
            "effective_year_coded",
            "publication_year_coded",
            "later_year",
            "timing_basis_later",
            "window_exposure_eligible_later",
            "B_coded",
            *[x for x in ("B1_coded", "B2_coded", "B3_coded", "O_coded", "O1_coded", "O2_coded", "O3_coded") if x in d],
        ]
    ].sort_values(["city_key", "later_year", "gid"], na_position="last")
    return pd.DataFrame(rows), document_dates


def build_policy_exposure() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel = _panel_skeleton(read_csv(PANEL_PATH))
    clauses = read_csv(CLAUSE_PATH)
    registry = read_csv(REGISTRY_PATH)
    b_documents = read_csv(B_DOCUMENT_PATH)
    rs = derive_rs_exposure(clauses, registry, panel)
    b, document_dates = derive_b_exposure(b_documents, panel)
    annual = rs.merge(b, on=["city_key", "year"], how="left", validate="one_to_one")
    meta = {
        "annual_rows": int(len(annual)),
        "annual_cities": int(annual.city_key.nunique()),
        "clause_rows": int(len(clauses)),
        "clause_gids": int(clauses.gid.nunique()),
        "registry_rows": int(len(registry)),
        "registry_qualified_G_rows": int(((registry.policy_type == "G") & registry.qualified_first_gid.map(_flag)).sum()) if "qualified_first_gid" in registry else None,
        "b_document_rows": int(len(b_documents)),
        "b_document_dates_rows": int(len(document_dates)),
        "inputs": {
            "clauses": str(CLAUSE_PATH.relative_to(REPOSITORY_ROOT).as_posix()),
            "registry": str(REGISTRY_PATH.relative_to(REPOSITORY_ROOT).as_posix()),
            "b_documents": str(B_DOCUMENT_PATH.relative_to(REPOSITORY_ROOT).as_posix()),
            "coverage": str(COVERAGE_PATH.relative_to(REPOSITORY_ROOT).as_posix()),
            "annual_panel": str(PANEL_PATH.relative_to(REPOSITORY_ROOT).as_posix()),
        },
        "input_sha256": {p.name: sha256(p) for p in (CLAUSE_PATH, REGISTRY_PATH, B_DOCUMENT_PATH, COVERAGE_PATH, PANEL_PATH)},
    }
    return annual, document_dates, meta


def check_reference(annual: pd.DataFrame, model_input: pd.DataFrame | None = None) -> dict[str, Any]:
    """Check the fixed 325-row policy input without rewriting it."""

    model = read_csv(MODEL_INPUT_PATH) if model_input is None else model_input.copy()
    required = {"analysis_row_id", "city_key", "year", "R_prev_observed_v9", "S_prev_observed_v9", "lag1_B_count_later"}
    missing = sorted(required.difference(model.columns))
    if missing:
        raise ValueError(f"policy_model_input.csv missing columns: {missing}")
    model["year"] = model["year"].map(_year).astype(int)
    merged = model.merge(annual, on=["city_key", "year"], how="left", validate="one_to_one", indicator=True, suffixes=("_reference", "_derived"))
    if not merged["_merge"].eq("both").all():
        raise ValueError("Derived annual exposure does not cover every saved policy-model row")
    mismatches: dict[str, int] = {}
    for name in ("R_prev_observed_v9", "S_prev_observed_v9", "lag1_B_count_later"):
        left = merged[name + "_reference"].map(_number)
        right = merged[name + "_derived"].map(_number)
        mismatches[name] = int((left.fillna(-999999.0) != right.fillna(-999999.0)).sum())
    id_mismatch = int((model["analysis_row_id"].astype(str) != (model["city_key"].astype(str) + "__" + model["year"].astype(str))).sum())
    result = {
        "status": "ok" if not any(mismatches.values()) and id_mismatch == 0 else "mismatch",
        "model_rows": int(len(model)),
        "model_cities": int(model.city_key.nunique()),
        "analysis_row_id_mismatch": id_mismatch,
        "field_mismatches": mismatches,
        "saved_input_sha256": sha256(MODEL_INPUT_PATH) if model_input is None else None,
        "note": "The saved 325-row input is read-only; derived exposure fields are compared, never overwritten.",
    }
    return result


def run(output_dir: Path | None = None, *, validate_reference: bool = True) -> dict[str, Any]:
    annual, document_dates, metadata = build_policy_exposure()
    reference = check_reference(annual) if validate_reference else {"status": "not_checked"}
    metadata["reference_check"] = reference
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        annual.to_csv(output_dir / "policy_exposure_annual.csv", index=False, encoding="utf-8", lineterminator="\n")
        document_dates.to_csv(output_dir / "policy_exposure_document_dates.csv", index=False, encoding="utf-8", lineterminator="\n")
        (output_dir / "policy_exposure_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata

