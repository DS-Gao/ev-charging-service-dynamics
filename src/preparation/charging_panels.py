"""Prepare annual and quarterly charging panels from the public EV-air file.

This module is the public raw-charging preparation route for the study.  It
implements the documented aggregation contract: the raw ``ev_air.csv`` rows are
mapped to the supplied city keys, source dates are parsed, compatible weeks
are the rows with a valid source date and positive ``cumulative_plugs``, and
compatible ``Total == 0`` rows are retained.  No weather, policy, COVID,
model-fitting, or imputation step is performed here.

The source calendar has 53 source dates in 2017 and 2023.  Calendar counts
are made from the parsed source dates, not from ``week_only``; this preserves
the source's duplicated Stata week-52 representation in those years.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


START_YEAR = 2016
END_YEAR = 2023
EXPECTED_CITY_COUNT = 334
EXPECTED_YEARS = tuple(range(START_YEAR, END_YEAR + 1))
EXTRA_DATE_YEARS = frozenset({2017, 2023})

RAW_COLUMNS = (
    "city",
    "date",
    "Total",
    "cumulative_plugs",
    "year_only",
    "month_only",
    "week_only",
)
CROSSWALK_COLUMNS = ("aq_city", "city_key")
MEASURE_COLUMNS = ("Q", "E", "W", "A", "N", "I")


class InputContractError(ValueError):
    """Raised when a supplied input does not satisfy the known source contract."""


def _read_csv(path: Path, role: str) -> pd.DataFrame:
    if not path.exists():
        if role == "raw charging input":
            raise FileNotFoundError(
                "Raw charging input was not found: "
                f"{path}\nDownload Zenodo record 15564121 (version 2), extract "
                "Replication_AQ_EV/Data/ev_air.csv, and pass that file with "
                "--input. Required fields: "
                + ", ".join(RAW_COLUMNS)
                + "."
            )
        if role == "city crosswalk":
            raise FileNotFoundError(
                "City crosswalk was not found: "
                f"{path}\nProvide the supplied 334-city crosswalk and pass it with "
                "--crosswalk. Required fields: "
                + ", ".join(CROSSWALK_COLUMNS)
                + "."
            )
        raise FileNotFoundError(f"{role} was not found: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise InputContractError(f"Could not read {role} as a UTF-8 CSV: {path}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path, role: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise InputContractError(
            f"{role} is missing required field(s) {missing}. "
            f"Available fields in {path.name}: {list(frame.columns)}"
        )


def _nonempty_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _normalise_city_columns(frame: pd.DataFrame, source: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in source:
        out[column] = out[column].astype("string").str.strip()
    return out


def _load_inputs(input_path: Path, crosswalk_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = _read_csv(input_path, "raw charging input")
    crosswalk = _read_csv(crosswalk_path, "city crosswalk")
    _require_columns(raw, RAW_COLUMNS, input_path, "Raw charging input")
    _require_columns(crosswalk, CROSSWALK_COLUMNS, crosswalk_path, "City crosswalk")

    crosswalk = _normalise_city_columns(crosswalk, ("aq_city", "city_key"))
    if crosswalk["aq_city"].eq("").any() or crosswalk["city_key"].eq("").any():
        raise InputContractError("City crosswalk contains blank aq_city or city_key values.")
    if crosswalk["aq_city"].duplicated().any():
        dup = crosswalk.loc[crosswalk["aq_city"].duplicated(False), "aq_city"].tolist()[:10]
        raise InputContractError(f"City crosswalk aq_city values are not unique; examples: {dup}")
    if crosswalk["city_key"].duplicated().any():
        dup = crosswalk.loc[crosswalk["city_key"].duplicated(False), "city_key"].tolist()[:10]
        raise InputContractError(f"City crosswalk city_key values are not unique; examples: {dup}")
    if len(crosswalk) != EXPECTED_CITY_COUNT:
        raise InputContractError(
            f"Expected the supplied {EXPECTED_CITY_COUNT}-city crosswalk; got {len(crosswalk)} rows."
        )

    raw = raw.copy()
    raw["source_city"] = raw["city"].map(_nonempty_text)
    raw["date"] = raw["date"].map(_nonempty_text)
    raw["dt"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["dt"].isna().any():
        examples = raw.loc[raw["dt"].isna(), "date"].head(5).tolist()
        raise InputContractError(f"Raw charging input contains unparseable date value(s): {examples}")

    for column in ("Total", "cumulative_plugs"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        if raw[column].isna().any():
            raise InputContractError(f"Raw charging field {column!r} contains nonnumeric or missing values.")
    for column in ("year_only", "month_only", "week_only"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        if raw[column].isna().any():
            raise InputContractError(f"Raw charging field {column!r} contains nonnumeric or missing values.")

    raw_year = raw["dt"].dt.year.astype(int)
    raw_month = raw["dt"].dt.month.astype(int)
    if not raw_year.eq(raw["year_only"].astype(int)).all():
        raise InputContractError("Raw date year does not agree with year_only; no date rule was inferred.")
    if not raw_month.eq(raw["month_only"].astype(int)).all():
        raise InputContractError("Raw date month does not agree with month_only; no date rule was inferred.")
    if not raw_year.isin(EXPECTED_YEARS).all():
        years = sorted(raw_year.unique().tolist())
        raise InputContractError(f"Raw charging dates are outside the supported 2016--2023 window: {years}")

    city_map = crosswalk.set_index("aq_city")["city_key"].to_dict()
    raw["city_key"] = raw["source_city"].map(city_map)
    if raw["city_key"].isna().any():
        examples = raw.loc[raw["city_key"].isna(), "source_city"].drop_duplicates().head(20).tolist()
        raise InputContractError(
            "Raw charging city labels did not map exactly through aq_city -> city_key; "
            f"examples: {examples}"
        )
    raw["city_key"] = raw["city_key"].astype("string")
    expected_keys = set(crosswalk["city_key"].tolist())
    observed_keys = set(raw["city_key"].tolist())
    missing_keys = sorted(expected_keys - observed_keys)
    extra_keys = sorted(observed_keys - expected_keys)
    if missing_keys or extra_keys:
        raise InputContractError(
            "Raw charging city coverage does not match the supplied crosswalk; "
            f"missing examples={missing_keys[:10]}, extra examples={extra_keys[:10]}"
        )

    raw["year"] = raw["dt"].dt.year.astype(int)
    raw["quarter_num"] = raw["dt"].dt.quarter.astype(int)
    raw["quarter"] = raw["dt"].dt.to_period("Q").astype(str)
    raw["compatible_week"] = raw["cumulative_plugs"].gt(0)
    raw["compatible_total_zero"] = raw["compatible_week"] & raw["Total"].eq(0)
    raw = raw.sort_values(["city_key", "dt"]).reset_index(drop=True)

    expected_per_year = {year: 53 if year in EXTRA_DATE_YEARS else 52 for year in EXPECTED_YEARS}
    per_city_year = raw.groupby(["city_key", "year"], as_index=False).agg(source_date_count=("dt", "nunique"))
    per_city_year["expected_source_date_count"] = per_city_year["year"].map(expected_per_year)
    bad_calendar = per_city_year.loc[
        per_city_year["source_date_count"].ne(per_city_year["expected_source_date_count"])
    ]
    if not bad_calendar.empty:
        raise InputContractError(
            "Source-date calendar counts do not match the original rule "
            "(53 dates in 2017 and 2023; 52 otherwise); examples: "
            f"{bad_calendar.head(10).to_dict('records')}"
        )
    expected_rows = EXPECTED_CITY_COUNT * sum(expected_per_year.values())
    if len(raw) != expected_rows:
        raise InputContractError(
            f"Expected {expected_rows:,} raw city-date rows under the 334-city source calendar; got {len(raw):,}."
        )

    metadata = {
        "raw_rows": int(len(raw)),
        "raw_city_count": int(raw["city_key"].nunique()),
        "crosswalk_rows": int(len(crosswalk)),
        "raw_date_min": raw["dt"].min().strftime("%Y-%m-%d"),
        "raw_date_max": raw["dt"].max().strftime("%Y-%m-%d"),
        "raw_years": sorted(raw["year"].unique().tolist()),
        "compatible_week_rows": int(raw["compatible_week"].sum()),
        "compatible_source_total_zero_rows": int(raw["compatible_total_zero"].sum()),
        "compatibility_rule": "valid source date and cumulative_plugs > 0; compatible Total=0 retained",
        "calendar_rule": "count parsed source dates; 53 dates in 2017 and 2023, 52 otherwise",
    }
    return raw, crosswalk, metadata


def _city_info(crosswalk: pd.DataFrame) -> pd.DataFrame:
    columns = ["city_key"]
    for column in ("city", "province", "admin_code", "crosswalk_status_v3"):
        if column in crosswalk.columns:
            columns.append(column)
    info = crosswalk[columns].copy()
    if "city" not in info.columns:
        info["city"] = crosswalk["aq_city"]
    info["city"] = info["city"].map(_nonempty_text)
    info["source_city"] = crosswalk["aq_city"].copy()
    return info[["city_key", "city", "source_city"] + [c for c in columns if c not in {"city_key", "city"}]]


def _skeleton(crosswalk: pd.DataFrame, period_type: str) -> pd.DataFrame:
    info = _city_info(crosswalk)
    if period_type == "year":
        periods = pd.DataFrame({"year": list(EXPECTED_YEARS)})
    elif period_type == "quarter":
        periods = pd.DataFrame(
            [(year, quarter) for year in EXPECTED_YEARS for quarter in range(1, 5)],
            columns=["year", "quarter_num"],
        )
    else:
        raise ValueError(f"Unsupported period_type: {period_type}")
    info = info.assign(_join=1)
    periods = periods.assign(_join=1)
    out = info.merge(periods, on="_join", how="inner", validate="many_to_many").drop(columns="_join")
    return out.sort_values(["city_key", "year"] + (["quarter_num"] if period_type == "quarter" else [])).reset_index(drop=True)


def _aggregate(raw: pd.DataFrame, crosswalk: pd.DataFrame, period_type: str) -> pd.DataFrame:
    if period_type == "year":
        keys = ["city_key", "year"]
    elif period_type == "quarter":
        keys = ["city_key", "year", "quarter_num"]
    else:
        raise ValueError(f"Unsupported period_type: {period_type}")

    calendar = (
        raw.groupby(keys, as_index=False, dropna=False)
        .agg(
            W_calendar=("dt", "nunique"),
            calendar_date_min=("dt", "min"),
            calendar_date_max=("dt", "max"),
            source_week_rows=("dt", "size"),
        )
        .assign(calendar_week_count=lambda frame: frame["W_calendar"])
    )
    compatible = raw.loc[raw["compatible_week"]].copy()
    aggregate = (
        compatible.groupby(keys, as_index=False, dropna=False)
        .agg(
            Q=("Total", "sum"),
            E=("cumulative_plugs", "sum"),
            W=("compatible_week", "sum"),
            source_total_zero_weeks=("Total", lambda values: int(values.eq(0).sum())),
        )
        if not compatible.empty
        else pd.DataFrame(columns=keys + ["Q", "E", "W", "source_total_zero_weeks"])
    )
    out = _skeleton(crosswalk, period_type).merge(calendar, on=keys, how="left", validate="one_to_one")
    out = out.merge(aggregate, on=keys, how="left", validate="one_to_one")
    out["W_calendar"] = pd.to_numeric(out["W_calendar"], errors="coerce").fillna(0).astype(int)
    out["calendar_week_count"] = out["W_calendar"]
    out["source_week_rows"] = pd.to_numeric(out["source_week_rows"], errors="coerce").fillna(0).astype(int)
    out["W"] = pd.to_numeric(out["W"], errors="coerce").fillna(0).astype(int)
    out["source_total_zero_weeks"] = pd.to_numeric(out["source_total_zero_weeks"], errors="coerce").fillna(0).astype(int)
    out.loc[out["W"].eq(0), ["Q", "E"]] = np.nan

    out["A"] = np.where(out["W"].gt(0), out["Q"] / out["W"], np.nan)
    out["N"] = np.where(out["W"].gt(0), out["E"] / out["W"], np.nan)
    out["I"] = np.where(out["E"].gt(0), out["Q"] / out["E"], np.nan)
    out["log_A"] = np.log(out["A"].where(out["A"].gt(0)))
    out["log_N"] = np.log(out["N"].where(out["N"].gt(0)))
    out["log_I"] = np.log(out["I"].where(out["I"].gt(0)))
    out["source_zero_semantics"] = np.select(
        [
            out["W"].eq(0),
            out["source_total_zero_weeks"].eq(out["W"]) & out["W"].gt(0),
            out["source_total_zero_weeks"].gt(0),
        ],
        [
            "no_compatible_weeks",
            "compatible_source_total_zero_preserved_not_recode",
            "mixed_compatible_zero_and_positive_preserved",
        ],
        default="compatible_source_total_positive",
    )
    out["q_status"] = np.select(
        [out["W"].eq(0), out["Q"].eq(0)],
        ["unknown_no_compatible_weeks", "observed_compatible_zero"],
        default="observed_compatible_positive",
    )
    out["e_status"] = np.select(
        [out["W"].eq(0), out["E"].eq(0)],
        ["unknown_no_compatible_weeks", "observed_compatible_zero"],
        default="observed_compatible_positive",
    )
    out["i_status"] = np.select(
        [out["W"].eq(0), out["E"].eq(0), out["Q"].eq(0)],
        ["unknown_no_compatible_weeks", "zero_E_denominator", "zero_Q"],
        default="observed_positive_ratio",
    )
    out["period_type"] = period_type
    out["outcome_definition"] = "Q=sum(Total), E=sum(cumulative_plugs), W=compatible source dates; compatible Total=0 retained"
    out["calendar_denominator_definition"] = "count of raw ev_air source dates in the natural calendar period"
    out["compatible_denominator_definition"] = "count of rows with valid date and cumulative_plugs > 0"
    out["source_date_rule"] = "date parsed from source date field; calendar counts use source dates, not week_only"

    if period_type == "quarter":
        out["quarter"] = out["year"].astype(str) + "Q" + out["quarter_num"].astype(str)
        periods = pd.PeriodIndex(out["quarter"], freq="Q")
        out["quarter_start"] = periods.start_time.strftime("%Y-%m-%d")
        out["quarter_end"] = periods.end_time.strftime("%Y-%m-%d")
        out["quarter_period_ordinal"] = periods.astype(int)

    for column in ("calendar_date_min", "calendar_date_max"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    order = [
        "city_key",
        "city",
        "source_city",
        "province",
        "admin_code",
        "crosswalk_status_v3",
        "year",
        "quarter_num",
        "quarter",
        "quarter_start",
        "quarter_end",
        "quarter_period_ordinal",
        "W_calendar",
        "calendar_date_min",
        "calendar_date_max",
        "source_week_rows",
        "calendar_week_count",
        "Q",
        "E",
        "W",
        "source_total_zero_weeks",
        "A",
        "N",
        "I",
        "log_A",
        "log_N",
        "log_I",
        "source_zero_semantics",
        "q_status",
        "e_status",
        "i_status",
        "period_type",
        "outcome_definition",
        "calendar_denominator_definition",
        "compatible_denominator_definition",
        "source_date_rule",
    ]
    return out[[column for column in order if column in out.columns]].sort_values(keys).reset_index(drop=True)


def _max_abs_difference(left: pd.Series, right: pd.Series) -> float:
    if not isinstance(left, pd.Series):
        left = pd.Series(left)
    if not isinstance(right, pd.Series):
        right = pd.Series(right, index=left.index)
    both = left.notna() & right.notna()
    if not both.any():
        return 0.0
    return float((left.loc[both].astype(float) - right.loc[both].astype(float)).abs().max())


def _identity_checks(annual: pd.DataFrame, quarterly: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "annual_rows": int(len(annual)),
        "annual_expected_rows": EXPECTED_CITY_COUNT * len(EXPECTED_YEARS),
        "quarterly_rows": int(len(quarterly)),
        "quarterly_expected_rows": EXPECTED_CITY_COUNT * len(EXPECTED_YEARS) * 4,
        "annual_unique_keys": int(annual.duplicated(["city_key", "year"]).sum()),
        "quarterly_unique_keys": int(quarterly.duplicated(["city_key", "quarter"]).sum()),
        "annual_city_count": int(annual["city_key"].nunique()),
        "quarterly_city_count": int(quarterly["city_key"].nunique()),
    }
    date_counts = annual.loc[annual["year"].isin(EXTRA_DATE_YEARS), "W_calendar"]
    checks["calendar_2017_2023_city_years_all_53"] = bool(date_counts.eq(53).all())
    checks["calendar_other_years_city_years_all_52"] = bool(
        annual.loc[~annual["year"].isin(EXTRA_DATE_YEARS), "W_calendar"].eq(52).all()
    )
    checks["annual_quarter_addup_max_Q_diff"] = None
    checks["annual_quarter_addup_max_E_diff"] = None
    checks["annual_quarter_addup_max_W_diff"] = None
    quarter_sum = quarterly.groupby(["city_key", "year"], as_index=False)[["Q", "E", "W"]].sum(min_count=1)
    merged = annual[["city_key", "year", "Q", "E", "W"]].merge(
        quarter_sum,
        on=["city_key", "year"],
        how="left",
        suffixes=("_annual", "_quarter"),
        validate="one_to_one",
    )
    for column in ("Q", "E", "W"):
        checks[f"annual_quarter_addup_max_{column}_diff"] = _max_abs_difference(
            merged[f"{column}_annual"], merged[f"{column}_quarter"]
        )
    derived_A = annual["Q"].div(annual["W"]).where(annual["W"].gt(0))
    derived_N = annual["E"].div(annual["W"]).where(annual["W"].gt(0))
    derived_I = annual["Q"].div(annual["E"]).where(annual["E"].gt(0))
    checks["derived_A_max_abs_error"] = _max_abs_difference(annual["A"], derived_A)
    checks["derived_N_max_abs_error"] = _max_abs_difference(annual["N"], derived_N)
    checks["derived_I_max_abs_error"] = _max_abs_difference(annual["I"], derived_I)
    checks["status"] = "PASS" if all(
        [
            checks["annual_rows"] == checks["annual_expected_rows"],
            checks["quarterly_rows"] == checks["quarterly_expected_rows"],
            checks["annual_unique_keys"] == 0,
            checks["quarterly_unique_keys"] == 0,
            checks["annual_city_count"] == EXPECTED_CITY_COUNT,
            checks["quarterly_city_count"] == EXPECTED_CITY_COUNT,
            checks["calendar_2017_2023_city_years_all_53"],
            checks["calendar_other_years_city_years_all_52"],
            all(checks[f"annual_quarter_addup_max_{column}_diff"] <= 1e-8 for column in ("Q", "E", "W")),
            checks["derived_A_max_abs_error"] <= 1e-8,
            checks["derived_N_max_abs_error"] <= 1e-8,
            checks["derived_I_max_abs_error"] <= 1e-8,
        ]
    ) else "FAIL"
    return checks


def _reference_checks(frame: pd.DataFrame, reference_path: Path | None, period_type: str) -> list[dict[str, Any]]:
    if reference_path is None:
        return [{"period_type": period_type, "status": "NOT_REQUESTED", "reference": ""}]
    if not reference_path.exists():
        return [{"period_type": period_type, "status": "NOT_AVAILABLE", "reference": reference_path.name}]
    reference = _read_csv(reference_path, f"{period_type} reference panel")
    keys = ["city_key", "year"] if period_type == "year" else ["city_key", "quarter"]
    required = keys + list(MEASURE_COLUMNS)
    _require_columns(reference, required, reference_path, f"{period_type} reference panel")
    left = frame[keys + list(MEASURE_COLUMNS)].copy()
    right = reference[keys + list(MEASURE_COLUMNS)].copy()
    left["city_key"] = left["city_key"].astype(str)
    right["city_key"] = right["city_key"].astype(str)
    if period_type == "year":
        left["year"] = pd.to_numeric(left["year"], errors="raise").astype(int)
        right["year"] = pd.to_numeric(right["year"], errors="raise").astype(int)
    if right.duplicated(keys).any():
        raise InputContractError(f"{period_type} reference panel has duplicate keys: {keys}")
    merged = left.merge(right, on=keys, how="outer", suffixes=("_raw", "_reference"), indicator=True)
    rows: list[dict[str, Any]] = []
    for column in MEASURE_COLUMNS:
        raw_values = merged[f"{column}_raw"]
        ref_values = merged[f"{column}_reference"]
        equal = (raw_values.isna() & ref_values.isna()) | (
            raw_values.notna() & ref_values.notna() & np.isclose(
                pd.to_numeric(raw_values, errors="coerce"),
                pd.to_numeric(ref_values, errors="coerce"),
                rtol=1e-10,
                atol=1e-8,
                equal_nan=False,
            )
        )
        rows.append(
            {
                "period_type": period_type,
                "reference": reference_path.name,
                "measure": column,
                "rows_compared": int((merged["_merge"] == "both").sum()),
                "raw_only_rows": int((merged["_merge"] == "left_only").sum()),
                "reference_only_rows": int((merged["_merge"] == "right_only").sum()),
                "max_abs_difference": _max_abs_difference(raw_values, ref_values),
                "mismatches": int((~equal).sum()),
                "status": "PASS" if bool(equal.all()) else "FAIL",
            }
        )
    return rows


def build_panels(
    input_path: Path,
    crosswalk_path: Path,
    output_dir: Path,
    annual_reference: Path | None = None,
    quarterly_reference: Path | None = None,
) -> dict[str, Any]:
    """Build raw-only annual/quarterly panels and write QC artifacts."""

    raw, crosswalk, input_metadata = _load_inputs(input_path, crosswalk_path)
    annual = _aggregate(raw, crosswalk, "year")
    quarterly = _aggregate(raw, crosswalk, "quarter")
    qc = _identity_checks(annual, quarterly)
    reference_rows = _reference_checks(annual, annual_reference, "year")
    reference_rows.extend(_reference_checks(quarterly, quarterly_reference, "quarter"))

    output_dir.mkdir(parents=True, exist_ok=True)
    annual_path = output_dir / "annual_panel_from_raw.csv"
    quarterly_path = output_dir / "quarterly_panel_from_raw.csv"
    checks_path = output_dir / "charging_panel_reference_checks.csv"
    qc_path = output_dir / "charging_panel_qc.json"
    metadata_path = output_dir / "charging_panel_metadata.json"
    annual.to_csv(annual_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    quarterly.to_csv(quarterly_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    pd.DataFrame(reference_rows).to_csv(checks_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    qc_payload = {
        "status": qc["status"],
        "checks": qc,
        "reference_checks_status": "PASS" if all(row.get("status") in {"PASS", "NOT_REQUESTED", "NOT_AVAILABLE"} for row in reference_rows) else "FAIL",
        "reference_checks": reference_rows,
    }
    qc_path.write_text(json.dumps(_as_jsonable(qc_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "script_role": "raw charging preparation only; no weather, policy, COVID, or model fitting",
        "source": {
            "charging_archive": "https://doi.org/10.5281/zenodo.15564121",
            "input_filename": input_path.name,
            "input_sha256": _sha256(input_path),
            "crosswalk_filename": crosswalk_path.name,
            "crosswalk_sha256": _sha256(crosswalk_path),
        },
        "date_rule": "parse date as source date; use source-date counts, retaining 53 dates in 2017 and 2023",
        "compatibility_rule": "valid source date and cumulative_plugs > 0; compatible Total=0 rows retained",
        "measures": "Q=sum(Total), E=sum(cumulative_plugs), W=count compatible source dates, A=Q/W, N=E/W, I=Q/E",
        "outputs": {
            "annual": annual_path.name,
            "quarterly": quarterly_path.name,
            "reference_checks": checks_path.name,
            "qc": qc_path.name,
        },
        "input_summary": input_metadata,
        "other_inputs_not_consumed": [
            "weather_city_year_v4.csv is required later for weather-adjusted models",
            "policy corpus metadata/indicators are required later for policy models",
            "no policy full text is read or copied by this preparation route",
        ],
    }
    metadata_path.write_text(json.dumps(_as_jsonable(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "annual_path": str(annual_path),
        "quarterly_path": str(quarterly_path),
        "qc_path": str(qc_path),
        "metadata_path": str(metadata_path),
        "qc_status": qc["status"],
        "reference_checks_status": qc_payload["reference_checks_status"],
        "annual_rows": int(len(annual)),
        "quarterly_rows": int(len(quarterly)),
    }

