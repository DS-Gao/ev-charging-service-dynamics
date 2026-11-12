"""Prepare the public weather and COVID context controls.

This module is the standalone public-input route for the two context inputs
used by the service-conversion analysis. It adapts the validated V4 NASA
POWER aggregation and V11 OxCGRT exposure construction without changing their
scientific rules:

* NASA POWER daily JSON responses are mapped to the nearest MERRA-2 grid using
  the supplied city-point file. Non-precipitation means require at least 95%
  valid calendar-day coverage, annual precipitation sums and HDD18/CDD24 are
  retained only for complete calendars, and missing days are never annualized.
* OxCGRT uses the compact StringencyIndex_Average for Chinese STATE_TOTAL
  province rows. The exact source-compatible EV-air dates are selected with
  cumulative_plugs > 0; Total is not a filter, so compatible Total-zero weeks
  remain in the exposure mean. Pre-2020 values are labelled structural zeros,
  while dates after 2023-02-28 remain missing.

The public route consumes user-supplied raw files rather than downloading a
large archive. NASA POWER documentation and OxCGRT source URLs, required
fields, the fixed OxCGRT source commit, and all missing-value rules are
recorded in the generated metadata and source registry. No model fitting,
weather imputation, policy-text download, or outage workflow is included.
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
YEARS = tuple(range(START_YEAR, END_YEAR + 1))

NASA_POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_DOCUMENTATION = (
    "https://power.larc.nasa.gov/docs/services/api/temporal/daily/",
    "https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/",
    "https://power.larc.nasa.gov/docs/services/",
)
NASA_PARAMETERS = ("T2M", "PRECTOTCORR", "WS10M", "RH2M")
NASA_UNITS = {
    "T2M": "C",
    "PRECTOTCORR": "mm/day",
    "WS10M": "m/s",
    "RH2M": "%",
}
NASA_DAILY_COVERAGE_THRESHOLD = 0.95
HDD_BASE_C = 18.0
CDD_BASE_C = 24.0

OXCGRT_COMMIT = "e7f66ee39654293b5c068efd2f195bd591dc27f6"
OXCGRT_COMPACT_URL = (
    "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/"
    f"{OXCGRT_COMMIT}/data/OxCGRT_compact_subnational_v1.csv"
)
OXCGRT_CODEBOOK_URL = (
    "https://github.com/OxCGRT/covid-policy-dataset/blob/"
    f"{OXCGRT_COMMIT}/documentation_and_codebook.md"
)
OXCGRT_TRACKER_URL = "https://www.bsg.ox.ac.uk/research/covid-19-government-response-tracker"
OXCGRT_CUTOFF = pd.Timestamp("2023-02-28")
OXCGRT_STRINGENCY_COLUMN = "StringencyIndex_Average"

WEATHER_COORDINATE_COLUMNS = ("city_key", "latitude", "longitude", "coordinate_status")
CITY_CROSSWALK_COLUMNS = ("aq_city", "city_key", "city", "province")
EV_AIR_COLUMNS = (
    "city",
    "date",
    "Total",
    "cumulative_plugs",
    "year_only",
    "month_only",
    "week_only",
)
OXCGRT_COLUMNS = (
    "CountryName",
    "CountryCode",
    "RegionName",
    "RegionCode",
    "Jurisdiction",
    "Date",
    OXCGRT_STRINGENCY_COLUMN,
    "C5M_Close.public.transport",
    "C5M_Flag",
    "C6M_Stay.at.home.requirements",
    "C6M_Flag",
    "C7M_Restrictions.on.internal.movement",
    "C7M_Flag",
)

PROVINCE_CROSSWALK = {
    "北京市": ("Beijing", "CHN_BJ"),
    "天津市": ("Tianjin", "CHN_TJ"),
    "河北省": ("Hebei", "CHN_HE"),
    "山西省": ("Shanxi", "CHN_SX"),
    "内蒙古自治区": ("Inner Mongolia", "CHN_NM"),
    "辽宁省": ("Liaoning", "CHN_LN"),
    "吉林省": ("Jilin", "CHN_JL"),
    "黑龙江省": ("Heilongjiang", "CHN_HL"),
    "上海市": ("Shanghai", "CHN_SH"),
    "江苏省": ("Jiangsu", "CHN_JS"),
    "浙江省": ("Zhejiang", "CHN_ZJ"),
    "安徽省": ("Anhui", "CHN_AH"),
    "福建省": ("Fujian", "CHN_FJ"),
    "江西省": ("Jiangxi", "CHN_JX"),
    "山东省": ("Shandong", "CHN_SD"),
    "河南省": ("Henan", "CHN_HA"),
    "湖北省": ("Hubei", "CHN_HB"),
    "湖南省": ("Hunan", "CHN_HN"),
    "广东省": ("Guangdong", "CHN_GD"),
    "广西壮族自治区": ("Guangxi", "CHN_GX"),
    "海南省": ("Hainan", "CHN_HI"),
    "重庆市": ("Chongqing", "CHN_CQ"),
    "四川省": ("Sichuan", "CHN_SC"),
    "贵州省": ("Guizhou", "CHN_GZ"),
    "云南省": ("Yunnan", "CHN_YN"),
    "西藏自治区": ("Tibet", "CHN_XZ"),
    "陕西省": ("Shaanxi", "CHN_SN"),
    "甘肃省": ("Gansu", "CHN_GS"),
    "青海省": ("Qinghai", "CHN_QH"),
    "宁夏回族自治区": ("Ningxia", "CHN_NX"),
    "新疆维吾尔自治区": ("Xinjiang", "CHN_XJ"),
}


class InputContractError(ValueError):
    """Raised when a supplied context input violates its known contract."""


def _read_csv(
    path: Path,
    role: str,
    required: Iterable[str] | None = None,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{role} was not found: {path}. "
            "Pass an explicit public-relative input path."
        )
    try:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
            usecols=list(usecols) if usecols is not None else None,
        )
    except ValueError as exc:
        raise InputContractError(
            f"Could not read required fields from {role} {path.name}: {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise InputContractError(f"Could not read {role} as UTF-8 CSV: {path}: {exc}") from exc
    if required is not None:
        _require_columns(frame, required, path, role)
    return frame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path, role: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise InputContractError(
            f"{role} is missing required field(s) {missing}. "
            f"Available fields in {path.name}: {list(frame.columns)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if not isinstance(value, (str, bytes, bool, list, tuple, dict)) and pd.isna(value):
        return None
    return value


def _public_path(path: Path) -> str:
    """Return a path token without leaking a machine-specific absolute path."""

    return path.name


def _normalise_date_series(values: pd.Series, role: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise InputContractError(f"{role} contains unparseable dates")
    return parsed


def build_city_grid_mapping(coordinates_path: Path) -> pd.DataFrame:
    """Apply the exact V4 coordinate eligibility and nearest-grid rules."""

    points = _read_csv(
        coordinates_path,
        "weather city-point input",
        required=WEATHER_COORDINATE_COLUMNS,
    ).copy()
    if points["city_key"].duplicated().any():
        raise InputContractError("Weather city-point input has duplicate city_key values")
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    # V4 retains the one unresolved registry city in the skeleton. Missing
    # coordinates therefore make a city ineligible rather than dropping it or
    # imputing a point.
    points["weather_coordinate_eligible"] = (
        points["coordinate_status"].astype(str).eq("available")
        & points["latitude"].between(15, 55)
        & points["longitude"].between(70, 140)
    )
    points["grid_latitude"] = (
        np.floor((points["latitude"] + 90) / 0.5 + 0.5) * 0.5 - 90
    ).where(points["weather_coordinate_eligible"])
    points["grid_longitude"] = (
        np.floor((points["longitude"] + 180) / 0.625 + 0.5) * 0.625 - 180
    ).where(points["weather_coordinate_eligible"])

    def grid_name(row: pd.Series) -> str:
        if not bool(row["weather_coordinate_eligible"]):
            return ""
        return f"MERRA2_{float(row['grid_latitude']):.3f}_{float(row['grid_longitude']):.3f}"

    points["weather_grid_id"] = points.apply(grid_name, axis=1)
    # Do not copy the historical absolute coordinate_source field into a public
    # output. It is provenance metadata, not a numerical input to the rule.
    drop_columns = [column for column in ("coordinate_source",) if column in points.columns]
    return points.drop(columns=drop_columns)


def _weather_grid_from_filename(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0] != "MERRA2":
        raise InputContractError(
            f"NASA POWER JSON filename {path.name} does not follow the expected "
            "MERRA2_<latitude>_<longitude>_<start>_<end>.json pattern"
        )
    return "_".join(parts[:3])


def load_weather_jsons(weather_json_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    """Load cached NASA POWER daily JSON responses without network access."""

    if not weather_json_dir.exists():
        raise FileNotFoundError(
            f"NASA POWER JSON directory was not found: {weather_json_dir}. "
            "Download/cache daily point JSON responses from "
            f"{NASA_POWER_ENDPOINT} using UTC, parameters "
            f"{', '.join(NASA_PARAMETERS)}, then pass the directory with --weather-json-dir."
        )
    paths = sorted(weather_json_dir.glob("MERRA2_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No MERRA2_*.json NASA POWER responses were found in {weather_json_dir}. "
            "Required JSON keys are properties.parameter.T2M, PRECTOTCORR, WS10M, RH2M; "
            "the response fill value must be treated as missing."
        )
    grid_data: dict[str, list[pd.DataFrame]] = {}
    metadata: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise InputContractError(f"Could not parse NASA POWER JSON {path.name}: {exc}") from exc
        header = payload.get("header", {})
        parameters = payload.get("properties", {}).get("parameter")
        if not isinstance(parameters, dict):
            raise InputContractError(f"NASA POWER JSON {path.name} lacks properties.parameter")
        missing = [parameter for parameter in NASA_PARAMETERS if parameter not in parameters]
        if missing:
            raise InputContractError(
                f"NASA POWER JSON {path.name} is missing parameter(s) {missing}; "
                f"required parameters are {list(NASA_PARAMETERS)}"
            )
        units = {
            parameter: payload.get("parameters", {}).get(parameter, {}).get("units")
            for parameter in NASA_PARAMETERS
        }
        if units != NASA_UNITS:
            raise InputContractError(
                f"NASA POWER JSON {path.name} has units {units}; expected {NASA_UNITS}"
            )
        try:
            daily = pd.DataFrame(parameters).reindex(columns=NASA_PARAMETERS)
            daily.index = pd.to_datetime(
                daily.index.astype(str), format="%Y%m%d", errors="coerce"
            )
        except Exception as exc:
            raise InputContractError(f"Could not construct daily NASA frame from {path.name}: {exc}") from exc
        if daily.index.isna().any():
            raise InputContractError(f"NASA POWER JSON {path.name} has unparseable daily keys")
        daily = daily.apply(pd.to_numeric, errors="coerce")
        fill_value = header.get("fill_value", -999)
        if fill_value is not None:
            daily = daily.replace(fill_value, np.nan)
        grid_id = _weather_grid_from_filename(path)
        grid_data.setdefault(grid_id, []).append(daily)
        metadata.append(
            {
                "grid_id": grid_id,
                "file": path.name,
                "api_version": header.get("api", {}).get("version", "unknown"),
                "sources": ";".join(header.get("sources", [])),
                "time_standard": header.get("time_standard", "unknown"),
                "fill_value": fill_value,
                "units": json.dumps(
                    {key: payload.get("parameters", {}).get(key, {}) for key in NASA_PARAMETERS},
                    ensure_ascii=False,
                ),
                "sha256": _sha256(path),
            }
        )
    frames = {
        grid_id: pd.concat(parts)
        .sort_index()
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
        for grid_id, parts in grid_data.items()
    }
    return frames, pd.DataFrame(metadata).sort_values("grid_id").reset_index(drop=True), paths


def build_weather_city_year(points: pd.DataFrame, grid_frames: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the V4 annual weather aggregation and missingness semantics."""

    rows: list[dict[str, Any]] = []
    for _, point in points.iterrows():
        daily_grid = grid_frames.get(str(point["weather_grid_id"]))
        for year in YEARS:
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
            expected_days = len(dates)
            if daily_grid is None:
                daily = pd.DataFrame(np.nan, index=dates, columns=NASA_PARAMETERS)
            else:
                daily = daily_grid.reindex(dates)
            grid_id = str(point["weather_grid_id"])
            versions = metadata.loc[metadata["grid_id"].eq(grid_id), "api_version"]
            record: dict[str, Any] = {
                "city_key": point["city_key"],
                "year": year,
                "weather_provider": "NASA_POWER_Daily",
                "weather_grid_id": grid_id,
                "weather_time_standard": "UTC",
                "weather_grid_resolution": "0.5deg_lat_x_0.625deg_lon",
                "weather_spatial_method": "nearest_grid_to_administrative_polygon_centroid",
                "weather_calendar_days": expected_days,
                "weather_api_version": "|".join(sorted(set(versions.astype(str)))) if len(versions) else "",
                "weather_source_dataset": "MERRA2;POWER",
            }
            field_names = {
                "T2M": ("temperature", "c"),
                "PRECTOTCORR": ("precipitation", None),
                "WS10M": ("wind_speed", "ms"),
                "RH2M": ("relative_humidity", "pct"),
            }
            for parameter, (name, unit) in field_names.items():
                values = daily[parameter]
                valid_days = int(values.notna().sum())
                coverage = valid_days / expected_days
                record[f"weather_{name}_valid_days"] = valid_days
                record[f"weather_{name}_coverage"] = coverage
                if parameter == "PRECTOTCORR":
                    record["weather_precipitation_observed_days_mm"] = values.sum(min_count=1)
                    record["weather_precipitation_sum_mm"] = (
                        values.sum(min_count=1) if valid_days == expected_days else np.nan
                    )
                else:
                    record[f"weather_{name}_mean_{unit}"] = (
                        values.mean() if coverage >= NASA_DAILY_COVERAGE_THRESHOLD else np.nan
                    )
            temperature = daily["T2M"]
            complete_temperature = int(temperature.notna().sum()) == expected_days
            record["weather_hdd18_degree_days"] = (
                np.maximum(HDD_BASE_C - temperature, 0).sum(min_count=1)
                if complete_temperature
                else np.nan
            )
            record["weather_cdd24_degree_days"] = (
                np.maximum(temperature - CDD_BASE_C, 0).sum(min_count=1)
                if complete_temperature
                else np.nan
            )
            record["weather_all_valid_days"] = int(daily.notna().all(axis=1).sum())
            eligible = bool(point["weather_coordinate_eligible"])
            record["weather_status"] = (
                "unresolved_city_coordinate"
                if not eligible
                else (
                    "complete"
                    if bool(daily.notna().all().all())
                    else ("partial_no_annualization" if bool(daily.notna().any().any()) else "fetch_missing")
                )
            )
            rows.append(record)
    return pd.DataFrame(rows).sort_values(["city_key", "year"]).reset_index(drop=True)


def read_oxcgrt_source(oxcgrt_path: Path) -> pd.DataFrame:
    """Read and validate the compact OxCGRT source at the fixed commit."""

    raw = _read_csv(
        oxcgrt_path,
        "OxCGRT compact source CSV",
        required=OXCGRT_COLUMNS,
        usecols=OXCGRT_COLUMNS,
    ).copy()
    raw = raw.loc[
        raw["CountryCode"].eq("CHN") & raw["Jurisdiction"].eq("STATE_TOTAL")
    ].copy()
    raw["source_date_dt"] = pd.to_datetime(
        raw["Date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    if raw["source_date_dt"].isna().any():
        raise InputContractError("OxCGRT CHN STATE_TOTAL rows contain unparseable Date values")
    expected_codes = {value[1] for value in PROVINCE_CROSSWALK.values()}
    observed_codes = set(raw["RegionCode"].dropna())
    if observed_codes != expected_codes:
        raise InputContractError(
            "OxCGRT CHN STATE_TOTAL RegionCode values do not match the fixed 31-province "
            f"crosswalk; missing={sorted(expected_codes - observed_codes)}, "
            f"extra={sorted(observed_codes - expected_codes)}"
        )
    raw["province_cn"] = raw["RegionCode"].map(
        {value[1]: key for key, value in PROVINCE_CROSSWALK.items()}
    )
    raw[OXCGRT_STRINGENCY_COLUMN] = pd.to_numeric(raw[OXCGRT_STRINGENCY_COLUMN], errors="coerce")
    if raw[OXCGRT_STRINGENCY_COLUMN].isna().any():
        raise InputContractError("OxCGRT StringencyIndex_Average has missing or non-numeric STATE_TOTAL values")
    raw["covid_stringency_source"] = raw[OXCGRT_STRINGENCY_COLUMN]
    raw["covid_source_status"] = np.where(
        raw[OXCGRT_STRINGENCY_COLUMN].eq(0), "source_measured_zero", "source_measured"
    )
    raw = raw.rename(
        columns={
            "C5M_Close.public.transport": "covid_c5m_close_public_transport",
            "C5M_Flag": "covid_c5m_targeting_flag",
            "C6M_Stay.at.home.requirements": "covid_c6m_stay_at_home",
            "C6M_Flag": "covid_c6m_targeting_flag",
            "C7M_Restrictions.on.internal.movement": "covid_c7m_internal_movement",
            "C7M_Flag": "covid_c7m_internal_movement_flag",
        }
    )
    raw["source_date"] = raw["source_date_dt"].dt.strftime("%Y-%m-%d")
    raw = raw[
        [
            "CountryName",
            "CountryCode",
            "RegionName",
            "RegionCode",
            "province_cn",
            "Jurisdiction",
            "source_date",
            "source_date_dt",
            "covid_stringency_source",
            OXCGRT_STRINGENCY_COLUMN,
            "covid_source_status",
            "covid_c5m_close_public_transport",
            "covid_c5m_targeting_flag",
            "covid_c6m_stay_at_home",
            "covid_c6m_targeting_flag",
            "covid_c7m_internal_movement",
            "covid_c7m_internal_movement_flag",
        ]
    ].sort_values(["RegionCode", "source_date"])
    raw = raw.rename(columns={"covid_c7m_internal_movement_flag": "covid_c7m_targeting_flag"})
    if raw["source_date"].min() != "2020-01-01" or raw["source_date"].max() != "2023-02-28":
        raise InputContractError(
            "OxCGRT source date range must be 2020-01-01 through 2023-02-28; "
            f"observed {raw['source_date'].min()} through {raw['source_date'].max()}"
        )
    if len(raw) != 35805 or raw["RegionCode"].nunique() != 31:
        raise InputContractError(
            "Unexpected OxCGRT CHN STATE_TOTAL shape; expected 35,805 rows across 31 regions, "
            f"observed {len(raw)} rows across {raw['RegionCode'].nunique()} regions"
        )
    return raw.reset_index(drop=True)


def build_province_crosswalk(city_crosswalk: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Verify the explicit Chinese-province to OxCGRT region mapping."""

    _require_columns(city_crosswalk, CITY_CROSSWALK_COLUMNS, Path("city_crosswalk"), "city crosswalk")
    province_counts = (
        city_crosswalk[["province", "city_key"]]
        .drop_duplicates()
        .groupby("province", as_index=False)
        .agg(panel_city_count=("city_key", "nunique"))
    )
    if province_counts["province"].isna().any() or province_counts["province"].eq("").any():
        raise InputContractError("City crosswalk has missing province values")
    province_counts["panel_row_count"] = province_counts["panel_city_count"].astype(int) * len(YEARS)
    province_counts["ox_region_name"] = province_counts["province"].map(
        {key: value[0] for key, value in PROVINCE_CROSSWALK.items()}
    )
    province_counts["ox_region_code"] = province_counts["province"].map(
        {key: value[1] for key, value in PROVINCE_CROSSWALK.items()}
    )
    if province_counts["ox_region_code"].isna().any():
        missing = province_counts.loc[province_counts["ox_region_code"].isna(), "province"].tolist()
        raise InputContractError(f"City crosswalk contains province(s) outside the fixed OxCGRT crosswalk: {missing}")
    source_regions = daily[["RegionName", "RegionCode"]].drop_duplicates()
    province_counts = province_counts.rename(columns={"province": "province_cn"})
    province_counts = province_counts.merge(
        source_regions,
        left_on=["ox_region_name", "ox_region_code"],
        right_on=["RegionName", "RegionCode"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    province_counts["crosswalk_status"] = np.where(
        province_counts["_merge"].eq("both"),
        "verified_exact_region_code_name",
        "mapping_error",
    )
    province_counts["country_code"] = "CHN"
    province_counts["jurisdiction"] = "STATE_TOTAL"
    province_counts["matching_rule"] = "explicit province_cn -> OxCGRT RegionCode and RegionName"
    output = province_counts[
        [
            "province_cn",
            "ox_region_name",
            "ox_region_code",
            "country_code",
            "jurisdiction",
            "panel_city_count",
            "panel_row_count",
            "crosswalk_status",
            "matching_rule",
        ]
    ].sort_values("ox_region_code")
    if output["crosswalk_status"].ne("verified_exact_region_code_name").any():
        raise InputContractError("At least one city-crosswalk province failed exact OxCGRT region verification")
    return output.reset_index(drop=True)


def prepare_compatible_week_details(
    ev_air_path: Path,
    city_crosswalk: pd.DataFrame,
    province_crosswalk: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Join source-compatible EV-air dates to exact province-date exposure."""

    ev = _read_csv(
        ev_air_path,
        "EV-air charging source",
        required=EV_AIR_COLUMNS,
        usecols=EV_AIR_COLUMNS,
    ).copy()
    ev["source_date_dt"] = _normalise_date_series(ev["date"], "EV-air date")
    ev["year"] = pd.to_numeric(ev["year_only"], errors="coerce")
    ev["cumulative_plugs"] = pd.to_numeric(ev["cumulative_plugs"], errors="coerce")
    ev["Total"] = pd.to_numeric(ev["Total"], errors="coerce")
    if ev[["year", "cumulative_plugs", "Total"]].isna().any().any():
        raise InputContractError(
            "EV-air source has missing or non-numeric year_only, cumulative_plugs, or Total values"
        )
    ev["year"] = ev["year"].astype(int)
    if not ev["year"].isin(YEARS).all():
        raise InputContractError(f"EV-air year_only values must be within {YEARS}")
    ev = ev.loc[ev["cumulative_plugs"].gt(0) & ev["source_date_dt"].notna()].copy()
    if ev.empty:
        raise InputContractError("EV-air source contains no compatible rows with cumulative_plugs > 0")

    city_map = city_crosswalk.rename(columns={"aq_city": "source_city"})[
        ["source_city", "city_key", "city", "province"]
    ].drop_duplicates()
    if city_map["source_city"].duplicated().any():
        raise InputContractError("City crosswalk has duplicate aq_city values")
    ev = ev.rename(columns={"city": "source_city"}).merge(
        city_map,
        on="source_city",
        how="left",
        validate="many_to_one",
    )
    if ev["city_key"].isna().any():
        missing = sorted(ev.loc[ev["city_key"].isna(), "source_city"].dropna().unique().tolist())
        raise InputContractError(
            "Some compatible EV-air city labels did not map exactly to the supplied crosswalk; "
            f"examples={missing[:10]}"
        )

    province_map = province_crosswalk.rename(columns={"ox_region_code": "RegionCode"})[
        ["province_cn", "RegionCode", "ox_region_name"]
    ]
    ev = ev.merge(
        province_map,
        left_on="province",
        right_on="province_cn",
        how="left",
        validate="many_to_one",
    )
    if ev["RegionCode"].isna().any():
        raise InputContractError("Some EV-air city provinces did not map to an OxCGRT region")

    daily_for_merge = daily.rename(columns={"RegionCode": "covid_region_code"}).copy()
    covid_columns = [
        "covid_region_code",
        "source_date_dt",
        "covid_stringency_source",
        "covid_c5m_close_public_transport",
        "covid_c5m_targeting_flag",
        "covid_c6m_stay_at_home",
        "covid_c6m_targeting_flag",
        "covid_c7m_internal_movement",
        "covid_c7m_targeting_flag",
        "covid_source_status",
    ]
    details = ev.merge(
        daily_for_merge[covid_columns],
        left_on=["RegionCode", "source_date_dt"],
        right_on=["covid_region_code", "source_date_dt"],
        how="left",
        validate="many_to_one",
    )
    details["source_date"] = details["source_date_dt"].dt.strftime("%Y-%m-%d")
    details["source_date_year"] = details["source_date_dt"].dt.year.astype(int)
    details["covid_source_date_status"] = np.select(
        [
            details["source_date_year"].le(2019),
            details["covid_stringency_source"].notna(),
            details["source_date_dt"].gt(OXCGRT_CUTOFF),
        ],
        [
            "pre_covid_structural_zero",
            details["covid_source_status"].fillna("source_measured"),
            "out_of_source_coverage",
        ],
        default="source_missing_on_in_range_date",
    )
    details["covid_stringency_source_date_value"] = np.where(
        details["source_date_year"].le(2019),
        0.0,
        details["covid_stringency_source"],
    )
    details["covid_stringency_source_date_is_zero"] = details[
        "covid_stringency_source_date_value"
    ].eq(0)
    details["compatible_week"] = True
    details["compatible_total_zero"] = details["Total"].eq(0)
    details = details.rename(columns={"RegionCode": "ox_region_code", "RegionName": "ox_region_name"})
    keep = [
        "city_key",
        "city",
        "source_city",
        "province",
        "province_cn",
        "ox_region_name",
        "ox_region_code",
        "year",
        "source_date",
        "source_date_dt",
        "month_only",
        "week_only",
        "Total",
        "cumulative_plugs",
        "compatible_week",
        "compatible_total_zero",
        "covid_stringency_source_date_value",
        "covid_stringency_source_date_is_zero",
        "covid_source_date_status",
        "covid_stringency_source",
        "covid_c5m_close_public_transport",
        "covid_c5m_targeting_flag",
        "covid_c6m_stay_at_home",
        "covid_c6m_targeting_flag",
        "covid_c7m_internal_movement",
        "covid_c7m_targeting_flag",
    ]
    return details[keep].sort_values(["city_key", "source_date_dt"]).reset_index(drop=True)


def build_exposure_coverage(city_crosswalk: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    """Build the complete city-year exposure skeleton with V11 statuses."""

    city_map = city_crosswalk.rename(columns={"aq_city": "source_city"})[
        ["city_key", "city", "province", "source_city"]
    ].drop_duplicates()
    year_frame = pd.DataFrame({"year": YEARS})
    city_map["_join"] = 1
    year_frame["_join"] = 1
    skeleton = city_map.merge(year_frame, on="_join", how="outer").drop(columns="_join")
    keys = ["city_key", "year"]
    if details.empty:
        grouped = pd.DataFrame(columns=keys)
    else:
        grouped = details.groupby(keys, as_index=False).agg(
            detail_week_count=("source_date", "size"),
            observed_source_week_count=(
                "covid_stringency_source_date_value",
                lambda values: int(values.notna().sum()),
            ),
            missing_source_week_count=(
                "covid_stringency_source_date_value",
                lambda values: int(values.isna().sum()),
            ),
            covid_stringency_source_date_mean=(
                "covid_stringency_source_date_value",
                "mean",
            ),
            covid_stringency_source_date_min=("source_date", "min"),
            covid_stringency_source_date_max=("source_date", "max"),
            covid_source_date_n=(
                "covid_stringency_source_date_value",
                lambda values: int(values.notna().sum()),
            ),
            covid_source_date_zero_count=(
                "covid_stringency_source_date_value",
                lambda values: int(values.eq(0).sum()),
            ),
            compatible_total_zero_week_count=("compatible_total_zero", "sum"),
        )
        statuses = details.groupby(keys)["covid_source_date_status"].agg(
            lambda values: set(values.dropna())
        ).reset_index(name="status_set")
        grouped = grouped.merge(statuses, on=keys, how="left", validate="one_to_one")
    output = skeleton.merge(grouped, on=keys, how="left", validate="one_to_one")
    count_columns = [
        "detail_week_count",
        "observed_source_week_count",
        "missing_source_week_count",
        "covid_source_date_n",
        "covid_source_date_zero_count",
        "compatible_total_zero_week_count",
    ]
    for column in count_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0).astype(int)
    output["compatible_week_count"] = output["detail_week_count"]
    output["W_zero_flag"] = output["compatible_week_count"].eq(0)
    output["exposure_coverage_ratio"] = np.where(
        output["compatible_week_count"].gt(0),
        output["observed_source_week_count"] / output["compatible_week_count"],
        np.nan,
    )
    output["exposure_complete"] = output["compatible_week_count"].gt(0) & output[
        "missing_source_week_count"
    ].eq(0)
    output["covid_exposure_complete"] = output["exposure_complete"]
    output["covid_stringency_source_date_mean_complete"] = np.where(
        output["exposure_complete"],
        output["covid_stringency_source_date_mean"],
        np.nan,
    )
    output["covid_exposure_status"] = np.select(
        [
            output["compatible_week_count"].eq(0),
            output["year"].le(2019) & output["exposure_complete"],
            output["exposure_complete"] & output["year"].between(2020, 2022),
            output["exposure_complete"] & output["year"].eq(2023),
            output["observed_source_week_count"].gt(0) & output["missing_source_week_count"].gt(0),
            output["observed_source_week_count"].eq(0) & output["missing_source_week_count"].gt(0),
        ],
        [
            "no_compatible_weeks_W0",
            "pre_covid_structural_zero",
            "source_measured_complete",
            "source_measured_complete_2023",
            "source_measured_partial_out_of_source_coverage",
            "out_of_source_coverage",
        ],
        default="source_date_status_unclassified",
    )
    status_values = output["status_set"] if "status_set" in output.columns else pd.Series("", index=output.index)
    output["covid_source_date_status_set"] = status_values.map(
        lambda value: ";".join(sorted(value)) if isinstance(value, set) else ""
    )
    if "status_set" in output.columns:
        output = output.drop(columns="status_set")
    output["covid_exposure_na_reason"] = np.select(
        [
            output["compatible_week_count"].eq(0),
            output["missing_source_week_count"].gt(0),
        ],
        ["W0_no_compatible_week", "source_date_missing_or_out_of_coverage"],
        default="not_na",
    )
    output["covid_stringency_source_date_mean_status"] = np.select(
        [
            output["compatible_week_count"].eq(0),
            output["year"].le(2019) & output["exposure_complete"],
            output["exposure_complete"],
            output["observed_source_week_count"].gt(0),
        ],
        [
            "NA_W0",
            "pre_covid_structural_zero",
            "source_measured_complete",
            "observed_part_mean_not_complete",
        ],
        default="NA_source_coverage",
    )
    # These aliases match the compact model-facing fields already used by the
    # public annual panel while retaining all V11 audit columns above.
    output["covid_mean"] = output["covid_stringency_source_date_mean"]
    output["covid_observed_weeks"] = output["observed_source_week_count"]
    output["covid_zero_weeks"] = output["covid_source_date_zero_count"]
    output["covid_coverage_ratio"] = output["exposure_coverage_ratio"]
    output["covid_coverage_complete"] = output["exposure_complete"]
    output["covid_status"] = output["covid_exposure_status"].map(
        {
            "no_compatible_weeks_W0": "no_compatible_weeks",
            "pre_covid_structural_zero": "complete_compatible_date_coverage",
            "source_measured_complete": "complete_compatible_date_coverage",
            "source_measured_complete_2023": "complete_compatible_date_coverage",
            "source_measured_partial_out_of_source_coverage": "partial_compatible_date_coverage",
            "out_of_source_coverage": "out_of_source_coverage",
        }
    ).fillna("source_date_status_unclassified")
    return output.sort_values(keys).reset_index(drop=True)


def _period_rows() -> Iterable[tuple[str, str, pd.Timestamp, pd.Timestamp, int, int | None]]:
    for quarters in (False, True):
        for year in YEARS:
            if not quarters:
                start = pd.Timestamp(year=year, month=1, day=1)
                end = pd.Timestamp(year=year, month=12, day=31)
                yield "year", str(year), start, end, year, None
            else:
                for quarter in range(1, 5):
                    start_month = (quarter - 1) * 3 + 1
                    start = pd.Timestamp(year=year, month=start_month, day=1)
                    end = start + pd.offsets.QuarterEnd()
                    yield "quarter", f"{year}Q{quarter}", start, end, year, quarter


def build_period_background(daily: pd.DataFrame, province_crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Build V11 calendar-year and calendar-quarter province backgrounds."""

    source = daily.copy()
    source["covid_stringency_source"] = pd.to_numeric(source["covid_stringency_source"], errors="coerce")
    source_lookup = source.set_index(["province_cn", "source_date_dt"])["covid_stringency_source"]
    rows: list[dict[str, Any]] = []
    for province_cn, region_name, region_code in province_crosswalk[
        ["province_cn", "ox_region_name", "ox_region_code"]
    ].itertuples(index=False, name=None):
        for period_type, period_key, start, end, year, quarter in _period_rows():
            expected_dates = pd.date_range(start, end, freq="D")
            expected_days = len(expected_dates)
            if year <= 2019:
                observed_days = 0
                source_min = None
                source_max = None
                status = "pre_covid_structural_zero"
                calendar_mean = 0.0
                observed_part_mean = np.nan
                complete_mean = 0.0
            else:
                index = pd.MultiIndex.from_product(
                    [[province_cn], expected_dates],
                    names=["province_cn", "source_date_dt"],
                )
                values = source_lookup.reindex(index).droplevel(0)
                observed = values.dropna()
                observed_days = int(observed.shape[0])
                source_dates = expected_dates[expected_dates.isin(source["source_date_dt"])]
                source_min = source_dates.min().strftime("%Y-%m-%d") if len(source_dates) else None
                source_max = source_dates.max().strftime("%Y-%m-%d") if len(source_dates) else None
                if observed_days == expected_days:
                    status = "source_measured_complete"
                    calendar_mean = float(observed.mean())
                    complete_mean = calendar_mean
                elif observed_days > 0:
                    status = "source_measured_partial"
                    calendar_mean = np.nan
                    complete_mean = np.nan
                else:
                    status = "out_of_source_coverage"
                    calendar_mean = np.nan
                    complete_mean = np.nan
                observed_part_mean = float(observed.mean()) if observed_days else np.nan
            rows.append(
                {
                    "province_cn": province_cn,
                    "ox_region_name": region_name,
                    "ox_region_code": region_code,
                    "country_code": "CHN",
                    "jurisdiction": "STATE_TOTAL",
                    "period_type": period_type,
                    "period_key": period_key,
                    "calendar_year": year,
                    "calendar_quarter": quarter,
                    "period_start": start.strftime("%Y-%m-%d"),
                    "period_end": end.strftime("%Y-%m-%d"),
                    "expected_calendar_days": expected_days,
                    "observed_source_days": observed_days,
                    "missing_source_days": expected_days - observed_days,
                    "source_coverage_ratio": observed_days / expected_days,
                    "background_status": status,
                    "covid_stringency_calendar_mean": calendar_mean,
                    "covid_stringency_observed_part_mean": observed_part_mean,
                    "covid_stringency_complete_mean": complete_mean,
                    "source_date_min": source_min,
                    "source_date_max": source_max,
                    "source_indicator": OXCGRT_STRINGENCY_COLUMN,
                    "source_average_scale": "0-100",
                    "is_source_observed": bool(year >= 2020 and observed_days > 0),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["period_type", "period_key", "ox_region_code"]
    ).reset_index(drop=True)


def build_source_registry(oxcgrt_path: Path) -> pd.DataFrame:
    """Write an auditable fixed-commit source registry for the public route."""

    rows = [
        {
            "source_id": "oxcgrt_compact_subnational_v1",
            "source_url": OXCGRT_COMPACT_URL,
            "official_tracker_page": OXCGRT_TRACKER_URL,
            "input_file": _public_path(oxcgrt_path),
            "input_sha256": _sha256(oxcgrt_path),
            "github_commit": OXCGRT_COMMIT,
            "analysis_use": "primary_analysis_source",
            "field_or_filter": (
                "CountryCode == CHN; Jurisdiction == STATE_TOTAL; "
                "RegionCode in fixed 31-province crosswalk; StringencyIndex_Average"
            ),
            "period_coverage": "source measured 2020-01-01 through 2023-02-28",
            "notes": "No post-source dates are forward filled; pre-2020 exposure is structural zero.",
        },
        {
            "source_id": "oxcgrt_codebook",
            "source_url": OXCGRT_CODEBOOK_URL,
            "official_tracker_page": OXCGRT_TRACKER_URL,
            "input_file": "user-supplied source documentation",
            "input_sha256": None,
            "github_commit": OXCGRT_COMMIT,
            "analysis_use": "official_definition",
            "field_or_filter": "Average index and C5M/C6M/C7M indicator definitions",
            "period_coverage": "source documentation",
            "notes": "The public route does not download or redistribute policy full texts.",
        },
    ]
    return pd.DataFrame(rows)


def _max_abs_difference(actual: pd.Series, expected: pd.Series) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    e = pd.to_numeric(expected, errors="coerce")
    both = a.notna() & e.notna()
    if not both.any():
        return None
    return float((a[both] - e[both]).abs().max())


def _numeric_equal(actual: pd.Series, expected: pd.Series, atol: float = 1e-8) -> tuple[bool, int, float | None]:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    equal = np.isclose(a, e, atol=atol, rtol=1e-10, equal_nan=True)
    return bool(equal.all()), int((~equal).sum()), _max_abs_difference(actual, expected)


def compare_weather_reference(actual: pd.DataFrame, reference_path: Path | None) -> dict[str, Any]:
    if reference_path is None:
        return {"status": "NOT_REQUESTED"}
    reference = _read_csv(reference_path, "weather reference panel")
    keys = ["city_key", "year"]
    _require_columns(reference, keys, reference_path, "weather reference panel")
    measures = [
        "weather_temperature_mean_c",
        "weather_precipitation_sum_mm",
        "weather_wind_speed_mean_ms",
        "weather_relative_humidity_mean_pct",
        "weather_hdd18_degree_days",
        "weather_cdd24_degree_days",
        "weather_all_valid_days",
        "weather_calendar_days",
    ]
    required = keys + measures + ["weather_status"]
    _require_columns(reference, required, reference_path, "weather reference panel")
    merged = actual.rename(columns={c: c + "_actual" for c in actual if c not in keys}).merge(
        reference.rename(columns={c: c + "_reference" for c in reference if c not in keys}),
        on=keys, how="outer", indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        return {
            "status": "FAIL",
            "reason": "weather reference key set differs",
            "actual_rows": int(len(actual)),
            "reference_rows": int(len(reference)),
        }
    checks: dict[str, Any] = {}
    failed = False
    for measure in measures:
        passed, mismatches, maximum = _numeric_equal(
            merged[f"{measure}_actual"], merged[f"{measure}_reference"]
        )
        checks[measure] = {"pass": passed, "mismatches": mismatches, "max_abs_difference": maximum}
        failed |= not passed
    status_equal = merged["weather_status_actual"].fillna("").eq(merged["weather_status_reference"].fillna(""))
    checks["weather_status"] = {"pass": bool(status_equal.all()), "mismatches": int((~status_equal).sum())}
    failed |= not bool(status_equal.all())
    return {
        "status": "FAIL" if failed else "PASS",
        "reference_file": _public_path(reference_path),
        "rows": int(len(merged)),
        "checks": checks,
    }


def _normalise_public_covid_status(value: pd.Series) -> pd.Series:
    return value.map(
        {
            "no_compatible_weeks_W0": "no_compatible_weeks",
            "pre_covid_structural_zero": "complete_compatible_date_coverage",
            "source_measured_complete": "complete_compatible_date_coverage",
            "source_measured_complete_2023": "complete_compatible_date_coverage",
            "source_measured_partial_out_of_source_coverage": "partial_compatible_date_coverage",
            "out_of_source_coverage": "out_of_source_coverage",
        }
    ).fillna(value)


def compare_covid_reference(actual: pd.DataFrame, reference_path: Path | None) -> dict[str, Any]:
    if reference_path is None:
        return {"status": "NOT_REQUESTED"}
    reference = _read_csv(reference_path, "COVID exposure reference panel")
    keys = ["city_key", "year"]
    _require_columns(reference, keys, reference_path, "COVID exposure reference panel")
    if "covid_stringency_source_date_mean" in reference.columns:
        mapping = {
            "mean": "covid_stringency_source_date_mean",
            "observed": "observed_source_week_count",
            "zero": "covid_source_date_zero_count",
            "ratio": "exposure_coverage_ratio",
            "complete": "exposure_complete",
            "status": "covid_exposure_status",
        }
        reference_status_is_v11 = True
    else:
        required = [
            "covid_mean",
            "covid_observed_weeks",
            "covid_zero_weeks",
            "covid_coverage_ratio",
            "covid_coverage_complete",
            "covid_status",
        ]
        _require_columns(reference, required, reference_path, "COVID exposure reference panel")
        mapping = {
            "mean": "covid_mean",
            "observed": "covid_observed_weeks",
            "zero": "covid_zero_weeks",
            "ratio": "covid_coverage_ratio",
            "complete": "covid_coverage_complete",
            "status": "covid_status",
        }
        reference_status_is_v11 = False
    merged = actual.rename(columns={c: c + "_actual" for c in actual if c not in keys}).merge(
        reference.rename(columns={c: c + "_reference" for c in reference if c not in keys}),
        on=keys, how="outer", indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        return {
            "status": "FAIL",
            "reason": "COVID reference key set differs",
            "actual_rows": int(len(actual)),
            "reference_rows": int(len(reference)),
        }
    checks: dict[str, Any] = {}
    failed = False
    for label in ("mean", "observed", "zero", "ratio", "complete"):
        actual_column = {
            "mean": "covid_stringency_source_date_mean",
            "observed": "observed_source_week_count",
            "zero": "covid_source_date_zero_count",
            "ratio": "exposure_coverage_ratio",
            "complete": "exposure_complete",
        }[label]
        reference_column = mapping[label]
        if label == "complete":
            actual_values = merged[f"{actual_column}_actual"].fillna(False).astype(bool)
            reference_values = merged[f"{reference_column}_reference"].fillna(False).astype(bool)
            passed = actual_values.eq(reference_values)
            check = {"pass": bool(passed.all()), "mismatches": int((~passed).sum())}
        else:
            passed_all, mismatches, maximum = _numeric_equal(
                merged[f"{actual_column}_actual"], merged[f"{reference_column}_reference"]
            )
            check = {"pass": passed_all, "mismatches": mismatches, "max_abs_difference": maximum}
            passed = passed_all
        checks[label] = check
        failed |= not check["pass"]
    actual_status = merged["covid_exposure_status_actual"]
    reference_status = merged[f"{mapping['status']}_reference"]
    if not reference_status_is_v11:
        actual_status = _normalise_public_covid_status(actual_status)
    status_equal = actual_status.fillna("").eq(reference_status.fillna(""))
    checks["status"] = {"pass": bool(status_equal.all()), "mismatches": int((~status_equal).sum())}
    failed |= not bool(status_equal.all())
    return {
        "status": "FAIL" if failed else "PASS",
        "reference_file": _public_path(reference_path),
        "rows": int(len(merged)),
        "checks": checks,
    }


def build_context_controls(
    coordinates_path: Path,
    weather_json_dir: Path,
    ev_air_path: Path,
    city_crosswalk_path: Path,
    oxcgrt_path: Path,
    output_dir: Path,
    weather_reference: Path | None = None,
    covid_reference: Path | None = None,
) -> dict[str, Any]:
    """Run the independent weather and COVID context preparation route."""

    for reference in (weather_reference, covid_reference):
        if reference is not None and reference.resolve().parent == output_dir.resolve():
            raise InputContractError("Output directory must differ from the reference input directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    city_crosswalk = _read_csv(
        city_crosswalk_path,
        "city crosswalk",
        required=CITY_CROSSWALK_COLUMNS,
    ).copy()
    if city_crosswalk["aq_city"].duplicated().any() or city_crosswalk["city_key"].duplicated().any():
        raise InputContractError("City crosswalk must have unique aq_city and city_key values")
    points = build_city_grid_mapping(coordinates_path)
    if "province" in points.columns:
        point_provinces = points[["city_key", "province"]].drop_duplicates()
        if point_provinces["city_key"].duplicated().any():
            raise InputContractError("Weather city-point input has conflicting province values for a city_key")
        city_crosswalk = city_crosswalk.merge(
            point_provinces.rename(columns={"province": "point_province"}),
            on="city_key",
            how="left",
            validate="one_to_one",
        )
        mismatch = (
            city_crosswalk["province"].notna()
            & city_crosswalk["point_province"].notna()
            & city_crosswalk["province"].ne(city_crosswalk["point_province"])
        )
        if mismatch.any():
            raise InputContractError("City crosswalk and dated city-point provinces disagree for city_key values")
        city_crosswalk["province"] = city_crosswalk["province"].fillna(city_crosswalk["point_province"])
        city_crosswalk = city_crosswalk.drop(columns=["point_province"])
    grid_frames, weather_metadata, weather_paths = load_weather_jsons(weather_json_dir)
    weather = build_weather_city_year(points, grid_frames, weather_metadata)
    daily = read_oxcgrt_source(oxcgrt_path)
    province_crosswalk = build_province_crosswalk(city_crosswalk, daily)
    details = prepare_compatible_week_details(ev_air_path, city_crosswalk, province_crosswalk, daily)
    exposure = build_exposure_coverage(city_crosswalk, details)
    background = build_period_background(daily, province_crosswalk)
    weather_keys = ["city_key", "year"]
    weather_join = weather.drop(columns=[column for column in ("city", "province") if column in weather.columns])
    annual_context = exposure.merge(weather_join, on=weather_keys, how="left", validate="one_to_one")
    if len(annual_context) != len(exposure) or annual_context[weather_keys].duplicated().any():
        raise InputContractError("Merged annual context controls are not a unique city-year skeleton")

    outputs = {
        "weather_city_grid_mapping_v4.csv": points,
        "weather_city_year_v4.csv": weather,
        "weather_response_metadata_v4.csv": weather_metadata,
        "covid_source_registry_v11.csv": build_source_registry(oxcgrt_path),
        "province_date_covid_v11.csv": daily.drop(columns=["source_date_dt"]),
        "covid_province_crosswalk_v11.csv": province_crosswalk,
        "covid_source_date_details_v11.csv": details.drop(columns=["source_date_dt"]),
        "covid_exposure_coverage_v11.csv": exposure,
        "covid_period_background_v11.csv": background,
        "annual_context_controls_v4_v11.csv": annual_context,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")

    weather_check = compare_weather_reference(weather, weather_reference)
    covid_check = compare_covid_reference(exposure, covid_reference)
    reference_checks = {"weather": weather_check, "covid": covid_check}
    (output_dir / "context_reference_checks.json").write_text(
        json.dumps(_jsonable(reference_checks), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    qc = {
        "weather_rows": int(len(weather)),
        "weather_city_count": int(points["city_key"].nunique()),
        "weather_unique_city_year_keys": int(weather[weather_keys].duplicated().sum()),
        "weather_coordinate_eligible_city_count": int(points["weather_coordinate_eligible"].sum()),
        "weather_status_counts": {str(k): int(v) for k, v in weather["weather_status"].value_counts().items()},
        "covid_daily_rows": int(len(daily)),
        "covid_region_count": int(daily["RegionCode"].nunique()),
        "covid_province_count": int(len(province_crosswalk)),
        "covid_detail_rows": int(len(details)),
        "covid_coverage_rows": int(len(exposure)),
        "covid_unique_city_year_keys": int(exposure[weather_keys].duplicated().sum()),
        "covid_status_counts": {str(k): int(v) for k, v in exposure["covid_exposure_status"].value_counts().items()},
        "covid_compatible_total_zero_week_rows": int(details["compatible_total_zero"].sum()),
        "covid_post_cutoff_nonmissing_values": int(
            details.loc[
                details["source_date_dt"].gt(OXCGRT_CUTOFF),
                "covid_stringency_source_date_value",
            ].notna().sum()
        ),
        "period_background_rows": int(len(background)),
        "annual_context_rows": int(len(annual_context)),
        "reference_checks": reference_checks,
    }
    structural_pass = (
        qc["weather_unique_city_year_keys"] == 0
        and qc["covid_unique_city_year_keys"] == 0
        and qc["covid_post_cutoff_nonmissing_values"] == 0
        and all(check.get("status") != "FAIL" for check in reference_checks.values())
    )
    qc["status"] = "PASS" if structural_pass else "FAIL"
    (output_dir / "context_qc.json").write_text(
        json.dumps(_jsonable(qc), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "module": "public context preparation",
        "algorithm_lineage": {
            "weather": "V4 build_weather.py offline annual aggregation",
            "covid": "V11 build_covid_v11.py source-date exposure and period background",
        },
        "inputs": {
            "coordinates": {"file": _public_path(coordinates_path), "sha256": _sha256(coordinates_path)},
            "city_crosswalk": {"file": _public_path(city_crosswalk_path), "sha256": _sha256(city_crosswalk_path)},
            "ev_air": {"file": _public_path(ev_air_path), "sha256": _sha256(ev_air_path)},
            "oxcgrt": {"file": _public_path(oxcgrt_path), "sha256": _sha256(oxcgrt_path)},
            "weather_json_count": len(weather_paths),
            "weather_json_files": [{"file": path.name, "sha256": _sha256(path)} for path in weather_paths],
        },
        "weather_rules": {
            "parameters": list(NASA_PARAMETERS),
            "units": NASA_UNITS,
            "endpoint": NASA_POWER_ENDPOINT,
            "documentation": list(NASA_POWER_DOCUMENTATION),
            "time_standard": "UTC",
            "daily_coverage_threshold": NASA_DAILY_COVERAGE_THRESHOLD,
            "precipitation_sum_requires_complete_calendar": True,
            "hdd_base_c": HDD_BASE_C,
            "cdd_base_c": CDD_BASE_C,
            "missing_days_are_not_annualized": True,
            "unresolved_city_coordinates_are_not_imputed": True,
        },
        "covid_rules": {
            "source_commit": OXCGRT_COMMIT,
            "source_url": OXCGRT_COMPACT_URL,
            "codebook_url": OXCGRT_CODEBOOK_URL,
            "indicator": OXCGRT_STRINGENCY_COLUMN,
            "source_filter": "CountryCode == CHN; Jurisdiction == STATE_TOTAL",
            "compatible_week_rule": "cumulative_plugs > 0; Total is not a filter",
            "pre_2020": "structural zero, not Oxford measured",
            "source_date_coverage": "2020-01-01 through 2023-02-28",
            "post_2023_02_28": "missing; no forward fill or zero fill",
            "scale": "0-100",
        },
        "outputs": list(outputs),
        "weather_reference": _public_path(weather_reference) if weather_reference is not None else None,
        "covid_reference": _public_path(covid_reference) if covid_reference is not None else None,
    }
    (output_dir / "context_metadata.json").write_text(
        json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": qc["status"],
        "weather_rows": qc["weather_rows"],
        "covid_daily_rows": qc["covid_daily_rows"],
        "covid_detail_rows": qc["covid_detail_rows"],
        "covid_coverage_rows": qc["covid_coverage_rows"],
        "period_background_rows": qc["period_background_rows"],
        "annual_context_rows": qc["annual_context_rows"],
        "reference_checks": reference_checks,
        "output_dir": str(output_dir),
    }

