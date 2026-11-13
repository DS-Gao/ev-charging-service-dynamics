"""Generate the predeclared 2022--2023 compatible-week comparison.

The public entrypoint reads the committed annual Q/E/W panel and optionally a
user-supplied weekly source. It preserves actual source-date support, keeps
the original 225-city endpoint comparison, and reports the 167-city subset
with W = 52 in 2022 and W = 53 in 2023. No model, bootstrap, or covariance
fit is run. By default it writes only CSV and JSON below
``data/results/recomputed_descriptive/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
YEARS = (2022, 2023)


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _default_annual_csv() -> Path:
    path = REPO_ROOT / "data" / "processed" / "annual_panel_v12.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Public annual panel not found: {path}")
    return path


def _default_weekly_csv() -> Path | None:
    # The raw weekly archive is not bundled; opt in with --weekly-csv.
    return None


def _default_crosswalk() -> Path | None:
    path = REPO_ROOT / "data" / "processed" / "city_crosswalk_v4.csv"
    return path if path.is_file() else None


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _read_annual(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    required = {"city_key", "year", "Q", "E", "W"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Annual panel is missing required columns: {missing}")
    frame = frame.loc[frame["year"].isin(YEARS)].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["city_key"] = frame["city_key"].astype(str)
    frame = _numeric(frame, ["Q", "E", "W"])
    if "N" not in frame.columns:
        frame["N"] = np.where(frame["W"].gt(0), frame["E"] / frame["W"], np.nan)
    else:
        frame["N"] = pd.to_numeric(frame["N"], errors="coerce")
    if frame.duplicated(["city_key", "year"]).any():
        raise ValueError("Annual Q/E/W panel has duplicate city_key/year rows")
    for column in ("city", "province"):
        if column not in frame.columns:
            frame[column] = ""
    for column in ("calendar_date_min", "calendar_date_max"):
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _normalise_city(value: object) -> str:
    text = str(value).strip()
    for suffix in ("市", "地区", "自治州", "盟"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.replace(" ", "").replace("　", "")


def _city_key_map(annual: pd.DataFrame, crosswalk: Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if crosswalk is not None:
        cross = _read_csv(crosswalk)
        if {"aq_city", "city_key"}.issubset(cross.columns):
            for row in cross[["aq_city", "city_key"]].dropna().itertuples(index=False):
                raw = str(row.aq_city).strip()
                key = str(row.city_key).strip()
                if raw and key:
                    mapping[raw] = key
                    mapping.setdefault(_normalise_city(raw), key)
    for row in annual[["city", "city_key"]].dropna().itertuples(index=False):
        raw = str(row.city).strip()
        key = str(row.city_key).strip()
        if raw and key:
            mapping.setdefault(raw, key)
            mapping.setdefault(_normalise_city(raw), key)
    return mapping


def _rebuild_from_weekly(
    weekly_path: Path, annual: pd.DataFrame, crosswalk: Path | None
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = ["city", "date", "Total", "cumulative_plugs"]
    raw = _read_csv(weekly_path, usecols=columns)
    raw["date_dt"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["date_dt"].isna().any():
        raise ValueError("Weekly source contains unparseable dates")
    raw = _numeric(raw, ["Total", "cumulative_plugs"])
    if raw[["Total", "cumulative_plugs"]].isna().any().any():
        raise ValueError("Weekly source contains non-numeric Total or cumulative_plugs")
    mapping = _city_key_map(annual, crosswalk)
    raw["city_key"] = raw["city"].astype(str).str.strip().map(
        lambda value: mapping.get(value, mapping.get(_normalise_city(value), np.nan))
    )
    if raw["city_key"].isna().any():
        examples = raw.loc[raw["city_key"].isna(), "city"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Weekly source city labels did not map to city_key; examples: {examples}")
    raw["city_key"] = raw["city_key"].astype(str)
    if raw.duplicated(["city_key", "date_dt"]).any():
        raise ValueError("Weekly source has duplicate city_key/source-date rows")
    raw = raw.loc[raw["date_dt"].dt.year.isin(YEARS)].copy()
    raw["year"] = raw["date_dt"].dt.year.astype(int)
    raw["compatible"] = raw["cumulative_plugs"].gt(0)
    compatible = raw.loc[raw["compatible"]].copy()
    grouped = (
        compatible.groupby(["city_key", "year"], as_index=False)
        .agg(
            Q=("Total", "sum"),
            E=("cumulative_plugs", "sum"),
            W=("compatible", "sum"),
            compatible_date_min=("date_dt", "min"),
            compatible_date_max=("date_dt", "max"),
            compatible_dates=("date_dt", lambda values: ";".join(sorted(values.dt.strftime("%Y-%m-%d")))),
        )
    )
    grouped["N"] = np.where(grouped["W"].gt(0), grouped["E"] / grouped["W"], np.nan)
    grouped["A"] = np.where(grouped["W"].gt(0), grouped["Q"] / grouped["W"], np.nan)
    # Add year rows with no compatible date, so the audit remains explicit.
    observed = raw.groupby(["city_key", "year"], as_index=False).agg(
        source_date_min=("date_dt", "min"), source_date_max=("date_dt", "max")
    )
    grouped = observed.merge(grouped, on=["city_key", "year"], how="left", validate="one_to_one")
    grouped["W"] = grouped["W"].fillna(0).astype(int)
    grouped.loc[grouped["W"].eq(0), ["Q", "E", "N", "A"]] = np.nan
    grouped["source_date_min"] = grouped["source_date_min"].dt.strftime("%Y-%m-%d")
    grouped["source_date_max"] = grouped["source_date_max"].dt.strftime("%Y-%m-%d")
    for column in ("compatible_date_min", "compatible_date_max"):
        grouped[column] = pd.to_datetime(grouped[column], errors="coerce").dt.strftime("%Y-%m-%d")
    grouped["compatible_dates"] = grouped["compatible_dates"].fillna("")

    reference = annual[["city_key", "year", "Q", "E", "W"]].copy()
    merged = reference.merge(
        grouped[["city_key", "year", "Q", "E", "W"]],
        on=["city_key", "year"],
        how="outer",
        suffixes=("_annual", "_weekly"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Weekly and annual panels do not have the same endpoint keys")
    for column in ("Q", "E"):
        if not np.allclose(
            merged[f"{column}_annual"].fillna(np.nan),
            merged[f"{column}_weekly"].fillna(np.nan),
            equal_nan=True,
            rtol=1e-10,
            atol=1e-8,
        ):
            raise ValueError(f"Weekly reconstruction does not match annual {column}")
    if not np.array_equal(
        merged["W_annual"].fillna(0).astype(int), merged["W_weekly"].fillna(0).astype(int)
    ):
        raise ValueError("Weekly reconstruction does not match annual W")
    metadata = {
        "weekly_source_rows": int(len(raw)),
        "weekly_source_cities": int(raw["city_key"].nunique()),
        "weekly_source_date_min": raw["date_dt"].min().strftime("%Y-%m-%d"),
        "weekly_source_date_max": raw["date_dt"].max().strftime("%Y-%m-%d"),
        "weekly_reconstruction_matches_annual": True,
        "source_date_rule": "parsed source date; compatible iff cumulative_plugs > 0; Total=0 retained",
    }
    return grouped, metadata


def _use_annual_only(annual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    out = annual.copy()
    out["A"] = np.where(out["W"].gt(0), out["Q"] / out["W"], np.nan)
    out["N"] = np.where(out["W"].gt(0), out["E"] / out["W"], np.nan)
    out["source_date_min"] = out["calendar_date_min"].astype(str).replace("nan", "")
    out["source_date_max"] = out["calendar_date_max"].astype(str).replace("nan", "")
    out["compatible_date_min"] = ""
    out["compatible_date_max"] = ""
    out["compatible_dates"] = ""
    return out, {
        "weekly_source_rows": None,
        "weekly_source_cities": None,
        "weekly_source_date_min": None,
        "weekly_source_date_max": None,
        "weekly_reconstruction_matches_annual": None,
        "source_date_rule": "annual panel W retained; provide --weekly-csv to reconstruct from parsed source dates",
    }


def _valid_endpoint_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["year"].isin(YEARS)
        & frame["Q"].ge(0)
        & frame["E"].gt(0)
        & frame["W"].gt(0)
        & frame["N"].gt(0)
    ].copy()


def _endpoint_wide(frame: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_endpoint_rows(frame)
    if valid.empty:
        raise ValueError("No valid endpoint rows remain after Q/E/W support filters")
    valid["A"] = valid["Q"] / valid["W"]
    valid["N"] = valid["E"] / valid["W"]
    counts = valid.groupby("city_key")["year"].nunique()
    keys = counts.loc[counts.eq(len(YEARS))].index
    valid = valid.loc[valid["city_key"].isin(keys)].copy()
    if valid.empty:
        raise ValueError("No cities have both endpoint years")
    value_columns = [
        "Q",
        "E",
        "W",
        "A",
        "N",
        "city",
        "province",
        "source_date_min",
        "source_date_max",
        "compatible_date_min",
        "compatible_date_max",
        "compatible_dates",
    ]
    wide = valid.pivot(index="city_key", columns="year", values=value_columns)
    wide.columns = [f"{column}_{int(year)}" for column, year in wide.columns]
    wide = wide.reset_index()
    return wide


def _growth(wide: pd.DataFrame, scope: str, mask: pd.Series) -> dict[str, object]:
    z = wide.loc[mask].copy()
    if z.empty:
        raise ValueError(f"No members for scope {scope}")
    a_start = float(z["A_2022"].sum())
    a_end = float(z["A_2023"].sum())
    n_start = float(z["N_2022"].sum())
    n_end = float(z["N_2023"].sum())
    i_start = a_start / n_start
    i_end = a_end / n_end
    return {
        "scope": scope,
        "city_count": int(len(z)),
        "year_start": YEARS[0],
        "year_end": YEARS[1],
        "complete_week_rule": "all endpoint rows" if scope == "original_endpoint" else "W_2022=52 and W_2023=53",
        "activity_growth_pct": 100.0 * (a_end / a_start - 1.0),
        "facility_growth_pct": 100.0 * (n_end / n_start - 1.0),
        "network_intensity_growth_pct": 100.0 * (i_end / i_start - 1.0),
        "sum_A_2022": a_start,
        "sum_A_2023": a_end,
        "sum_N_2022": n_start,
        "sum_N_2023": n_end,
        "network_I_2022": i_start,
        "network_I_2023": i_end,
    }


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(annual_path: Path, weekly_path: Path | None, crosswalk_path: Path | None, output_dir: Path) -> dict[str, object]:
    annual = _read_annual(annual_path)
    if weekly_path is not None:
        frame, source_meta = _rebuild_from_weekly(weekly_path, annual, crosswalk_path)
        # Retain the annual panel's descriptive labels and province values.
        labels = annual[["city_key", "year", "city", "province"]]
        frame = frame.drop(columns=["city", "province"], errors="ignore").merge(
            labels, on=["city_key", "year"], how="left", validate="one_to_one"
        )
    else:
        frame, source_meta = _use_annual_only(annual)
    wide = _endpoint_wide(frame)
    original_mask = pd.Series(True, index=wide.index)
    complete_mask = wide["W_2022"].eq(52) & wide["W_2023"].eq(53)
    summary = pd.DataFrame(
        [
            _growth(wide, "original_endpoint", original_mask),
            _growth(wide, "complete_compatible_week", complete_mask),
        ]
    )
    member_columns = [
        "city_key",
        "city_2022",
        "province_2022",
        "W_2022",
        "W_2023",
        "Q_2022",
        "Q_2023",
        "E_2022",
        "E_2023",
        "A_2022",
        "A_2023",
        "N_2022",
        "N_2023",
        "source_date_min_2022",
        "source_date_max_2022",
        "source_date_min_2023",
        "source_date_max_2023",
        "compatible_date_min_2022",
        "compatible_date_max_2022",
        "compatible_date_min_2023",
        "compatible_date_max_2023",
        "compatible_dates_2022",
        "compatible_dates_2023",
    ]
    members = []
    for scope, mask in (
        ("original_endpoint", original_mask),
        ("complete_compatible_week", complete_mask),
    ):
        part = wide.loc[mask, [column for column in member_columns if column in wide.columns]].copy()
        part.insert(0, "scope", scope)
        members.append(part)
    members_frame = pd.concat(members, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "complete_weeks_summary.csv"
    members_path = output_dir / "complete_weeks_members.csv"
    metadata_path = output_dir / "complete_weeks_metadata.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.14g")
    members_frame.to_csv(members_path, index=False, encoding="utf-8-sig", float_format="%.14g")
    metadata = {
        "script": "scripts/compare_compatible_weeks.py",
        "annual_input": "data/processed/annual_panel_v12.csv",
        "annual_input_sha256": _sha256(annual_path),
        "weekly_input": "user-supplied --weekly-csv" if weekly_path else None,
        "weekly_input_sha256": _sha256(weekly_path),
        "crosswalk_input": "data/processed/city_crosswalk_v4.csv" if crosswalk_path else None,
        "output_dir": "data/results/recomputed_descriptive/",
        "years": list(YEARS),
        "source_meta": source_meta,
        "summary": json.loads(summary.to_json(orient="records")),
        "member_rows_by_scope": {
            scope: int((members_frame["scope"] == scope).sum())
            for scope in ("original_endpoint", "complete_compatible_week")
        },
        "network_intensity_definition": "sum(A_i)/sum(N_i), not mean(I_i)",
        "models_reestimated": False,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "summary": summary,
        "members": members_frame,
        "metadata": metadata,
        "summary_path": summary_path,
        "members_path": members_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-csv", type=Path, default=None, help="Annual Q/E/W panel CSV")
    parser.add_argument("--weekly-csv", type=Path, default=None, help="Optional weekly ev_air.csv for parsed source-date reconstruction")
    parser.add_argument("--crosswalk-csv", type=Path, default=None, help="Optional aq_city-to-city_key crosswalk")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for scientific CSV and metadata outputs")
    args = parser.parse_args()
    annual_path = args.annual_csv or _default_annual_csv()
    weekly_path = args.weekly_csv if args.weekly_csv is not None else _default_weekly_csv()
    crosswalk_path = args.crosswalk_csv if args.crosswalk_csv is not None else _default_crosswalk()
    output_dir = args.output_dir or (REPO_ROOT / "data" / "results" / "recomputed_descriptive")
    result = run(annual_path, weekly_path, crosswalk_path, output_dir)
    print(result["summary"].to_string(index=False))
    print(f"Wrote {result['summary_path']}")
    print(f"Wrote {result['members_path']}")


if __name__ == "__main__":
    main()
