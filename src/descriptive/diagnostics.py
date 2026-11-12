"""Recompute the fixed descriptive state, transition, and contribution analyses.

This module is the public counterpart of the V17 descriptive builder and the
V20 recorded-state/transition builder.  It is deliberately descriptive: it
does not fit a model, add cities, change window definitions, impute zeroes, or
alter the saved reference members.  The default input is the public annual
panel, from which the fixed positive-support windows and endpoint states are
recomputed.  The saved descriptive CSVs are read only as reconciliation
targets when ``--check-reference`` is supplied.

Run from the repository root with::

    py -3.12 -X utf8 src/descriptive/diagnostics.py --check-reference

The command writes ``descriptive_*`` files next to the existing window
recomputation outputs under ``data/results/recomputed_descriptive``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
REPOSITORY = HERE.parents[2]
DEFAULT_INPUT = REPOSITORY / "data" / "processed" / "annual_panel_v12.csv"
DEFAULT_OUTPUT = REPOSITORY / "data" / "results" / "recomputed_descriptive"
DEFAULT_REFERENCE = REPOSITORY / "data" / "results" / "reference" / "descriptive"
DEFAULT_FIGURE_SOURCE = REPOSITORY / "data" / "figure_source"

TOL = 1e-10
YEARS = list(range(2016, 2024))
STATES = ["++", "0+", "+0", "00"]
CATEGORIES = ["Faster", "Slower", "Contraction", "Other"]

WINDOWS: list[tuple[str, int, int]] = [
    ("continuous_2019_2023", 2019, 2023),
    ("adjacent_2019_2020", 2019, 2020),
    ("adjacent_2020_2021", 2020, 2021),
    ("adjacent_2021_2022", 2021, 2022),
    ("adjacent_2022_2023", 2022, 2023),
    ("continuous_2019_2021", 2019, 2021),
    ("continuous_2020_2022", 2020, 2022),
    ("continuous_2021_2023", 2021, 2023),
]
WINDOW_SPECS = {name: (start, end) for name, start, end in WINDOWS}
EXPECTED_WINDOW_N = {
    "continuous_2019_2023": 36,
    "adjacent_2019_2020": 39,
    "adjacent_2020_2021": 55,
    "adjacent_2021_2022": 77,
    "adjacent_2022_2023": 93,
    "continuous_2019_2021": 38,
    "continuous_2020_2022": 53,
    "continuous_2021_2023": 73,
}
REGIME_TYPES = [
    "facility_expansion_activity_faster",
    "facility_expansion_activity_following",
    "facility_expansion_activity_contraction",
    "facility_expansion_activity_flat",
    "facility_expansion_activity_same_rate",
    "facility_unchanged_activity_increase",
    "facility_unchanged_activity_contraction",
    "facility_unchanged_activity_flat",
    "facility_contraction_activity_increase",
    "facility_contraction_activity_decrease",
    "facility_contraction_activity_flat",
    "facility_contraction_activity_same_rate",
]
REGIME_TO_CATEGORY = {
    "facility_expansion_activity_faster": "Faster",
    "facility_expansion_activity_following": "Slower",
    "facility_expansion_activity_contraction": "Contraction",
}


def resolve_path(value: str | Path, base: Path = REPOSITORY) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def status(value: float, tol: float = TOL) -> str:
    if value > tol:
        return "increase"
    if value < -tol:
        return "decrease"
    return "unchanged"


def signed_label(value: float, tol: float = TOL) -> str:
    if value > tol:
        return "positive"
    if value < -tol:
        return "negative"
    return "zero"


def classify_regime(x: float, y: float, tol: float = TOL) -> str:
    """Use the frozen V17 economic boundaries x=0, y=0, and y=x."""

    if x > tol:
        if y > x + tol:
            return "facility_expansion_activity_faster"
        if y > tol and y < x - tol:
            return "facility_expansion_activity_following"
        if y < -tol:
            return "facility_expansion_activity_contraction"
        if abs(y) <= tol:
            return "facility_expansion_activity_flat"
        return "facility_expansion_activity_same_rate"
    if x < -tol:
        if y > tol:
            return "facility_contraction_activity_increase"
        if y < -tol:
            return "facility_contraction_activity_decrease"
        if abs(y) <= tol:
            return "facility_contraction_activity_flat"
        return "facility_contraction_activity_same_rate"
    if y > tol:
        return "facility_unchanged_activity_increase"
    if y < -tol:
        return "facility_unchanged_activity_contraction"
    return "facility_unchanged_activity_flat"


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_panel(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = read_csv(path)
    require_columns(
        panel,
        {"city_key", "city", "province", "year", "Q", "E", "W", "N"},
        path.name,
    )
    for column in ("year", "Q", "E", "W", "N"):
        panel[column] = numeric(panel[column])
    panel["year"] = panel["year"].astype("Int64")
    if panel[["city_key", "year"]].duplicated().any():
        raise ValueError(f"{path.name} has duplicate city-year rows")
    if sorted(panel["year"].dropna().astype(int).unique()) != YEARS:
        raise ValueError(f"Expected years {YEARS} in {path.name}")

    eligible = (
        np.isfinite(panel[["Q", "E", "W", "N"]]).all(axis=1)
        & panel["Q"].ge(0)
        & panel["E"].gt(0)
        & panel["W"].gt(0)
        & panel["N"].gt(0)
    )
    valid = panel.loc[eligible].copy()
    valid["A"] = valid["Q"] / valid["W"]
    valid["N"] = valid["E"] / valid["W"]
    valid["I"] = valid["Q"] / valid["E"]
    if not np.isfinite(valid[["A", "N", "I"]]).all(axis=None):
        raise ValueError("Positive-support panel contains non-finite A/N/I")
    checks = {
        "panel_rows": int(len(panel)),
        "panel_cities": int(panel["city_key"].nunique()),
        "panel_years": YEARS,
        "positive_support_rows": int(len(valid)),
        "positive_support_cities": int(valid["city_key"].nunique()),
        "positive_support_q_zero_rows": int(valid["Q"].eq(0).sum()),
        "positive_support_q_positive_rows": int(valid["Q"].gt(0).sum()),
    }
    return valid, checks


def build_window_membership(valid: pd.DataFrame) -> pd.DataFrame:
    # The fixed growth windows use the positive-support rule from windows.py:
    # Q, E and W must be positive in every intervening year.  Recorded-state
    # endpoints are intentionally broader and may retain Q == 0; they do not
    # use this window membership filter.
    window_valid = valid.loc[valid["Q"].gt(0)].copy()
    members: list[pd.DataFrame] = []
    for window_id, start, end in WINDOWS:
        frame = window_valid.loc[window_valid["year"].between(start, end)].copy()
        counts = frame.groupby("city_key")["year"].nunique()
        cities = counts.loc[counts.eq(end - start + 1)].index
        frame = frame.loc[frame["city_key"].isin(cities)].sort_values(
            ["city_key", "year"], kind="mergesort"
        )
        member = frame[
            ["city_key", "city", "province", "year", "A", "N", "I", "Q", "E", "W"]
        ].copy()
        member.insert(0, "window_id", window_id)
        members.append(member)
    result = pd.concat(members, ignore_index=True)
    counts = result.groupby("window_id")["city_key"].nunique().to_dict()
    for window_id, expected in EXPECTED_WINDOW_N.items():
        if int(counts.get(window_id, 0)) != expected:
            raise ValueError(f"{window_id}: expected {expected} cities, found {counts.get(window_id)}")
    return result


def endpoint_join(window_membership: pd.DataFrame, window_id: str) -> pd.DataFrame:
    start_year, end_year = WINDOW_SPECS[window_id]
    current = window_membership.loc[window_membership["window_id"].eq(window_id)].copy()
    start = current.loc[current["year"].eq(start_year)].copy()
    end = current.loc[current["year"].eq(end_year)].copy()
    if set(start["city_key"]) != set(end["city_key"]):
        raise ValueError(f"{window_id}: endpoint city sets differ")
    value_columns = ["A", "N", "I", "Q", "E", "W"]
    start = start.rename(columns={column: f"{column}_start" for column in value_columns})
    end = end.rename(columns={column: f"{column}_end" for column in value_columns})
    merged = start[
        ["city_key", "city", "province"] + [f"{column}_start" for column in value_columns]
    ].merge(
        end[["city_key"] + [f"{column}_end" for column in value_columns]],
        on="city_key",
        how="inner",
        validate="one_to_one",
    )
    merged.insert(0, "window_id", window_id)
    merged["start_year"] = start_year
    merged["end_year"] = end_year
    return merged


def build_recorded_states(
    valid: pd.DataFrame, input_label: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    memberships: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for year in range(2019, 2023):
        columns = ["city_key", "city", "province", "Q", "E", "W", "A", "N", "I"]
        start = valid.loc[valid["year"].eq(year), columns]
        end = valid.loc[valid["year"].eq(year + 1), columns]
        joined = start.merge(end, on="city_key", suffixes=("_start", "_end"), validate="one_to_one")
        joined["city"] = joined["city_start"]
        joined["province"] = joined["province_start"]
        joined["state"] = np.where(
            joined["Q_start"].gt(0), "+", "0"
        ) + np.where(joined["Q_end"].gt(0), "+", "0")
        window_id = f"adjacent_{year}_{year + 1}"
        joined["window_id"] = window_id
        joined["window"] = window_id
        joined["start_year"] = year
        joined["end_year"] = year + 1
        joined["source_file"] = input_label
        memberships.append(joined)

        base = float(joined["A_start"].sum())
        net = float(joined["A_end"].sum() - base)
        n_start = float(joined["N_start"].sum())
        n_end = float(joined["N_end"].sum())
        for state_name in STATES:
            group = joined.loc[joined["state"].eq(state_name)]
            delta = float(group["A_end"].sum() - group["A_start"].sum())
            summaries.append(
                {
                    "window_id": window_id,
                    "window": window_id,
                    "start_year": year,
                    "end_year": year + 1,
                    "scope": "all_members",
                    "state": state_name,
                    "n_cities": int(len(group)),
                    "total_cities": int(len(joined)),
                    "city_share": float(len(group) / len(joined)),
                    "N_start": float(group["N_start"].sum()),
                    "N_end": float(group["N_end"].sum()),
                    "N_share_start": float(group["N_start"].sum() / n_start),
                    "N_share_end": float(group["N_end"].sum() / n_end),
                    "A_start": float(group["A_start"].sum()),
                    "A_end": float(group["A_end"].sum()),
                    "I_start": float(group["A_start"].sum() / group["N_start"].sum())
                    if len(group)
                    else np.nan,
                    "I_end": float(group["A_end"].sum() / group["N_end"].sum())
                    if len(group)
                    else np.nan,
                    "source_file": input_label,
                }
            )
            components.append(
                {
                    "window_id": window_id,
                    "window": window_id,
                    "start_year": year,
                    "end_year": year + 1,
                    "scope": "state_component",
                    "state": state_name,
                    "n_cities": int(len(group)),
                    "delta_A": delta,
                    "baseline_all_A": base,
                    "contribution_index_points": 100 * delta / base if base else np.nan,
                    "total_delta_A": net,
                    "total_change_percent": 100 * net / base if base else np.nan,
                    "denominator": "all common members initial summed A",
                    "source_file": input_label,
                }
            )
        for scope, group in (("all_members", joined), ("positive_positive", joined.loc[joined["state"].eq("++")])):
            row: dict[str, Any] = {
                "window_id": window_id,
                "window": window_id,
                "start_year": year,
                "end_year": year + 1,
                "scope": scope,
                "state": "ALL" if scope == "all_members" else "++",
                "n_cities": int(len(group)),
                "source_file": input_label,
            }
            for variable in ("A", "N", "I"):
                value_start = (
                    float(group[f"{variable}_start"].sum())
                    if variable != "I"
                    else float(group["A_start"].sum() / group["N_start"].sum())
                )
                value_end = (
                    float(group[f"{variable}_end"].sum())
                    if variable != "I"
                    else float(group["A_end"].sum() / group["N_end"].sum())
                )
                row.update(
                    {
                        f"{variable}_start": value_start,
                        f"{variable}_end": value_end,
                        f"delta_{variable}": value_end - value_start,
                        f"delta_{variable}_percent": 100 * (value_end / value_start - 1)
                        if value_start
                        else np.nan,
                    }
                )
            row["relative_change_status"] = "defined" if row["A_start"] else "zero_baseline_relative_change_missing"
            summaries.append(row)

        component_rows = [r for r in components if r["window_id"] == window_id]
        residual = net - sum(float(row["delta_A"]) for row in component_rows)
        counts = [int(joined["state"].eq(state_name).sum()) for state_name in STATES]
        if not np.isclose(residual, 0, atol=1e-8 * max(1, abs(net))):
            raise ValueError(f"{window_id}: state decomposition residual {residual}")
        state_summary = [
            row
            for row in summaries
            if row["window_id"] == window_id
            and row["scope"] == "all_members"
            and row["state"] in STATES
        ]
        if not all(
            np.isclose(sum(float(row[f"N_share_{period}"]) for row in state_summary), 1)
            for period in ("start", "end")
        ):
            raise ValueError(f"{window_id}: facility shares do not sum to one")
        for row in component_rows:
            row["decomposition_residual"] = float(residual)
            row["relative_change_status"] = "defined" if base else "zero_baseline_relative_change_missing"
        checks.append(
            {
                "window_id": window_id,
                "n": int(len(joined)),
                "counts": counts,
                "facility_state_order": STATES,
                "decomposition_residual": float(residual),
                "exhaustive_disjoint": bool(sum(counts) == len(joined)),
            }
        )
    return (
        pd.concat(memberships, ignore_index=True),
        pd.DataFrame(summaries),
        pd.DataFrame(components),
        checks,
    )


def build_growth_regimes(window_membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for window_id, _, _ in WINDOWS:
        endpoint = endpoint_join(window_membership, window_id)
        for record in endpoint.sort_values("city_key", kind="mergesort").to_dict("records"):
            x = float(np.log(record["N_end"] / record["N_start"]))
            y = float(np.log(record["A_end"] / record["A_start"]))
            d_i = y - x
            regime = classify_regime(x, y)
            rows.append(
                {
                    **record,
                    "x_delta_lnN": x,
                    "y_delta_lnA": y,
                    "dI_delta_lnI": d_i,
                    "x_lnN": x,
                    "y_lnA": y,
                    "dI_lnI": d_i,
                    "x_status": status(x),
                    "y_status": status(y),
                    "dI_status": status(d_i),
                    "growth_line_status": (
                        "above_y_equals_x"
                        if y > x + TOL
                        else "below_y_equals_x"
                        if y < x - TOL
                        else "on_y_equals_x"
                    ),
                    "growth_regime": regime,
                    "regime_type": regime,
                    "x_is_flat": abs(x) <= TOL,
                    "y_is_flat": abs(y) <= TOL,
                    "dI_is_flat": abs(d_i) <= TOL,
                    "classification_tolerance": TOL,
                }
            )
    regimes = pd.DataFrame(rows)
    expected_rows = int(sum(EXPECTED_WINDOW_N.values()))
    if len(regimes) != expected_rows:
        raise ValueError(f"Expected {expected_rows} regime rows, found {len(regimes)}")

    summaries: list[dict[str, Any]] = []
    for window_id, start, end in WINDOWS:
        group = regimes.loc[regimes["window_id"].eq(window_id)]
        row: dict[str, Any] = {
            "window_id": window_id,
            "start_year": start,
            "end_year": end,
            "n_cities": int(len(group)),
            "classification_tolerance": TOL,
            "classification_complete": bool(group["growth_regime"].notna().all()),
        }
        for axis in ("x", "y", "dI"):
            for label in ("increase", "unchanged", "decrease"):
                row[f"{axis}_{label}_n"] = int(group[f"{axis}_status"].eq(label).sum())
        for regime in REGIME_TYPES:
            count = int(group["growth_regime"].eq(regime).sum())
            row[f"n_{regime}"] = count
            row[f"share_{regime}"] = count / len(group) if len(group) else np.nan
        row["primary_three_n"] = int(
            group["growth_regime"].isin(list(REGIME_TO_CATEGORY)).sum()
        )
        row["classification_count_check"] = int(
            sum(row[f"n_{regime}"] for regime in REGIME_TYPES)
        )
        summaries.append(row)
    return regimes, pd.DataFrame(summaries)


def _direction_mismatch(row: pd.Series) -> bool:
    city_direction = row["city_I_direction"]
    net_direction = row["net_contribution_direction"]
    if city_direction != "unchanged":
        return signed_label(float(row["I_end"] - row["I_start"])) != net_direction
    return net_direction != "zero"


def build_contributions(
    window_membership: pd.DataFrame,
    figure_source: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    reference_decomp_path = figure_source / "figS_window_decomposition_v20.csv"
    reference_results_path = figure_source / "fig3_window_results_v20.csv"
    reference_decomp = read_csv(reference_decomp_path) if reference_decomp_path.exists() else pd.DataFrame()
    reference_results = read_csv(reference_results_path) if reference_results_path.exists() else pd.DataFrame()

    def ref_value(frame: pd.DataFrame, window_id: str, **selectors: str) -> float:
        if frame.empty:
            return float("nan")
        mask = frame["window_id"].eq(window_id)
        for key, value in selectors.items():
            mask &= frame[key].eq(value)
        selected = frame.loc[mask]
        return float(selected.iloc[0]["change_original"]) if len(selected) == 1 else float("nan")

    def result_value(window_id: str, column: str) -> float:
        if reference_results.empty:
            return float("nan")
        selected = reference_results.loc[reference_results["window_id"].eq(window_id)]
        return float(selected.iloc[0][column]) if len(selected) == 1 else float("nan")

    for window_id, start_year, end_year in WINDOWS:
        endpoint = endpoint_join(window_membership, window_id)
        n_start = float(endpoint["N_start"].sum())
        n_end = float(endpoint["N_end"].sum())
        endpoint["w_start"] = endpoint["N_start"] / n_start
        endpoint["w_end"] = endpoint["N_end"] / n_end
        i_aggregate_start = float((endpoint["w_start"] * endpoint["I_start"]).sum())
        i_aggregate_end = float((endpoint["w_end"] * endpoint["I_end"]).sum())
        endpoint["x_lnN"] = np.log(endpoint["N_end"] / endpoint["N_start"])
        endpoint["y_lnA"] = np.log(endpoint["A_end"] / endpoint["A_start"])
        endpoint["dI_lnI"] = endpoint["y_lnA"] - endpoint["x_lnN"]
        endpoint["within"] = ((endpoint["w_start"] + endpoint["w_end"]) / 2) * (
            endpoint["I_end"] - endpoint["I_start"]
        )
        endpoint["composition"] = ((endpoint["I_start"] + endpoint["I_end"]) / 2) * (
            endpoint["w_end"] - endpoint["w_start"]
        )
        endpoint["total"] = endpoint["w_end"] * endpoint["I_end"] - endpoint["w_start"] * endpoint["I_start"]
        for component in ("within", "composition", "total"):
            endpoint[f"{component}_index_points"] = 100 * endpoint[component] / i_aggregate_start
        endpoint["total_identity_error"] = endpoint["total"] - endpoint["within"] - endpoint["composition"]
        endpoint["city_I_direction"] = [
            status(float(value)) for value in (endpoint["I_end"] - endpoint["I_start"])
        ]
        endpoint["net_contribution_direction"] = [signed_label(float(value)) for value in endpoint["total"]]
        endpoint["within_direction"] = [signed_label(float(value)) for value in endpoint["within"]]
        endpoint["composition_direction"] = [signed_label(float(value)) for value in endpoint["composition"]]
        endpoint["I_vs_net_direction_mismatch"] = endpoint.apply(_direction_mismatch, axis=1)
        endpoint["within_composition_opposite_sign"] = endpoint["within"] * endpoint["composition"] < -(TOL**2)
        endpoint["I_aggregate_start"] = i_aggregate_start
        endpoint["I_aggregate_end"] = i_aggregate_end
        endpoint["N_sum_start"] = n_start
        endpoint["N_sum_end"] = n_end
        endpoint["contribution_index_base"] = 100.0
        rows.extend(endpoint.sort_values("city_key", kind="mergesort").to_dict("records"))

        sum_within = float(endpoint["within"].sum())
        sum_composition = float(endpoint["composition"].sum())
        sum_total = float(endpoint["total"].sum())
        top_abs = endpoint.assign(_abs_total=endpoint["total"].abs()).sort_values(
            ["_abs_total", "city_key"], ascending=[False, True], kind="mergesort"
        )
        top_abs5 = top_abs.head(5)
        positive = endpoint.loc[endpoint["total"].gt(TOL)].copy()
        positive["_positive"] = positive["total"]
        top_positive5 = positive.sort_values(
            ["_positive", "city_key"], ascending=[False, True], kind="mergesort"
        ).head(5)
        negative = endpoint.loc[endpoint["total"].lt(-TOL)].copy()
        negative["_abs_negative"] = -negative["total"]
        top_negative5 = negative.sort_values(
            ["_abs_negative", "city_key"], ascending=[False, True], kind="mergesort"
        ).head(5)
        total_negative_abs = float((-endpoint.loc[endpoint["total"].lt(0), "total"]).sum())
        v15_within = ref_value(reference_decomp, window_id, component="within")
        v15_composition = ref_value(reference_decomp, window_id, component="composition")
        v15_total = ref_value(reference_decomp, window_id, component="total")
        summaries.append(
            {
                "window_id": window_id,
                "start_year": start_year,
                "end_year": end_year,
                "n_cities": int(len(endpoint)),
                "N_sum_start": n_start,
                "N_sum_end": n_end,
                "I_aggregate_start": i_aggregate_start,
                "I_aggregate_end": i_aggregate_end,
                "aggregate_change_original": i_aggregate_end - i_aggregate_start,
                "aggregate_change_index_points": 100 * (i_aggregate_end - i_aggregate_start) / i_aggregate_start,
                "aggregate_percent_change": 100 * (i_aggregate_end / i_aggregate_start - 1),
                "within_sum_original": sum_within,
                "composition_sum_original": sum_composition,
                "total_sum_original": sum_total,
                "within_sum_index_points": 100 * sum_within / i_aggregate_start,
                "composition_sum_index_points": 100 * sum_composition / i_aggregate_start,
                "total_sum_index_points": 100 * sum_total / i_aggregate_start,
                "sum_identity_error": sum_total - sum_within - sum_composition,
                "max_abs_city_identity_error": float(endpoint["total_identity_error"].abs().max()),
                "n_city_I_increase": int(endpoint["city_I_direction"].eq("increase").sum()),
                "n_city_I_decrease": int(endpoint["city_I_direction"].eq("decrease").sum()),
                "n_city_I_unchanged": int(endpoint["city_I_direction"].eq("unchanged").sum()),
                "n_net_positive": int(endpoint["net_contribution_direction"].eq("positive").sum()),
                "n_net_negative": int(endpoint["net_contribution_direction"].eq("negative").sum()),
                "n_net_zero": int(endpoint["net_contribution_direction"].eq("zero").sum()),
                "positive_total_sum_original": float(endpoint.loc[endpoint["total"].gt(TOL), "total"].sum()),
                "negative_total_sum_original": float(endpoint.loc[endpoint["total"].lt(-TOL), "total"].sum()),
                "negative_total_abs_sum_original": total_negative_abs,
                "positive_total_sum_index_points": float(100 * endpoint.loc[endpoint["total"].gt(TOL), "total"].sum() / i_aggregate_start),
                "negative_total_sum_index_points": float(100 * endpoint.loc[endpoint["total"].lt(-TOL), "total"].sum() / i_aggregate_start),
                "negative_total_abs_sum_index_points": float(100 * total_negative_abs / i_aggregate_start),
                "top5_positive_net_sum_original": float(top_positive5["total"].sum()),
                "top5_positive_net_share_of_positive": float(top_positive5["total"].sum() / endpoint.loc[endpoint["total"].gt(TOL), "total"].sum()),
                "top5_positive_net_sum_index_points": float(100 * top_positive5["total"].sum() / i_aggregate_start),
                "top5_positive_net_city_keys": "|".join(top_positive5["city_key"].astype(str)),
                "top5_positive_net_cities": "|".join(top_positive5["city"].astype(str)),
                "top5_abs_net_contribution_abs_sum_original": float(top_abs5["total"].abs().sum()),
                "top5_abs_net_contribution_share_of_abs_total": float(top_abs5["total"].abs().sum() / endpoint["total"].abs().sum()),
                "top5_negative_net_abs_sum_original": float(top_negative5["_abs_negative"].sum()),
                "top5_negative_net_abs_sum_index_points": float(100 * top_negative5["_abs_negative"].sum() / i_aggregate_start),
                "top5_negative_share_of_negative_abs": float(top_negative5["_abs_negative"].sum() / total_negative_abs) if total_negative_abs else np.nan,
                "top5_abs_net_city_keys": "|".join(top_abs5["city_key"].astype(str)),
                "top5_abs_net_cities": "|".join(top_abs5["city"].astype(str)),
                "top5_negative_net_city_keys": "|".join(top_negative5["city_key"].astype(str)),
                "top5_negative_net_cities": "|".join(top_negative5["city"].astype(str)),
                "direction_mismatch_n": int(endpoint["I_vs_net_direction_mismatch"].sum()),
                "within_composition_opposite_sign_n": int(endpoint["within_composition_opposite_sign"].sum()),
                "reference_within_original": v15_within,
                "reference_composition_original": v15_composition,
                "reference_total_original": v15_total,
                "difference_vs_reference_within": sum_within - v15_within if np.isfinite(v15_within) else np.nan,
                "difference_vs_reference_composition": sum_composition - v15_composition if np.isfinite(v15_composition) else np.nan,
                "difference_vs_reference_total": sum_total - v15_total if np.isfinite(v15_total) else np.nan,
                "reference_aggregate_percent_change": result_value(window_id, "I_aggregate_percent_change"),
                "difference_vs_reference_aggregate_percent_change": (
                    100 * (i_aggregate_end / i_aggregate_start - 1) - result_value(window_id, "I_aggregate_percent_change")
                    if np.isfinite(result_value(window_id, "I_aggregate_percent_change"))
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def build_transitions(regimes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    transitions: list[pd.DataFrame] = []
    matrix_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    other_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    comparisons = [
        ("main_52", "adjacent_2020_2021", "adjacent_2022_2023", 52),
        ("supplement_73", "adjacent_2021_2022", "adjacent_2022_2023", 73),
    ]
    for comparison, early, late, expected in comparisons:
        early_frame = regimes.loc[regimes["window_id"].eq(early)].copy()
        late_frame = regimes.loc[regimes["window_id"].eq(late)].copy()
        joined = early_frame.merge(late_frame, on="city_key", suffixes=("_early", "_late"), validate="one_to_one")
        if joined.empty or len(joined) != expected:
            raise ValueError(f"{comparison}: expected {expected} matched cities, found {len(joined)}")
        joined["comparison"] = comparison
        joined["early_window"] = early
        joined["late_window"] = late
        joined["city"] = joined["city_early"]
        joined["province"] = joined["province_early"]
        joined["source_file"] = "data/processed/annual_panel_v12.csv"
        for period in ("early", "late"):
            joined[f"{period}_category"] = joined[f"regime_type_{period}"].map(REGIME_TO_CATEGORY).fillna("Other")
            for variable in ("x_lnN", "y_lnA"):
                joined[f"{period}_{variable}"] = joined[f"{variable}_{period}"]
            for subtype, group in joined.loc[joined[f"{period}_category"].eq("Other")].groupby(f"regime_type_{period}"):
                other_rows.append(
                    {
                        "comparison": comparison,
                        "period": period,
                        "subtype": subtype,
                        "n": int(len(group)),
                        "city_keys": ";".join(group["city_key"].astype(str)),
                    }
                )
        transitions.append(joined)
        for early_category in CATEGORIES:
            denominator = int(joined["early_category"].eq(early_category).sum())
            for late_category in CATEGORIES:
                n = int(
                    (joined["early_category"].eq(early_category) & joined["late_category"].eq(late_category)).sum()
                )
                matrix_rows.append(
                    {
                        "comparison": comparison,
                        "early_window": early,
                        "late_window": late,
                        "early_category": early_category,
                        "late_category": late_category,
                        "n": n,
                        "early_column_n": denominator,
                        "late_row_n": int(joined["late_category"].eq(late_category).sum()),
                        "n_cities": int(len(joined)),
                        "conditional_pct": 100 * n / denominator if denominator else np.nan,
                        "denominator": "early-category column",
                        "probability_status": "defined" if denominator else "no_cities_in_early_category",
                    }
                )
        def add_metric(name: str, mask: pd.Series, denominator: int, definition: str) -> None:
            summary_rows.append(
                {
                    "comparison": comparison,
                    "metric": name,
                    "numerator": int(mask.sum()),
                    "denominator": int(denominator),
                    "percent": 100 * mask.sum() / denominator if denominator else np.nan,
                    "definition": definition,
                }
            )
        add_metric(
            "same_category",
            joined["early_category"].eq(joined["late_category"]),
            len(joined),
            "All four diagonal category pairs / all matched cities",
        )
        add_metric(
            "faster_to_slower",
            joined["early_category"].eq("Faster") & joined["late_category"].eq("Slower"),
            int(joined["early_category"].eq("Faster").sum()),
            "Faster to Slower / early Faster",
        )
        add_metric(
            "slower_to_faster",
            joined["early_category"].eq("Slower") & joined["late_category"].eq("Faster"),
            int(joined["early_category"].eq("Slower").sum()),
            "Slower to Faster / early Slower",
        )
        recovery = joined["early_category"].eq("Contraction") & joined["late_category"].isin(["Faster", "Slower"])
        if not joined.loc[recovery, "late_y_lnA"].gt(0).all():
            raise ValueError(f"{comparison}: recovery cell has non-positive late activity growth")
        add_metric(
            "contraction_to_positive_growth",
            recovery,
            int(joined["early_category"].eq("Contraction").sum()),
            "Contraction to Faster or Slower with positive late log activity change / early Contraction",
        )
        checks.append(
            {
                "comparison": comparison,
                "n": int(len(joined)),
                "expected_count_matches": bool(len(joined) == expected),
                "unique_city_keys": bool(joined["city_key"].is_unique),
                "matrix_total_matches": bool(
                    sum(row["n"] for row in matrix_rows if row["comparison"] == comparison) == len(joined)
                ),
                "nonempty_columns_sum_100": bool(
                    all(
                        np.isclose(
                            sum(
                                row["conditional_pct"]
                                for row in matrix_rows
                                if row["comparison"] == comparison
                                and row["early_category"] == category
                            ),
                            100,
                        )
                        for category in CATEGORIES
                        if any(
                            row["early_category"] == category
                            and row["early_column_n"] > 0
                            and row["comparison"] == comparison
                            for row in matrix_rows
                        )
                    )
                ),
                "other_is_not_missing": bool(joined["late_category"].notna().all()),
            }
        )
    return (
        pd.concat(transitions, ignore_index=True),
        pd.DataFrame(matrix_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(other_rows),
        checks,
    )


def compare_frames(
    generated: pd.DataFrame,
    reference_path: Path,
    key_columns: list[str],
    ignored_columns: set[str] | None = None,
) -> dict[str, Any]:
    ignored = ignored_columns or set()
    result: dict[str, Any] = {
        "reference": relative_path(reference_path),
        "generated_rows": int(len(generated)),
        "reference_rows": None,
        "key_match": False,
        "numeric_max_abs_difference": None,
        "categorical_mismatch_count": None,
        "passed": False,
    }
    if not reference_path.exists():
        result["status"] = "reference_missing"
        return result
    reference = read_csv(reference_path)
    result["reference_rows"] = int(len(reference))
    if not set(key_columns).issubset(generated.columns) or not set(key_columns).issubset(reference.columns):
        result["status"] = "key_columns_missing"
        return result
    g = generated.copy()
    r = reference.copy()
    g_keys = g[key_columns].astype(str).sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    r_keys = r[key_columns].astype(str).sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    result["key_match"] = bool(g_keys.equals(r_keys))
    if not result["key_match"]:
        result["status"] = "key_mismatch"
        return result
    common = [column for column in generated.columns if column in reference.columns and column not in ignored and column not in key_columns]
    numeric_max = 0.0
    categorical_mismatch = 0
    g_sorted = g.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    r_sorted = r.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    for column in common:
        if pd.api.types.is_numeric_dtype(g_sorted[column]) and pd.api.types.is_numeric_dtype(r_sorted[column]):
            left = pd.to_numeric(g_sorted[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(r_sorted[column], errors="coerce").to_numpy(float)
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                numeric_max = max(numeric_max, float(np.max(np.abs(left[finite] - right[finite]))))
            if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.allclose(
                left[finite], right[finite], rtol=1e-8, atol=1e-8
            ):
                categorical_mismatch += 1
        else:
            left = g_sorted[column].fillna("<NA>").astype(str).to_numpy()
            right = r_sorted[column].fillna("<NA>").astype(str).to_numpy()
            categorical_mismatch += int(np.sum(left != right))
    result["numeric_max_abs_difference"] = numeric_max
    result["categorical_mismatch_count"] = categorical_mismatch
    result["passed"] = bool(result["key_match"] and categorical_mismatch == 0)
    result["status"] = "matched" if result["passed"] else "value_mismatch"
    return result


def run_diagnostics(
    input_path: Path,
    output_dir: Path,
    reference_dir: Path,
    figure_source_dir: Path,
    check_reference: bool,
) -> dict[str, Any]:
    valid, input_checks = prepare_panel(input_path)
    window_membership = build_window_membership(valid)
    recorded_states, state_summary, state_components, state_checks = build_recorded_states(
        valid, relative_path(input_path)
    )
    regimes, regime_summary = build_growth_regimes(window_membership)
    city_contributions, contribution_summary = build_contributions(window_membership, figure_source_dir)
    transitions, transition_matrix, transition_summary, transition_other, transition_checks = build_transitions(regimes)

    output_frames = {
        "descriptive_window_membership.csv": window_membership,
        "descriptive_record_state_membership.csv": recorded_states,
        "descriptive_record_state_summary.csv": state_summary,
        "descriptive_activity_change_components.csv": state_components,
        "descriptive_growth_regime_membership.csv": regimes,
        "descriptive_growth_regime_summary.csv": regime_summary,
        "descriptive_city_contributions.csv": city_contributions,
        "descriptive_contribution_summary.csv": contribution_summary,
        "descriptive_growth_transition_membership.csv": transitions,
        "descriptive_growth_transition_matrix.csv": transition_matrix,
        "descriptive_growth_transition_summary.csv": transition_summary,
        "descriptive_growth_transition_other_subtypes.csv": transition_other,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in output_frames.items():
        write_csv(frame, output_dir / name)

    reference_checks: list[dict[str, Any]] = []
    if check_reference:
        # The figure-source export uses presentation labels for three grouped
        # regimes; the executable keeps the original V17 regime names for the
        # transition algorithm.  Normalize only this comparison field.
        regime_reference_labels = {
            "facility_expansion_activity_following": "both_grow_activity_lags",
            "facility_expansion_activity_contraction": "facility_expansion_activity_contracts",
            "facility_unchanged_activity_increase": "activity_flat_or_facility_flat",
            "facility_unchanged_activity_contraction": "activity_flat_or_facility_flat",
        }
        regimes_for_reference = regimes.copy()
        regimes_for_reference["growth_regime"] = regimes_for_reference["growth_regime"].map(
            lambda value: regime_reference_labels.get(value, value)
        )
        reference_checks.extend(
            [
                compare_frames(
                    recorded_states,
                    reference_dir / "record_state_membership.csv",
                    ["window_id", "city_key", "state"],
                    {"source_file"},
                ),
                compare_frames(
                    state_summary,
                    reference_dir / "record_state_summary.csv",
                    ["window_id", "scope", "state"],
                    {"source_file"},
                ),
                compare_frames(
                    state_components,
                    reference_dir / "activity_change_components.csv",
                    ["window_id", "scope", "state"],
                    {"source_file"},
                ),
                compare_frames(
                    transition_matrix,
                    reference_dir / "growth_transition_matrix.csv",
                    ["comparison", "early_category", "late_category"],
                ),
                compare_frames(
                    transition_summary,
                    reference_dir / "growth_transition_summary.csv",
                    ["comparison", "metric"],
                ),
                compare_frames(
                    transition_other,
                    reference_dir / "growth_transition_other_subtypes.csv",
                    ["comparison", "period", "subtype"],
                ),
                compare_frames(
                    city_contributions,
                    figure_source_dir / "fig5_city_contributions_v20.csv",
                    ["window_id", "city_key"],
                ),
                compare_frames(
                    contribution_summary,
                    figure_source_dir / "fig5_contribution_summary_v20.csv",
                    ["window_id"],
                    {"reference_within_original", "reference_composition_original", "reference_total_original", "difference_vs_reference_within", "difference_vs_reference_composition", "difference_vs_reference_total", "reference_aggregate_percent_change", "difference_vs_reference_aggregate_percent_change"},
                ),
                compare_frames(
                    regimes_for_reference,
                    figure_source_dir / "fig4_city_growth_regimes_v20.csv",
                    ["window_id", "city_key"],
                    {"x_lnN", "y_lnA", "dI_lnI", "regime_type"},
                ),
                compare_frames(
                    regime_summary,
                    figure_source_dir / "fig4_growth_regime_summary_v20.csv",
                    ["window_id"],
                ),
            ]
        )
        checks_path = output_dir / "descriptive_reference_checks.csv"
        write_csv(pd.DataFrame(reference_checks), checks_path)

    metadata = {
        "analysis_id": "public_descriptive_diagnostics",
        "algorithm_sources": {
            "recorded_states_and_activity_components": "src/descriptive/diagnostics.py",
            "growth_classification_and_city_contributions": "src/descriptive/diagnostics.py",
            "transition_category_matrix": "src/descriptive/diagnostics.py",
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": relative_path(HERE),
        "script_sha256": sha256(HERE),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "command": "py -3.12 -X utf8 src/descriptive/diagnostics.py --check-reference",
        "inputs": {
            "annual_panel": {"path": relative_path(input_path), "sha256": sha256(input_path)},
            "figure_source_directory": relative_path(figure_source_dir),
            "reference_directory": relative_path(reference_dir),
        },
        "rules": {
            "positive_support": "finite Q/E/W/N; Q >= 0; E > 0; W > 0; N > 0",
            "windows": [
                {"window_id": name, "start_year": start, "end_year": end, "required_years": end - start + 1}
                for name, start, end in WINDOWS
            ],
            "recorded_states": {"Q > 0": "+", "Q == 0": "0", "state_order": STATES},
            "growth_boundaries": {"facility": "x = ln(N_end/N_start)", "activity": "y = ln(A_end/A_start)", "flat_tolerance": TOL, "activity_faster": "y > x"},
            "contribution_formula": {
                "w_it": "N_it / sum_j N_jt",
                "within_i": "((w_i0+w_i1)/2) * (I_i1-I_i0)",
                "composition_i": "((I_i0+I_i1)/2) * (w_i1-w_i0)",
                "total_i": "w_i1*I_i1 - w_i0*I_i0",
                "index_points": "100 * component_i / I_aggregate_start",
            },
            "reference_status": "saved CSVs are reconciliation targets; they are not used to define new members",
        },
        "input_checks": input_checks,
        "window_city_counts": window_membership.groupby("window_id")["city_key"].nunique().to_dict(),
        "record_state_checks": state_checks,
        "transition_checks": transition_checks,
        "reference_checks": reference_checks,
        "outputs": {name: int(len(frame)) for name, frame in output_frames.items()},
    }
    metadata_path = output_dir / "descriptive_diagnostics_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Public annual panel CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Saved descriptive reference CSV directory")
    parser.add_argument("--figure-source", type=Path, default=DEFAULT_FIGURE_SOURCE, help="Saved figure-source CSV directory")
    parser.add_argument("--check-reference", action="store_true", help="Compare generated outputs with saved references")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = run_diagnostics(
        resolve_path(args.input),
        resolve_path(args.output),
        resolve_path(args.reference),
        resolve_path(args.figure_source),
        bool(args.check_reference),
    )
    print(json.dumps({"outputs": metadata["outputs"], "reference_checks": metadata["reference_checks"]}, ensure_ascii=False, indent=2))
    if args.check_reference and not all(check.get("passed", False) for check in metadata["reference_checks"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
