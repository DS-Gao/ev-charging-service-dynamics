"""Recompute the prespecified common-city growth windows.

This is the V15 window algorithm with repository-relative input and output
paths.  It uses only the public annual panel, requires positive ``Q``, ``E``
and ``W`` in every year of a window, and never imputes a missing compatible
week or converts an unknown observation state to zero.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
REPOSITORY = HERE.parents[2]
INPUT = REPOSITORY / "data" / "processed" / "annual_panel_v12.csv"
OUTPUT = REPOSITORY / "data" / "results" / "recomputed_descriptive"
OUTPUT.mkdir(parents=True, exist_ok=True)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def main() -> dict[str, object]:
    panel = _read(INPUT)
    required = {"city_key", "city", "province", "year", "A", "N", "I", "Q", "E", "W"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise KeyError(f"annual_panel_v12.csv missing required columns: {missing}")
    for c in ("year", "A", "N", "I", "Q", "E", "W"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["year"] = panel["year"].astype("Int64")
    valid = panel.loc[panel[["Q", "E", "W", "N", "A"]].gt(0).all(axis=1)].copy()
    windows = [("continuous_2019_2023", 2019, 2023)] + [
        (f"adjacent_{year}_{year + 1}", year, year + 1) for year in range(2019, 2023)
    ] + [(f"continuous_{year}_{year + 2}", year, year + 2) for year in range(2019, 2022)]

    members: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    changes: list[pd.DataFrame] = []
    yearly: list[dict[str, object]] = []
    decomposition: list[dict[str, object]] = []

    for window_id, start, end in windows:
        frame = valid.loc[valid["year"].between(start, end)].copy()
        counts = frame.groupby("city_key")["year"].nunique()
        cities = counts.loc[counts.eq(end - start + 1)].index
        frame = frame.loc[frame["city_key"].isin(cities)].sort_values(["city_key", "year"])
        member = frame[["city_key", "city", "province", "year", "A", "N", "I", "Q", "E", "W"]].copy()
        member.insert(0, "window_id", window_id)
        members.append(member)

        start_frame = frame.loc[frame["year"].eq(start)].set_index("city_key")
        end_frame = frame.loc[frame["year"].eq(end)].set_index("city_key")
        joined = start_frame.join(end_frame, lsuffix="_start", rsuffix="_end")
        change = pd.DataFrame(
            {
                "window_id": window_id,
                "city_key": joined.index,
                "city": joined["city_start"].to_numpy(),
                "province": joined["province_start"].to_numpy(),
                "start_year": start,
                "end_year": end,
            }
        )
        summary: dict[str, object] = {
            "window_id": window_id,
            "start_year": start,
            "end_year": end,
            "continuous_years_required": end - start + 1,
            "n_cities": int(len(cities)),
            "n_city_years": int(len(frame)),
            "sample_type": "D",
        }
        for variable in ("A", "N", "I"):
            change[f"{variable}_start"] = joined[f"{variable}_start"].to_numpy()
            change[f"{variable}_end"] = joined[f"{variable}_end"].to_numpy()
            change[f"delta_ln{variable}"] = np.log(change[f"{variable}_end"] / change[f"{variable}_start"])
            change[f"percent_change_{variable}"] = 100 * (change[f"{variable}_end"] / change[f"{variable}_start"] - 1)
            delta = change[f"delta_ln{variable}"]
            for stat, quantile in (("p25", 0.25), ("median", 0.5), ("p75", 0.75)):
                summary[f"{variable}_delta_ln_{stat}"] = float(delta.quantile(quantile))
            for label, mask in (
                ("increase", delta > 1e-12),
                ("decrease", delta < -1e-12),
                ("unchanged", delta.abs() <= 1e-12),
            ):
                summary[f"{variable}_{label}_n"] = int(mask.sum())
                summary[f"{variable}_{label}_share"] = float(mask.mean())
        for variable in ("A", "N"):
            summary[f"{variable}_start_sum"] = float(change[f"{variable}_start"].sum())
            summary[f"{variable}_end_sum"] = float(change[f"{variable}_end"].sum())
            summary[f"{variable}_aggregate_percent_change"] = 100 * (
                summary[f"{variable}_end_sum"] / summary[f"{variable}_start_sum"] - 1
            )
        summary["I_aggregate_start"] = summary["A_start_sum"] / summary["N_start_sum"]
        summary["I_aggregate_end"] = summary["A_end_sum"] / summary["N_end_sum"]
        summary["I_aggregate_percent_change"] = 100 * (
            summary["I_aggregate_end"] / summary["I_aggregate_start"] - 1
        )
        weights_start = change["N_start"] / change["N_start"].sum()
        weights_end = change["N_end"] / change["N_end"].sum()
        within = (((weights_start + weights_end) / 2) * (change["I_end"] - change["I_start"])).sum()
        composition = (((change["I_start"] + change["I_end"]) / 2) * (weights_end - weights_start)).sum()
        for component, value in (("within", within), ("composition", composition), ("total", within + composition)):
            decomposition.append(
                {
                    "window_id": window_id,
                    "n_cities": int(len(cities)),
                    "component": component,
                    "change_original": float(value),
                    "change_index_points": float(100 * value / summary["I_aggregate_start"]),
                    "identity_error": float(within + composition - (summary["I_aggregate_end"] - summary["I_aggregate_start"])),
                }
            )
        for year, group in frame.groupby("year"):
            a_sum = group["A"].sum()
            n_sum = group["N"].sum()
            yearly.append(
                {
                    "window_id": window_id,
                    "year": int(year),
                    "n_cities": int(group["city_key"].nunique()),
                    "A_sum": float(a_sum),
                    "N_sum": float(n_sum),
                    "I_aggregate": float(a_sum / n_sum),
                    "A_index": float(100 * a_sum / summary["A_start_sum"]),
                    "N_index": float(100 * n_sum / summary["N_start_sum"]),
                    "I_index": float(100 * (a_sum / n_sum) / summary["I_aggregate_start"]),
                }
            )
        summary["source_status"] = "recomputed_from_public_annual_panel"
        summaries.append(summary)
        changes.append(change)

    all_changes = pd.concat(changes, ignore_index=True)
    recent = all_changes.loc[all_changes["window_id"].eq("adjacent_2022_2023")].copy()
    cuts = recent["N_start"].quantile([1 / 3, 2 / 3]).to_numpy()
    if not cuts[0] < cuts[1]:
        raise ValueError(f"Expected distinct tercile boundaries, got {cuts.tolist()}")
    recent["size_group"] = pd.cut(recent["N_start"], [-np.inf, *cuts, np.inf], labels=["Low", "Middle", "High"], right=True).astype(str)
    recent["lower_tercile_boundary_N"] = cuts[0]
    recent["upper_tercile_boundary_N"] = cuts[1]
    recent["group_rule"] = "2022 N empirical terciles; equal boundary values remain together; no outcome-dependent regrouping"
    size_summary: list[dict[str, object]] = []
    for group_name, group in recent.groupby("size_group", sort=False):
        for variable in ("A", "N", "I"):
            delta = group[f"delta_ln{variable}"]
            size_summary.append(
                {
                    "size_group": group_name,
                    "variable": variable,
                    "n_cities": int(len(group)),
                    "baseline_N_min": float(group["N_start"].min()),
                    "baseline_N_max": float(group["N_start"].max()),
                    "delta_ln_median": float(delta.median()),
                    "delta_ln_p25": float(delta.quantile(0.25)),
                    "delta_ln_p75": float(delta.quantile(0.75)),
                }
            )

    outputs = {
        "window_membership.csv": pd.concat(members, ignore_index=True),
        "window_summary.csv": pd.DataFrame(summaries),
        "city_changes.csv": all_changes,
        "window_trends.csv": pd.DataFrame(yearly),
        "window_decomposition.csv": pd.DataFrame(decomposition),
        "city_size_groups.csv": recent,
        "city_size_summary.csv": pd.DataFrame(size_summary),
    }
    for name, data in outputs.items():
        data.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig")
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": "data/processed/annual_panel_v12.csv",
        "positive_support_rule": "Q>0 and E>0 and W>0; all intervening years required",
        "window_ids": [name for name, _, _ in windows],
        "candidate_rows": int(len(valid)),
        "tercile_boundaries_N": [float(value) for value in cuts],
        "outputs": list(outputs),
    }
    (OUTPUT / "windows_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries)[["window_id", "n_cities", "A_aggregate_percent_change", "N_aggregate_percent_change", "I_aggregate_percent_change"]].to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
