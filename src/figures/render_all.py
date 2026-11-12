"""Render the complete public figure set from committed saved values.

This is a figure-only consumer. It redraws the saved-value layouts with
Python/matplotlib and writes only below
data/results/recomputed_figures. No model, bootstrap, or covariance fit is
run here.  The descriptive transition panel is intentionally gated on the
saved A/B CSVs; a missing worker output produces a pending manifest entry
rather than a placeholder chart.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SOURCE_DIR = REPO / "data" / "figure_source"
OUT = REPO / "data" / "results" / "recomputed_figures"
FIG_DIR = OUT
SRC_DIR = OUT / "derived_sources"
GEOMETRY_SOURCE = REPO / "data" / "processed" / "city_geometry.geojson"

_GEO_CACHE: gpd.GeoDataFrame | None = None

FIG_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)


# One publication palette reused across every redrawn panel.
COL = {
    "A": "#246aa2",
    "N": "#c47a2c",
    "I": "#7655a8",
    "p": "#4f8f83",
    "within": "#7655a8",
    "composition": "#839a8b",
    "gain": "#2a8070",
    "loss": "#b8574f",
    "accent": "#b46b32",
    "dark": "#303030",
    "muted": "#606060",
    "grid": "#d9d9d9",
}

REGIME_COLORS = {
    "facility_expansion_activity_faster": "#2a8070",
    "both_grow_activity_lags": "#c9993d",
    "facility_expansion_activity_contracts": "#b8574f",
    "facility_decline_activity_grows": "#6d8fb3",
    "facility_decline_activity_declines": "#9a6a8f",
    "activity_flat_or_facility_flat": "#929292",
    "other": "#929292",
}
REGIME_SHORT = {
    "facility_expansion_activity_faster": "N expands; A faster",
    "both_grow_activity_lags": "N expands; A lags",
    "facility_expansion_activity_contracts": "N expands; A falls",
    "facility_decline_activity_grows": "N falls; A grows",
    "facility_decline_activity_declines": "Both fall",
    "activity_flat_or_facility_flat": "Flat component",
    "other": "Other",
}
REGIME_ORDER = [
    "facility_expansion_activity_faster",
    "both_grow_activity_lags",
    "facility_expansion_activity_contracts",
    "other",
]
DISPLAY_BOUND = 3.503090

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def num(s: pd.Series | Iterable[object]) -> pd.Series:
    if isinstance(s, pd.Series):
        out = s.copy()
    else:
        out = pd.Series(s)
    return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan)


def boolish(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Write editable vectors and a high-resolution review raster."""
    # Enforce the declared minimum after all inherited per-panel styling.
    # Final page placement is close to the 170 mm design width.
    from matplotlib.text import Text
    for artist in fig.findobj(Text):
        if artist.get_text() and artist.get_fontsize() < 7.5:
            artist.set_fontsize(7.5)
    from matplotlib.collections import QuadMesh
    for mesh in fig.findobj(QuadMesh):
        mesh.set_rasterized(False)
        mesh.set_edgecolor("face")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}_preview.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_frozen_assets() -> dict[str, list[str]]:
    """The public renderer consumes the committed V20 source tables in place."""
    return {"figures": [], "sources": []}


def _code(value: object) -> str:
    """Normalize administrative codes without changing their identity."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text.replace(".0", "")


def load_geometries() -> gpd.GeoDataFrame:
    """Load the licensed source polygons once, retaining their source CRS."""
    global _GEO_CACHE
    if _GEO_CACHE is None:
        if not GEOMETRY_SOURCE.exists():
            raise FileNotFoundError(GEOMETRY_SOURCE)
        geo = gpd.read_file(GEOMETRY_SOURCE)
        if geo.crs is None:
            # The source GeoJSON is lon/lat and omits a CRS member.
            geo = geo.set_crs("EPSG:4326")
        if "adcode" not in geo.columns:
            raise KeyError("city_geometry.geojson must contain adcode")
        geo["admin_code_norm"] = geo["adcode"].map(_code)
        if geo["admin_code_norm"].duplicated().any():
            raise ValueError("city_geometry.geojson contains duplicate normalized adcodes")
        _GEO_CACHE = geo
    return _GEO_CACHE.copy()


def _map_key(data: pd.DataFrame) -> pd.Series:
    """Use the saved numeric geometry key for exact polygon matching."""
    if "map_geometry_key" in data.columns:
        return data["map_geometry_key"].map(_code)
    if "admin_code_norm" in data.columns:
        return data["admin_code_norm"].map(_code)
    raise KeyError("map data must include map_geometry_key or admin_code_norm")


def draw_status_map(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    status_col: str,
    palette: dict[str, str],
    categories: list[str],
    title: str,
) -> None:
    """Render saved status values on the source-backed city polygons."""
    geo = load_geometries()
    values = data[["map_geometry_key", status_col]].copy()
    values["_key"] = _map_key(values)
    values = values.drop_duplicates("_key").set_index("_key")[status_col]
    geo["_key"] = geo["admin_code_norm"].map(_code)
    geo["_plot_status"] = geo["_key"].map(values)
    geo.plot(ax=ax, color="#d7d7d7", edgecolor="white", linewidth=0.16, zorder=1)
    for category in categories:
        z = geo.loc[geo["_plot_status"].eq(category)]
        if not z.empty:
            z.plot(ax=ax, color=palette.get(category, "#929292"), edgecolor="white", linewidth=0.16, zorder=2)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, loc="left")


def draw_white_value_map(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    *,
    norm,
    cmap: str,
    title: str,
) -> None:
    """Render saved spatial values on exact numeric-key-matched polygons."""
    geo = load_geometries()
    geo.plot(ax=ax, color="#d7d7d7", edgecolor="white", linewidth=0.16, zorder=1)
    values = data[["map_geometry_key", value_col]].copy()
    values["_key"] = _map_key(values)
    values[value_col] = num(values[value_col])
    values = values.drop_duplicates("_key").set_index("_key")[value_col]
    geo["_key"] = geo["admin_code_norm"].map(_code)
    geo["_plot_value"] = geo["_key"].map(values)
    finite = geo["_plot_value"].notna()
    if finite.any():
        geo.loc[finite].plot(ax=ax, column="_plot_value", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.16, zorder=2)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(handles=[Patch(facecolor="#d7d7d7", label="Outside comparison")], loc="lower left", fontsize=7.5, handlelength=1)


def v18(name: str) -> pd.DataFrame:
    return read_csv(SOURCE_DIR / name.replace("_v18", "_v20"))


def write_source(frame: pd.DataFrame, name: str) -> str:
    """Return a source-table name without rewriting committed inputs."""
    committed = SOURCE_DIR / name
    if committed.exists():
        return committed.name
    target = SRC_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig", float_format="%.14g")
    return str(target.relative_to(OUT))


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=10, weight="bold", va="bottom", ha="left")


def _status_labels() -> tuple[dict[str, str], dict[str, str]]:
    labels = {
        "positive_activity": "Positive activity",
        "source_encoded_zero": "Source-coded zero",
        "q_missing_with_facility": "Q missing, facilities observed",
        "no_compatible_weeks": "No compatible weeks",
    }
    colors = {
        "positive_activity": COL["A"],
        "source_encoded_zero": "#d4ad62",
        "q_missing_with_facility": "#8f78a5",
        "no_compatible_weeks": "#e3e3e3",
    }
    return labels, colors


def figure1_coverage() -> dict[str, object]:
    maps = v18("fig1_map_members_v18.csv")
    coverage = v18("fig1_annual_coverage_v18.csv")
    map_source = write_source(maps, "fig1_map_members_v20.csv")
    cov_source = write_source(coverage, "fig1_annual_coverage_v20.csv")
    labels, colors = _status_labels()
    display = maps.copy()
    display["status_label"] = display["map_status_2023"].map(labels).fillna(display["map_status_2023"])
    fig = plt.figure(figsize=(6.69, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.0], hspace=0.40, wspace=0.55)
    ax = fig.add_subplot(gs[0, :])
    map_palette = {labels[k]: colors[k] for k in labels}
    draw_status_map(ax, display, status_col="status_label", palette=map_palette, categories=list(map_palette), title="2023 observation state")
    for txt in list(ax.texts):
        txt.remove()
    matched = display.loc[display["map_match_status"].eq("matched_unique_admin_code")]
    handles = []
    for key, label in labels.items():
        count = int(matched["status_label"].eq(label).sum())
        if count:
            handles.append(Patch(facecolor=colors[key], edgecolor="none", label=f"{label} (n={count})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=4, fontsize=6.8, handlelength=1.0, columnspacing=0.8)
    ax.text(0.99, 0.015, "333 matched city polygons; 334-city registry", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=COL["muted"])
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    for col, label, color in [
        ("facility_observable_city_n", "Facilities observed", COL["N"]),
        ("positive_activity_city_n", "Positive activity", COL["A"]),
        ("source_encoded_zero_city_n", "Source-coded zero", "#d4ad62"),
        ("no_compatible_weeks_city_n", "No compatible weeks", "#9b9b9b"),
    ]:
        if col in coverage:
            ax.plot(coverage["year"], num(coverage[col]), "o-", ms=2.4, lw=1.1, color=color, label=label)
    ax.set(xlabel="Year", ylabel="Cities (denominator 334)", title="Annual coverage states")
    ax.set_ylim(-5, 350)
    ax.set_xticks([2016, 2018, 2020, 2022, 2023])
    ax.grid(axis="y", color=COL["grid"], lw=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.30),fontsize=7.5,ncol=2,handlelength=1,columnspacing=0.6)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    evidence = pd.DataFrame(
        {
            "label": ["Registry", "Facilities observed", "Positive activity", "Policy clauses"],
            "cities": [334, 293, 123, 88],
            "records": [2672, 890, 464, 466],
            "record_label": ["city-years", "city-years", "city-years", "clauses"],
        }
    )
    yy = np.arange(len(evidence))
    ax.hlines(yy, 0, evidence["cities"], color="#ccd4d9", lw=4.5)
    ax.scatter(evidence["cities"], yy, s=20, color=[COL["dark"], COL["N"], COL["A"], COL["accent"]], zorder=3)
    for y, row in zip(yy, evidence.itertuples()):
        ax.text(row.cities + 7, y, f"{row.cities} cities\n{row.records:,} {row.record_label}", va="center", fontsize=7.5)
    ax.set(yticks=yy, yticklabels=["Registry","Facilities","Positive activity","Policy clauses"], xlabel="Cities", title="Evidence coverage")
    ax.set_xlim(0, 620);ax.set_ylim(-.5,3.5);ax.set_xticks([0,150,300,450,600])
    ax.grid(axis="x", color=COL["grid"], lw=0.55)
    panel_label(ax, "c")
    save_figure(fig, "fig1_coverage")
    return {"stem": "fig1_coverage", "core_conclusion": "Registry, charging, activity and policy evidence have distinct denominators.", "source_tables": [map_source, cov_source], "status": "complete"}


def figure2_observation() -> dict[str, object]:
    profile = v18("fig2_observation_profile_v18.csv")
    dist = v18("fig2_outcome_distributions_v18.csv")
    psrc = write_source(profile, "fig2_observation_profile_v20.csv")
    dsrc = write_source(dist, "fig2_outcome_distributions_v20.csv")
    states = ["no_compatible_weeks", "q_missing_with_facility", "source_encoded_zero", "positive_activity"]
    colors = ["#e3e3e3", "#8f78a5", "#d4ad62", COL["A"]]
    codes = {state: i for i, state in enumerate(states)}
    fig = plt.figure(figsize=(6.69, 6.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.0], wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    years = sorted(profile["year"].unique())
    heat = profile.pivot(index="city_order", columns="year", values="observation_state").sort_index().reindex(columns=years)
    arr = heat.replace(codes).to_numpy(dtype=float)
    ax.imshow(arr, aspect="auto", interpolation="none", cmap=ListedColormap(colors), vmin=-0.5, vmax=len(states) - 0.5)
    ax.set(xticks=np.arange(len(years)), xticklabels=years, xlabel="Year", ylabel="Registry cities (n=334)", title="Observation state")
    ax.set_yticks([0, 100, 200, 300, 333], ["1", "101", "201", "301", "334"])
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=colors[i], label=state.replace("_", " ").title()) for i, state in enumerate(states)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=6.4)
    panel_label(ax, "a")
    right = gs[0, 1].subgridspec(3, 1, hspace=0.56)
    for j, variable in enumerate(["N", "A", "I"]):
        ax = fig.add_subplot(right[j, 0])
        d = dist.loc[dist["variable"].eq(variable)].sort_values("year")
        x = num(d["year"]).to_numpy()
        p10, p25, med, p75, p90 = [num(d[c]).to_numpy() for c in ["p10", "p25", "median", "p75", "p90"]]
        color = COL[variable]
        ax.fill_between(x, p10, p90, color=color, alpha=0.10, label="P10–P90")
        ax.fill_between(x, p25, p75, color=color, alpha=0.26, label="IQR")
        ax.plot(x, med, "o-", color=color, lw=1.2, ms=2.8, label="Median")
        for k,(xx, yy, n) in enumerate(zip(x, med, d["n"])):
            ax.annotate(('n=' if k==0 else '')+str(int(n)), (xx, yy), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=COL["muted"])
        ax.set_ylabel(f"ln {variable}")
        ax.set_title({'N':'Mean facilities N','A':'Positive activity A','I':'Positive intensity I'}[variable], loc="left")
        ax.set_xticks([2016, 2018, 2020, 2022, 2023])
        if j < 2:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Year")
        ax.grid(axis="y", color=COL["grid"], lw=0.55)
        if j==0:
            ax.legend(loc="upper center",bbox_to_anchor=(0.5,1.40),fontsize=7.5,ncol=3,handlelength=1.2)
        panel_label(ax, chr(98+j))
    save_figure(fig, "fig2_observation")
    return {"stem": "fig2_observation", "core_conclusion": "Annual observation states and outcome distributions are heterogeneous on their appropriate positive supports.", "source_tables": [psrc, dsrc], "status": "complete"}


WINDOW_ORDER = [
    "adjacent_2019_2020", "adjacent_2020_2021", "adjacent_2021_2022", "adjacent_2022_2023",
    "continuous_2019_2021", "continuous_2020_2022", "continuous_2021_2023", "continuous_2019_2023",
]
WINDOW_SHORT = {
    "adjacent_2019_2020": "19–20", "adjacent_2020_2021": "20–21", "adjacent_2021_2022": "21–22", "adjacent_2022_2023": "22–23",
    "continuous_2019_2021": "19–21", "continuous_2020_2022": "20–22", "continuous_2021_2023": "21–23", "continuous_2019_2023": "19–23",
}


def figure3_growth() -> dict[str, object]:
    changes = v18("fig3_city_changes_v18.csv")
    results = v18("fig3_window_results_v18.csv")
    trends = v18("fig3_window_trends_v18.csv")
    csrc = write_source(changes, "fig3_city_changes_v20.csv")
    rsrc = write_source(results, "fig3_window_results_v20.csv")
    tsrc = write_source(trends, "fig3_window_trends_v20.csv")
    results = results.set_index("window_id").reindex(WINDOW_ORDER).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(6.69, 5.4))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.13, top=0.92, wspace=0.38, hspace=0.53)
    ax = axes[0, 0]
    fixed = trends.loc[trends["window_id"].eq("continuous_2019_2023")].sort_values("year")
    for var, label, color in [("A", "A: activity", COL["A"]), ("N", "N: facilities", COL["N"]), ("I", "I: intensity", COL["I"])]:
        ax.plot(fixed["year"], fixed[f"{var}_index"], "o-", ms=2.8, lw=1.15, color=color, label=label)
    ax.axhline(100, color="#777", lw=0.6, ls="--")
    ax.set(xticks=[2019, 2020, 2021, 2022, 2023], xlabel="Year", ylabel="Index (2019 = 100)", title="Fixed 36-city path")
    ax.grid(axis="y", color=COL["grid"], lw=0.55); ax.legend(loc="upper left", fontsize=6.3); panel_label(ax, "a")
    x = np.arange(len(results))
    ax = axes[0, 1]
    for var, label, color in [("A", "A", COL["A"]), ("N", "N", COL["N"]), ("I", "I", COL["I"])]:
        vals = num(results[f"{var}_aggregate_percent_change"])
        ax.scatter(x, vals, s=17, color=color, label=label, zorder=3)
    for xx, n in zip(x, results["n_cities"]):
        ax.text(xx, ax.get_ylim()[0] if np.isfinite(ax.get_ylim()[0]) else 0, f"n={int(n)}", ha="center", va="bottom", fontsize=5.8, color=COL["muted"])
    ax.axhline(0, color="#666", lw=0.6); ax.set(xticks=x, xticklabels=[WINDOW_SHORT[w] for w in WINDOW_ORDER], xlabel="Window", ylabel="Aggregate change (%)", title="Network aggregates")
    ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", color=COL["grid"], lw=0.55); ax.legend(loc="best", ncol=3, fontsize=6.3); panel_label(ax, "b")
    ax = axes[1, 0]
    for var, label, color in [("delta_lnA", "Δ ln A", COL["A"]), ("delta_lnN", "Δ ln N", COL["N"]), ("delta_lnI", "Δ ln I", COL["I"])]:
        med, lo, hi = [], [], []
        for wid in WINDOW_ORDER:
            z = num(changes.loc[changes["window_id"].eq(wid), var]).dropna()
            med.append(z.median()); lo.append(z.quantile(0.25)); hi.append(z.quantile(0.75))
        ax.scatter(x, med, s=15, color=color, label=label, zorder=3); ax.vlines(x, lo, hi, color=color, lw=1.7, alpha=0.65)
    ax.axhline(0, color="#666", lw=0.6); ax.set(xticks=x, xticklabels=[WINDOW_SHORT[w] for w in WINDOW_ORDER], xlabel="Window", ylabel="City log change (median; IQR)", title="Equal-city distributions")
    ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", color=COL["grid"], lw=0.55); ax.legend(loc="best", ncol=3, fontsize=6.3); panel_label(ax, "c")
    ax = axes[1, 1]
    aggregate = num(results["I_aggregate_percent_change"])
    med_pct = []
    for wid in WINDOW_ORDER:
        z = num(changes.loc[changes["window_id"].eq(wid), "delta_lnI"]).dropna(); med_pct.append(100 * (np.exp(z.median()) - 1))
    ax.bar(x - 0.18, aggregate, width=0.34, color=COL["I"], alpha=0.84, label="Aggregate I")
    ax.scatter(x + 0.18, med_pct, s=16, color=COL["dark"], label="Median city I", zorder=3)
    ax.axhline(0, color="#666", lw=0.6); ax.set(xticks=x, xticklabels=[WINDOW_SHORT[w] for w in WINDOW_ORDER], xlabel="Window", ylabel="Intensity change (%)", title="Network versus typical city")
    ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", color=COL["grid"], lw=0.55); ax.legend(loc="best", fontsize=6.3); panel_label(ax, "d")
    save_figure(fig, "fig3_growth")
    return {"stem": "fig3_growth", "core_conclusion": "Facilities generally grow faster than recorded activity, while aggregate and equal-city summaries differ.", "source_tables": [csrc, rsrc, tsrc], "status": "complete"}


def regime_scatter(ax: plt.Axes, data: pd.DataFrame, title: str, legend: bool = False) -> None:
    for regime, z in data.groupby("growth_regime", sort=False):
        ax.scatter(num(z["x_delta_lnN"]), num(z["y_delta_lnA"]), s=19, color=REGIME_COLORS.get(regime, "#929292"), edgecolor="white", linewidth=0.25, alpha=0.85, label=REGIME_SHORT.get(regime, regime))
    vals = np.r_[num(data["x_delta_lnN"]), num(data["y_delta_lnA"])]; vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max()); span = max(hi - lo, 0.5); lo -= 0.04 * span; hi += 0.04 * span
    ax.plot([lo, hi], [lo, hi], ls="--", color="#777", lw=0.7); ax.axvline(0, color="#b8b8b8", lw=0.55); ax.axhline(0, color="#b8b8b8", lw=0.55)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Δ ln N", ylabel="Δ ln A", title=title)
    ax.grid(axis="both", color=COL["grid"], lw=0.5)
    if legend:
        ax.legend(loc="upper left", fontsize=5.8, handlelength=1.0, labelspacing=0.25)


def figure4_regimes() -> dict[str, object]:
    regimes = v18("fig4_city_growth_regimes_v18.csv")
    summary = v18("fig4_growth_regime_summary_v18.csv")
    recent = v18("fig4_spatial_recent_93_v18.csv")
    size = v18("fig4_initial_scale_summary_v18.csv")
    srcs = [
        write_source(regimes, "fig4_city_growth_regimes_v20.csv"),
        write_source(summary, "fig4_growth_regime_summary_v20.csv"),
        write_source(recent, "fig4_spatial_recent_93_v20.csv"),
        write_source(size, "fig4_initial_scale_summary_v20.csv"),
    ]
    fig = plt.figure(figsize=(6.69, 7.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.65], hspace=0.35, wspace=0.40)
    fixed = regimes.loc[regimes["window_id"].eq("continuous_2019_2023")].copy()
    ax = fig.add_subplot(gs[0, 0]); regime_scatter(ax, fixed, "Fixed cohort (n=36)", legend=True); panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); regime_scatter(ax, recent, "Recent window (n=93)", legend=True)
    # Zoom inset retains the complete cloud in the parent panel while making
    # the dense central region legible at final size.
    core = recent.loc[num(recent["x_delta_lnN"]).abs().lt(0.65) & num(recent["y_delta_lnA"]).abs().lt(0.65)]
    if len(core) > 4:
        ins = inset_axes(ax, width="43%", height="43%", loc="lower left", borderpad=1.0)
        for regime, z in core.groupby("growth_regime", sort=False):
            ins.scatter(num(z["x_delta_lnN"]), num(z["y_delta_lnA"]), s=8, color=REGIME_COLORS.get(regime, "#929292"), edgecolor="white", linewidth=0.15)
        ins.axvline(0, color="#b8b8b8", lw=0.4); ins.axhline(0, color="#b8b8b8", lw=0.4); ins.set_xlim(-0.65, 0.65); ins.set_ylim(-0.65, 0.65); ins.set_xticks([]);ins.set_yticks([]);ins.set_title("Zoom ±0.65", fontsize=7.5)
    panel_label(ax, "b")
    ax = fig.add_subplot(gs[1, 0:])
    norm = TwoSlopeNorm(vmin=-DISPLAY_BOUND, vcenter=0.0, vmax=DISPLAY_BOUND)
    map_data = recent.copy(); map_data["delta_lnI"] = num(map_data["dI_delta_lnI"])
    draw_white_value_map(ax, map_data, "delta_lnI", norm=norm, cmap="PuOr_r", title="Recent intensity change across all 93 cities")
    sm = ScalarMappable(norm=norm, cmap="PuOr_r"); sm.set_array(num(map_data["delta_lnI"]).dropna().to_numpy())
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.036, pad=0.06, extend="both"); cb.set_ticks([-3.5, 0, 3.5]); cb.set_label("Δ ln I", fontsize=7)
    over = map_data.loc[num(map_data["delta_lnI"]).abs().gt(DISPLAY_BOUND)]
    for row in over.itertuples():
        if np.isfinite(row.geometry_centroid_longitude) and np.isfinite(row.geometry_centroid_latitude):
            ax.scatter(row.geometry_centroid_longitude, row.geometry_centroid_latitude, marker="^" if row.delta_lnI > 0 else "v", s=24, facecolor="white", edgecolor="black", lw=0.5, zorder=8)
    panel_label(ax, "c")
    # Place baseline-size support in the SI moved panel; the main map is the
    # required spatial hero and keeps the city cloud readable.
    save_figure(fig, "fig4_regimes")
    return {"stem": "fig4_regimes", "core_conclusion": "City growth regimes remain heterogeneous under facility expansion, with spatially varying intensity change.", "source_tables": srcs, "status": "complete"}


CITY_NAMES = {
    "成都": "Chengdu", "福州": "Fuzhou", "海口": "Haikou", "南充": "Nanchong", "绍兴": "Shaoxing", "深圳": "Shenzhen", "苏州": "Suzhou", "杭州": "Hangzhou", "西安": "Xi'an", "南京": "Nanjing", "广州": "Guangzhou", "重庆": "Chongqing", "太原": "Taiyuan", "无锡": "Wuxi", "宁波": "Ningbo", "合肥": "Hefei", "青岛": "Qingdao", "厦门": "Xiamen", "天津": "Tianjin", "武汉": "Wuhan", "长沙": "Changsha", "郑州": "Zhengzhou", "济南": "Jinan", "石家庄": "Shijiazhuang", "东莞": "Dongguan", "佛山": "Foshan", "南宁": "Nanning", "昆明": "Kunming", "贵阳": "Guiyang", "兰州": "Lanzhou", "乌鲁木齐": "Urumqi", "长春": "Changchun", "沈阳": "Shenyang", "哈尔滨": "Harbin", "大连": "Dalian", "珠海": "Zhuhai", "中山": "Zhongshan", "百色": "Baise", "汕头": "Shantou", "惠州": "Huizhou", "温州": "Wenzhou", "金华": "Jinhua", "台州": "Taizhou", "烟台": "Yantai", "南昌": "Nanchang", "桂林": "Guilin", "三亚": "Sanya", "临沂": "Linyi", "绵阳": "Mianyang", "宜昌": "Yichang", "襄阳": "Xiangyang", "扬州": "Yangzhou", "北京": "Beijing", "三明": "Sanming", "莆田": "Putian",
}


def city_name(value: object, fallback: str) -> str:
    raw = str(value).replace("市", "")
    for key, label in CITY_NAMES.items():
        if key in raw:
            return label
    return fallback


def contribution_rank(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy(); data["total_index_points"] = num(data["total_index_points"])
    fallback_names = data["city"].astype(str).str.replace("市", "", regex=False) if "city" in data.columns else data["city_key"].astype(str)
    data["label"] = [city_name(v, fallback_names.iloc[i]) for i, v in enumerate(data["city_key"])]
    pos = data.loc[data["total_index_points"].gt(0)].sort_values("total_index_points", ascending=False)
    neg = data.loc[data["total_index_points"].lt(0)].sort_values("total_index_points")
    rows = []
    for i, (_, row) in enumerate(pos.head(5).iterrows(), 1): rows.append({**row.to_dict(), "display_order": i, "group": "named +"})
    rows.append({"label": "Other +", "city_key": "__other_positive__", "total_index_points": pos.iloc[5:]["total_index_points"].sum(), "within_index_points": np.nan, "composition_index_points": np.nan, "display_order": 6, "group": "other +"})
    rows.append({"label": "Other −", "city_key": "__other_negative__", "total_index_points": neg.iloc[5:]["total_index_points"].sum(), "within_index_points": np.nan, "composition_index_points": np.nan, "display_order": 7, "group": "other −"})
    for i, (_, row) in enumerate(neg.head(5).iterrows(), 8): rows.append({**row.to_dict(), "display_order": i, "group": "named −"})
    return pd.DataFrame(rows)


def figure5_contributions() -> dict[str, object]:
    data = v18("fig5_city_contributions_v18.csv"); summary = v18("fig5_contribution_summary_v18.csv")
    srcs = [write_source(data, "fig5_city_contributions_v20.csv"), write_source(summary, "fig5_contribution_summary_v20.csv")]
    fig, axes = plt.subplots(1, 3, figsize=(6.69, 4.75), gridspec_kw={"width_ratios": [1, 1, 0.86]})
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.19, top=0.82, wspace=0.48)
    for ax, wid, letter in zip(axes[:2], ["adjacent_2020_2021", "adjacent_2022_2023"], ["a", "b"]):
        d = data.loc[data["window_id"].eq(wid)]; rank = contribution_rank(d); y = np.arange(len(rank)); vals = num(rank["total_index_points"]).to_numpy()
        ax.axhspan(4.48, 6.52, color="#f3f5f4", zorder=0)
        ax.axhline(4.5, color="#aeb9b5", lw=0.55, zorder=1); ax.axhline(6.5, color="#aeb9b5", lw=0.55, zorder=1)
        ax.barh(y, vals, height=0.60, color=[COL["gain"] if x >= 0 else COL["loss"] for x in vals], alpha=0.80)
        selected = rank.loc[~rank["city_key"].astype(str).str.startswith("__")]
        for _, row in selected.iterrows():
            j = int(row["display_order"] - 1); ax.scatter(row["within_index_points"], j - 0.12, s=14, color=COL["within"], zorder=4); ax.scatter(row["composition_index_points"], j + 0.12, s=14, marker="s", color=COL["composition"], zorder=4)
        ax.axvline(0, color="#666", lw=0.6); ax.set(yticks=y, yticklabels=rank["label"], xlabel="Net contribution (index points)", title={"adjacent_2020_2021": "2020–21", "adjacent_2022_2023": "2022–23"}[wid] + f" (n={len(d)})"); ax.invert_yaxis(); ax.grid(axis="x", color=COL["grid"], lw=0.5); ax.tick_params(axis="y", labelsize=6.3)
        for tick in ax.get_yticklabels(): tick.set_fontname("Arial"); tick.set_fontsize(6.3)
        panel_label(ax, letter)
    ax = axes[2]
    if "component" in summary.columns:
        focus = summary.loc[summary["window_id"].isin(["adjacent_2020_2021", "adjacent_2022_2023"]) & summary["component"].isin(["within", "composition", "total"])].copy()
        pivot = focus.pivot(index="window_id", columns="component", values="sum_index_points").reindex(["adjacent_2020_2021", "adjacent_2022_2023"])
    else:
        focus = summary.loc[summary["window_id"].isin(["adjacent_2020_2021", "adjacent_2022_2023"])].copy().set_index("window_id")
        pivot = focus.rename(columns={"within_sum_index_points": "within", "composition_sum_index_points": "composition", "total_sum_index_points": "total"})[["within", "composition", "total"]].reindex(["adjacent_2020_2021", "adjacent_2022_2023"])
    x = np.arange(len(pivot))
    for j, comp in enumerate(["within", "composition", "total"]):
        if comp in pivot:
            ax.bar(x + (j - 1) * 0.23, pivot[comp], width=0.21, color=[COL["within"], COL["composition"], COL["dark"]][j], label=comp.title(), alpha=0.85)
    ax.axhline(0, color="#666", lw=0.6); ax.set(xticks=x, xticklabels=["20–21", "22–23"], ylabel="Index points (base 100)", title="Exact decomposition"); ax.grid(axis="y", color=COL["grid"], lw=0.5); ax.legend(handles=[Patch(facecolor=COL["within"], label="Within-city"), Patch(facecolor=COL["composition"], label="Changing weight"), Patch(facecolor=COL["dark"], label="Net change")], loc="lower left", fontsize=6.1)
    panel_label(ax, "c")
    ax.get_legend().remove()
    fig.legend(handles=[Patch(facecolor=COL["dark"], label="Net change"), Patch(facecolor=COL["gain"], label="Net contribution"), Line2D([0], [0], marker="o", linestyle="none", color=COL["within"], label="Within-city", markersize=4), Line2D([0], [0], marker="s", linestyle="none", color=COL["composition"], label="Changing weight", markersize=4)], loc="upper center", bbox_to_anchor=(0.53, 0.965), ncol=4, fontsize=7.5, handlelength=1.0, columnspacing=0.8)
    save_figure(fig, "fig5_contributions")
    return {"stem": "fig5_contributions", "core_conclusion": "Network intensity change separates within-city change from changing facility weights.", "source_tables": srcs, "status": "complete"}


def find_transition_sources() -> tuple[Path | None, Path | None]:
    members = SOURCE_DIR / "growth_transition_membership_v20.csv"
    summary = SOURCE_DIR / "growth_transition_summary_v20.csv"
    return (members if members.exists() else None, summary if summary.exists() else None)


def _col(frame: pd.DataFrame, names: list[str]) -> str | None:
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in frame.columns}
    for name in names:
        if re.sub(r"[^a-z0-9]", "", name.lower()) in norm:
            return norm[re.sub(r"[^a-z0-9]", "", name.lower())]
    for name in names:
        token = re.sub(r"[^a-z0-9]", "", name.lower())
        hits = [c for c in frame.columns if token and token in re.sub(r"[^a-z0-9]", "", str(c).lower())]
        if len(hits) == 1:
            return hits[0]
    return None


def _pick_comparison(df: pd.DataFrame, target: int) -> tuple[pd.DataFrame, str]:
    comp = _col(df, ["comparison", "comparison_id", "sample", "window", "set_id"])
    if comp is None:
        return df.copy(), "single comparison"
    vals = df[comp].astype(str)
    hits = [v for v in vals.dropna().unique() if str(target) in v or (target == 52 and ("main" in v.lower() or "common" in v.lower()))]
    if hits:
        return df.loc[vals.eq(hits[0])].copy(), str(hits[0])
    # If city counts are supplied, use the exact target count.
    ncol = _col(df, ["n_cities", "city_n", "cities"])
    if ncol:
        z = df.loc[num(df[ncol]).eq(target)]
        if not z.empty:
            return df.loc[vals.eq(z.iloc[0][comp])].copy(), str(z.iloc[0][comp])
    unique = list(vals.dropna().unique())
    if len(unique) == 1:
        return df.copy(), str(unique[0])
    raise ValueError(f"Could not select {target}-city transition comparison; observed {unique}")


def _transition_cells(raw: pd.DataFrame, target: int) -> tuple[pd.DataFrame, str, list[str], str]:
    data, comparison = _pick_comparison(raw, target)
    early = _col(data, ["early_regime", "regime_early", "early_category", "early_state", "regime_start", "start_regime"])
    late = _col(data, ["late_regime", "regime_late", "late_category", "late_state", "regime_end", "end_regime"])
    if early is None or late is None:
        raise ValueError(f"Transition membership lacks early/late categories: {list(data.columns)}")
    data = data.copy()
    def short(v: object) -> str:
        text = str(v)
        if text in REGIME_ORDER or text in REGIME_SHORT:
            return text if text in REGIME_ORDER else "other"
        low = text.lower()
        if "faster" in low: return REGIME_ORDER[0]
        if "lag" in low or "follow" in low or "slower" in low: return REGIME_ORDER[1]
        if "contract" in low or "fall" in low or "declin" in low: return REGIME_ORDER[2]
        return "other"
    data["early_category"] = data[early].map(short)
    data["late_category"] = data[late].map(short)
    # Keep the full membership source and use a compact derived table only for plotting.
    cells = data.groupby(["late_category", "early_category"], dropna=False).size().rename("n").reset_index()
    cats = REGIME_ORDER
    grid = pd.MultiIndex.from_product([cats, cats], names=["late_category", "early_category"]).to_frame(index=False).merge(cells, how="left").fillna({"n": 0})
    grid["n"] = grid["n"].astype(int)
    coltot = grid.groupby("early_category")["n"].transform("sum")
    grid["conditional_pct"] = np.where(coltot > 0, 100 * grid["n"] / coltot, np.nan)
    grid["comparison"] = comparison
    grid["n_cities"] = int(data[_col(data, ["city_key", "city", "municipality"])].nunique()) if _col(data, ["city_key", "city", "municipality"]) else len(data)
    return grid, comparison, cats, f"{early} / {late}"


def _plot_transition(
    ax: plt.Axes,
    cells: pd.DataFrame,
    cats: list[str],
    title: str,
    *,
    x_window: str,
    y_window: str,
) -> None:
    matrix = cells.pivot(index="late_category", columns="early_category", values="conditional_pct").reindex(index=cats, columns=cats).to_numpy(dtype=float)
    counts = cells.pivot(index="late_category", columns="early_category", values="n").reindex(index=cats, columns=cats).to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(100, np.nanmax(matrix) if np.isfinite(matrix).any() else 100), aspect="equal")
    early_margins = cells.groupby("early_category")["n"].sum().reindex(cats).astype(int).tolist()
    late_margins = cells.groupby("late_category")["n"].sum().reindex(cats).astype(int).tolist()
    xlabels = [f"{REGIME_SHORT[c]}\nn={n}" for c, n in zip(cats, early_margins)]
    ylabels = [f"{REGIME_SHORT[c]}\nn={n}" for c, n in zip(cats, late_margins)]
    ax.set(
        xticks=np.arange(len(cats)),
        yticks=np.arange(len(cats)),
        xticklabels=xlabels,
        yticklabels=ylabels,
        xlabel=f"Early window: {x_window}",
        ylabel=f"Late window: {y_window}",
        title=title,
    )
    ax.tick_params(axis="x", rotation=28, labelsize=6.8); ax.tick_params(axis="y", labelsize=6.8)
    for i in range(len(cats)):
        for j in range(len(cats)):
            if np.isfinite(matrix[i, j]):
                text = f"{int(counts[i, j])}\n{matrix[i, j]:.0f}%"
                ax.text(j, i, text, ha="center", va="center", fontsize=7.2, color="white" if matrix[i, j] > 54 else COL["dark"], linespacing=0.95)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.047, pad=0.03); cb.set_label("Column-conditional share (%)", fontsize=7); cb.ax.tick_params(labelsize=6.4)


def figure5_growth_transitions() -> tuple[dict[str, object], dict[str, object]]:
    member_path, summary_path = find_transition_sources()
    if member_path is None:
        pending = {"stem": "fig5_growth_transitions", "core_conclusion": "Pending saved A/B growth-transition membership table.", "source_tables": [], "status": "pending: growth_transition_membership.csv not found"}
        return pending, {"stem": "growth_transitions_supplement", "core_conclusion": pending["core_conclusion"], "source_tables": [], "status": pending["status"]}
    raw = read_csv(member_path)
    member_src = write_source(raw, "growth_transition_membership_v20.csv")
    extra_src: list[str] = []
    if summary_path is not None:
        extra = read_csv(summary_path); extra_src.append(write_source(extra, "growth_transition_summary_v20.csv"))
    main_cells, main_comp, cats, mapping = _transition_cells(raw, 52)
    main_cell_src = write_source(main_cells, "fig5_growth_transition_cells_52_v20.csv")
    fig, ax = plt.subplots(figsize=(6.69, 4.55)); _plot_transition(ax, main_cells, cats, "Growth-category transitions (52-city comparison)", x_window="2020–2021", y_window="2022–2023")
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.31, top=0.86)
    save_figure(fig, "fig5_growth_transitions")
    main_contract = {"stem": "fig5_growth_transitions", "core_conclusion": "Same-city growth categories persist or change between early and late windows.", "source_tables": [member_src, *extra_src, main_cell_src], "status": "complete", "comparison": main_comp, "category_mapping": mapping}

    supp_cells, supp_comp, _, _ = _transition_cells(raw, 73)
    supp_src = write_source(supp_cells, "fig5_growth_transition_cells_73_v20.csv")
    fig, ax = plt.subplots(figsize=(6.69, 4.55)); _plot_transition(ax, supp_cells, cats, "Growth-category transitions (73-city comparison)", x_window="2021–2022", y_window="2022–2023")
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.31, top=0.86)
    save_figure(fig, "growth_transitions_supplement")
    supp_contract = {"stem": "growth_transitions_supplement", "core_conclusion": "The 73-city transition comparison is retained as supplementary support.", "source_tables": [member_src, *extra_src, supp_src], "status": "complete", "comparison": supp_comp, "category_mapping": mapping}
    return main_contract, supp_contract


def figure6_recording() -> dict[str, object]:
    comp = v18("fig6_observation_components_v18.csv")
    source = write_source(comp, "fig6_observation_components_v20.csv")
    focus = ["continuous_2019_2023", "adjacent_2022_2023"]
    colors = {focus[0]: COL["A"], focus[1]: COL["accent"]}; labels = {focus[0]: "2019–2023 (36)", focus[1]: "2022–2023 (93)"}
    cols = ["delta_ln_p", "delta_ln_q_positive_mean", "delta_ln_N", "delta_ln_I"]
    names = ["Δ ln p", "Δ ln mean activity", "Δ ln N", "Δ ln I"]
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.55), gridspec_kw={"width_ratios": [1.06, 1]})
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.24, top=0.85, wspace=0.46)
    ax = axes[0]; x = np.arange(4)
    for wid, off in zip(focus, [-0.17, 0.17]):
        vals = [num(comp.loc[comp["window_id"].eq(wid), c]).dropna().to_numpy() for c in cols]
        bp = ax.boxplot(vals, positions=x + off, widths=0.27, patch_artist=True, showfliers=False, medianprops={"color": "#222", "lw": 0.8}, whiskerprops={"color": colors[wid], "lw": 0.7}, capprops={"color": colors[wid], "lw": 0.7})
        for box in bp["boxes"]: box.set(facecolor=colors[wid], edgecolor=colors[wid], alpha=0.4)
        rng = np.random.default_rng(20261111 + focus.index(wid))
        for i, arr in enumerate(vals): ax.scatter(np.repeat(x[i] + off, len(arr)) + rng.uniform(-0.04, 0.04, len(arr)), arr, s=3.4, color=colors[wid], alpha=0.35, lw=0)
    ax.axhline(0, color="#666", lw=0.6); ax.set(xticks=x, xticklabels=["Δ ln p", "Δ ln mean\nactivity", "Δ ln N", "Δ ln I"], ylabel="City-level log change", title="Component distributions"); ax.grid(axis="y", color=COL["grid"], lw=0.5); ax.legend(handles=[Patch(facecolor=colors[w], alpha=0.4, label=labels[w]) for w in focus], loc="upper left", fontsize=6.2); panel_label(ax, "a")
    ax = axes[1]
    for j, wid in enumerate(focus):
        d = comp.loc[comp["window_id"].eq(wid)]
        vals = [d["delta_ln_p"].mean(), d["delta_ln_q_positive_mean"].mean(), -d["delta_ln_N"].mean(), d["delta_ln_I"].mean()]
        xx = x + (j - 0.5) * 0.27
        ax.bar(xx, vals, width=0.25, color=colors[wid], alpha=0.82, label=labels[wid])
        # Exact component means are reported in the accompanying source table.
    ax.axhline(0, color="#666", lw=0.6); ax.set(xticks=x, xticklabels=["Δ ln p", "Δ ln mean\nactivity", "−Δ ln N", "Δ ln I"], ylabel="Equal-city mean log change", title="Additive mean identity"); ax.grid(axis="y", color=COL["grid"], lw=0.5); ax.legend(loc="lower left", fontsize=6.2); panel_label(ax, "b")
    save_figure(fig, "fig6_recording")
    return {"stem": "fig6_recording", "core_conclusion": "Recorded intensity change separates positive-record frequency, mean activity, and facility exposure.", "source_tables": [source], "status": "complete"}


def figureS_recording_coverage() -> dict[str, object]:
    """Move compatible-week coverage and coded-zero distributions to SI."""

    comp = v18("fig6_observation_components_v18.csv")
    source = write_source(comp, "figS_recording_coverage_v20.csv")
    focus = ["continuous_2019_2023", "adjacent_2022_2023"]
    colors = {focus[0]: COL["A"], focus[1]: COL["accent"]}
    labels = {focus[0]: "2019–2023 (n=36)", focus[1]: "2022–2023 (n=93)"}
    endpoint_positions = {
        (focus[0], "start"): 0,
        (focus[0], "end"): 1,
        (focus[1], "start"): 2,
        (focus[1], "end"): 3,
    }
    xlabels = ["2019–2023\nstart", "2019–2023\nend", "2022–2023\nstart", "2022–2023\nend"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.55), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.82, wspace=0.38)
    handles = [Patch(facecolor=colors[w], alpha=0.45, label=labels[w]) for w in focus]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, fontsize=6.8)

    ax = axes[0]
    for wid in focus:
        d = comp.loc[comp["window_id"].astype(str).eq(wid)]
        for endpoint in ["start", "end"]:
            col = f"coverage_ratio_{endpoint}"
            vals = num(d[col]).dropna().to_numpy()
            pos = endpoint_positions[(wid, endpoint)]
            if vals.size:
                ax.boxplot(
                    [vals],
                    positions=[pos],
                    widths=0.34,
                    patch_artist=True,
                    showfliers=True,
                    boxprops={"facecolor": colors[wid], "alpha": 0.42, "edgecolor": colors[wid]},
                    medianprops={"color": "#222", "lw": 0.8},
                    whiskerprops={"color": colors[wid], "lw": 0.7},
                    capprops={"color": colors[wid], "lw": 0.7},
                    flierprops={"marker": ".", "markersize": 2.2, "markerfacecolor": colors[wid], "markeredgecolor": colors[wid], "alpha": 0.42},
                )
                rng = np.random.default_rng(20261112 + pos)
                ax.scatter(np.repeat(pos, len(vals)) + rng.uniform(-0.07, 0.07, len(vals)), vals, s=3.8, color=colors[wid], alpha=0.42, lw=0)
    ax.axhline(1, color="#666", lw=0.6, ls="--")
    ax.set(xticks=np.arange(4), xticklabels=xlabels, ylim=(-0.03, 1.08), ylabel="Compatible / calendar weeks", title="Compatible-week coverage")
    ax.grid(axis="y", color=COL["grid"], lw=0.5)
    panel_label(ax, "a")

    ax = axes[1]
    max_zero = 1.0
    for wid in focus:
        d = comp.loc[comp["window_id"].astype(str).eq(wid)]
        for endpoint in ["start", "end"]:
            col = f"W_zero_{endpoint}"
            vals = num(d[col]).dropna().to_numpy()
            pos = endpoint_positions[(wid, endpoint)]
            if vals.size:
                max_zero = max(max_zero, float(np.nanmax(vals)))
                ax.boxplot(
                    [vals],
                    positions=[pos],
                    widths=0.34,
                    patch_artist=True,
                    showfliers=True,
                    boxprops={"facecolor": colors[wid], "alpha": 0.42, "edgecolor": colors[wid]},
                    medianprops={"color": "#222", "lw": 0.8},
                    whiskerprops={"color": colors[wid], "lw": 0.7},
                    capprops={"color": colors[wid], "lw": 0.7},
                    flierprops={"marker": ".", "markersize": 2.2, "markerfacecolor": colors[wid], "markeredgecolor": colors[wid], "alpha": 0.42},
                )
                rng = np.random.default_rng(20261120 + pos)
                ax.scatter(np.repeat(pos, len(vals)) + rng.uniform(-0.07, 0.07, len(vals)), vals, s=3.8, color=colors[wid], alpha=0.42, lw=0)
    ax.set(xticks=np.arange(4), xticklabels=xlabels, ylim=(-1, max(5, np.ceil(max_zero + 2))), ylabel="Source-coded-zero weeks", title="Source-coded-zero weeks")
    ax.grid(axis="y", color=COL["grid"], lw=0.5)
    panel_label(ax, "b")
    save_figure(fig, "figS_recording_coverage")
    return {
        "stem": "figS_recording_coverage",
        "core_conclusion": "Compatible-week coverage and source-coded-zero weeks are distinct observation dimensions moved to SI.",
        "source_tables": [source],
        "status": "complete",
    }


def figure7_quarterly_dynamic() -> dict[str, object]:
    paths = v18("fig7_quarterly_dynamic_paths_v18.csv"); contrasts = v18("fig7_quarterly_dynamic_contrasts_v18.csv"); support = v18("fig7_quarterly_dynamic_support_v18.csv")
    sources = [write_source(paths, "fig7_quarterly_dynamic_paths_v20.csv"), write_source(contrasts, "fig7_quarterly_dynamic_contrasts_v20.csv"), write_source(support, "fig7_quarterly_dynamic_support_v20.csv")]
    d3 = paths.loc[paths["model_id"].astype(str).eq("D3")].copy().sort_values("h")
    contrast = contrasts.loc[contrasts["model_id"].astype(str).eq("D3") & contrasts["comparison"].astype(str).eq("c3_minus_c0")]
    if d3.empty or contrast.empty:
        return {"stem": "fig7_quarterly_dynamic", "core_conclusion": "Saved D3 path or joint c3−c0 contrast is unavailable.", "source_tables": sources, "status": "pending: required saved D3 rows missing"}
    contrast = contrast.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.20), gridspec_kw={"width_ratios": [1.15, 0.85]})
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.25, top=0.84, wspace=0.55)
    ax = axes[0]
    x = num(d3["h"]).to_numpy(); y = num(d3["estimate"]).to_numpy(); lo = num(d3["ci_low"]).to_numpy(); hi = num(d3["ci_high"]).to_numpy()
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=4.6, lw=1.05, capsize=2.5, color=COL["A"], ecolor=COL["A"], zorder=3)
    ax.axhline(0, color="#4f8f83", lw=0.7); ax.axhline(1, color="#777", lw=0.7, ls="--"); ax.set(xticks=x, xlabel="Quarters after persistent difference", ylabel="Conditional path elasticity, $c_h$", title="Three-quarter lag model"); ax.grid(axis="y", color=COL["grid"], lw=0.5); panel_label(ax, "a")
    ax = axes[1]
    est, low, high = float(contrast["estimate"]), float(contrast["ci_low"]), float(contrast["ci_high"])
    ax.errorbar([0], [est], yerr=[[est - low], [high - est]], fmt="o", ms=5.0, lw=1.2, capsize=3, color=COL["accent"], ecolor=COL["accent"], zorder=3)
    ax.axhline(0, color="#777", lw=0.7, ls="--"); ax.set(xticks=[0], xticklabels=["c3 − c0"], ylabel="Additional path elasticity", title="Joint lag contrast"); ax.set_xlim(-0.65, 0.65); ax.grid(axis="y", color=COL["grid"], lw=0.5); panel_label(ax, "b")
    fig.text(0.50, 1.0, f"n={int(contrast['n']):,} city-quarters; {int(contrast['clusters'])} cities; 95% CI", ha="center", va="center", fontsize=7.5, color=COL["muted"])
    save_figure(fig, "fig7_quarterly_dynamic")
    return {"stem": "fig7_quarterly_dynamic", "core_conclusion": "The saved three-quarter lag path and joint c3−c0 contrast distinguish the endpoint relationship from additional follow-through.", "source_tables": sources, "status": "complete", "contrast": {"estimate": est, "ci_low": low, "ci_high": high}}


def figure8_policy() -> dict[str, object]:
    categories = v18("fig8_policy_categories_v18.csv"); sequence = v18("fig8_document_sequence_v18.csv"); design = v18("fig8_responsibility_clauses_v18.csv")
    sources = [write_source(categories, "fig8_policy_categories_v20.csv"), write_source(sequence, "fig8_document_sequence_v20.csv"), write_source(design, "fig8_responsibility_clauses_v20.csv")]
    p = categories.loc[categories["row_type"].eq("category")].copy()
    order = ["information_supervision", "operation_maintenance", "continuous_operation_payment", "measurable_subsidy"]
    p = p.set_index("category_key").reindex(order).reset_index()
    labels = ["Information / supervision", "Operation / maintenance", "Continuing operation / payment", "Assessable / subsidy-linked"]
    fig, axes = plt.subplots(2, 1, figsize=(6.69, 6.3), gridspec_kw={"height_ratios": [1.0, 1.3]})
    fig.subplots_adjust(left=0.36, right=0.94, bottom=0.09, top=0.94, hspace=0.48)
    ax = axes[0]; y = np.arange(len(p)); pct = 100 * num(p["city_count"]) / 88
    ax.barh(y, pct, color=[COL["A"], COL["N"], COL["I"], COL["accent"]], alpha=0.84)
    for yy, pp, cc in zip(y, pct, p["city_count"]): ax.text(pp + 1.5, yy, f"{int(cc)} / 88 ({pp:.0f}%)", va="center", fontsize=7)
    ax.set(yticks=y, yticklabels=labels, xlabel="Cities in content scope (percent; n=88)", title="Operational functions")
    ax.invert_yaxis(); ax.set_xlim(0, max(100, float(pct.max()) + 20)); ax.grid(axis="x", color=COL["grid"], lw=0.5); panel_label(ax, "a")
    comp_col = _col(sequence, ["B_vs_G_document_effective_year_sequence", "B_vs_G_document_year_sequence"])
    seq_counts = sequence[comp_col].astype(str).value_counts() if comp_col else pd.Series(dtype=int)
    before = int(sum(v for k, v in seq_counts.items() if "before" in k.lower()))
    same = int(sum(v for k, v in seq_counts.items() if "same" in k.lower()))
    assert before+same==len(sequence)==95
    identity=num(sequence['first_B_and_G_same_gid'])
    vals=[before,same,int(identity.eq(1).sum()),int(identity.eq(0).sum()),int(identity.isna().sum())]
    labels=['Construction before governance','Same first effective year','Same first document','Different first documents','Document order unresolved']
    yy=[0,1,2.6,3.6,4.6]
    ax=axes[1];ax.barh(yy,vals,height=.65,color=[COL['A'],'#a9b7bf',COL['p'],COL['N'],'#b5b5b5'])
    for y0,v in zip(yy,vals):ax.text(v+1,y0,str(v),va='center',fontsize=8)
    ax.set(yticks=yy,yticklabels=labels,xlabel='Cities',title='First construction and governance records (n=95)',xlim=(0,65))
    ax.invert_yaxis();ax.axhline(1.8,color='#aaaaaa',lw=.5);ax.grid(axis='x',color=COL['grid'],lw=.5);panel_label(ax,'b')
    sources.append(write_source(pd.DataFrame({'relation':labels,'cities':vals,'denominator':95}),'fig8_sequence_counts_v20.csv'))
    save_figure(fig, "fig8_policy")
    return {"stem": "fig8_policy", "core_conclusion": "Information/supervision is the broadest documented function; the 95-city timing registry is a separate scope.", "source_tables": sources, "status": "complete"}


def figureS_annual_models() -> dict[str, object]:
    """Redraw the saved annual relationship display with explicit city-year notes."""

    curves = v18("figS_annual_models_curves_v18.csv")
    coeff = v18("figS_annual_models_coefficients_v18.csv")
    tests = v18("figS_annual_models_nonlinearity_tests_v18.csv")
    l0_all = v18("figS_annual_models_ols_coefficients_v18.csv")
    l0 = l0_all.loc[l0_all["term"].astype(str).eq("x")].copy()
    sources = [
        write_source(curves, "figS_annual_models_curves_v20.csv"),
        write_source(coeff, "figS_annual_models_coefficients_v20.csv"),
        write_source(tests, "figS_annual_models_nonlinearity_tests_v20.csv"),
        write_source(l0, "figS_annual_models_ols_coefficients_v20.csv"),
    ]

    fig=plt.figure(figsize=(6.69,5.8))
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],hspace=.65,wspace=.40)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    fig.subplots_adjust(left=.22,right=.97,bottom=.10,top=.94)
    for ax, model, title, color, label in [
        (axes[0], "L0", "OLS: ln A", COL["dark"], "Linear"),
        (axes[0], "L1", "OLS: ln A", COL["A"], "Restricted cubic"),
        (axes[1], "P1", "PPML: mean Q with ln W offset", COL["accent"], "Restricted cubic"),
    ]:
        z = curves.loc[curves["model"].astype(str).eq(model)].sort_values("x")
        if z.empty:
            continue
        axes[0 if model in {"L0", "L1"} else 1].plot(z["x"], z["eta"], lw=1.2, color=color, label=label)
        axes[0 if model in {"L0", "L1"} else 1].fill_between(z["x"], z["eta_ci_low"], z["eta_ci_high"], color=color, alpha=0.12)

    for ax in axes[:2]:
        ax.axhline(1, color="#777", lw=0.6, ls="--")
        ax.set(xlabel="ln N", ylabel="Elasticity (95% CI)")
        ax.grid(color=COL["grid"], lw=0.5)
    axes[0].set_title("OLS: ln A")
    axes[0].legend(loc="upper right", fontsize=5.8)
    axes[1].set_title("PPML with exposure offset")
    axes[1].legend(loc="upper right", fontsize=5.8)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")

    ax = axes[2]
    intervals: list[tuple[str, float, float, float, str]] = []
    if not l0.empty:
        row = l0.iloc[0]
        intervals.append(("Log OLS", float(row["coefficient"]), float(row["ci_low"]), float(row["ci_high"]), f"{int(row['cities'])} cities; {int(row['n'])} city-years"))
    for model_id, label in [("P0_positive", "PPML positive"), ("P0_with_zeros", "PPML with zeros")]:
        z = coeff.loc[coeff["model_id"].astype(str).eq(model_id)]
        if z.empty:
            continue
        row = z.iloc[0]
        intervals.append((label, float(row["beta_lnN"]), float(row["ci_low_lnN"]), float(row["ci_high_lnN"]), f"{int(row['cities'])} cities; {int(row['n'])} city-years"))
    yy = np.arange(len(intervals))
    for i, (label, estimate, low, high, note) in enumerate(intervals):
        color = [COL["dark"], COL["A"], COL["accent"]][i % 3]
        ax.plot([low, high], [i, i], color=color, lw=1.25)
        ax.plot(estimate, i, "o", color=color, ms=4)
    ax.axvline(1, color="#777", lw=0.6, ls="--")
    ax.set(
        yticks=yy,
        yticklabels=[row[0]+'\n'+row[4] for row in intervals],
        xlabel="Facility coefficient / elasticity",
        title="Constant elasticities (95% CI)",
        ylim=(len(intervals) - 0.5, -0.65),
    )
    ax.grid(axis="x", color=COL["grid"], lw=0.5)
    panel_label(ax, "c")
    save_figure(fig, "figS_annual_models")
    return {
        "stem": "figS_annual_models",
        "core_conclusion": "Saved annual curves, intervals and curvature tests are retained in SI without refitting.",
        "source_tables": sources,
        "status": "complete",
    }


def figureS_window_decomposition() -> dict[str, object]:
    """Redraw the saved window decomposition with a clear net-gap legend."""

    decomposition = v18("figS_window_decomposition_v18.csv")
    summary = v18("figS_window_summary_v18.csv")
    sources = [
        write_source(decomposition, "figS_window_decomposition_v20.csv"),
        write_source(summary, "figS_window_summary_v20.csv"),
    ]
    order = summary["window_id"].astype(str).tolist()
    decomposition = decomposition.set_index(["window_id", "component"]).loc[
        [(window, component) for window in order for component in ["within", "composition", "total"]]
    ].reset_index()
    y = np.arange(len(summary))
    labels = [f"{int(row.start_year)}–{int(row.end_year)} (n={int(row.n_cities)})" for row in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.17, top=0.87)
    colors = {"within": COL["within"], "composition": COL["composition"], "total": COL["dark"]}
    names = {"within": "Within-city", "composition": "Changing weight", "total": "Net gap"}
    for component, offset in [("within", -0.20), ("composition", 0.0), ("total", 0.20)]:
        z = decomposition.loc[decomposition["component"].eq(component)].set_index("window_id").reindex(order)
        ax.barh(y + offset, num(z["change_index_points"]), height=0.17, color=colors[component], label=names[component])
    ax.axvline(0, color="#666", lw=0.6)
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Index points (initial intensity = 100)",
        title="Saved window decomposition",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", color=COL["grid"], lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=6.2)
    panel_label(ax, "a")
    save_figure(fig, "figS_window_decomposition")
    return {
        "stem": "figS_window_decomposition",
        "core_conclusion": "Saved within-city, changing-weight and net-gap accounting terms remain distinct across windows.",
        "source_tables": sources,
        "status": "complete",
    }


def figureS_spatial_scale() -> dict[str, object]:
    recent = v18("figS_spatial_scale_v18.csv"); size = v18("figS_spatial_scale_groups_v18.csv")
    sources = [write_source(recent, "figS_spatial_scale_v20.csv"), write_source(size, "figS_spatial_scale_groups_v20.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.35), gridspec_kw={"width_ratios": [1.05, 0.95]})
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.84, wspace=0.38)
    norm = TwoSlopeNorm(vmin=-DISPLAY_BOUND, vcenter=0, vmax=DISPLAY_BOUND); d = recent.copy(); d["delta_lnI"] = num(d["dI_delta_lnI"])
    draw_white_value_map(axes[0], d, "delta_lnI", norm=norm, cmap="PuOr_r", title="Recent Δ ln I (93 cities)")
    sm = ScalarMappable(norm=norm, cmap="PuOr_r"); sm.set_array(num(d["delta_lnI"]).dropna().to_numpy()); cb = fig.colorbar(sm, ax=axes[0], orientation="horizontal", fraction=0.055, pad=0.08, extend="both"); cb.set_ticks([-3.5, 0, 3.5]); cb.set_label("Δ ln I", fontsize=7); panel_label(axes[0], "a")
    ax = axes[1]
    for j, group in enumerate(["Low", "Middle", "High"]):
        for k, var in enumerate(["A", "N", "I"]):
            row = size.loc[size["size_group"].eq(group) & size["variable"].eq(var)]
            if row.empty: continue
            row = row.iloc[0]; yy = j + (k - 1) * 0.22; ax.plot([row["delta_ln_p25"], row["delta_ln_p75"]], [yy, yy], color=COL[var], lw=1.6); ax.plot(row["delta_ln_median"], yy, "o", color=COL[var], ms=3.2)
    ax.axvline(0, color="#777", lw=0.6); ax.set(yticks=[0, 1, 2], yticklabels=["Low N (31)", "Middle N (31)", "High N (31)"], ylim=(2.5, -0.5), xlabel="City-level log change", title="Initial facility scale"); ax.grid(axis="x", color=COL["grid"], lw=0.5); ax.legend(handles=[Line2D([0], [0], color=COL[v], marker="o", label=v, ms=3) for v in ["A", "N", "I"]], loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=6.3); panel_label(ax, "b")
    save_figure(fig, "figS_spatial_scale")
    return {"stem": "figS_spatial_scale", "core_conclusion": "The complete recent spatial and initial-scale support remains available in SI.", "source_tables": sources, "status": "complete"}



def figureS_governance() -> dict[str, object]:
    """Redraw saved governance timing estimates and paired facility context."""
    paired = v18("figS_governance_paired_v18.csv")
    leads = v18("figS_governance_leads_v18.csv")
    paired = paired.loc[paired["paired_support"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    paired["early_N_mean"] = num(paired["early_N_mean"])
    paired["late_N_mean"] = num(paired["late_N_mean"])
    leads["event_h"] = num(leads["event_h"])
    for col in ["estimate", "ci_low", "ci_high"]:
        leads[col] = num(leads[col])
    leads = leads.sort_values("event_h")
    if len(paired) != 13 or leads["event_h"].tolist() != [-4, -3, -2, -1]:
        raise ValueError("Saved governance support does not match the declared 13-city / four-lead display")
    paired = paired.sort_values("late_N_mean").reset_index(drop=True)
    y = np.arange(len(paired))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.65), gridspec_kw={"width_ratios": [1.35, 0.95]})
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.88, wspace=0.52)
    ax = axes[0]
    early = paired["early_N_mean"].to_numpy(float)
    late = paired["late_N_mean"].to_numpy(float)
    for yy, x0, x1 in zip(y, early, late):
        ax.plot([x0, x1], [yy, yy], color="#b8b8b8", lw=1.1, zorder=1)
    ax.scatter(early, y, s=24, color="#5a7897", zorder=3, label="Early window")
    ax.scatter(late, y, s=24, marker="s", color=COL["accent"], zorder=3, label="Late window")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(paired["city_label"].astype(str))
    ax.set_xlabel("Mean facilities $N$ (log scale)")
    ax.set_title("Paired facility context", loc="left")
    ax.grid(axis="x", color=COL["grid"], lw=0.55)
    ax.legend(loc="lower right", fontsize=7.5, handlelength=1.2)
    ax.set_xlim(max(0.7, float(np.nanmin(early)) * 0.72), float(np.nanmax(late)) * 1.35)
    panel_label(ax, "a")
    ax = axes[1]
    x = leads["estimate"].to_numpy(float)
    lo = leads["ci_low"].to_numpy(float)
    hi = leads["ci_high"].to_numpy(float)
    yy = np.arange(len(leads))
    ax.errorbar(x, yy, xerr=[x - lo, hi - x], fmt="o", ms=4.2, lw=1.05, capsize=2.4, color=COL["A"], ecolor=COL["A"], zorder=3)
    ax.axvline(0, color="#777", lw=0.7, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"h = {int(h)}" for h in leads["event_h"]])
    ax.set_xlabel("Facility lead contrast, log $N$")
    ax.set_title("Pre-policy facility leads (95% CI)", loc="left")
    ax.grid(axis="x", color=COL["grid"], lw=0.55)
    panel_label(ax, "b")
    save_figure(fig, "figS_governance")
    return {
        "stem": "figS_governance",
        "core_conclusion": "Saved paired facility context and pre-policy facility leads are shown without refitting.",
        "source_tables": [
            "figS_governance_paired_v20.csv",
            "figS_governance_leads_v20.csv",
        ],
        "status": "complete",
    }


TRAJECTORY_CITIES = [
    "东莞", "厦门", "宜春", "成都", "武汉", "海口", "温州",
    "湖州", "福州", "绍兴", "苏州",
]
TRAJECTORY_LABELS = {
    "东莞": "Dongguan", "厦门": "Xiamen", "宜春": "Yichun",
    "成都": "Chengdu", "武汉": "Wuhan", "海口": "Haikou",
    "温州": "Wenzhou", "湖州": "Huzhou", "福州": "Fuzhou",
    "绍兴": "Shaoxing", "苏州": "Suzhou",
}


def _read_trajectory_source() -> pd.DataFrame:
    source = SOURCE_DIR / "city_trajectories.csv"
    paths = read_csv(source)
    required = {"city_key", "event_h", "A", "N", "I", "time_coverage", "coded_zero_share", "event_state"}
    missing = required.difference(paths.columns)
    if missing:
        raise ValueError(f"Trajectory source is missing columns: {sorted(missing)}")
    if len(paths) != 160 or paths["city_key"].nunique() != 11:
        raise ValueError("Expected the saved 160-row, 11-city trajectory source")
    if set(paths["city_key"].astype(str)) != set(TRAJECTORY_CITIES):
        raise ValueError("Trajectory city membership differs from the saved C11 source")
    event_h = num(paths["event_h"])
    if event_h.isna().any() or not event_h.mod(1).eq(0).all() or set(event_h.astype(int)) != set(range(-8, 9)):
        raise ValueError("Event-quarter support must remain h = -8, ..., 8")
    if paths.duplicated(["city_key", "event_h"]).any():
        raise ValueError("Trajectory source contains duplicate city/event-quarter rows")
    return paths


def _render_trajectory_page(paths: pd.DataFrame, group: list[str], page: int) -> None:
    fig, axes = plt.subplots(len(group), 4, figsize=(12, 2.25 * len(group)), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.09, top=0.90, wspace=0.4, hspace=0.5)
    state_col = "G_policy_state" if "G_policy_state" in paths.columns else "event_state"
    y_colors = [COL["A"], COL["N"], COL["I"]]
    for row, city in enumerate(group):
        city_rows = paths.loc[paths["city_key"].eq(city)].copy()
        city_rows["_event_h_num"] = num(city_rows["event_h"])
        city_rows = city_rows.sort_values("_event_h_num")
        full = city_rows.set_index("_event_h_num").reindex(range(-8, 9))
        for col, key in enumerate(("A", "N", "I", "observation")):
            ax = axes[row, col]
            if col < 3:
                ax.plot(full.index, full[key], "-o", color=y_colors[col], lw=1, ms=2)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 4))
                ax.set_title(f"{TRAJECTORY_LABELS[city]} / {key}", fontsize=9)
            else:
                ax.plot(full.index, full["time_coverage"], "-o", color="#333333", lw=1, ms=2, label="Time coverage")
                ax.plot(full.index, full["coded_zero_share"], "-s", color=COL["accent"], lw=1, ms=2, label="Coded-zero share")
                ax.set_ylim(-0.05, 1.08)
                ax.set_title(f"{TRAJECTORY_LABELS[city]} / observation", fontsize=9)
                if row == 0:
                    ax.legend(fontsize=6, loc="best")
            transition = city_rows[state_col].astype(str).str.contains("transition", na=False)
            for event_h in city_rows.loc[transition, "_event_h_num"]:
                ax.axvspan(event_h - 0.5, event_h + 0.5, color=".8", alpha=0.5, lw=0)
            ax.axvline(0, color=".5", ls="--", lw=0.65)
            ax.set_xticks([-8, -4, 0, 4, 8])
            ax.grid(axis="y", alpha=0.2)
            if row == len(group) - 1:
                ax.set_xlabel("Event quarter h")
    fig.suptitle("City trajectories and observation coverage", weight="bold", fontsize=11, y=0.96)
    save_figure(fig, f"figS_city_paths_{page}")


def figureS_city_paths() -> list[dict[str, object]]:
    paths = _read_trajectory_source()
    contracts = []
    for page, start in enumerate(range(0, len(TRAJECTORY_CITIES), 4), 1):
        group = TRAJECTORY_CITIES[start : start + 4]
        _render_trajectory_page(paths, group, page)
        contracts.append({
            "stem": f"figS_city_paths_{page}",
            "core_conclusion": "Saved city trajectories retain A/N/I paths, observation coverage, transition shading and event-quarter markers.",
            "source_tables": ["city_trajectories.csv"],
            "status": "complete",
        })
    return contracts



def build_manifest(contracts: list[dict[str, object]], copied: dict[str, list[str]]) -> None:
    rows: list[dict[str, object]] = []
    for contract in contracts:
        stem = str(contract["stem"])
        srcs = ";".join(str(x) for x in contract.get("source_tables", []))
        for ext in ["pdf", "svg", "png"]:
            path = FIG_DIR / f"{stem}.{ext}"
            rows.append({"figure_id": stem, "filename": path.name, "role": "main" if not stem.startswith("figS") and stem != "growth_transitions_supplement" else "supplement", "format": ext.upper(), "status": contract.get("status", "unknown"), "source_data": srcs, "core_conclusion": contract.get("core_conclusion", "")})
    # Preserve the copied SI vector files that are not redrawn this turn.
    redrawn = {str(c["stem"]) for c in contracts}
    for name in sorted(set(copied["figures"])):
        stem = Path(name).stem
        if stem.endswith("_preview") or stem in redrawn or Path(name).suffix.lower() != ".pdf":
            continue
        rows.append({"figure_id": stem, "filename": name, "role": "supplement", "format": "PDF", "status": "copied_frozen_vector", "source_data": "retained V18/V19 source table", "core_conclusion": "Retained supplementary display"})
    pd.DataFrame(rows).drop_duplicates(["figure_id", "filename"]).to_csv(OUT / "figure_manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    copied = copy_frozen_assets()
    contracts: list[dict[str, object]] = []
    contracts.extend([
        figure1_coverage(),
        figure2_observation(),
        figure3_growth(),
        figure4_regimes(),
        figure5_contributions(),
    ])
    transition_main, transition_si = figure5_growth_transitions()
    contracts.extend([transition_main, transition_si])
    contracts.extend([
        figure6_recording(),
        figure7_quarterly_dynamic(),
        figure8_policy(),
        figureS_annual_models(),
        *figureS_city_paths(),
        figureS_governance(),
        figureS_recording_coverage(),
        figureS_spatial_scale(),
        figureS_window_decomposition(),
    ])
    build_manifest(contracts, copied)
    print(json.dumps({
        "output": str(OUT),
        "figures": [c["stem"] for c in contracts],
        "pending": [c["stem"] for c in contracts if str(c.get("status", "")).startswith("pending")],
        "model_refit": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


