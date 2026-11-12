"""Redraw a representative growth figure from the shipped figure-source CSV.

No model is fitted here.  The CSV is the authoritative numeric source for the
delivered figure; this script only applies labels and a reproducible layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve()
REPOSITORY = HERE.parents[2]
SOURCE = REPOSITORY / "data" / "figure_source" / "fig3_window_results_v20.csv"
OUTPUT = REPOSITORY / "data" / "results" / "recomputed_figures"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> Path:
    frame = pd.read_csv(SOURCE)
    required = {"window_id", "A_aggregate_percent_change", "N_aggregate_percent_change", "I_aggregate_percent_change"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"fig3_window_results_v20.csv missing required columns: {missing}")
    frame = frame.copy()
    labels = frame["window_id"].str.replace("adjacent_", "", regex=False).str.replace("continuous_", "", regex=False)
    frame["display_window"] = labels
    long = frame.melt(
        id_vars=["window_id", "display_window"],
        value_vars=["A_aggregate_percent_change", "N_aggregate_percent_change", "I_aggregate_percent_change"],
        var_name="metric",
        value_name="percent_change",
    )
    long["metric"] = long["metric"].map(
        {
            "A_aggregate_percent_change": "Activity per compatible week (A)",
            "N_aggregate_percent_change": "Facilities per compatible week (N)",
            "I_aggregate_percent_change": "Activity intensity (I)",
        }
    )
    long.to_csv(OUTPUT / "fig3_growth_values.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.0, 4.7), constrained_layout=True)
    for metric, group in long.groupby("metric", sort=False):
        ax.plot(group["display_window"], group["percent_change"], marker="o", linewidth=1.8, label=metric)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Aggregate change (%)")
    ax.set_xlabel("Predeclared comparison window")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Aggregate service-conversion changes")
    fig.savefig(OUTPUT / "fig3_growth_recomputed.png", dpi=180)
    plt.close(fig)
    print(f"wrote {OUTPUT / 'fig3_growth_recomputed.png'}")
    return OUTPUT / "fig3_growth_recomputed.png"


if __name__ == "__main__":
    main()
