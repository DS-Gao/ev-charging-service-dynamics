"""Render saved figures with released-field labels and annotations.

This entrypoint imports the repository's saved-value renderer and applies only
visible labels, clipping notes, endpoint notation, and policy-construct
annotations. It never refits a model, recalculates source values, or runs a
bootstrap. The default output is Git-ignored
``data/results/recomputed_figures_labels``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Iterable
from render_figure_corrections import _revise_fig4_correction, _revise_fig8_correction


DISPLAY_BOUND = 3.503090


def _locate_repo(explicit: str | None) -> tuple[Path, bool]:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not (candidate / "src" / "figures" / "render_all.py").exists():
            raise FileNotFoundError(candidate / "src" / "figures" / "render_all.py")
        return candidate, False
    env_repo = os.environ.get("EV_CHARGING_PUBLIC_REPO")
    if env_repo:
        candidate = Path(env_repo).expanduser().resolve()
        if (candidate / "src" / "figures" / "render_all.py").exists():
            return candidate, False
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "src" / "figures" / "render_all.py").exists():
        return candidate, False
    raise FileNotFoundError("Could not locate the public package root")


def _default_output(from_candidate: bool, repo: Path) -> Path:
    explicit = os.environ.get("EV_CHARGING_FIGURE_OUTPUT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return repo / "data" / "results" / "recomputed_figures_labels"


def _load_renderer(repo: Path):
    source = repo / "src" / "figures" / "render_all.py"
    spec = importlib.util.spec_from_file_location("ev_charging_public_renderer", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import renderer from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numeric_text_replacements() -> dict[str, str]:
    """Visible-label replacements that make the released scale explicit."""

    return {
        "Facilities observed": "Released facility indicator observed",
        "Positive activity": "Positive released activity",
        "Positive Activity": "Positive Released Activity",
        "Facilities": "Released facility indicators",
        "Mean facilities N": "Mean released facility indicator (N)",
        "Positive activity A": "Positive released activity indicator (A)",
        "Positive intensity I": "Positive released intensity indicator (I)",
        "A: activity": "A: released activity indicator",
        "N: facilities": "N: released facility indicator",
        "I: intensity": "I: released intensity indicator",
        "Aggregate I": "Aggregate released intensity (I)",
        "Median city I": "Median city released intensity (I)",
        "Δ ln N": "Δ ln released facility indicator (N)",
        "Δ ln A": "Δ ln released activity indicator (A)",
        "Δ ln I": "Δ ln released intensity indicator (I)",
        "Recent intensity change across all 93 cities": "Recent released-intensity change across all 93 cities",
        "Recent Δ ln I (93 cities)": "Recent Δ ln released intensity indicator (I; 93 cities)",
        "Facility coefficient / elasticity": "Released-facility coefficient / elasticity",
        "Mean facilities $N$ (log scale)": "Mean released facility indicator $N$ (log scale)",
        "Facility lead contrast, log $N$": "Released-facility lead contrast, log $N$",
        "ln N": "ln released facility indicator (N)",
        "ln A": "ln released activity indicator (A)",
        "ln I": "ln released intensity indicator (I)",
        "OLS: ln A": "OLS: ln released activity (A)",
        "N expands; A lags": "Released N expands; released A lags",
        "N expands; A faster": "Released N expands; released A faster",
        "N expands; A falls": "Released N expands; released A falls",
        "Flat component": "Flat released component",
    }


def _replace_visible_text(fig, replacements: dict[str, str]) -> None:
    from matplotlib.text import Text

    for artist in fig.findobj(Text):
        value = artist.get_text()
        if value in replacements:
            artist.set_text(replacements[value])
            continue
        revised = value
        for old, new in replacements.items():
            if old in revised:
                revised = revised.replace(old, new)
        if revised != value:
            artist.set_text(revised)


def _revise_fig1_labels(fig) -> None:
    """Make status labels explicit on the evidence-coverage panel."""

    if len(fig.axes) < 3:
        return
    evidence = fig.axes[2]
    tick_map = {
        "Facilities": "Released\nfacility indicator",
        "Positive activity": "Released\nactivity indicator",
    }
    evidence.set_yticklabels([tick_map.get(tick.get_text(), tick.get_text()) for tick in evidence.get_yticklabels()])
    for axis in fig.axes:
        legend = axis.get_legend()
        if legend is None:
            continue
        for text in legend.get_texts():
            if text.get_text().startswith("Positive released activity"):
                suffix = text.get_text()[len("Positive released activity") :]
                text.set_text("Released activity indicator" + suffix)
            elif text.get_text().startswith("Released facility indicator"):
                text.set_text("Released facility indicator")


def _revise_fig2_labels(fig) -> None:
    """Keep observation-state legend wording on the released scale."""

    for axis in fig.axes:
        legend = axis.get_legend()
        if legend is None:
            continue
        for text in legend.get_texts():
            if text.get_text() == "Positive Released Activity":
                text.set_text("Positive released activity indicator")


def _revise_fig3_labels(fig) -> None:
    """Keep compact released-scale labels without expanding the page bbox."""

    if len(fig.axes) < 2:
        return
    legend = fig.axes[1].get_legend()
    if legend is not None:
        # The panel-b legend is intentionally vertical and internal.  The
        # panel-a legend carries the full released-indicator definitions.
        compact = {"A": "A", "N": "N", "I": "I"}
        for text in legend.get_texts():
            if text.get_text() in compact:
                text.set_text(compact[text.get_text()])
            text.set_fontsize(5.8)
        legend.set_bbox_to_anchor((0.98, 0.88), transform=fig.axes[1].transAxes)
        legend._loc = 1  # upper right, inside panel b
        legend.set_ncols(1)
        legend.set_frame_on(False)
    if len(fig.axes) >= 3:
        distribution_legend = fig.axes[2].get_legend()
        if distribution_legend is not None:
            # Panel a already provides the color-coded A/N/I definitions.  The
            # compact footer below repeats their released-indicator scope, so
            # removing this redundant panel-c legend avoids the narrow c/d gap.
            distribution_legend.remove()
    fig.subplots_adjust(bottom=0.17)
    fig.text(
        0.50,
        0.022,
        "A/N/I: released activity, released facility, and released intensity indicators.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#606060",
    )


def _add_fig4_explanation(fig, renderer) -> None:
    """Explain clipped-map direction triangles without changing map values."""

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # The display bound is the saved-value 95th percentile used by baseline.  Keep
    # the rounded publication value explicit, but assert it against the source
    # table so a future source replacement cannot silently change the legend.
    import numpy as np
    import pandas as pd

    map_source = renderer.SOURCE_DIR / "fig4_spatial_recent_93_v20.csv"
    source_bound = pd.to_numeric(pd.read_csv(map_source)["dI_delta_lnI"], errors="coerce").abs().quantile(0.95)
    if not np.isclose(source_bound, DISPLAY_BOUND, atol=1e-6, rtol=0):
        raise ValueError(
            f"Figure 4 display bound {DISPLAY_BOUND} does not match source 95th percentile {source_bound}"
        )

    map_axes = [
        ax
        for ax in fig.axes
        if ax.get_title(loc="left").lower().startswith("recent") and not ax.get_xaxis().get_visible()
    ]
    if not map_axes:
        # The map is the wide lower panel; this fallback is only for older
        # matplotlib/inset-axes ordering where axis visibility is unavailable.
        map_axes = [ax for ax in fig.axes if "released-intensity" in ax.get_title(loc="left").lower()]
    if not map_axes:
        return
    ax = map_axes[0]
    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()
    handles = [
        Patch(facecolor="#d7d7d7", edgecolor="none", label="Outside comparison"),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label=f"Positive beyond ±{DISPLAY_BOUND:.6f}",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label=f"Negative beyond ±{DISPLAY_BOUND:.6f}",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        fontsize=6.4,
        frameon=False,
        handlelength=1.1,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    # The colorbar label is the most stable place to retain the exact clipping
    # rule when the figure is viewed without its manuscript caption.
    for artist in fig.findobj():
        if hasattr(artist, "get_text") and artist.get_text() in {
            "Δ ln released intensity indicator (I)",
            "Δ ln released intensity (I)",
        }:
            artist.set_text(f"Δ ln released intensity indicator (I); color clipped at ±{DISPLAY_BOUND:.6f}")


def _revise_fig5_contributions(fig) -> None:
    axes = fig.axes
    if len(axes) < 3:
        return
    panel = axes[2]
    panel.set_ylabel("Change in index points\n(each window start = 100)")
    panel.set_title("Exact decomposition\n(each window start = 100)")


def _revise_fig6_recording(fig) -> None:
    axes = fig.axes
    if len(axes) < 2:
        return
    labels = [
        "Δ ln p\npositive\nweek\nshare",
        "Δ ln q+\npositive-week\nmean\nactivity",
        "Δ ln N\nreleased\nfacility\nindicator",
        "Δ ln I\nreleased\nintensity\nindicator",
    ]
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Component distributions (P25–P75 city distribution)")
    axes[1].set_xticklabels(
        [
            "Δ ln p\npositive\nweek\nshare",
            "Δ ln q+\npositive-week\nmean\nactivity",
            "−Δ ln N\nreleased\nfacility\nindicator",
            "Δ ln I\nreleased\nintensity\nindicator",
        ]
    )
    axes[0].tick_params(axis="x", labelsize=5.6, pad=1)
    axes[1].tick_params(axis="x", labelsize=5.6, pad=1)
    for axis in axes:
        for label in axis.get_xticklabels():
            label.set_rotation(0)
            label.set_ha("center")
    fig.subplots_adjust(bottom=0.44)
    fig.text(
        0.50,
        0.055,
        "p: share of compatible weeks with positive released activity indicator; q+: mean released activity indicator in those weeks.\n"
        "Boxes show P25–P75 city distributions, not confidence intervals.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#606060",
    )


def _revise_fig7_quarterly(fig) -> None:
    axes = fig.axes
    if len(axes) < 2:
        return
    axes[0].set_ylabel("Conditional path coefficient, c_h^(L)")
    axes[1].set_xticklabels(["c_3^(3) − c_0^(3)"])
    axes[1].set_ylabel("Within-D3 contrast, c_3^(3) − c_0^(3)")
    axes[1].set_title("D3 within-model lag contrast")
    for artist in fig.texts:
        if artist.get_text().startswith("n="):
            artist.set_text(
                artist.get_text()
                + "\n"
                + "c_h^(L) = sum(j=0..h) beta_j^(L), 0 ≤ h ≤ L; D3: c_3^(3) − c_0^(3)"
            )
            artist.set_fontsize(6.8)
            artist.set_y(1.01)
            artist.set_va("bottom")


def _revise_fig8_policy(fig) -> None:
    axes = fig.axes
    if not axes:
        return
    axes[0].set_title("Operational content functions (n=88)")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.36,
        0.008,
        "Content category: operation and maintenance. Policy-estimation S: operating-service responsibility\n"
        "(broader construct; not the narrow content category above).",
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#606060",
    )


def _revision_hook(renderer):
    replacements = _numeric_text_replacements()
    original_save = renderer.save_figure

    def save_figure(fig, stem: str) -> None:
        if stem not in {"fig4_regimes", "fig7_quarterly_dynamic"}:
            _replace_visible_text(fig, replacements)
        if stem == "fig1_coverage":
            _revise_fig1_labels(fig)
        elif stem == "fig2_observation":
            _revise_fig2_labels(fig)
        elif stem == "fig3_growth":
            _revise_fig3_labels(fig)
        elif stem == "fig4_regimes":
            _revise_fig4_correction(fig, renderer)
        elif stem == "fig5_contributions":
            _revise_fig5_contributions(fig)
        elif stem == "fig6_recording":
            _revise_fig6_recording(fig)
        elif stem == "fig7_quarterly_dynamic":
            _revise_fig8_correction(fig)
        elif stem == "fig8_policy":
            _revise_fig8_policy(fig)
        original_save(fig, stem)

    renderer.save_figure = save_figure


def _configure_renderer(renderer, source_repo: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    renderer.SOURCE_DIR = source_repo / "data" / "figure_source"
    renderer.GEOMETRY_SOURCE = source_repo / "data" / "processed" / "city_geometry.geojson"
    renderer.OUT = output
    renderer.FIG_DIR = output
    renderer.SRC_DIR = output / "derived_sources"
    renderer.SRC_DIR.mkdir(parents=True, exist_ok=True)
    renderer._GEO_CACHE = None
    _revision_hook(renderer)


def render(repo: Path, output: Path) -> list[dict[str, object]]:
    renderer = _load_renderer(repo)
    _configure_renderer(renderer, repo, output)
    contracts: list[dict[str, object]] = []
    # Main figures with released-scale labels or N1/N2 policy/notation changes.
    for function in (
        renderer.figure1_coverage,
        renderer.figure2_observation,
        renderer.figure3_growth,
        renderer.figure4_regimes,
        renderer.figure5_contributions,
        renderer.figure6_recording,
        renderer.figure7_quarterly_dynamic,
        renderer.figure8_policy,
        renderer.figureS_annual_models,
        renderer.figureS_spatial_scale,
        renderer.figureS_governance,
    ):
        contracts.append(function())
    manifest = {
        "renderer": "src/figures/render_labels.py",
        "source_directory": "data/figure_source",
        "output_directory": "data/results/recomputed_figures_labels",
        "model_refit": False,
        "bootstrap": False,
        "source_values_changed": False,
        "figures": [
            {
                "stem": c.get("stem"),
                "status": c.get("status"),
                "source_tables": c.get("source_tables", []),
            }
            for c in contracts
        ],
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return contracts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="public-package root containing src/figures/render_all.py")
    parser.add_argument("--output", help="directory for released-scale PDF/SVG/PNG outputs")
    args = parser.parse_args()
    repo, from_candidate = _locate_repo(args.repo)
    output = Path(args.output).expanduser().resolve() if args.output else _default_output(from_candidate, repo)
    contracts = render(repo, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "figures": [c.get("stem") for c in contracts],
                "model_refit": False,
                "source_values_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
