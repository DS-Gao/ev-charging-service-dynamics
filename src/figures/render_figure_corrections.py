"""Public saved-value corrections for Figures 4 and 8.

This module changes visible layout and mathematical labels only. It does not
fit models, bootstrap, or alter the committed figure-source values.
"""

from __future__ import annotations

DISPLAY_BOUND = 3.503090
DISPLAY_BOUND_LABEL = "≈ ±3.50"


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


def _revise_fig4_correction(fig, renderer) -> None:
    """Shorten Figure 4 labels and move regime keys outside the point clouds.

    The exact source threshold remains ``DISPLAY_BOUND`` for both the color
    normalization and the out-of-bound triangle test in the inherited public
    renderer.  Only its visible publication label is rounded to approximately
    plus/minus 3.50.
    """

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd

    map_source = renderer.SOURCE_DIR / "fig4_spatial_recent_93_v20.csv"
    source_bound = pd.to_numeric(pd.read_csv(map_source)["dI_delta_lnI"], errors="coerce").abs().quantile(0.95)
    if not np.isclose(source_bound, DISPLAY_BOUND, atol=1e-6, rtol=0):
        raise ValueError(
            f"Figure 4 display bound {DISPLAY_BOUND} does not match source 95th percentile {source_bound}"
        )

    # The inherited renderer creates one legend on each scatter axis.  Keep
    # their handles/labels, remove both in-panel legends, and add a single
    # figure-level key in the whitespace between the scatter row and map.
    scatter_axes = []
    regime_handles = []
    regime_labels = []
    for axis in fig.axes:
        title = axis.get_title()
        if title.startswith(("Fixed cohort", "Recent window")):
            scatter_axes.append(axis)
            old_legend = axis.get_legend()
            if old_legend is not None:
                for handle, label in zip(old_legend.legend_handles, [t.get_text() for t in old_legend.get_texts()]):
                    if label not in regime_labels:
                        regime_handles.append(handle)
                        regime_labels.append(label)
                old_legend.remove()

    if regime_handles:
        fig.legend(
            regime_handles,
            regime_labels,
            loc="upper center",
            bbox_to_anchor=(0.50, 0.575),
            ncol=min(4, len(regime_labels)),
            fontsize=6.4,
            frameon=False,
            handlelength=1.0,
            columnspacing=0.9,
            handletextpad=0.35,
            borderaxespad=0.0,
        )
        # Give the shared key a clean band below the top-row x labels and
        # above panel c without changing the map or point coordinates.
        fig.subplots_adjust(hspace=0.72)

    # Keep the map key separate from the regime key and use a publication-
    # scale threshold.  The triangles themselves are still drawn by the
    # inherited Figure 4 function using the exact DISPLAY_BOUND value.
    map_axes = [
        ax
        for ax in fig.axes
        if ax.get_title(loc="left").lower().startswith("recent") and not ax.get_xaxis().get_visible()
    ]
    if not map_axes:
        map_axes = [ax for ax in fig.axes if "intensity change" in ax.get_title(loc="left").lower()]
    if not map_axes:
        raise ValueError("Figure 4 map axis was not found")
    map_axis = map_axes[0]
    old_map_legend = map_axis.get_legend()
    if old_map_legend is not None:
        old_map_legend.remove()
    map_axis.legend(
        handles=[
            Patch(facecolor="#d7d7d7", edgecolor="none", label="Outside comparison"),
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=5,
                label="Above ≈ +3.50",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=5,
                label="Below ≈ −3.50",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        fontsize=6.4,
        frameon=False,
        handlelength=1.1,
        labelspacing=0.25,
        borderaxespad=0.0,
    )

    # Restore concise mathematical variable labels after the inherited
    # renderer's released-scale wording, and round only the visible threshold.
    replacements = {
        "Δ ln released facility indicator (N)": "Δ ln N",
        "Δ ln released activity indicator (A)": "Δ ln A",
        "Recent intensity change across all 93 cities": "Recent intensity change (93 cities)",
        "Recent released-intensity change across all 93 cities": "Recent intensity change (93 cities)",
        "Δ ln I": f"Δ ln I; color clipped at {DISPLAY_BOUND_LABEL}",
        "Δ ln released intensity indicator (I); color clipped at ±3.503090": f"Δ ln I; color clipped at {DISPLAY_BOUND_LABEL}",
        "Δ ln released intensity indicator (I); color clipped at ±3.50": f"Δ ln I; color clipped at {DISPLAY_BOUND_LABEL}",
    }
    _replace_visible_text(fig, replacements)


def _revise_fig8_correction(fig) -> None:
    """Use publication mathtext and a reader-facing within-model contrast."""

    axes = fig.axes
    if len(axes) < 2:
        raise ValueError("Figure 8 quarterly dynamic renderer returned fewer than two axes")

    axes[0].set_ylabel(r"Conditional path coefficient, $c_h^{(L)}$")
    axes[1].set_xticklabels([r"$c_3^{(3)} - c_0^{(3)}$"])
    axes[1].set_ylabel(r"Within-model contrast, $c_3^{(3)} - c_0^{(3)}$")
    axes[1].set_title("Within-model cumulative contrast")

    # Preserve the sample size, city-cluster count, confidence-interval
    # meaning, and quarter interpretation in the existing annotation while
    # replacing code-like notation with rendered math.
    formula = (
        r"$c_h^{(L)}=\sum_{j=0}^{h}\beta_j^{(L)}$ ($0\leq h\leq L$); "
        r"within-model comparison: $c_3^{(3)}-c_0^{(3)}$"
    )
    for artist in fig.texts:
        if artist.get_text().startswith("n="):
            sample_note = artist.get_text().split("\n", 1)[0]
            artist.set_text(f"{sample_note}\n{formula}")
            artist.set_fontsize(6.8)
            artist.set_y(1.01)
            artist.set_va("bottom")
