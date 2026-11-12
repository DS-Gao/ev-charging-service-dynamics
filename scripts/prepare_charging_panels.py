"""CLI for the public raw charging panel preparation route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preparation.charging_panels import build_panels  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build raw-only annual and quarterly Q/E/W/A/N/I charging panels "
            "from Zenodo ev_air.csv and the supplied city crosswalk."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to the downloaded Zenodo version-2 Data/ev_air.csv. Required "
            "fields: city, date, Total, cumulative_plugs, year_only, month_only, week_only."
        ),
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        required=True,
        help="Path to the supplied 334-city crosswalk with aq_city and city_key fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed",
        help="Directory for raw-derived panels and QC artifacts (default: data/processed).",
    )
    parser.add_argument(
        "--annual-reference",
        type=Path,
        default=None,
        help=(
            "Optional bundled/reference annual panel to compare on city_key/year and "
            "Q, E, W, A, N, I. If omitted, data/processed/annual_panel_v12.csv "
            "is checked when present."
        ),
    )
    parser.add_argument(
        "--quarterly-reference",
        type=Path,
        default=None,
        help=(
            "Optional bundled/reference quarterly panel to compare on city_key/quarter "
            "and Q, E, W, A, N, I. If omitted, data/processed/quarterly_panel.csv "
            "is checked when present."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    annual_reference = args.annual_reference
    quarterly_reference = args.quarterly_reference
    if annual_reference is None:
        candidate = REPOSITORY_ROOT / "data" / "processed" / "annual_panel_v12.csv"
        annual_reference = candidate if candidate.exists() else None
    if quarterly_reference is None:
        candidate = REPOSITORY_ROOT / "data" / "processed" / "quarterly_panel.csv"
        quarterly_reference = candidate if candidate.exists() else None
    try:
        summary = build_panels(
            input_path=args.input,
            crosswalk_path=args.crosswalk,
            output_dir=args.output_dir,
            annual_reference=annual_reference,
            quarterly_reference=quarterly_reference,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

