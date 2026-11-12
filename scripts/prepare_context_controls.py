"""CLI for the standalone V4 weather and V11 COVID preparation route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preparation.context_controls import (  # noqa: E402
    OXCGRT_COMMIT,
    EV_AIR_COLUMNS,
    NASA_PARAMETERS,
    build_context_controls,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the annual NASA POWER weather controls and OxCGRT-compatible "
            "COVID exposure controls from user-supplied cached raw inputs. "
            "No network download is performed."
        )
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        required=True,
        help=(
            "Dated city-point CSV. Required fields: city_key, latitude, longitude, "
            "coordinate_status. V4 marks coordinate_status=available within "
            "15<=latitude<=55 and 70<=longitude<=140 as weather-eligible."
        ),
    )
    parser.add_argument(
        "--weather-json-dir",
        type=Path,
        required=True,
        help=(
            "Directory of cached NASA POWER JSON responses named "
            "MERRA2_<latitude>_<longitude>_<start>_<end>.json. Each JSON must "
            f"contain properties.parameter {list(NASA_PARAMETERS)} with the V4 units. "
            "Source endpoint: https://power.larc.nasa.gov/api/temporal/daily/point."
        ),
    )
    parser.add_argument(
        "--ev-air",
        type=Path,
        required=True,
        help=(
            "Original EV-air weekly CSV. Required fields: "
            + ", ".join(EV_AIR_COLUMNS)
            + ". Compatibility is cumulative_plugs > 0; Total is not filtered."
        ),
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        required=True,
        help=(
            "City crosswalk CSV. Required fields: aq_city, city_key, city, province. "
            "The province mapping is explicit and is checked against OxCGRT names/codes."
        ),
    )
    parser.add_argument(
        "--oxcgrt",
        type=Path,
        required=True,
        help=(
            "OxCGRT compact subnational CSV at fixed commit "
            f"{OXCGRT_COMMIT}. Required fields include CountryCode, RegionCode, "
            "Jurisdiction, Date, StringencyIndex_Average, and C5M/C6M/C7M fields. "
            "The route uses CHN STATE_TOTAL rows dated 2020-01-01 through 2023-02-28."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "results" / "recomputed_context",
        help="Directory for weather, COVID, context, metadata, and QC outputs.",
    )
    parser.add_argument(
        "--weather-reference",
        type=Path,
        default=None,
        help=(
            "Optional weather_city_year_v4.csv reference. If omitted, the bundled "
            "data/processed/weather_city_year_v4.csv is checked when present."
        ),
    )
    parser.add_argument(
        "--covid-reference",
        type=Path,
        default=None,
        help=(
            "Optional COVID exposure or annual-panel reference. If omitted, the "
            "bundled data/processed/annual_panel_v12.csv is checked when present."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    weather_reference = args.weather_reference
    covid_reference = args.covid_reference
    if weather_reference is None:
        candidate = REPOSITORY_ROOT / "data" / "processed" / "weather_city_year_v4.csv"
        weather_reference = candidate if candidate.exists() else None
    if covid_reference is None:
        candidate = REPOSITORY_ROOT / "data" / "processed" / "annual_panel_v12.csv"
        covid_reference = candidate if candidate.exists() else None
    try:
        summary = build_context_controls(
            coordinates_path=args.coordinates,
            weather_json_dir=args.weather_json_dir,
            ev_air_path=args.ev_air,
            city_crosswalk_path=args.crosswalk,
            oxcgrt_path=args.oxcgrt,
            output_dir=args.output_dir,
            weather_reference=weather_reference,
            covid_reference=covid_reference,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

