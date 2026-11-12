"""CLI for the public coded policy-exposure preparation route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preparation.policy_exposure import run  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive annual R/S preceding-year exposures and the later-date "
            "construction-support B lag from public coded metadata."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "results" / "recomputed_policy",
        help="Directory for derived annual exposure, document-date, and metadata outputs.",
    )
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="Compare derived fields with the read-only saved 325-row policy input.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run derivation and checks without writing output files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(None if args.no_write else args.output_dir, validate_reference=args.check_reference)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary.get("reference_check", {}).get("status") == "mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())
