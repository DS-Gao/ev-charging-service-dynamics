"""Redraw the saved figure set without fitting any statistical model.

The default route consumes committed numeric figure-source tables and the
bundled source geometry.  ``--compile`` is an explicit optional handoff to the
standard manuscript compiler; it does not trigger an estimator or data
preparation stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redraw saved figures without fitting models; optionally compile the manuscript."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="After figure rendering, run manuscript/compile_manuscript.py.",
    )
    return parser


def _run(relative: str) -> None:
    command = [sys.executable, relative]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY, check=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _run("scripts/quick_figures.py")
    if args.compile:
        _run("manuscript/compile_manuscript.py")
    print("saved-result route complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
