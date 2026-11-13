"""Redraw the saved figure set without fitting any statistical model.

The route consumes committed numeric figure-source tables and the bundled
source geometry.  It never compiles a manuscript or runs data preparation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def _run(relative: str) -> None:
    command = [sys.executable, relative]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY, check=True)


def main(argv: list[str] | None = None) -> int:
    supplied = sys.argv[1:] if argv is None else argv
    if supplied == ["--compile"]:
        print(
            "Manuscript compilation is not part of the public package; "
            "running the saved-figure route only. Use the separate local "
            "submission delivery for compilation.",
            flush=True,
        )
    elif supplied:
        raise SystemExit(f"unrecognized option(s): {' '.join(supplied)}")
    _run("scripts/quick_figures.py")
    print("saved-result route complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
