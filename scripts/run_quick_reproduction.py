"""Replay the independent estimator/descriptive models and saved figures.

This is the model-replay route.  Use ``scripts/reproduce_saved_results.py``
when only saved-result figure regeneration is wanted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
COMMANDS = [
    ["src/models/fit_annual_models.py"],
    ["src/models/fit_annual_zero.py"],
    ["src/models/fit_policy_association.py"],
    ["src/descriptive/windows.py"],
    ["scripts/quick_figures.py"],
]


def main() -> None:
    for relative in COMMANDS:
        command = [sys.executable, *relative]
        print("$", " ".join(command))
        subprocess.run(command, cwd=REPOSITORY, check=True)
    print("quick reproduction complete")


if __name__ == "__main__":
    main()
