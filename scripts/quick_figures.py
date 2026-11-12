"""Render every main and supplementary figure from committed saved values.

This entry point deliberately imports only src/figures/render_all.py. It
does not call model, bootstrap, descriptive, or input-preparation stages.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FIGURE_MODULE = REPO / "src" / "figures"
if str(FIGURE_MODULE) not in sys.path:
    sys.path.insert(0, str(FIGURE_MODULE))

from render_all import main  # noqa: E402


if __name__ == "__main__":
    main()
