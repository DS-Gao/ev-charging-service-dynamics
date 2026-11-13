"""Render saved-value figures with an optional released-scale label pass.

The default route imports ``src/figures/render_all.py`` only. The optional
``--released-scale-labels`` route applies visible labels and explanatory
annotations without fitting models, bootstrapping, or changing source values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FIGURE_MODULE = REPO / "src" / "figures"
if str(FIGURE_MODULE) not in sys.path:
    sys.path.insert(0, str(FIGURE_MODULE))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--released-scale-labels",
        action="store_true",
        help="Apply released-field labels and explanatory figure annotations.",
    )
    args = parser.parse_args()
    if args.released_scale_labels:
        sys.argv = [sys.argv[0]]
        from render_labels import main as render_main
    else:
        from render_all import main as render_main
    render_main()


if __name__ == "__main__":
    main()
