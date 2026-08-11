"""
HIPO conformation model — published figure suite
================================================

Every figure in the manuscript and the SI, one function per figure, all drawn
from `hipo.core` so the joints, the profiles and the marginals cannot disagree.

    cd src && PYTHONPATH=. python -m hipo.plots

The figure bodies live in `hipo/_figblock.py`, which is inserted verbatim into
`hipo_model_standalone.py` as well.  `tests/test_figure_parity.py` asserts the
two copies are identical, so a fix applied here cannot silently miss the
standalone file (or vice versa).

See FIGURE_MAP.md for which published figure each function produces.
"""
import numpy as np
from hipo.core import *          # noqa: F401,F403  — model API, incl. PHI_GRID, DPHI, LABEL
from hipo import core as H

from hipo._figblock import *     # noqa: F401,F403  — the figure functions
from hipo._figblock import run_figures

MODEL = H.MODEL

if __name__ == "__main__":
    print("Figures:")
    run_figures(MODEL, outdir=str(H.Path(__file__).resolve().parents[2] / "figures"))
