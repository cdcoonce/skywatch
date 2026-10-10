"""Night splitting: the one place the gap rule lives.

A detector that windows by sample index must never let a window span the idle hours
between nights: the telescope re-points, so the level on one side of the gap says
nothing about the other.
"""

from __future__ import annotations

import itertools

import numpy as np


def split_nights(t: np.ndarray, cadence_s: int, gap_cadences: int = 100) -> list[slice]:
    """Contiguous index slices of `t`, one per night, in order.

    A new night starts wherever consecutive samples are more than
    ``gap_cadences * cadence_s`` seconds apart. A gap of exactly that size does not
    split (a within-night dropout is shorter than a night gap by a wide margin). The
    slices tile ``range(len(t))``; an empty `t` gives ``[]``.
    """
    n = len(t)
    if n == 0:
        return []
    breaks = np.flatnonzero(np.diff(t) > gap_cadences * cadence_s) + 1
    bounds = [0, *(int(b) for b in breaks), n]
    return [slice(lo, hi) for lo, hi in itertools.pairwise(bounds)]
