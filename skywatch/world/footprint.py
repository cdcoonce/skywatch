"""Footprint: how many samples an injected event actually changed.

An event's truth label spans a time interval, but the scene only has samples on a grid.
An event that falls between two samples, or peaks after the night's last one, is labelled
yet changes nothing, so no detector could ever see it. ``footprint_samples`` records that
fact at injection time, measured from the scene itself (flux before against flux after,
mask before against mask after) rather than from span arithmetic, so it stays true whatever
the injector's own shape code does.

Only samples observed when the event was injected count: a change to a cell an earlier
injector already masked out is invisible. A later injector (dropouts run last) may still
hide samples this event changed; that is not knowable here and is not counted.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from skywatch.types import TruthLabel

KEY = "footprint_samples"


class Footprint:
    """Result holder filled in when the ``touched`` block exits."""

    samples: int = 0


@contextmanager
def touched(scene, row: int | None = None) -> Iterator[Footprint]:
    """Measure how many samples the block changes in ``scene``.

    ``row`` set: samples of that star (cells of its row). ``row`` None: samples of the
    shared grid at which any star changed (columns), for field-wide events. A sample counts
    if its flux changed or the event cleared it from the mask, and it was observed on entry.
    """
    rows = slice(None) if row is None else row
    flux0 = scene.flux[rows].copy()
    mask0 = scene.mask[rows].copy()
    out = Footprint()
    try:
        yield out
    finally:
        changed = ((scene.flux[rows] != flux0) | (mask0 & ~scene.mask[rows])) & mask0
        out.samples = int(np.count_nonzero(changed.any(axis=0) if row is None else changed))


def with_footprint(label: TruthLabel, samples: int) -> TruthLabel:
    """``label`` with ``params["footprint_samples"]`` set."""
    return replace(label, params={**label.params, KEY: int(samples)})
