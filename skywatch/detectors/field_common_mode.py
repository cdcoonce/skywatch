"""Field-wide detector: the common-mode flux trend across all stars, plus field dropouts.

This is the only detector that looks across stars, and it emits field-wide detections
(``star_id=None``). Clouds, tracking drift and dropouts are the glitches that hit every
star at once; a single light curve cannot tell a cloud from a transit, the ensemble can.

Algorithm (per night; nights are split where the union time grid has a gap larger than
``night_gap_cadences * cadence_s``):

1. Dropout: a gap between consecutive union-grid times larger than ``cadence_s`` means
   no star was observed, so it is a field-wide dropout spanning
   ``[t_prev + cadence_s, t_next)``; score = gap length in cadences. A dropout of one
   star leaves the union grid intact and is not this detector's job.
2. Common mode: each star is divided by a centered running median of its own flux,
   computed per night (``baseline_window`` samples, shrinking at the night edges; the
   window never crosses a night gap); the per-time median across
   stars is the common mode ``c(t)`` (NaN where fewer than ``min_stars`` contribute),
   smoothed with a NaN-aware centered moving average of ``smooth`` samples (so
   ``min_run=2`` is effectively inert at ``smooth=3``: any single-sample excursion
   that clears the threshold smears over three samples; ``min_run`` only binds when
   ``smooth=1``). Samples
   with ``|c - 1| >= threshold`` are merged into runs of at least ``min_run`` grid
   samples; span ``[t_first, t_last + cadence_s)``, score = max ``|c - 1| / threshold``.

Limitations, stated honestly:

* The running-median baseline is centered, so it partially absorbs a slow field-wide
  event. Long shallow clouds (3600 s at depth 0.05) are detected over roughly half
  their span. A cloud of an hour or more is attenuated in a 2 h window, and a held drift is
  absorbed progressively: the longer the departure persists, the more the baseline
  follows it, and near the end of a night the shrunken window can absorb it entirely.
* ``kind_guess`` is a heuristic on the shape of the run. A negative-dominated run that
  ends before the night's last grid sample is a "cloud"; a run that reaches the night's
  last grid sample (the departure persists to the end, as a held drift does) is
  "tracking_drift"; a positive excursion that ends earlier is also "tracking_drift"
  (upward drift); anything else gets None. A cloud that straddles the night end will
  therefore be guessed as drift.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from skywatch.detectors import register
from skywatch.detectors.nights import split_nights
from skywatch.types import Detection, LightCurve


def _running_median(flux: np.ndarray, window: int) -> np.ndarray:
    """Centered running median over `window` samples, shrinking at the edges."""
    n = len(flux)
    h = window // 2
    if n <= 2 * h:
        return np.full(n, np.median(flux))
    out = np.empty(n)
    out[h : n - h] = np.median(sliding_window_view(flux, 2 * h + 1), axis=1)
    for i in range(h):
        out[i] = np.median(flux[: i + h + 1])
        out[n - 1 - i] = np.median(flux[n - 1 - i - h :])
    return out


def _smooth(c: np.ndarray, width: int) -> np.ndarray:
    """NaN-aware centered moving average; NaN stays NaN."""
    if width <= 1:
        return c
    ok = ~np.isnan(c)
    kernel = np.ones(width)
    num = np.convolve(np.where(ok, c, 0.0), kernel, mode="same")
    den = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~ok] = np.nan
    return out


@register
class FieldCommonModeDetector:
    """Flags field-wide departures of the ensemble median flux from 1, and field dropouts."""

    name = "field_common_mode"

    def __init__(
        self,
        threshold: float = 0.02,
        baseline_window: int = 121,
        smooth: int = 3,
        min_run: int = 2,
        night_gap_cadences: int = 100,
        min_stars: int = 5,
    ) -> None:
        self.threshold = threshold
        self.baseline_window = baseline_window
        self.smooth = smooth
        self.min_run = min_run
        self.night_gap_cadences = night_gap_cadences
        self.min_stars = min_stars

    def _baseline(self, t: np.ndarray, flux: np.ndarray, cadence_s: int) -> np.ndarray:
        """Running median computed per night, so a window never spans a night gap."""
        base = np.empty(len(flux))
        for night in split_nights(t, cadence_s, self.night_gap_cadences):
            base[night] = _running_median(flux[night], self.baseline_window)
        return base

    def _common_mode(
        self, curves: Mapping[int, LightCurve], grid: np.ndarray, cadence_s: int
    ) -> np.ndarray:
        mat = np.full((len(curves), len(grid)), np.nan)
        for row, star_id in enumerate(sorted(curves)):
            c = curves[star_id]
            if len(c) == 0:
                continue
            flux = np.asarray(c.flux, dtype=float)
            base = self._baseline(np.asarray(c.t), flux, cadence_s)
            with np.errstate(invalid="ignore", divide="ignore"):
                norm = np.where(base != 0, flux / base, np.nan)
            mat[row, np.searchsorted(grid, c.t)] = norm
        counts = np.sum(~np.isnan(mat), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            cm = np.nanmedian(mat, axis=0)
        cm[counts < self.min_stars] = np.nan
        return cm

    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]:
        nonempty = [curves[s].t for s in sorted(curves) if len(curves[s]) > 0]
        if not nonempty:
            return []
        grid = np.unique(np.concatenate(nonempty))
        cm = self._common_mode(curves, grid, cadence_s)

        out: list[Detection] = []
        for night, sl in enumerate(split_nights(grid, cadence_s, self.night_gap_cadences)):
            lo, hi = sl.start, sl.stop
            tg = grid[sl]
            events: list[tuple[int, int, float, str | None]] = []

            for g in np.flatnonzero(np.diff(tg) > cadence_s):
                gap = int(tg[g + 1] - tg[g])
                events.append((int(tg[g]) + cadence_s, int(tg[g + 1]), gap / cadence_s, "dropout"))

            c = _smooth(cm[lo:hi], self.smooth)
            dev = c - 1.0
            with np.errstate(invalid="ignore"):
                flagged = np.abs(dev) >= self.threshold
            padded = np.concatenate(([False], flagged, [False])).astype(np.int8)
            edges = np.flatnonzero(np.diff(padded))
            for a, b in zip(edges[::2], edges[1::2], strict=True):  # run is [a, b)
                if b - a < self.min_run:
                    continue
                seg = dev[a:b]
                score = float(np.abs(seg).max() / self.threshold)
                if b == len(tg):
                    guess = "tracking_drift"
                elif np.mean(seg < 0) > 0.5:
                    guess = "cloud"
                elif np.mean(seg > 0) > 0.5:
                    guess = "tracking_drift"
                else:
                    guess = None
                events.append((int(tg[a]), int(tg[b - 1]) + cadence_s, score, guess))
            events.sort(key=lambda e: (e[0], e[1]))
            for k, (t0, t1, score, guess) in enumerate(events):
                out.append(
                    Detection(
                        detector=self.name,
                        det_id=f"{self.name}-n{night}-{k}",
                        star_id=None,
                        t_start=t0,
                        t_end=t1,
                        score=score,
                        kind_guess=guess,
                    )
                )
        return out
