"""Centered running-median residual detector with robust scale and kind guesses.

Unlike rolling_z (trailing window, usable online), this detector is offline: the
baseline for sample i is the median over a window centered on i, so it sees the
future. An event longer than about window/2 samples is absorbed into the baseline
and shows only at its edges. Windows work by sample index within a night; nights are
split where t has a gap larger than ``night_gap_cadences * cadence_s``, and a window
(and a run) never crosses that gap: the first and last samples of a night get the
same shrinking window as the ends of a whole curve.

NaN: a NaN sample poisons the median of every window that touches it, so the
baseline and sigma are NaN there and z = 0 (sigma > 0 fails): the detector is blind
to all samples within half a window of a NaN, even a large spike next to it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from skywatch.detectors import register
from skywatch.detectors.nights import split_nights
from skywatch.types import Detection, LightCurve


def _running_median(x: np.ndarray, window: int) -> np.ndarray:
    """Centered running median; the window shrinks at the ends (no padding)."""
    n = len(x)
    half = window // 2
    out = np.empty(n)
    if n <= window:
        for i in range(n):
            out[i] = np.median(x[max(0, i - half) : i + half + 1])
        return out
    out[half : n - half] = np.median(sliding_window_view(x, window), axis=1)
    for i in range(half):
        out[i] = np.median(x[: i + half + 1])
        j = n - 1 - i
        out[j] = np.median(x[j - half :])
    return out


def median_residual_scores(flux: np.ndarray, flux_err: np.ndarray, window: int) -> np.ndarray:
    """Signed robust z of each sample against a centered running median.

    sigma = 1.4826 * running median of |residual|, floored at flux_err; z = 0 where
    sigma is 0.
    """
    flux = np.asarray(flux, dtype=float)
    if len(flux) == 0:
        return np.zeros(0)
    residual = flux - _running_median(flux, window)
    sigma = np.maximum(
        1.4826 * _running_median(np.abs(residual), window), np.asarray(flux_err, dtype=float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = residual / sigma
    return np.where(sigma > 0, z, 0.0)


@register
class MedianResidualDetector:
    """Flags runs where |z| >= threshold (see `median_residual_scores`).

    A NaN sample blinds every window touching it (z = 0 within half a window). The
    per-night median shrinks at night ends, so an event there is partly absorbed.
    """

    name = "median_residual"

    def __init__(
        self,
        window: int = 61,
        threshold: float = 4.0,
        min_run: int = 1,
        night_gap_cadences: int = 100,
    ) -> None:
        if window % 2 == 0:
            raise ValueError(f"window must be odd, got {window}")
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if not threshold > 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if min_run < 1:
            raise ValueError(f"min_run must be >= 1, got {min_run}")
        self.window = window
        self.threshold = threshold
        self.min_run = min_run
        self.night_gap_cadences = night_gap_cadences

    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]:
        out: list[Detection] = []
        for star_id in sorted(curves):
            curve = curves[star_id]
            k = 0
            for night in split_nights(curve.t, cadence_s, self.night_gap_cadences):
                t = curve.t[night]
                z = median_residual_scores(curve.flux[night], curve.flux_err[night], self.window)
                flagged = np.abs(z) >= self.threshold
                n = len(t)
                i = 0
                while i < n:
                    if not flagged[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 < n and flagged[j + 1]:
                        j += 1
                    length = j - i + 1
                    if length >= self.min_run:
                        run = z[i : j + 1]
                        peak = float(np.abs(run).max())
                        if run.max() >= self.threshold and run.mean() > 0:
                            guess = "cosmic_ray" if length == 1 else "flare"
                        else:
                            guess = None
                        out.append(
                            Detection(
                                detector=self.name,
                                det_id=f"{self.name}-s{star_id}-{k}",
                                star_id=star_id,
                                t_start=int(t[i]),
                                t_end=int(t[j]) + cadence_s,
                                score=peak,
                                kind_guess=guess,
                            )
                        )
                        k += 1
                    i = j + 1
        return out
