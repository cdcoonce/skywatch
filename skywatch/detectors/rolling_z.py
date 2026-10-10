"""Baseline detector: trailing rolling mean/std z-score (the predecessor's method)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from skywatch.detectors import register
from skywatch.detectors.nights import split_nights
from skywatch.types import Detection, LightCurve


def rolling_z_scores(flux: np.ndarray, flux_err: np.ndarray, window: int) -> np.ndarray:
    """Signed z-score of each sample against the trailing `window` samples.

    The window for sample i is flux[i - window : i]; it excludes sample i. The std is
    floored at the sample's flux_err. Curves with n <= window produce all zeros, and
    the first `window` samples are always 0 (never flagged). A NaN anywhere in a
    window (or the sample itself) gives z = 0, so that sample is unflaggable.
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    z = np.zeros(n)
    w = window
    if n <= w:
        return z
    # windows[i] = flux[i : i + w]; the window for sample i + w excludes it
    windows = sliding_window_view(flux, w)[: n - w]
    mean = windows.mean(axis=1)
    std = np.maximum(windows.std(axis=1), np.asarray(flux_err, dtype=float)[w:])
    with np.errstate(divide="ignore", invalid="ignore"):
        zz = (flux[w:] - mean) / std
    z[w:] = np.where(std > 0, zz, 0.0)
    return z


@register
class RollingZDetector:
    """Flags runs where |z| >= z_threshold (see `rolling_z_scores` for edge cases)."""

    name = "rolling_z"

    def __init__(
        self,
        window: int = 30,
        z_threshold: float = 4.0,
        min_run: int = 1,
        night_gap_cadences: int = 100,
    ) -> None:
        self.window = window
        self.z_threshold = z_threshold
        self.min_run = min_run
        self.night_gap_cadences = night_gap_cadences

    def _z(self, curve: LightCurve) -> np.ndarray:
        return rolling_z_scores(curve.flux, curve.flux_err, self.window)

    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]:
        """Windows and runs never cross a night gap; each night warms up on its own."""
        out: list[Detection] = []
        for star_id in sorted(curves):
            curve = curves[star_id]
            k = 0
            for night in split_nights(curve.t, cadence_s, self.night_gap_cadences):
                t = curve.t[night]
                absz = np.abs(
                    rolling_z_scores(curve.flux[night], curve.flux_err[night], self.window)
                )
                flagged = absz >= self.z_threshold
                i = 0
                n = len(t)
                while i < n:
                    if not flagged[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 < n and flagged[j + 1]:
                        j += 1
                    if j - i + 1 >= self.min_run:
                        out.append(
                            Detection(
                                detector=self.name,
                                det_id=f"{self.name}-s{star_id}-{k}",
                                star_id=star_id,
                                t_start=int(t[i]),
                                t_end=int(t[j]) + cadence_s,
                                score=float(absz[i : j + 1].max()),
                            )
                        )
                        k += 1
                    i = j + 1
        return out
