"""Baseline detector: trailing rolling mean/std z-score (the predecessor's method)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from skywatch.detectors import register
from skywatch.detectors.nights import split_nights
from skywatch.types import Detection, LightCurve


def rolling_z_scores(
    flux: np.ndarray, flux_err: np.ndarray, window: int, warmup: int = 5
) -> np.ndarray:
    """Signed z-score of each sample against the trailing `window` samples.

    The window for sample i is flux[i - window : i]; it excludes sample i. The std is
    floored at the sample's flux_err. A NaN anywhere in a window (or the sample itself)
    gives z = 0, so that sample is unflaggable.

    Start of the curve: the first ``min(warmup, window)`` samples are always 0 (never
    flagged). For samples ``warmup <= i < window`` the window is the ``i`` samples seen so
    far (a growing window), and the z-score is multiplied by ``sqrt(i / window)`` so it is
    comparable to a full-window z against the same threshold: a short window has a noisier
    mean and std, and unscaled it fires on noise. ``warmup <= 0`` or ``warmup >= window``
    switches the growing window off: the first `window` samples are all 0, and curves with
    n <= window produce all zeros.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(flux_err, dtype=float)
    n = len(flux)
    z = np.zeros(n)
    w = window
    if warmup > 0:
        for i in range(warmup, min(w, n)):
            head = flux[:i]
            std = max(head.std(), err[i])
            with np.errstate(divide="ignore", invalid="ignore"):
                zi = (flux[i] - head.mean()) / std * np.sqrt(i / w)
            z[i] = zi if std > 0 and np.isfinite(zi) else 0.0
    if n <= w:
        return z
    # windows[i] = flux[i : i + w]; the window for sample i + w excludes it
    windows = sliding_window_view(flux, w)[: n - w]
    mean = windows.mean(axis=1)
    std = np.maximum(windows.std(axis=1), err[w:])
    with np.errstate(divide="ignore", invalid="ignore"):
        zz = (flux[w:] - mean) / std
    z[w:] = np.where(std > 0, zz, 0.0)
    return z


@register
class RollingZDetector:
    """Flags runs where |z| >= z_threshold (see `rolling_z_scores` for edge cases).

    The window is per night. The first `warmup` samples of every night are blind; from
    sample `warmup` to `window` the window grows with the samples seen so far (with its
    z-score scaled to match a full window), so an event early in a night can still fire.
    `warmup=0` or `warmup >= window` gives the older rule: the first `window` samples are
    blind and a night shorter than the window is entirely unflaggable.
    """

    name = "rolling_z"

    def __init__(
        self,
        window: int = 30,
        z_threshold: float = 4.0,
        min_run: int = 1,
        night_gap_cadences: int = 100,
        warmup: int = 5,
    ) -> None:
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        self.window = window
        self.z_threshold = z_threshold
        self.min_run = min_run
        self.night_gap_cadences = night_gap_cadences
        self.warmup = warmup

    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]:
        """Windows and runs never cross a night gap; each night warms up on its own."""
        out: list[Detection] = []
        for star_id in sorted(curves):
            curve = curves[star_id]
            k = 0
            for night in split_nights(curve.t, cadence_s, self.night_gap_cadences):
                t = curve.t[night]
                absz = np.abs(
                    rolling_z_scores(
                        curve.flux[night], curve.flux_err[night], self.window, self.warmup
                    )
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
