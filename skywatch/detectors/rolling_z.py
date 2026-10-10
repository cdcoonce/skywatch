"""Baseline detector: trailing rolling mean/std z-score (the predecessor's method)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from skywatch.detectors import register
from skywatch.types import Detection, LightCurve


@register
class RollingZDetector:
    name = "rolling_z"

    def __init__(self, window: int = 30, z_threshold: float = 4.0, min_run: int = 1) -> None:
        self.window = window
        self.z_threshold = z_threshold
        self.min_run = min_run

    def _z(self, curve: LightCurve) -> np.ndarray:
        n = len(curve)
        z = np.zeros(n)
        w = self.window
        if n <= w:
            return z
        flux = np.asarray(curve.flux, dtype=float)
        # windows[i] = flux[i : i + w]; the window for sample i + w excludes it
        windows = sliding_window_view(flux, w)[: n - w]
        mean = windows.mean(axis=1)
        std = np.maximum(windows.std(axis=1), np.asarray(curve.flux_err, dtype=float)[w:])
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = (flux[w:] - mean) / std
        z[w:] = np.where(std > 0, zz, 0.0)
        return z

    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]:
        out: list[Detection] = []
        for star_id in sorted(curves):
            curve = curves[star_id]
            absz = np.abs(self._z(curve))
            flagged = absz >= self.z_threshold
            k = 0
            i = 0
            n = len(curve)
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
                            t_start=int(curve.t[i]),
                            t_end=int(curve.t[j]) + cadence_s,
                            score=float(absz[i : j + 1].max()),
                        )
                    )
                    k += 1
                i = j + 1
        return out
