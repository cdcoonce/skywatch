"""Data model for skywatch, as defined in docs/design.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCIENCE_KINDS = ("transit", "flare", "eclipse", "pulsation")
INSTRUMENT_KINDS = ("cosmic_ray", "cloud", "tracking_drift", "hot_pixel", "dropout")

ORIGIN_OF_KIND: dict[str, str] = {
    **{k: "science" for k in SCIENCE_KINDS},
    **{k: "instrument" for k in INSTRUMENT_KINDS},
}


@dataclass(frozen=True)
class Star:
    star_id: int
    flux0: float
    noise_floor: float
    variability: str
    ra: float
    dec: float
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Planet:
    planet_id: int
    star_id: int
    period_s: float
    epoch_s: float
    depth: float
    duration_s: float


@dataclass(frozen=True, eq=False)
class LightCurve:
    star_id: int
    t: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.t)
        if not (len(self.flux) == len(self.flux_err) == len(self.quality) == n):
            raise ValueError("t, flux, flux_err and quality must have equal lengths")
        if n > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError("t must be strictly increasing")

    def __len__(self) -> int:
        return len(self.t)


@dataclass(frozen=True)
class TruthLabel:
    label_id: str
    kind: str
    origin: str
    star_id: int | None
    t_start: int
    t_end: int
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ORIGIN_OF_KIND:
            raise ValueError(f"unknown kind: {self.kind!r}")
        if self.origin not in ("science", "instrument"):
            raise ValueError(f"unknown origin: {self.origin!r}")
        if ORIGIN_OF_KIND[self.kind] != self.origin:
            raise ValueError(
                f"kind {self.kind!r} has origin {ORIGIN_OF_KIND[self.kind]!r}, not {self.origin!r}"
            )

    @property
    def has_footprint(self) -> bool:
        """True if the label records ``params["footprint_samples"]`` (runs since #5)."""
        return self.params.get("footprint_samples") is not None

    @property
    def observable(self) -> bool:
        """False only when the event changed zero samples, so no detector could see it.

        A label with no ``footprint_samples`` (an older run) is treated as observable.
        """
        return self.params.get("footprint_samples") != 0


@dataclass(frozen=True)
class Detection:
    detector: str
    det_id: str
    star_id: int | None
    t_start: int
    t_end: int
    score: float
    kind_guess: str | None = None
