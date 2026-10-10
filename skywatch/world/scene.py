"""The per-night scene that injectors mutate. See docs/design.md, "Injector contract"."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from skywatch.types import Star, TruthLabel


@dataclass(eq=False)
class Scene:
    """One night's true (pre-noise) fluxes for the whole field.

    Row ``i`` of ``flux`` and ``mask`` belongs to ``stars[i]``. ``t`` is the shared
    integer-second grid for the night. Injectors mutate ``flux`` and ``mask`` in
    place and return the truth labels for what they placed.
    """

    t: np.ndarray
    flux: np.ndarray
    mask: np.ndarray
    stars: list[Star]
    night_index: int
    cadence_s: int

    def __post_init__(self) -> None:
        n_stars, n_t = len(self.stars), len(self.t)
        if self.flux.shape != (n_stars, n_t):
            raise ValueError(f"flux shape {self.flux.shape} != {(n_stars, n_t)}")
        if self.mask.shape != (n_stars, n_t):
            raise ValueError(f"mask shape {self.mask.shape} != {(n_stars, n_t)}")
        if n_t > 1 and not np.all(np.diff(self.t) == self.cadence_s):
            raise ValueError("t must step by exactly cadence_s")

    @classmethod
    def quiet(
        cls, stars: list[Star], t_start: int, n_samples: int, cadence_s: int, night_index: int
    ) -> Scene:
        """A scene with every star at its quiescent flux and every sample observed."""
        t = t_start + cadence_s * np.arange(n_samples, dtype=np.int64)
        flux0 = np.array([s.flux0 for s in stars], dtype=np.float64)
        flux = np.repeat(flux0[:, None], n_samples, axis=1)
        mask = np.ones((len(stars), n_samples), dtype=bool)
        return cls(
            t=t, flux=flux, mask=mask, stars=stars, night_index=night_index, cadence_s=cadence_s
        )

    def row(self, star_id: int) -> int:
        """Row index of ``star_id`` in ``flux``/``mask``."""
        for i, s in enumerate(self.stars):
            if s.star_id == star_id:
                return i
        raise KeyError(star_id)


@runtime_checkable
class Injector(Protocol):
    """Places one kind of event into a scene and reports the truth."""

    kind: str

    def inject(self, scene: Scene, rng: np.random.Generator) -> list[TruthLabel]: ...
