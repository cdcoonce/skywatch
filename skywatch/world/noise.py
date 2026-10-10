"""Observation noise model. See docs/design.md, "Noise" and "Injector contract"."""

from __future__ import annotations

import numpy as np


def apply_noise(
    flux: np.ndarray, noise_floor: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(flux_obs, flux_err)`` for ``flux`` of shape (n_stars, n_t).

    ``flux_obs = flux * (1 + N(0, noise_floor)) + N(0, sqrt(flux))`` and
    ``flux_err = sqrt(flux + (noise_floor * flux) ** 2)``. Negative flux is
    clipped to zero before the square roots.
    """
    f = np.clip(flux, 0.0, None)
    nf = np.asarray(noise_floor, dtype=np.float64)[:, None]
    scint = rng.normal(0.0, 1.0, size=f.shape) * nf
    photon = rng.normal(0.0, 1.0, size=f.shape) * np.sqrt(f)
    flux_obs = flux * (1.0 + scint) + photon
    flux_err = np.sqrt(f + (nf * f) ** 2)
    return flux_obs, flux_err
