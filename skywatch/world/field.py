"""Field generation: stars and planets. See docs/design.md, "World model"."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skywatch.types import Planet, Star

DAY_S = 86400.0


def _default_weights() -> dict[str, float]:
    return {"constant": 0.70, "sinusoidal": 0.12, "eclipsing_binary": 0.08, "flare_star": 0.10}


@dataclass(frozen=True)
class FieldConfig:
    n_stars: int = 200
    flux0_log10_range: tuple[float, float] = (3.0, 6.0)
    noise_floor_range: tuple[float, float] = (0.002, 0.02)
    variability_weights: dict[str, float] = field(default_factory=_default_weights)
    planet_rate: float = 0.15
    ra_center: float = 0.0
    dec_center: float = 0.0
    field_radius_deg: float = 1.0


def _params(variability: str, rng: np.random.Generator) -> dict:
    if variability == "eclipsing_binary":
        period = float(rng.uniform(0.3, 5.0) * DAY_S)
        primary = float(rng.uniform(0.05, 0.5))
        return {
            "period_s": period,
            "epoch_s": float(rng.uniform(0.0, period)),
            "depth_primary": primary,
            "depth_secondary": float(rng.uniform(0.2, 0.8) * primary),
            "duration_s": float(rng.uniform(0.05, 0.15) * period),
        }
    if variability == "sinusoidal":
        return {
            "period_s": float(rng.uniform(1.0, 12.0) * 3600.0),
            "amplitude": float(rng.uniform(0.001, 0.02)),
            "phase": float(rng.uniform(0.0, 2 * np.pi)),
        }
    if variability == "flare_star":
        return {"flare_rate_per_day": float(rng.uniform(0.5, 5.0))}
    return {}


def generate_field(cfg: FieldConfig, rng: np.random.Generator) -> tuple[list[Star], list[Planet]]:
    """Draw the stars and planets of a field from a single ``rng``."""
    names = list(cfg.variability_weights)
    w = np.array([cfg.variability_weights[k] for k in names], dtype=np.float64)
    w = w / w.sum()
    lo, hi = cfg.flux0_log10_range
    stars: list[Star] = []
    for i in range(cfg.n_stars):
        flux0 = float(10 ** rng.uniform(lo, hi))
        noise_floor = float(rng.uniform(*cfg.noise_floor_range))
        variability = names[int(rng.choice(len(names), p=w))]
        r = cfg.field_radius_deg * float(np.sqrt(rng.uniform()))
        theta = float(rng.uniform(0.0, 2 * np.pi))
        stars.append(
            Star(
                star_id=i,
                flux0=flux0,
                noise_floor=noise_floor,
                variability=variability,
                ra=cfg.ra_center + r * float(np.cos(theta)),
                dec=cfg.dec_center + r * float(np.sin(theta)),
                params=_params(variability, rng),
            )
        )
    planets: list[Planet] = []
    for s in stars:
        for _ in range(int(rng.poisson(cfg.planet_rate))):
            period = float(rng.uniform(0.5, 20.0) * DAY_S)
            planets.append(
                Planet(
                    planet_id=len(planets),
                    star_id=s.star_id,
                    period_s=period,
                    epoch_s=float(rng.uniform(0.0, period)),
                    depth=float(10 ** rng.uniform(-4.0, np.log10(3e-2))),
                    duration_s=float(rng.uniform(0.5, 6.0) * 3600.0),
                )
            )
    return stars, planets
