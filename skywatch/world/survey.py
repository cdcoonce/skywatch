"""The survey loop: field -> nights -> injectors -> noise -> light curves.

Random streams (fixed; see docs/design.md, "Determinism"):

* ``SeedSequence(seed).spawn(3)``: child 0 -> field stars, child 1 -> observation noise,
  child 2 -> planets (kept apart so ``planet_rate`` never perturbs the stars); ``spawn(n_nights)`` gives one stream per night.
* each injector, each night -> ``SeedSequence([seed, night_index, stable_hash(kind)])``,
  so adding or removing an injector never perturbs another. Injector ``kind`` values must
  therefore be unique within a run: two injectors of the same kind share one stream.

Each night holds ``night_length_s // cadence_s`` samples; a partial last cadence is dropped.

A label's span is inclusive on both ends when setting ``quality`` bit 0.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from skywatch.types import LightCurve, Planet, Star, TruthLabel
from skywatch.world.field import FieldConfig, generate_field
from skywatch.world.noise import apply_noise
from skywatch.world.scene import Injector, Scene


@dataclass(frozen=True)
class SurveyConfig:
    seed: int = 42
    n_nights: int = 3
    cadence_s: int = 60
    night_length_s: int = 8 * 3600
    night_gap_s: int = 16 * 3600
    field: FieldConfig = field(default_factory=FieldConfig)

    def __post_init__(self) -> None:
        if self.cadence_s < 1:
            raise ValueError(f"cadence_s must be >= 1, got {self.cadence_s}")
        if self.night_length_s < self.cadence_s:
            raise ValueError(
                f"night_length_s must be >= cadence_s, got {self.night_length_s} < {self.cadence_s}"
            )
        if self.night_gap_s < 0:
            raise ValueError(f"night_gap_s must be >= 0, got {self.night_gap_s}")
        if self.n_nights < 1:
            raise ValueError(f"n_nights must be >= 1, got {self.n_nights}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")


@dataclass
class SurveyResult:
    config: SurveyConfig
    stars: list[Star]
    planets: list[Planet]
    curves: dict[int, LightCurve]
    truth: list[TruthLabel]
    night_starts: list[int]


def stable_hash(kind: str) -> int:
    return int.from_bytes(hashlib.sha256(kind.encode()).digest()[:4], "big")


def _injector_rng(seed: int, night_index: int, kind: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, night_index, stable_hash(kind)]))


def simulate(
    cfg: SurveyConfig,
    science: Sequence[Injector] = (),
    instrument: Sequence[Injector] = (),
) -> SurveyResult:
    root = np.random.SeedSequence(cfg.seed)
    stars_ss, noise_ss, planets_ss = root.spawn(3)
    stars, planets = generate_field(
        cfg.field, np.random.default_rng(stars_ss), np.random.default_rng(planets_ss)
    )
    noise_nights = noise_ss.spawn(cfg.n_nights)
    n_samples = cfg.night_length_s // cfg.cadence_s
    noise_floor = np.array([s.noise_floor for s in stars], dtype=np.float64)

    truth: list[TruthLabel] = []
    night_starts: list[int] = []
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for k in range(cfg.n_nights):
        t_start = k * (cfg.night_length_s + cfg.night_gap_s)
        night_starts.append(t_start)
        scene = Scene.quiet(stars, t_start, n_samples, cfg.cadence_s, k)
        quality = np.zeros(n_samples, dtype=np.uint8)
        for inj in science:
            truth.extend(inj.inject(scene, _injector_rng(cfg.seed, k, inj.kind)))
        for inj in instrument:
            labels = inj.inject(scene, _injector_rng(cfg.seed, k, inj.kind))
            truth.extend(labels)
            for lab in labels:
                if lab.star_id is None:
                    quality[(scene.t >= lab.t_start) & (scene.t <= lab.t_end)] |= 1
        obs, err = apply_noise(scene.flux, noise_floor, np.random.default_rng(noise_nights[k]))
        parts.append((scene.t, obs, err, np.broadcast_to(quality, obs.shape), scene.mask))

    curves: dict[int, LightCurve] = {}
    for i, s in enumerate(stars):
        sel = [p[4][i] for p in parts]
        curves[s.star_id] = LightCurve(
            star_id=s.star_id,
            t=np.concatenate([p[0][m] for p, m in zip(parts, sel, strict=True)]),
            flux=np.concatenate([p[1][i][m] for p, m in zip(parts, sel, strict=True)]),
            flux_err=np.concatenate([p[2][i][m] for p, m in zip(parts, sel, strict=True)]),
            quality=np.concatenate([p[3][i][m] for p, m in zip(parts, sel, strict=True)]),
        )
    return SurveyResult(cfg, stars, planets, curves, truth, night_starts)
