"""Science injectors: transits, eclipses, pulsations, flares. See docs/design.md."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from skywatch.types import Planet, TruthLabel
from skywatch.world.footprint import touched, with_footprint
from skywatch.world.scene import Scene

_FLARE_AMP_RANGE = (0.01, 0.5)
_FLARE_DECAY_RANGE = (120.0, 1800.0)
_FLARE_TAIL_DECAYS = 3.0  # exp(-3) ~ 5% of amplitude


def _night_bounds(scene: Scene) -> tuple[int, int]:
    """[start, end) of the night in integer seconds."""
    return int(scene.t[0]), int(scene.t[-1]) + scene.cadence_s


def _clip_label(
    scene: Scene,
    kind: str,
    star_id: int,
    k: int,
    start: float,
    end: float,
    params: dict,
) -> TruthLabel | None:
    """Label for an event over [start, end) clipped to the night; None if wholly outside."""
    ns, ne = _night_bounds(scene)
    if end <= ns or start >= ne:
        return None
    t_start = max(math.floor(start), ns)
    t_end = min(math.ceil(end), ne)
    if start < ns or end > ne:
        params = {**params, "clipped": True}
    return TruthLabel(
        label_id=f"{kind}-n{scene.night_index}-s{star_id}-{k}",
        kind=kind,
        origin="science",
        star_id=star_id,
        t_start=t_start,
        t_end=t_end,
        params=params,
    )


def _event_indices(scene: Scene, epoch: float, period: float, half_width: float) -> range:
    """Cycle numbers n whose event [mid - hw, mid + hw] can overlap the night."""
    ns, ne = _night_bounds(scene)
    lo = math.floor((ns - half_width - epoch) / period)
    hi = math.ceil((ne + half_width - epoch) / period)
    return range(lo, hi + 1)


class TransitInjector:
    kind = "transit"

    def __init__(self, planets: Sequence[Planet]) -> None:
        self.planets = list(planets)

    def inject(self, scene: Scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        counts: dict[int, int] = {}
        present = {s.star_id for s in scene.stars}
        for planet in self.planets:
            if planet.star_id not in present:
                continue
            row = scene.row(planet.star_id)
            half = planet.duration_s / 2
            ramp = 0.1 * planet.duration_s
            for n in _event_indices(scene, planet.epoch_s, planet.period_s, half):
                t_mid = planet.epoch_s + n * planet.period_s
                start, end = t_mid - half, t_mid + half
                k = counts.get(planet.star_id, 0)
                label = _clip_label(
                    scene,
                    self.kind,
                    planet.star_id,
                    k,
                    start,
                    end,
                    {
                        "depth": planet.depth,
                        "duration_s": planet.duration_s,
                        "t_mid": t_mid,
                        "planet_id": planet.planet_id,
                    },
                )
                if label is None:
                    continue
                counts[planet.star_id] = k + 1
                profile = np.clip(
                    np.minimum((scene.t - start) / ramp, (end - scene.t) / ramp), 0.0, 1.0
                )
                with touched(scene, row) as fp:
                    scene.flux[row] *= 1.0 - planet.depth * profile
                labels.append(with_footprint(label, fp.samples))
        return labels


class EclipseInjector:
    kind = "eclipse"

    def inject(self, scene: Scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        for row, star in enumerate(scene.stars):
            if star.variability != "eclipsing_binary":
                continue
            p = star.params
            period, epoch, dur = p["period_s"], p["epoch_s"], p["duration_s"]
            half = dur / 2
            k = 0
            for n in _event_indices(scene, epoch, period, half + period / 2):
                for which, offset, depth in (
                    ("primary", 0.0, p["depth_primary"]),
                    ("secondary", period / 2, p["depth_secondary"]),
                ):
                    t_mid = epoch + n * period + offset
                    label = _clip_label(
                        scene,
                        self.kind,
                        star.star_id,
                        k,
                        t_mid - half,
                        t_mid + half,
                        {"depth": depth, "duration_s": dur, "t_mid": t_mid, "which": which},
                    )
                    if label is None:
                        continue
                    k += 1
                    profile = np.clip(1.0 - np.abs(scene.t - t_mid) / half, 0.0, 1.0)
                    with touched(scene, row) as fp:
                        scene.flux[row] *= 1.0 - depth * profile
                    labels.append(with_footprint(label, fp.samples))
        return labels


class PulsationInjector:
    kind = "pulsation"

    def inject(self, scene: Scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        ns, ne = _night_bounds(scene)
        for row, star in enumerate(scene.stars):
            if star.variability != "sinusoidal":
                continue
            p = star.params
            phase = p.get("phase", 0.0)
            with touched(scene, row) as fp:
                scene.flux[row] *= 1.0 + p["amplitude"] * np.sin(
                    2.0 * np.pi * scene.t / p["period_s"] + phase
                )
            labels.append(
                TruthLabel(
                    label_id=f"{self.kind}-n{scene.night_index}-s{star.star_id}-0",
                    kind=self.kind,
                    origin="science",
                    star_id=star.star_id,
                    t_start=ns,
                    t_end=ne,
                    params={
                        "period_s": p["period_s"],
                        "amplitude": p["amplitude"],
                        "footprint_samples": fp.samples,
                    },
                )
            )
        return labels


class FlareInjector:
    kind = "flare"

    def __init__(self, default_rate_per_day: float = 0.05) -> None:
        self.default_rate_per_day = default_rate_per_day

    def inject(self, scene: Scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        ns, ne = _night_bounds(scene)
        days = (ne - ns) / 86400.0
        log_lo, log_hi = (math.log(a) for a in _FLARE_AMP_RANGE)
        for row, star in enumerate(scene.stars):
            rate = star.params.get("flare_rate_per_day", self.default_rate_per_day)
            count = int(rng.poisson(rate * days))
            k = 0
            for _ in range(count):
                t_peak = float(rng.uniform(ns, ne))
                amp = float(math.exp(rng.uniform(log_lo, log_hi)))
                decay = float(rng.uniform(*_FLARE_DECAY_RANGE))
                label = _clip_label(
                    scene,
                    self.kind,
                    star.star_id,
                    k,
                    t_peak,
                    t_peak + _FLARE_TAIL_DECAYS * decay,
                    {"amplitude": amp, "decay_s": decay, "t_peak": t_peak},
                )
                if label is None:  # t_peak < ne always, so unreachable; kept for safety
                    continue
                k += 1
                after = scene.t >= t_peak
                dt = scene.t[after] - t_peak
                with touched(scene, row) as fp:
                    scene.flux[row, after] *= 1.0 + amp * np.exp(-dt / decay)
                labels.append(with_footprint(label, fp.samples))
        return labels
