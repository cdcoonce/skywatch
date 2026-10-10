"""Instrument glitches: cosmic rays, clouds, tracking drift, hot pixels, dropouts.

Each class implements the ``Injector`` protocol from ``skywatch.world.scene``. All
labels carry ``origin="instrument"``; field-wide events use ``star_id=None``.
Label ids are ``f"{kind}-n{night}-s{star_id or 'field'}-{k}"`` with ``k`` counting
from 0 per (star or field) per night. Spans are ``[t_start, t_end)`` in integer
seconds, clipped to the night (``params["clipped"]=True`` when clipped).

Determinism: every random draw comes from the passed ``rng`` in the order stated in
each class docstring. Per-star injectors iterate stars in ``scene.stars`` order and
draw the Poisson count first, then each event's draws in full before the next.
"""

from __future__ import annotations

import numpy as np

from skywatch.types import TruthLabel
from skywatch.world.footprint import KEY, touched


def _night_bounds(scene) -> tuple[int, int]:
    return int(scene.t[0]), int(scene.t[-1]) + scene.cadence_s


def _clip(scene, start: int, end: int) -> tuple[int, int, bool]:
    lo, hi = _night_bounds(scene)
    s, e = max(start, lo), min(end, hi)
    return s, e, (s != start or e != end)


def _uniform_time(scene, rng: np.random.Generator) -> int:
    lo, hi = _night_bounds(scene)
    return lo + int(rng.uniform(0.0, hi - lo))


def _label(
    scene,
    kind: str,
    star_id: int | None,
    k: int,
    t_start: int,
    t_end: int,
    params: dict,
    footprint: int,
) -> TruthLabel:
    sid = "field" if star_id is None else star_id
    return TruthLabel(
        label_id=f"{kind}-n{scene.night_index}-s{sid}-{k}",
        kind=kind,
        origin="instrument",
        star_id=star_id,
        t_start=t_start,
        t_end=t_end,
        params={**params, KEY: footprint},
    )


class CosmicRayInjector:
    """Single-sample positive spikes, additive on the true flux.

    Draw order per star: count ~ Poisson(rate); per hit: sample index (integers),
    then log-uniform amplitude in [0.2, 5.0].
    """

    kind = "cosmic_ray"

    def __init__(self, rate_per_star_per_night: float = 0.3) -> None:
        self.rate = rate_per_star_per_night

    def inject(self, scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        n_t = len(scene.t)
        for row, star in enumerate(scene.stars):
            for k in range(int(rng.poisson(self.rate))):
                j = int(rng.integers(n_t))
                amp = float(np.exp(rng.uniform(np.log(0.2), np.log(5.0))))
                with touched(scene, row) as fp:
                    scene.flux[row, j] += amp * scene.flux[row, j]
                t0 = int(scene.t[j])
                labels.append(
                    _label(
                        scene,
                        self.kind,
                        star.star_id,
                        k,
                        t0,
                        t0 + scene.cadence_s,
                        {"amplitude": amp},
                        fp.samples,
                    )
                )
        return labels


class CloudInjector:
    """Field-wide raised-cosine dimming.

    Draw order: count ~ Poisson(rate); per cloud: t_mid, duration_s, depth (uniform).
    ``t_mid`` and ``duration_s`` are rounded to integer seconds.
    """

    kind = "cloud"

    def __init__(self, rate_per_night: float = 0.8) -> None:
        self.rate = rate_per_night

    def inject(self, scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        for k in range(int(rng.poisson(self.rate))):
            t_mid = _uniform_time(scene, rng)
            dur = round(rng.uniform(300, 3600))
            depth = float(rng.uniform(0.05, 0.6))
            s, e, clipped = _clip(scene, t_mid - dur // 2, t_mid - dur // 2 + dur)
            sel = (scene.t >= s) & (scene.t < e)
            w = 0.5 * (1 + np.cos(np.pi * (scene.t[sel] - t_mid) / (dur / 2)))
            with touched(scene) as fp:
                scene.flux[:, sel] *= 1 - depth * w
            params = {"depth": depth, "duration_s": dur, "t_mid": t_mid}
            if clipped:
                params["clipped"] = True
            labels.append(_label(scene, self.kind, None, k, s, e, params, fp.samples))
        return labels


class TrackingDriftInjector:
    """Field-wide linear ramp that persists to the end of the night.

    Draw order: count ~ Poisson(rate); per event: t_start, duration_s, slope (uniform).
    The label spans the ramp only; samples after the ramp keep factor ``1 + slope``.
    If the ramp is clipped by the night end, the factor reached at night end stands.
    """

    kind = "tracking_drift"

    def __init__(self, rate_per_night: float = 0.3) -> None:
        self.rate = rate_per_night

    def inject(self, scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        for k in range(int(rng.poisson(self.rate))):
            t_start = _uniform_time(scene, rng)
            dur = round(rng.uniform(1800, 7200))
            slope = float(rng.uniform(-0.3, 0.3))
            s, e, clipped = _clip(scene, t_start, t_start + dur)
            factor = np.ones(len(scene.t))
            ramp = (scene.t >= s) & (scene.t < e)
            factor[ramp] = 1 + slope * (scene.t[ramp] - t_start) / dur
            factor[scene.t >= t_start + dur] = 1 + slope
            with touched(scene) as fp:
                scene.flux *= factor
            params = {"slope": slope, "duration_s": dur}
            if clipped:
                params["clipped"] = True
            labels.append(_label(scene, self.kind, None, k, s, e, params, fp.samples))
        return labels


class HotPixelInjector:
    """Flux stuck at ``flux0 * k`` over a span (overwrites).

    Draw order per star: count ~ Poisson(rate); per event: t_start, duration_s, k.
    """

    kind = "hot_pixel"

    def __init__(self, rate_per_star_per_night: float = 0.05) -> None:
        self.rate = rate_per_star_per_night

    def inject(self, scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        for row, star in enumerate(scene.stars):
            for k in range(int(rng.poisson(self.rate))):
                t_start = _uniform_time(scene, rng)
                dur = round(rng.uniform(300, 3600))
                value = float(star.flux0 * rng.uniform(1.5, 4.0))
                s, e, clipped = _clip(scene, t_start, t_start + dur)
                with touched(scene, row) as fp:
                    scene.flux[row, (scene.t >= s) & (scene.t < e)] = value
                params = {"value": value, "duration_s": dur}
                if clipped:
                    params["clipped"] = True
                labels.append(_label(scene, self.kind, star.star_id, k, s, e, params, fp.samples))
        return labels


class DropoutInjector:
    """Missing cadences: clears ``scene.mask`` over a span, flux untouched.

    Draw order: count ~ Poisson(rate); per event: scope uniform (field-wide if
    ``< field_wide_fraction``), star index (integers; only when scope is star),
    t_start, duration_s.
    """

    kind = "dropout"

    def __init__(self, rate_per_night: float = 0.5, field_wide_fraction: float = 0.5) -> None:
        self.rate = rate_per_night
        self.field_wide_fraction = field_wide_fraction

    def inject(self, scene, rng: np.random.Generator) -> list[TruthLabel]:
        labels: list[TruthLabel] = []
        counters: dict[int | None, int] = {}
        for _ in range(int(rng.poisson(self.rate))):
            field_wide = bool(rng.uniform() < self.field_wide_fraction)
            row = None if field_wide else int(rng.integers(len(scene.stars)))
            t_start = _uniform_time(scene, rng)
            dur = round(rng.uniform(60, 1800))
            s, e, clipped = _clip(scene, t_start, t_start + dur)
            sel = (scene.t >= s) & (scene.t < e)
            sid = None if row is None else scene.stars[row].star_id
            with touched(scene, row) as fp:
                if row is None:
                    scene.mask[:, sel] = False
                else:
                    scene.mask[row, sel] = False
            params: dict = {"duration_s": dur, "scope": "field" if field_wide else "star"}
            if clipped:
                params["clipped"] = True
            k = counters.get(sid, 0)
            counters[sid] = k + 1
            labels.append(_label(scene, self.kind, sid, k, s, e, params, fp.samples))
        return labels
