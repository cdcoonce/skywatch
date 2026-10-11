"""Which planets in a stored run a periodic search could find: events and stacked S/N.

A truth-side tool like ``score`` and ``report``: detectors never import it. It reads the
run's ``planets``, ``stars``, ``truth`` and ``observations`` tables and, for every planet,
counts the in-night transit events and measures how loud each is, in matched-filter units.

Per-event S/N is ``depth * sqrt(sum(profile(t_i)**2)) / sigma_tot`` over the star's
*observed* samples inside the label's span. ``profile`` is the transit injector's box with
linear ramps (see ``skywatch.world.science.TransitInjector``) and
``sigma_tot = sqrt(noise_floor**2 + 1/flux0)`` is the per-sample noise in relative flux
(scintillation plus photon noise, see ``skywatch.world.noise``). A label whose
``footprint_samples`` is 0 changed no sample, so it counts as an event with S/N 0.
``snr_stack`` is the quadrature sum of a planet's event S/Ns, ``snr_max`` the largest one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from skywatch.io import store

DAY_S = 86400.0
HOUR_S = 3600.0
_STRONG_SNR = 7.0
_HEADER = ("planet_id", "star", "period_d", "depth", "dur_h", "events", "snr_stack", "snr_max")


@dataclass(frozen=True)
class PlanetRow:
    planet_id: int
    star_id: int
    period_d: float
    depth: float
    dur_h: float
    events: int
    snr_stack: float
    snr_max: float


def _event_snr(t: np.ndarray, depth: float, duration_s: float, t_mid: float, sigma: float) -> float:
    """Matched-filter S/N of one transit over the observed sample times ``t``."""
    ramp = 0.1 * duration_s
    start = t_mid - duration_s / 2
    end = t_mid + duration_s / 2
    profile = np.clip(np.minimum((t - start) / ramp, (end - t) / ramp), 0.0, 1.0)
    return depth * math.sqrt(float(np.sum(profile**2))) / sigma


def planet_rows(run_dir: Path) -> list[PlanetRow]:
    """One row per planet, sorted by ``snr_stack`` descending then ``planet_id``."""
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"no run at {run_dir} (manifest.json missing)")
    planets = store.read_planets(run_dir)
    sigma = {
        s.star_id: math.sqrt(s.noise_floor**2 + 1.0 / s.flux0) for s in store.read_stars(run_dir)
    }
    labels = [
        lab
        for lab in store.read_truth(run_dir)
        if lab.kind == "transit" and "planet_id" in lab.params
    ]
    star_ids = sorted({lab.star_id for lab in labels})
    t_by_star: dict[int, np.ndarray] = {}
    if star_ids:
        tbl = pq.read_table(
            run_dir / "observations.parquet",
            columns=["star_id", "t"],
            filters=[("star_id", "in", star_ids)],
        )
        star_col = tbl["star_id"].to_numpy()
        t_col = tbl["t"].to_numpy().astype(np.float64)
        for sid in star_ids:
            t_by_star[sid] = np.sort(t_col[star_col == sid])

    snrs: dict[int, list[float]] = {p.planet_id: [] for p in planets}
    for lab in labels:
        prm = lab.params
        if prm.get("footprint_samples") == 0:
            snr = 0.0
        else:
            t = t_by_star[lab.star_id]
            t = t[(t >= lab.t_start) & (t <= lab.t_end)]
            snr = _event_snr(t, prm["depth"], prm["duration_s"], prm["t_mid"], sigma[lab.star_id])
        snrs.setdefault(prm["planet_id"], []).append(snr)

    rows = []
    for p in planets:
        s = snrs[p.planet_id]
        rows.append(
            PlanetRow(
                planet_id=p.planet_id,
                star_id=p.star_id,
                period_d=p.period_s / DAY_S,
                depth=p.depth,
                dur_h=p.duration_s / HOUR_S,
                events=len(s),
                snr_stack=math.sqrt(sum(x * x for x in s)),
                snr_max=max(s, default=0.0),
            )
        )
    rows.sort(key=lambda r: (-r.snr_stack, r.planet_id))
    return rows


def format_planets(rows: list[PlanetRow], min_events: int = 1, min_snr: float = 0.0) -> str:
    """The table (filtered by ``min_events`` and ``min_snr``) and a summary of every row."""
    shown = [r for r in rows if r.events >= min_events and r.snr_stack >= min_snr]
    cells = [
        (
            str(r.planet_id),
            str(r.star_id),
            f"{r.period_d:.3f}",
            f"{r.depth:.5f}",
            f"{r.dur_h:.2f}",
            str(r.events),
            f"{r.snr_stack:.1f}",
            f"{r.snr_max:.1f}",
        )
        for r in shown
    ]
    widths = [max([len(h), *(len(c[i]) for c in cells)]) for i, h in enumerate(_HEADER)]
    lines = ["  ".join(h.rjust(w) for h, w in zip(_HEADER, widths, strict=True))]
    lines += ["  ".join(v.rjust(w) for v, w in zip(c, widths, strict=True)) for c in cells]
    multi = [r for r in rows if r.events >= 2]
    strong = [r for r in multi if r.snr_stack >= _STRONG_SNR]
    lines.append(
        f"{len(rows)} planets; {len(multi)} with >= 2 events; "
        f"{len(strong)} with >= 2 events and snr_stack >= 7"
    )
    return "\n".join(lines)
