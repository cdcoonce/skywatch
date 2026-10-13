"""What the pulsation detrending of `bls` leaves, for every sinusoidal star that reaches it.

Usage::

    uv run python scripts/bls_detrend_audit.py runs/seed112-200s-20n [more runs ...]
    uv run python scripts/bls_detrend_audit.py --simulate runs 370 20 0.0   # planet-free pulsator field
    uv run python scripts/bls_detrend_audit.py --simulate runs 374 20 0.5   # with planets

Every sinusoidal star whose best box the pulsation guard rejects is detrended
(:meth:`BlsDetector._detrended`); this wraps that method, with the repeat test of issue #44
switched off, and prints one line for each star that ends with an accepted box: the injected and
fitted frequency (cycles/day) and amplitude, the Fourier power left at the fundamental and at
twice it in units of the guard's false-alarm level, the box, its S/N, the night-level sine ratio,
the S/N without the strongest night (``Candidate.snr_repeat``), how many of its events overlap a
planet's transit, and what the shipping bar does with it. A final line counts the stars that
reached the detrending and what became of them. It reads the light curves, the stars and the
planets of the run and writes nothing. This is the instrument behind the "Repeat test of a
detrended box" section of the `skywatch.detectors.bls` docstring (issue #44).

``--simulate OUT SEED NIGHTS PLANET_RATE`` writes a 200-star field in which every star is
sinusoidal (run id ``puls-seedN-200s-Mn``, ``puls-pf-seedN-...`` when the rate is 0): the
design runs of seeds 370-377.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from skywatch.detectors import bls
from skywatch.detectors.bls import BlsDetector
from skywatch.io import store

DAY_S = 86400.0


class _Audit(BlsDetector):
    """A `BlsDetector` that remembers what the detrending did to the star it is searching."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.record: dict | None = None

    def _left_over(self, a: bls._Arrays, flux: np.ndarray, cadence_s: int, f: float) -> dict:
        """Power left at `f` and `2 f` (best of +-0.15 cycles/day) in false-alarm levels."""
        sigma = self._white_sigma(flux, a.night)
        spectrum = self._band_power(a.t, flux - 1.0, sigma, cadence_s)
        if spectrum is None or not sigma > 0:
            return {"f": math.nan, "2f": math.nan}
        td = (a.t - float(a.t[0])) / DAY_S
        k = np.unique(a.night, return_inverse=True)[1]
        x = bls._demean_by_night(flux - 1.0, k)
        out = {}
        for name, centre in (("f", f), ("2f", 2.0 * f)):
            grid = [g for g in np.arange(centre - 0.15, centre + 0.1501, 0.01) if g > 0]
            gain = max(bls._sine_fit(td, x, k, float(g))[2] for g in grid)
            out[name] = gain / (2.0 * sigma**2) / spectrum[2]
        return out

    def _detrended(self, a, c, cadence_s, plan_mask):  # type: ignore[no-untyped-def]
        rec: dict = {}
        self.record = rec
        fit = self._pulsation_trend(a, cadence_s)
        if fit is None:
            rec["outcome"] = "no sinusoid"
            return None
        flux = bls._renormalise_nights(a.flux - fit[0], a.night)
        cand = self._search_arrays(replace(a, flux=flux), c, cadence_s, plan_mask)
        if cand is not None and cand.rejected != "snr":
            fit = self._pulsation_trend(a, cadence_s, cand.box)
            if fit is None:
                rec["outcome"] = "no sinusoid"
                return None
            flux = bls._renormalise_nights(a.flux - fit[0], a.night)
            cand = self._search_arrays(replace(a, flux=flux), c, cadence_s, plan_mask)
        rec["fit"] = fit[1]
        rec["left"] = self._left_over(a, flux, cadence_s, fit[1][0])
        rec["cand"] = cand
        if cand is None or cand.rejected is not None:
            rec["outcome"] = f"rejected: {'no box' if cand is None else cand.rejected}"
            return None
        if min(cand.snr, cand.snr_local) < self.trend_snr_threshold:
            rec["outcome"] = "below the detrended S/N bar"
            return None
        rec["outcome"] = "accepted"
        return replace(cand, trend=fit[1][:2])


def _events_on_planets(cand: bls.Candidate, planets: list) -> int:
    hits = 0
    for t0, t1, _ in cand.events:
        for p in planets:
            mid = p.epoch_s + round((t0 - p.epoch_s) / p.period_s) * p.period_s
            if abs(mid - (t0 + t1) / 2) < p.duration_s / 2 + (t1 - t0) / 2:
                hits += 1
                break
    return hits


def audit(run_dir: Path) -> tuple[list[str], Counter]:
    stars = {s.star_id: s for s in store.read_stars(run_dir)}
    planets = store.read_planets(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    curves = store.read_curves(run_dir)
    det = _Audit(trend_repeat_snr=None)
    bar = BlsDetector().trend_repeat_snr
    assert bar is not None
    cleaned = {}
    for star_id in sorted(curves):
        c = det._clean(curves[star_id], cadence_s)
        if c is not None:
            cleaned[star_id] = c
    common_mode = det._common_mode(cleaned)
    lines: list[str] = []
    tally: Counter = Counter()
    for star_id, c in cleaned.items():
        star = stars[star_id]
        if star.variability != "sinusoidal":
            continue
        det.record = None
        det._search(c, common_mode, cadence_s)
        rec = det.record
        if rec is None:
            continue
        tally[rec["outcome"]] += 1
        if rec["outcome"] != "accepted":
            continue
        cand = rec["cand"]
        mine = [p for p in planets if p.star_id == star_id]
        hits = _events_on_planets(cand, mine)
        verdict = "kept" if cand.snr_repeat >= bar else "refused by the repeat test"
        lines.append(
            f"{run_dir.name} star {star_id:4d} f {DAY_S / star.params['period_s']:5.2f} "
            f"(fit {rec['fit'][0]:6.3f}) amp {100 * star.params['amplitude']:4.2f}% "
            f"(fit {100 * rec['fit'][1]:4.2f}%) left f {rec['left']['f']:5.2f} 2f "
            f"{rec['left']['2f']:5.2f} | P {cand.box.period_s / DAY_S:6.3f} d "
            f"dur {cand.box.duration_s / 3600:3.1f} h events {len(cand.events):2d} "
            f"on a planet {hits:2d} S/N {cand.snr:5.1f} local {cand.snr_local:5.1f} "
            f"ratio {cand.sine_ratio or 0:4.2f} repeat {cand.snr_repeat:5.2f} | {verdict}"
        )
    return lines, tally


def simulate(out: Path, seed: int, nights: int, planet_rate: float) -> Path:
    from skywatch import run
    from skywatch.world.field import FieldConfig
    from skywatch.world.survey import SurveyConfig

    run_id = f"puls{'-pf' if planet_rate == 0 else ''}-seed{seed}-200s-{nights}n"
    field = FieldConfig(
        n_stars=200, planet_rate=planet_rate, variability_weights={"sinusoidal": 1.0}
    )
    cfg = SurveyConfig(seed=seed, n_nights=nights, cadence_s=60, field=field)
    return run.simulate_run(cfg, out, run_id, False)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--simulate"]:
        out_dir, seed_, nights_, rate = sys.argv[2:6]
        print(simulate(Path(out_dir), int(seed_), int(nights_), float(rate)))
        raise SystemExit(0)
    total: Counter = Counter()
    for path in sys.argv[1:]:
        run_lines, run_tally = audit(Path(path))
        for line in run_lines:
            print(line)
        total += run_tally
    print("reached the detrending:", sum(total.values()), dict(total))
