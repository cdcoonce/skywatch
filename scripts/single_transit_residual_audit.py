"""Which stars `single_transit` refuses on the detrended path, and why (issue #61).

Usage::

    uv run python scripts/single_transit_residual_audit.py [--jobs 6] [--detail] runs/puls-seed809-200s-20n ...

For every star whose strongest night box guard (a) (the ``bls`` pulsation guard read as a one-event
box) refuses on the raw flux, the detrended path of :meth:`SingleTransitDetector._detrended` is
replayed step by step and the star is put in one outcome class:

* ``reported``  the residual has a box at the threshold that survives the flare gate and guard (a);
* ``left_out``  as ``reported``, but only because the star's other boxes were left out of the guard's
               data (``leave_out_events``, issue #61): its plain read of the residual refuses it;
* ``nofit``     no sinusoid stands out (``_pulsation_trend`` is None), nothing is subtracted;
* ``noboxes``   no night of the residual reaches the threshold;
* ``flared``    every box of the residual is refused by the flare gate (``_stands_unflared``);
* ``fails_a``   the residual has boxes and guard (a) refuses them again, with the other boxes out too.

The plain read is what ``main`` did before issue #61, so ``fails_a`` + ``left_out`` is the table of
its comment.

The last line of a run is the tally of outcomes by the host's variability class. With ``--detail``
one line is printed for every ``fails_a`` star: the host's variability class and, for a sinusoidal
one, its true frequency and amplitude against the fitted ones; how many nights of the residual
reach the threshold and the S/N of the best; the truth under the best box (the kinds of the labels
that overlap its span, ``-`` when none); how many observable transit labels (and planets) the star
holds and whether one lies under a night of the residual that reaches the threshold; which of the
two tests of guard (a) refuses the box (``resid``: a sinusoid left in the residual, ``sine``: the
box-versus-sine ratio) with the residual's peak power over its false-alarm level and the sine
power over the box's chi-square; and the strongest sinusoid of the residual itself (frequency
relative to the fitted one, power over the false-alarm level). It reads the light curves, the stars
and the truth labels and writes nothing.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from skywatch.detectors.bls import DAY_S, _Arrays, _demean_by_night, in_box
from skywatch.detectors.single_transit import SingleTransitDetector
from skywatch.io import store

OUTCOMES = ("reported", "left_out", "nofit", "noboxes", "flared", "fails_a")


def _guard_a(det: SingleTransitDetector, t, flux, night, boxes, cadence_s):  # type: ignore[no-untyped-def]
    """(which test refuses the star's strongest box, residual power / level, sine / box chi2)."""
    from dataclasses import replace

    from skywatch.detectors.bls import _GUARD_CORE_WIDTH, _GUARD_MASK_WIDTH, _MAD_SCALE

    best = max(boxes, key=lambda b: b.snr)
    flux = det._level_corrected(t, flux, night, [best])
    box = det._event_box(best, cadence_s, len(t))
    box = replace(box, n_in=int(in_box(t, box).sum()))
    bls = det._bls
    core = in_box(t, replace(box, duration_s=_GUARD_CORE_WIDTH * box.duration_s))
    ring = in_box(t, replace(box, duration_s=_GUARD_MASK_WIDTH * box.duration_s)) & ~core
    keep = ~ring
    outside = keep & ~core
    out = flux[~in_box(t, box)]
    sigma = _MAD_SCALE * float(np.median(np.abs(out - np.median(out))))
    depth = float(flux[outside].mean() - flux[core].mean()) if core.any() else 0.0
    resid = flux - flux[outside].mean() + depth * core
    r_power, r_crit = bls._peak_power(t[keep], resid[keep], sigma, cadence_s)
    n, n_in = len(t), box.n_in
    box_chi2 = box.depth**2 * n_in * (n - n_in) / (n * sigma**2)
    s_power, _ = bls._peak_power(t, flux - 1.0, sigma, cadence_s)
    which = []
    if r_power >= r_crit:
        which.append("resid")
    if 2.0 * s_power >= bls.guard_ratio * box_chi2:
        which.append("sine")
    return "+".join(which) or "none", r_power / r_crit, 2.0 * s_power / box_chi2


def _kinds_under(labels: list, t0: int, t1: int) -> str:
    kinds = sorted(
        {lab.kind for lab in labels if lab.t_end > t0 and lab.t_start < t1} - {"pulsation"}
    )
    return "+".join(kinds) if kinds else "-"


def audit_run(args: tuple[str, bool]) -> tuple[list[str], Counter]:
    run_name, detail = args
    run = Path(run_name)
    stars = {s.star_id: s for s in store.read_stars(run)}
    labels: dict[int, list] = {}
    for lab in store.read_truth(run):
        if lab.star_id is not None:
            labels.setdefault(lab.star_id, []).append(lab)
    cadence_s = store.read_manifest(run)["config"]["cadence_s"]
    curves = store.read_curves(run)
    det = SingleTransitDetector()
    bls = det._bls
    cleaned = {}
    for star_id in sorted(curves):
        c = bls._clean(curves[star_id], cadence_s)
        if c is not None:
            cleaned[star_id] = c
    common_mode = bls._common_mode(cleaned, cadence_s)
    lines: list[str] = []
    tally: Counter = Counter()
    for star_id, c in cleaned.items():
        t, flux, night, level = bls._prepare(c, common_mode)
        if len(t) < det.min_in + det.min_out:
            continue
        boxes = det._boxes(t, flux, night, cadence_s)
        if not boxes or not det._pulsating(t, flux, night, boxes, cadence_s):
            continue
        star = stars[star_id]
        host = star.variability
        a = _Arrays(t, flux, night, level, c.t, c.night_of)
        fit = bls._pulsation_trend(a, cadence_s)
        if fit is None:
            tally[(host, "nofit")] += 1
            continue
        resid = det._residual(c, t, flux, night, level, cadence_s)
        if resid is None:
            tally[(host, "nofit")] += 1
            continue
        rboxes = det._boxes(t, resid, night, cadence_s)
        if not rboxes:
            tally[(host, "noboxes")] += 1
            continue
        kept = [b for b in rboxes if det._stands_unflared(t, resid, night, b)]
        if not kept:
            tally[(host, "flared")] += 1
            continue
        if not det._pulsating(t, resid, night, kept, cadence_s):
            tally[(host, "reported")] += 1
            continue
        if not det._residual_pulsating(t, resid, night, kept, cadence_s):
            tally[(host, "left_out")] += 1
            continue
        tally[(host, "fails_a")] += 1
        if not detail:
            continue
        best = max(kept, key=lambda b: b.snr)
        mine = labels.get(star_id, [])
        transits = [lab for lab in mine if lab.kind == "transit"]
        planets = {lab.params.get("planet_id") for lab in transits}
        high = {b.night for b in rboxes}
        t_nights = {int(lab.t_start // 86400) for lab in transits}  # night k starts at k * 86400 s
        under = len(t_nights & high)
        which, rp, sp = _guard_a(det, t, resid, night, kept, cadence_s)
        # the strongest sinusoid left in the residual, against the one subtracted
        sigma = bls._white_sigma(resid, night)
        td = (t - float(t[0])) / DAY_S
        k = np.unique(night, return_inverse=True)[1]
        x = _demean_by_night(resid - 1.0, k)
        left = bls._strongest_sinusoid(t, td, x, k, sigma, cadence_s, float(td[-1]))
        if left is None or not sigma > 0:
            lf = "-"
        else:
            gain, f2, sa, ca, lvl = left
            lf = f"left f {f2:5.2f} ({f2 / fit[1][0]:4.2f}x) amp {math.hypot(sa, ca) * 100:4.2f}% pwr {gain / (2 * sigma**2) / lvl:5.1f}x"
        if host == "sinusoidal":
            p = star.params
            truef = DAY_S / p["period_s"]
            host_s = f"sin f {truef:5.2f} A {p['amplitude'] * 100:4.2f}% fit {fit[1][0]:5.2f} {fit[1][1] * 100:4.2f}%"
        else:
            host_s = f"{host} fit {fit[1][0]:5.2f} {fit[1][1] * 100:4.2f}%"
        lines.append(
            f"{run.name} {star_id:4d} | {host_s} | nights>=8 {len(high):2d} best {best.snr:6.1f} "
            f"under {_kinds_under(mine, best.t_start, best.t_end + cadence_s)} | "
            f"transit labels {len(transits)} planets {len(planets)} on a high night {under} | "
            f"{which} resid {rp:5.1f}x sine {sp:5.2f} | {lf}"
        )
    return lines, tally


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    total: Counter = Counter()
    jobs = [(r, args.detail) for r in args.runs]
    with ProcessPoolExecutor(args.jobs) as pool:
        for lines, tally in pool.map(audit_run, jobs):
            for line in lines:
                print(line, flush=True)
            total += tally
    hosts = sorted({h for h, _ in total})
    print(f"{'host':14s} " + " ".join(f"{o:>9s}" for o in OUTCOMES))
    for h in hosts:
        print(f"{h:14s} " + " ".join(f"{total[(h, o)]:9d}" for o in OUTCOMES))


if __name__ == "__main__":
    main()
