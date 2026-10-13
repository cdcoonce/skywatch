"""What `single_transit` finds on the pulsating hosts it searches after the detrending (issue #51).

Usage::

    uv run python scripts/single_transit_pulsator_audit.py runs/seed44-200s-20n [more runs ...]
    uv run python scripts/single_transit_pulsator_audit.py --min-snr 0 runs/puls-seed393-200s-20n

For every star whose strongest night box the pulsation guard of `single_transit` refuses (guard
(a)), the pulsation is subtracted as `single_transit` does (:meth:`SingleTransitDetector._residual`)
and the per-night search is read on the residual. One line is printed for each such star whose
best night reaches ``--min-snr`` (default 5): the fitted frequency (cycles/day) and amplitude, the
S/N of the residual's best night and of the second best, how many nights reach the detector's
threshold, the truth label under the best night (transit, eclipse, flare, another glitch, or
"none" when only the pulsation is there), whether the residual passes the pulsation guard, the
shape statistic of the best night, and what the detector reports (``reported``, or why not). A
final block counts the hosts by variability class and outcome, and the reported detections by
truth label. It reads the light curves, the stars and the truth labels of the run and writes
nothing; it is the instrument behind the measurement table of issue #51.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from skywatch.detectors.bls import _Arrays
from skywatch.detectors.single_transit import HOUR_S, SingleTransitDetector, search_night
from skywatch.io import store

LABEL_ORDER = ("transit", "eclipse", "flare", "hot_pixel", "cosmic_ray", "cloud", "dropout")


def truth_kind(labels: list, t0: int, t1: int) -> str:
    """The kind of truth under [t0, t1), by priority; the pulsation alone reads "none"."""
    kinds = {lab.kind for lab in labels if lab.t_end > t0 and lab.t_start < t1}
    for k in LABEL_ORDER:
        if k in kinds:
            return k
    return "other" if kinds - {"pulsation"} else "none"


def audit(run_dir: Path, min_snr: float) -> tuple[list[str], Counter, Counter]:
    stars = {s.star_id: s for s in store.read_stars(run_dir)}
    by_star: dict[int, list] = {}
    for lab in store.read_truth(run_dir):
        if lab.star_id is not None:
            by_star.setdefault(lab.star_id, []).append(lab)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    curves = store.read_curves(run_dir)
    det = SingleTransitDetector()
    bls = det._bls
    cleaned = {}
    for star_id in sorted(curves):
        c = bls._clean(curves[star_id], cadence_s)
        if c is not None:
            cleaned[star_id] = c
    common_mode = bls._common_mode(cleaned)
    durations = tuple(h * HOUR_S for h in det.durations_h)
    lines: list[str] = []
    hosts: Counter = Counter()
    reported: Counter = Counter()
    for star_id, c in cleaned.items():
        t, flux, night, level = bls._prepare(c, common_mode)
        if len(t) < det.min_in + det.min_out:
            continue
        boxes = det._boxes(t, flux, night, cadence_s)
        if not boxes or not det._pulsating(t, flux, night, boxes, cadence_s):
            continue
        star = stars[star_id]
        resid = det._residual(c, t, flux, night, level, cadence_s)
        if resid is None:
            hosts[(star.variability, "no sinusoid")] += 1
            continue
        nights = []
        for k in np.unique(night):
            idx = np.flatnonzero(night == k)
            f = search_night(
                t[idx], resid[idx], cadence_s, durations, det.min_in, det.min_out, det.contrast
            )
            if f is not None:
                nights.append((f[0], int(t[idx[f[5]]]), int(t[idx[f[6]]]) + cadence_s))
        nights.sort(reverse=True)
        top = nights[0] if nights else (0.0, 0, 0)
        second = nights[1][0] if len(nights) > 1 else 0.0
        n_hit = sum(1 for n in nights if n[0] >= det.snr_threshold)
        label = truth_kind(by_star.get(star_id, []), top[1], top[2])
        found = det._detrended(c, t, flux, night, level, cadence_s)
        shape = None
        if found is None:
            outcome = "below the threshold" if n_hit == 0 else "residual guard"
        else:
            kept, resid = found
            shapes = [det._shape(t, resid, night, b, cadence_s) for b in kept]
            shape = max(shapes)
            if det.shape_threshold is not None and shape >= det.shape_threshold:
                outcome = "shape test"
            else:
                outcome = "reported"
                for b in kept:
                    reported[truth_kind(by_star.get(star_id, []), b.t_start, b.t_end)] += 1
        hosts[(star.variability, outcome)] += 1
        if top[0] >= min_snr:
            fit = bls._pulsation_trend(_Arrays(t, flux, night, level, c.t, c.night_of), cadence_s)
            freq = "      -" if fit is None else f"{fit[1][0]:5.2f}/{100 * fit[1][1]:4.2f}%"
            shape_s = "    -" if shape is None else f"{shape:5.2f}"
            lines.append(
                f"{run_dir.name} star {star_id:4d} {star.variability[:5]:5s} f/amp {freq} "
                f"best {top[0]:6.1f} 2nd {second:5.1f} nights>=thr {n_hit:2d} "
                f"label {label:9s} shape {shape_s} | {outcome}"
            )
    return lines, hosts, reported


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-snr", type=float, default=5.0, help="print hosts at or above this S/N")
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    hosts: Counter = Counter()
    reported: Counter = Counter()
    for path in args.runs:
        lines, h, r = audit(path, args.min_snr)
        for line in lines:
            print(line)
        hosts += h
        reported += r
    print("pulsating hosts searched, by class and outcome:")
    for (var, outcome), n in sorted(hosts.items()):
        print(f"  {var:17s} {outcome:24s} {n:5d}")
    print("detections reported on them, by truth label:", dict(reported))


if __name__ == "__main__":
    main()
