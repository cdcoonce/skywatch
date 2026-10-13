"""What lies under every flare-sized masked run `bls` could open a flare window for (issue #48).

Usage::

    uv run python scripts/bls_flare_run_audit.py runs/seed61-200s-20n [more runs ...]
    uv run python scripts/bls_flare_run_audit.py --dump runs.jsonl runs/seed61-200s-20n

For every star of a run, the samples the one-sided clip of ``positive_outliers`` masks are split
into runs of consecutive samples within a night, as ``flare_windows`` does; a run of
``flare_min_run`` (10) or more samples is *flare-sized*. One line per run goes to ``--dump``: its
length, its peak in sigmas of the night and where in the run the peak is, the share of the run's positive excess the peak sample
carries, the ratio of the run's median to its peak, the decay time and residual per degree of
freedom of the exponential ``decay_fit`` puts through it, whether ``bls`` opens a window for it (it
decays), and the truth under it (the glitch and flare labels of the star that overlap the run's
span; never read by the detector, only a label for this table).

The summary groups the runs by that truth and by a description of the run that does not use the
fit: a *spike* (the peak sample carries half the run's excess or more), a *plateau* (the run's
median is at least half its peak) or a *decay* (neither: a peak with more mass after it than a
spike has and a median far under it), and counts how many of each the fit lets open a window. It
then counts the stars that carry a window under each definition: every flare-sized run (issue #43)
and decaying runs only (issue #48). This is the instrument behind the "Flare-sized runs" section of
the `skywatch.detectors.bls` docstring; it reads no detections or scores.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from skywatch.detectors.bls import (
    _MAD_SCALE,
    BlsDetector,
    decay_fit,
    flare_windows,
    masked_runs,
)
from skywatch.io import store

KINDS = ("flare", "hot_pixel", "cosmic_ray")


def describe(x: np.ndarray) -> str:
    """spike, plateau or decay, from the run's excess `x` (sigmas) without the fit."""
    peak = float(x.max())
    if peak >= 0.5 * float(np.clip(x, 0, None).sum()):
        return "spike"
    if float(np.median(x)) >= 0.5 * peak:
        return "plateau"
    return "decay"


def audit(run_dir: Path, det: BlsDetector) -> tuple[list[dict], dict[str, int]]:
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    labels: dict[int, list] = defaultdict(list)
    for lab in store.read_truth(run_dir):
        if lab.star_id is not None and lab.kind in KINDS:
            labels[lab.star_id].append(lab)
    curves = store.read_curves(run_dir)
    cleaned = {}
    for star_id in sorted(curves):
        c = det._clean(curves[star_id], cadence_s)
        if c is not None:
            cleaned[star_id] = c
    common_mode = det._common_mode(cleaned)
    rows: list[dict] = []
    stars = {"stars": len(cleaned), "old": 0, "new": 0}
    for star_id, c in cleaned.items():
        flux, _ = det._unclipped(c, common_mode)
        kept = np.isin(c.t, det._prepare(c, common_mode)[0])
        window_s = (det.flare_window_h or 0.0) * 3600.0
        old = flare_windows(c.t, c.night_of, kept, det.flare_min_run, window_s)
        new = flare_windows(
            c.t,
            c.night_of,
            kept,
            det.flare_min_run,
            window_s,
            flux,
            cadence_s,
            det.flare_tau_s,
            det.flare_fit_chi2,
        )
        stars["old"] += len(old) > 0
        stars["new"] += len(new) > 0
        for k in np.unique(c.night_of):
            idx = np.flatnonzero(c.night_of == k)
            x = flux[idx]
            scale = _MAD_SCALE * float(np.median(np.abs(x - np.median(x))))
            for a, b in masked_runs(~kept[idx], det.flare_min_run):
                t_run = c.t[idx[a:b]]
                xs = (x[a:b] - 1.0) / scale
                tau, chi2 = decay_fit(t_run, xs, cadence_s)
                ok = det.flare_tau_s is not None and (
                    det.flare_tau_s[0] <= tau <= det.flare_tau_s[1] and chi2 <= det.flare_fit_chi2
                )
                under = {
                    lab.kind
                    for lab in labels.get(star_id, ())
                    if lab.t_start <= t_run[-1] and lab.t_end >= t_run[0]
                }
                rows.append(
                    {
                        "run": run_dir.name,
                        "star": star_id,
                        "night": int(k),
                        "n": b - a,
                        "peak": float(xs.max()),
                        "peak_at": int(np.argmax(xs)),
                        "share": float(xs.max() / max(np.clip(xs, 0, None).sum(), 1e-9)),
                        "median_over_peak": float(np.median(xs) / xs.max()),
                        "tau_s": tau,
                        "chi2_per_dof": chi2,
                        "shape": describe(xs),
                        "opens": bool(ok),
                        "truth": "+".join(sorted(under)) or "none",
                    }
                )
    return rows, stars


def summarise(rows: list[dict], stars: dict[str, int]) -> list[str]:
    lines = [f"{len(rows)} flare-sized runs; stars: {stars}"]
    lines.append(
        f"{'truth under the run':36s} {'runs':>6s} {'share':>6s} {'n':>5s} {'peak':>7s}"
        f" {'spike':>6s} {'plateau':>8s} {'decay':>6s} {'opens':>6s}"
    )
    by = Counter(r["truth"] for r in rows)
    for truth, n in by.most_common():
        sel = [r for r in rows if r["truth"] == truth]
        shape = Counter(r["shape"] for r in sel)
        lines.append(
            f"{truth:36s} {n:6d} {100 * n / len(rows):5.1f}% "
            f"{np.median([r['n'] for r in sel]):5.0f} {np.median([r['peak'] for r in sel]):7.1f} "
            f"{shape['spike']:6d} {shape['plateau']:8d} {shape['decay']:6d} "
            f"{sum(r['opens'] for r in sel):6d}"
        )
    lines.append(
        f"{'all':36s} {len(rows):6d} {'':>6s} {'':>5s} {'':>7s} "
        f"{sum(r['shape'] == 'spike' for r in rows):6d} "
        f"{sum(r['shape'] == 'plateau' for r in rows):8d} "
        f"{sum(r['shape'] == 'decay' for r in rows):6d} {sum(r['opens'] for r in rows):6d}"
    )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, help="write one JSON line per run here")
    ap.add_argument("--jobs", type=int, default=1, help="runs audited in parallel")
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    det = BlsDetector()
    rows: list[dict] = []
    stars: Counter = Counter()
    with ProcessPoolExecutor(args.jobs) as pool:
        for r, s in pool.map(audit, args.runs, [det] * len(args.runs)):
            rows += r
            stars.update(s)
    if args.dump:
        with args.dump.open("w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    print("\n".join(summarise(rows, dict(stars))))


if __name__ == "__main__":
    main()
