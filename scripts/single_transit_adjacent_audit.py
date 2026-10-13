"""What sits beside a `single_transit` box that came through the pulsation subtraction (issue #54).

Usage::

    # the measurement table: every detection of the detrended path, the nearest clipped run
    uv run python scripts/single_transit_adjacent_audit.py table runs/puls-seed712-200s-20n ...

    # detections of one detector on many runs, written as JSON (several runs in parallel)
    uv run python scripts/single_transit_adjacent_audit.py detect --detector single_transit \\
        --out scratch/branch --jobs 6 runs/seed420-200s-20n ...

    # the headline counts of `single_transit` alone and of its union with `bls`
    uv run python scripts/single_transit_adjacent_audit.py tally --single scratch/branch \\
        --bls scratch/bls runs/seed420-200s-20n ...

``table`` reruns the detector and, for each detection on a star that went through the pulsation
subtraction (:meth:`SingleTransitDetector._detrended`), prints the nearest *clipped run* in the same
night: a run of consecutive samples that were left out of the search as positive outliers, either by
the first clip (the one ``bls`` applies to the raw flux, whose scale the pulsation inflates) or by
the same clip applied again to the residual (which is white noise plus whatever the pulsation fit
did not remove). A run is described by its length in samples, whether the first clip or only the
second one caught it, its distance to the box (``before``: the run ends before the box starts,
``after``: it starts after the box ends, ``inside``: it overlaps the box; seconds) and, as a
measurement only (the gate never reads truth), what truth label lies under it. The last column is
the matched truth kind of the detection under the scorer's matcher (``-`` for a false positive).

``tally`` is the table every change to a detector quotes: transit labels bounded and
bounded-or-touched, false positives, the detector's transit guesses on eclipses, pulsations and
flares, detections, and the number of *planets* with at least one transit event bounded or touched.
The union row counts a label once when either detector found it; fp, confusion and detections are
summed over the two detectors. It reads no stored scores: matching is done again in memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from skywatch.io import store
from skywatch.scoring.match import match_detections
from skywatch.scoring.metrics import confusion, score
from skywatch.types import Detection

DET_FIELDS = ("detector", "det_id", "star_id", "t_start", "t_end", "score", "kind_guess")


# -- detect ----------------------------------------------------------------------------------


def _make(detector: str):  # type: ignore[no-untyped-def]
    from skywatch.detectors import REGISTRY, load_all

    load_all()
    return REGISTRY[detector]()


def _detect_one(args: tuple[str, str, str]) -> str:
    run, detector, out = args
    run_dir = Path(run)
    dest = Path(out) / f"{run_dir.name}.{detector}.json"
    curves = store.read_curves(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    dets = _make(detector).detect(curves, cadence_s)
    dest.write_text(json.dumps([{f: getattr(d, f) for f in DET_FIELDS} for d in dets]))
    return f"{run_dir.name} {detector} {len(dets)}"


def cmd_detect(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(r), args.detector, str(out)) for r in args.runs]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            for line in pool.map(_detect_one, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(_detect_one(job), flush=True)


# -- tally -----------------------------------------------------------------------------------


def _load(dirpath: Path, run: Path, detector: str) -> list[Detection]:
    rows = json.loads((dirpath / f"{run.name}.{detector}.json").read_text())
    return [Detection(**r) for r in rows]


def _stats(run_dir: Path, dets: list[Detection], detector: str) -> dict:
    truth = store.read_truth(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    matches = match_detections(dets, truth, cadence_s)
    scores = {s.kind: s for s in score(matches, dets, truth, detectors=[detector])}
    conf = confusion(matches, dets, truth, detectors=[detector])
    grade = {m.label_id: m.grade for m in matches if m.grade in ("bounded", "touched")}
    kind = {t.label_id: t.kind for t in truth}
    transit = [t for t in truth if t.kind == "transit" and t.observable]
    plan = {t.label_id: t.params.get("planet_id") for t in transit}
    return {
        "bounded": {lab for lab, g in grade.items() if g == "bounded" and kind[lab] == "transit"},
        "touched": {lab for lab, g in grade.items() if kind[lab] == "transit"},
        "fp": scores["all"].fp,
        "pulsation>transit": conf.get((detector, "pulsation", "transit"), 0),
        "eclipse>transit": conf.get((detector, "eclipse", "transit"), 0),
        "flare>transit": conf.get((detector, "flare", "transit"), 0),
        "dets": len(dets),
        "plan": plan,
    }


COLS = ("bounded", "b-or-t", "planets", "fp", "puls>t", "ecl>t", "flare>t", "dets")


def _row(parts: list[tuple[str, dict]]) -> tuple[int, ...]:
    """Counts of one run from one or two detectors' stats (a union when there are two)."""
    bounded = set().union(*(s["bounded"] for _, s in parts))
    touched = set().union(*(s["touched"] for _, s in parts))
    plan = parts[0][1]["plan"]
    planets = {plan[lab] for lab in touched}
    return (
        len(bounded),
        len(touched),
        len(planets),
        sum(s["fp"] for _, s in parts),
        sum(s["pulsation>transit"] for _, s in parts),
        sum(s["eclipse>transit"] for _, s in parts),
        sum(s["flare>transit"] for _, s in parts),
        sum(s["dets"] for _, s in parts),
    )


def cmd_tally(args: argparse.Namespace) -> None:
    single_dirs = [Path(p) for p in args.single]
    bls_dir = Path(args.bls) if args.bls else None
    header = f"{'run':26s} " + " ".join(f"{c:>8s}" for c in COLS)
    for sdir in single_dirs:
        for label in ("single_transit",) + (("union",) if bls_dir else ()):
            print(f"== {label}: {sdir}")
            print(header)
            total = [0] * len(COLS)
            for run in map(Path, args.runs):
                parts = [
                    (
                        "single_transit",
                        _stats(run, _load(sdir, run, "single_transit"), "single_transit"),
                    )
                ]
                if label == "union" and bls_dir:
                    parts.append(("bls", _stats(run, _load(bls_dir, run, "bls"), "bls")))
                row = _row(parts)
                total = [a + b for a, b in zip(total, row, strict=True)]
                if args.per_run:
                    print(f"{run.name:26s} " + " ".join(f"{v:8d}" for v in row))
            print(f"{'TOTAL':26s} " + " ".join(f"{v:8d}" for v in total))
    if bls_dir:
        print("== bls alone")
        print(header)
        total = [0] * len(COLS)
        for run in map(Path, args.runs):
            row = _row([("bls", _stats(run, _load(bls_dir, run, "bls"), "bls"))])
            total = [a + b for a, b in zip(total, row, strict=True)]
        print(f"{'TOTAL':26s} " + " ".join(f"{v:8d}" for v in total))


# -- table -----------------------------------------------------------------------------------


def _truth_under(labels: list, t0: int, t1: int) -> str:
    kinds = sorted(
        {lab.kind for lab in labels if lab.t_end > t0 and lab.t_start < t1} - {"pulsation"}
    )
    return "+".join(kinds) if kinds else "-"


def _tracking_detector():  # type: ignore[no-untyped-def]
    from skywatch.detectors.single_transit import SingleTransitDetector

    class Tracking(SingleTransitDetector):
        def __init__(self, **kw: object) -> None:
            super().__init__(**kw)  # type: ignore[arg-type]
            self.sid = -1
            self.detrended: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        def _star_boxes(self, star_id, c, common_mode, cadence_s):  # type: ignore[no-untyped-def]
            self.sid = star_id
            return super()._star_boxes(star_id, c, common_mode, cadence_s)

        def _detrended(self, c, t, flux, night, level, cadence_s):  # type: ignore[no-untyped-def]
            found = super()._detrended(c, t, flux, night, level, cadence_s)
            if found is not None:
                self.detrended[self.sid] = (t, found[1], night)
            return found

    return Tracking()


def _runs_of_night(c, kept_first: np.ndarray, kept_after: np.ndarray, k: int, cadence_s: int):  # type: ignore[no-untyped-def]
    """Clipped runs of night `k` as (t0, t1, n, by_first_clip_share), t1 exclusive (s)."""
    from skywatch.detectors.bls import masked_runs

    idx = np.flatnonzero(c.night_of == k)
    out = []
    for a, b in masked_runs(~kept_after[idx]):
        first = float(np.mean(~kept_first[idx[a:b]]))
        out.append((float(c.t[idx[a]]), float(c.t[idx[b - 1]]) + cadence_s, b - a, first))
    return out


def _clean_snr(t: np.ndarray, x: np.ndarray, drop: np.ndarray, t0: int, t1: int) -> float:
    """S/N of the box [t0, t1) of one night with the samples in `drop` left out (nan if none)."""
    from skywatch.detectors.bls import _MAD_SCALE

    keep = ~drop
    box = keep & (t >= t0) & (t < t1)
    out = keep & ~((t >= t0) & (t < t1))
    n_in, n_out = int(box.sum()), int(out.sum())
    if n_in < 15 or n_out < 30:
        return math.nan
    sigma = _MAD_SCALE * float(np.median(np.abs(x[out] - np.median(x[out]))))
    depth = float(x[out].mean() - x[box].mean())
    return depth * math.sqrt(n_in * n_out / (n_in + n_out)) / sigma


def _table_run(run_name: str) -> list[str]:
    from skywatch.detectors.bls import positive_outliers

    run = Path(run_name)
    lines = []
    det = _tracking_detector()
    bls = det._bls
    curves = store.read_curves(run)
    cadence_s = store.read_manifest(run)["config"]["cadence_s"]
    labels: dict[int, list] = defaultdict(list)
    truth = store.read_truth(run)
    for lab in truth:
        if lab.star_id is not None:
            labels[lab.star_id].append(lab)
    dets = det.detect(curves, cadence_s)
    matches = match_detections(dets, truth, cadence_s)
    kind = {t.label_id: t.kind for t in truth}
    matched = {m.det_id: (kind[m.label_id] if m.label_id else "-") for m in matches}
    cleaned = {s: bls._clean(curves[s], cadence_s) for s in sorted(curves)}
    for d in dets:
        if d.star_id not in det.detrended:
            continue
        c = cleaned[d.star_id]
        t, resid, night = det.detrended[d.star_id]
        kept_first = np.isin(c.t, t)
        kept_after = kept_first.copy()
        pos = np.searchsorted(c.t, t)
        for k in np.unique(night):
            m = night == k
            bad = positive_outliers(resid[m], bls.clip_sigma, bls.tail_sigma)
            kept_after[pos[m][bad]] = False
        inside = (c.t >= d.t_start) & (c.t < d.t_end)
        k = int(c.night_of[np.flatnonzero(inside)[0]])
        runs = _runs_of_night(c, kept_first, kept_after, k, cadence_s)

        def describe(rs: list, d: Detection = d) -> str:
            if not rs:
                return "none"
            best = None
            for r0, r1, n, first in rs:
                if r0 >= d.t_end:
                    side, gap = "after", r0 - d.t_end
                elif r1 <= d.t_start:
                    side, gap = "before", d.t_start - r1
                else:
                    side, gap = "inside", 0
                if best is None or gap < best[1]:
                    best = (side, gap, n, first, r0, r1)
            side, gap, n, first, r0, r1 = best
            tr = _truth_under(labels[d.star_id], int(r0), int(r1))
            return f"{side:6s} {gap:7.0f} {n:3d} {first:4.2f} {tr}"

        big = [r for r in runs if r[2] >= 10]
        # the box's S/N with the residual's clipped samples left out, and with the 1 h / 3 h
        # after each flare-sized run left out as well
        mk = night == k
        tk, xk = t[mk], resid[mk]
        bad = positive_outliers(xk, bls.clip_sigma, bls.tail_sigma)
        from skywatch.detectors.bls import masked_runs

        cleans = []
        for hours in (None, 1.0, 3.0):
            drop = bad.copy()
            if hours is not None:
                for a, b in masked_runs(bad, 10):
                    drop |= (tk >= tk[a]) & (tk <= tk[b - 1] + hours * 3600.0)
            cleans.append(_clean_snr(tk, xk, drop, d.t_start, d.t_end))
        clean = "/".join("nan" if math.isnan(v) else f"{v:.1f}" for v in cleans)
        lines.append(
            f"{run.name} {d.star_id:4d} n{k:<2d} {d.score:6.1f} | {describe(runs)} | "
            f"{describe(big)} | clean {clean} | {matched.get(d.det_id, '-')}"
        )
    return lines


def cmd_table(args: argparse.Namespace) -> None:
    print(
        "run star night S/N | nearest clipped run: side gap_s len first-clip-share truth | "
        "nearest run of 10+: same fields | matched truth"
    )
    with ProcessPoolExecutor(args.jobs) as pool:
        for lines in pool.map(_table_run, args.runs):
            for line in lines:
                print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_table)
    p = sub.add_parser("detect")
    p.add_argument("--detector", required=True, choices=("bls", "single_transit"))
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_detect)
    p = sub.add_parser("tally")
    p.add_argument("--single", nargs="+", required=True, help="directories of single_transit json")
    p.add_argument("--bls", help="directory of bls json")
    p.add_argument("--per-run", action="store_true")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_tally)
    args = ap.parse_args()
    args.fn(args)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
