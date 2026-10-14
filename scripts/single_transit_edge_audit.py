"""How `single_transit` reads a long night-edge event, on the detector's own path (issue #63).

Usage::

    # the records of the detector's own searches, one JSON-lines file per run
    uv run python scripts/single_transit_edge_audit.py records --out scratch/rec --jobs 9 runs/...

    # the single-event planets at matched-filter S/N >= 10: what the detector read, found or not
    uv run python scripts/single_transit_edge_audit.py planets --records scratch/rec \\
        --dets scratch/dets runs/...

    # the boxes at S/N 6-8 that overlap no transit label (and every planet-free detection)
    uv run python scripts/single_transit_edge_audit.py near-miss --records scratch/rec \\
        --dets scratch/dets runs/...

    # the headline counts of `single_transit` alone and of its union with `bls`, plus the
    # single-event planets at S/N >= 10 found
    uv run python scripts/single_transit_edge_audit.py tally --single scratch/dets \\
        [--bls scratch/bls] runs/...

``records`` runs a real ``detect`` with a subclass of :class:`SingleTransitDetector` whose
``search_night`` is wrapped, so every night search the detector makes on its own path is recorded
(the contrast S/N it ranks and reports, the S/N of the plain form, ``n_in``, ``n_out``, the night's
length, whether the box touches the night's first or last sample, and which search it was: the raw
curve, a pulsation-subtracted residual inside the detrend, or the residual of a star the detrend
never ran on). For a star whose raw best box reads from ``--floor`` (5) up and which the detector
did not detrend, the residual is also computed by the detector's own ``_residual`` on the arrays
the real ``_prepare`` returned (the common mode of the whole field included) and searched, tagged
``forced``: that is the number the detrend would have read had the raw prefilter let it run.
Nothing a record holds is read by the detector. A replay of ``_prepare`` without the field's
common mode is not the detector's path and gives different numbers (PR #55's false cause).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import single_transit_adjacent_audit as adj

from skywatch.io import store
from skywatch.scoring.match import match_detections
from skywatch.types import Detection

DAY_S = 86400
FLOOR = 4.0  # a search whose best box reads under this is not recorded
FORCE_FROM = 5.0  # a star whose raw best box reads this much gets its residual searched too
STRONG = 10.0  # matched-filter S/N of the planets in the table


# -- records ---------------------------------------------------------------------------------


def _recording_detector(floor: float):  # type: ignore[no-untyped-def]
    import skywatch.detectors.single_transit as mod
    from skywatch.detectors.single_transit import SingleTransitDetector

    orig = mod.search_night
    rec: list[dict] = []
    stars: list[dict] = []

    class Recording(SingleTransitDetector):
        def __init__(self, **kw: object) -> None:
            super().__init__(**kw)  # type: ignore[arg-type]
            self.sid = -1
            self.ctx = "raw"
            self.refused: bool | None = None
            self.detrended = False

        def _pulsating(self, t, flux, night, boxes, cadence_s):  # type: ignore[no-untyped-def]
            r = super()._pulsating(t, flux, night, boxes, cadence_s)
            if self.ctx == "raw" and self.refused is None:
                self.refused = r
            return r

        def _residual(self, c, t, flux, night, level, cadence_s):  # type: ignore[no-untyped-def]
            before, self.ctx = self.ctx, "pre" if self.ctx == "det" else "forced-pre"
            try:
                return super()._residual(c, t, flux, night, level, cadence_s)
            finally:
                self.ctx = before

        def _detrended(self, c, t, flux, night, level, cadence_s):  # type: ignore[no-untyped-def]
            self.ctx, self.detrended = "det", True
            try:
                return super()._detrended(c, t, flux, night, level, cadence_s)
            finally:
                self.ctx = "post"

        def _star_boxes(self, star_id, c, common_mode, cadence_s):  # type: ignore[no-untyped-def]
            self.sid, self.ctx, self.refused, self.detrended = star_id, "raw", None, False
            n0 = len(rec)
            out = super()._star_boxes(star_id, c, common_mode, cadence_s)
            raw = [r["snr"] for r in rec[n0:] if r["sid"] == star_id and r["ctx"] == "raw"]
            best = max(raw, default=0.0)
            forced = False
            if best >= FORCE_FROM and not self.detrended:
                t, flux, night, level = self._bls._prepare(c, common_mode)
                self.ctx = "forced-pre"
                resid = self._residual(c, t, flux, night, level, cadence_s)
                if resid is not None:
                    self.ctx, forced = "forced", True
                    self._boxes(t, resid, night, cadence_s)
            stars.append(
                {
                    "sid": star_id,
                    "raw_best": best,
                    "refused": self.refused,
                    "detrended": self.detrended,
                    "forced": forced,
                    "n_boxes": len(out[0]),
                }
            )
            return out

    det = Recording()

    def wrapper(t, flux, cadence_s, durations_s, min_in, min_out, contrast=True):  # type: ignore[no-untyped-def]
        r = orig(t, flux, cadence_s, durations_s, min_in, min_out, contrast)
        if r is None or det.sid < 0:
            return r
        plain = orig(t, flux, cadence_s, durations_s, min_in, min_out, False)
        if r[0] >= floor or (plain is not None and plain[0] >= floor + 1.0):
            snr, depth, n_in, n_out, sigma, first, last = r
            rec.append(
                {
                    "sid": det.sid,
                    "ctx": det.ctx,
                    "night": int(t[0] // DAY_S),
                    "snr": snr,
                    "snr_plain": None if plain is None else plain[0],
                    "depth": depth,
                    "n_in": n_in,
                    "n_out": n_out,
                    "n_night": len(t),
                    "sigma": sigma,
                    "t0": int(t[first]),
                    "t1": int(t[last]) + cadence_s,
                    "touch": ("s" if first == 0 else "") + ("e" if last == len(t) - 1 else ""),
                }
            )
        return r

    mod.search_night = wrapper
    return det, rec, stars


def _records_one(args: tuple[str, str, float]) -> str:
    run, out, floor = args
    run_dir = Path(run)
    curves = store.read_curves(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    det, rec, stars = _recording_detector(floor)
    dets = det.detect(curves, cadence_s)
    var = {s.star_id: s.variability for s in store.read_stars(run_dir)}
    for r in rec:
        r["var"] = var[r["sid"]]
    for s in stars:
        s["var"] = var[s["sid"]]
    dest = Path(out) / f"{run_dir.name}.jsonl"
    lines = [json.dumps({"kind": "box", **r}) for r in rec]
    lines += [json.dumps({"kind": "star", **s}) for s in stars]
    dest.write_text("\n".join(lines))
    return f"{run_dir.name} boxes {len(rec)} detections {len(dets)}"


def cmd_records(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(r), str(out), args.floor) for r in args.runs]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            for line in pool.map(_records_one, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(_records_one(job), flush=True)


# -- detect with parameters --------------------------------------------------------------------


def _detect_one(args: tuple[str, str, dict]) -> str:
    run, out, params = args
    run_dir = Path(run)
    from skywatch.detectors import REGISTRY, load_all

    load_all()
    curves = store.read_curves(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    dets = REGISTRY["single_transit"](**params).detect(curves, cadence_s)
    dest = Path(out) / f"{run_dir.name}.single_transit.json"
    dest.write_text(json.dumps([{f: getattr(d, f) for f in adj.DET_FIELDS} for d in dets]))
    return f"{run_dir.name} single_transit {len(dets)}"


def cmd_detect(args: argparse.Namespace) -> None:
    """`single_transit` with constructor parameters (--set name=value), json like `adjacent`."""
    params = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(r), str(out), params) for r in args.runs]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            for line in pool.map(_detect_one, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(_detect_one(job), flush=True)


# -- joins -----------------------------------------------------------------------------------


def _load_records(rec_dir: Path, run: Path) -> tuple[list[dict], dict[int, dict]]:
    boxes, stars = [], {}
    for line in (rec_dir / f"{run.name}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.pop("kind") == "box":
            boxes.append(r)
        else:
            stars[r["sid"]] = r
    return boxes, stars


def _star_times(run: Path, star_ids: list[int]) -> dict[int, np.ndarray]:
    if not star_ids:
        return {}
    tbl = pq.read_table(
        run / "observations.parquet",
        columns=["star_id", "t"],
        filters=[("star_id", "in", star_ids)],
    )
    sid, t = tbl["star_id"].to_numpy(), tbl["t"].to_numpy().astype(np.float64)
    return {s: np.sort(t[sid == s]) for s in star_ids}


def _planet_events(run: Path) -> dict[int, list]:
    """The observable transit labels of each planet (footprint above 0), by planet id."""
    events: dict[int, list] = defaultdict(list)
    for lab in store.read_truth(run):
        if lab.kind == "transit" and "planet_id" in lab.params and lab.observable:
            events[lab.params["planet_id"]].append(lab)
    return events


def _overlaps(r: dict, t0: float, t1: float) -> bool:
    return r["t0"] < t1 and r["t1"] > t0


def _best(boxes: list[dict], sid: int, ctxs: tuple[str, ...], t0: float, t1: float) -> dict | None:
    cand = [r for r in boxes if r["sid"] == sid and r["ctx"] in ctxs and _overlaps(r, t0, t1)]
    return max(cand, key=lambda r: r["snr"], default=None)


def _fmt(r: dict | None) -> str:
    if r is None:
        return "-"
    return f"{r['snr']:.2f} ({r['n_in']}/{r['n_out']})"


def _load_dets(dets_dir: Path, run: Path, name: str = "single_transit") -> list[Detection]:
    return adj._load(dets_dir, run, name)


def single_event_planets(run: Path) -> list[tuple]:
    """(planet row, its one observable label) for planets with one observable event, S/N >= 10."""
    from skywatch.planets import planet_rows

    events = _planet_events(run)
    out = []
    for row in planet_rows(run):
        labs = events.get(row.planet_id, [])
        if len(labs) == 1 and row.snr_stack >= STRONG:
            out.append((row, labs[0]))
    return out


def cmd_planets(args: argparse.Namespace) -> None:
    rec_dir, dets_dir = Path(args.records), Path(args.dets)
    print(
        "| run | star | host | MF S/N | depth % | event h | night n | clipped | raw (S/N, n_in/n_out) "
        "| residual (forced) | plain raw | path | found |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    n = found = bounded = 0
    for run in map(Path, args.runs):
        pl = single_event_planets(run)
        if not pl:
            continue
        boxes, stars = _load_records(rec_dir, run)
        cadence_s = store.read_manifest(run)["config"]["cadence_s"]
        truth = store.read_truth(run)
        dets = _load_dets(dets_dir, run)
        matches = match_detections(dets, truth, cadence_s)
        grade = {m.label_id: m.grade for m in matches if m.grade in ("bounded", "touched")}
        ts = _star_times(run, sorted({lab.star_id for _, lab in pl}))
        var = {s.star_id: s.variability for s in store.read_stars(run)}
        for row, lab in pl:
            t = ts[lab.star_id]
            k = int(lab.params["t_mid"] // DAY_S)
            night = t[(t // DAY_S).astype(int) == k]
            half = lab.params["duration_s"] / 2
            lo, hi = lab.params["t_mid"] - half, lab.params["t_mid"] + half
            clipped = len(night) > 0 and (lo < night[0] or hi > night[-1])
            raw = _best(boxes, lab.star_id, ("raw",), lab.t_start, lab.t_end)
            res = _best(boxes, lab.star_id, ("det", "forced"), lab.t_start, lab.t_end)
            plain = max(
                (
                    r["snr_plain"]
                    for r in boxes
                    if r["sid"] == lab.star_id
                    and r["ctx"] == "raw"
                    and r["snr_plain"] is not None
                    and _overlaps(r, lab.t_start, lab.t_end)
                ),
                default=None,
            )
            g = grade.get(lab.label_id)
            st = stars[lab.star_id]
            path = (
                "raw<8"
                if st["raw_best"] < 8
                else ("detrend" if st["detrended"] else ("refused" if st["refused"] else "raw"))
            )
            n += 1
            found += g is not None
            bounded += g == "bounded"
            print(
                f"| {run.name} | {lab.star_id} | {var[lab.star_id]} | {row.snr_stack:.1f} | "
                f"{100 * row.depth:.2f} | {row.dur_h:.2f} | {len(night)} | "
                f"{'edge' if clipped else '-'} | {_fmt(raw)} | {_fmt(res)} | "
                f"{'-' if plain is None else f'{plain:.2f}'} | {path} | {g or 'missed'} |"
            )
    print(
        f"\n{n} single-event planets at S/N >= {STRONG:g}; bounded {bounded}, bounded or touched {found}"
    )


def _kinds_under(truth: list, sid: int, t0: int, t1: int) -> list[str]:
    return sorted(
        {
            lab.kind
            for lab in truth
            if lab.star_id in (sid, None) and lab.t_start < t1 and lab.t_end > t0
        }
        - {"pulsation"}
    )


def cmd_near_miss(args: argparse.Namespace) -> None:
    """Best box per (star, night, search) at S/N 6 to 8 with no transit label under it."""
    lo, hi = args.lo, args.hi
    rec_dir = Path(args.records)
    rows = []
    det_rows = []
    for run in map(Path, args.runs):
        boxes, _ = _load_records(rec_dir, run)
        truth = store.read_truth(run)
        cadence_s = store.read_manifest(run)["config"]["cadence_s"]
        best: dict[tuple, dict] = {}
        for r in boxes:
            key = (r["sid"], r["night"], r["ctx"] if r["ctx"] != "forced" else "det")
            if r["ctx"] in ("pre", "forced-pre", "post"):
                continue
            if key not in best or r["snr"] > best[key]["snr"]:
                best[key] = r
        for r in best.values():
            kinds = _kinds_under(truth, r["sid"], r["t0"], r["t1"])
            if lo <= r["snr"] < hi and "transit" not in kinds:
                rows.append((run.name, r, kinds))
        if args.dets:
            dets = _load_dets(Path(args.dets), run)
            matches = match_detections(dets, truth, cadence_s)
            kind = {t.label_id: t.kind for t in truth}
            by = {m.det_id: (kind[m.label_id] if m.label_id else "-") for m in matches}
            for d in dets:
                det_rows.append((run.name, d, by.get(d.det_id, "-")))
    frac = [r["n_in"] / r["n_night"] for _, r, _ in rows]
    print(f"{len(rows)} boxes at S/N {lo:g} to {hi:g} under no transit label")
    if rows:
        edge = sum(1 for _, r, _ in rows if r["touch"])
        print(
            f"touching a night's edge {edge}; n_in/n_night median {np.median(frac):.2f}, "
            f"over 0.3: {sum(f > 0.3 for f in frac)}, over 0.4: {sum(f > 0.4 for f in frac)}"
        )
        by_ctx: dict[str, int] = defaultdict(int)
        for _, r, _ in rows:
            by_ctx[r["ctx"]] += 1
        print("by search:", dict(by_ctx))
        under: dict[str, int] = defaultdict(int)
        for _, _, k in rows:
            under["+".join(k) or "nothing"] += 1
        print("truth under the box:", dict(sorted(under.items(), key=lambda kv: -kv[1])))
    if args.dets:
        print(f"\n{len(det_rows)} detections on these runs")
        for name, d, k in det_rows:
            print(f"  {name} star {d.star_id} S/N {d.score:.2f} matched {k}")


# -- tally with the single-event planets ------------------------------------------------------

COLS = ("se10 b", "se10 b|t", *adj.COLS)


def _se10_labels(run: Path) -> set[str]:
    return {lab.label_id for _, lab in single_event_planets(run)}


def cmd_tally(args: argparse.Namespace) -> None:
    single_dir = Path(args.single)
    bls_dir = Path(args.bls) if args.bls else None
    labels = ("single_transit",) + (("union",) if bls_dir else ())
    for label in labels:
        print(f"== {label}")
        print(f"{'run':28s} " + " ".join(f"{c:>8s}" for c in COLS))
        total = [0] * len(COLS)
        for run in map(Path, args.runs):
            parts = [
                (
                    "single_transit",
                    adj._stats(run, _load_dets(single_dir, run), "single_transit"),
                )
            ]
            if label == "union" and bls_dir:
                parts.append(("bls", adj._stats(run, _load_dets(bls_dir, run, "bls"), "bls")))
            se = _se10_labels(run)
            bounded = set().union(*(s["bounded"] for _, s in parts))
            touched = set().union(*(s["touched"] for _, s in parts))
            row = (len(se & bounded), len(se & touched), *adj._row(parts))
            total = [a + b for a, b in zip(total, row, strict=True)]
            if args.per_run:
                print(f"{run.name:28s} " + " ".join(f"{v:8d}" for v in row))
        print(f"{'TOTAL':28s} " + " ".join(f"{v:8d}" for v in total))
    if args.detections_per_run:
        print("== detections per run (single_transit)")
        for run in map(Path, args.runs):
            print(f"{run.name} {len(_load_dets(single_dir, run))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("records")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--floor", type=float, default=FLOOR)
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_records)
    p = sub.add_parser("detect")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--set", action="append", default=[], help="constructor parameter, name=json")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_detect)
    p = sub.add_parser("planets")
    p.add_argument("--records", required=True)
    p.add_argument("--dets", required=True)
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_planets)
    p = sub.add_parser("near-miss")
    p.add_argument("--records", required=True)
    p.add_argument("--dets")
    p.add_argument("--lo", type=float, default=6.0)
    p.add_argument("--hi", type=float, default=8.0)
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_near_miss)
    p = sub.add_parser("tally")
    p.add_argument("--single", required=True)
    p.add_argument("--bls")
    p.add_argument("--per-run", action="store_true")
    p.add_argument("--detections-per-run", action="store_true")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_tally)
    args = ap.parse_args()
    args.fn(args)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
