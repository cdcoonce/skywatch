"""What subtracting a significant sinusoid before the first search does for `single_transit` (issue #69).

Usage::

    # one JSON-lines file per run: every star whose strongest sinusoid is significant, on the
    # detector's own path (a real `detect`, the field's common mode included)
    uv run python scripts/single_transit_presub_audit.py records --out scratch/rec runs/...

    # the detections of two directories (main, the branch) on the stars without a significant
    # sinusoid, written filtered so that `cmp` can compare them
    uv run python scripts/single_transit_presub_audit.py identical --a scratch/main --b scratch/br \\
        --out scratch/ident runs/...

    # the measurement table of issue #69 from those records
    uv run python scripts/single_transit_presub_audit.py table --records scratch/rec runs/...

``records`` runs a real ``detect`` with a subclass of :class:`SingleTransitDetector` whose
``_star_boxes`` is wrapped. For each star it takes the arrays the real ``_prepare`` returns (the
field's common mode included; a replay of ``_prepare`` without it is not the detector's path),
reads the strongest sinusoid of ``bls`` with the guard's own test (``peak`` is the strongest
Fourier peak of the guard band over the false-alarm level, ``ratio`` the fitted sinusoid's power
over it, the multiple ``BlsDetector._pulsation_trend`` compares with ``trend_min_ratio``), and, when
the peak is at least 1, records the best box of every night on the raw flux and, when the ratio
reaches the bar the first-pass subtraction uses for the star, the best box of every night on the
detector's own ``_residual`` (the subtraction with its refit round the best box) and the CPU time of
that call. The detector is built with ``subtract_first=False``, so what it reports and the path
spies record are those of ``main``. Nothing a record holds is read by the detector.

``table`` joins the records to the truth of each run: for every observable transit label on a
star with a significant sinusoid, the matched-filter S/N of the event (``planets._event_snr``),
the best raw and subtracted box that overlaps it, and, for every subtracted box at the detector's
bar, what truth label lies under it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from skywatch.io import store
from skywatch.planets import _event_snr

DAY_S = 86400.0
MIN_PEAK = (
    1.0  # the guard's own false-alarm level: a star whose Fourier peak reaches it is recorded
)


def _sinusoid(bls, t, flux, night, cadence_s):  # type: ignore[no-untyped-def]
    """``(peak, ratio)`` of the star's strongest sinusoid, both over the false-alarm level.

    ``peak`` is the strongest Fourier peak of the guard band (the guard's own test; the detector's
    significance test of the first-pass subtraction), ``ratio`` the fitted sinusoid's power, the
    multiple ``BlsDetector._pulsation_trend`` compares with ``trend_min_ratio``.
    """
    from skywatch.detectors.bls import _demean_by_night

    if len(t) < 4:
        return 0.0, 0.0
    sigma = bls._white_sigma(flux, night)
    if not sigma > 0:
        return 0.0, 0.0
    td = (t - float(t[0])) / DAY_S
    k = np.unique(night, return_inverse=True)[1]
    x = _demean_by_night(flux - 1.0, k)
    spectrum = bls._band_power(t, x, sigma, cadence_s)
    if spectrum is None:
        return 0.0, 0.0
    peak = float(spectrum[1].max()) / spectrum[2]
    fit = bls._strongest_sinusoid(t, td, x, k, sigma, cadence_s, float(td[-1]))
    if fit is None:
        return peak, 0.0
    gain, _, _, _, level = fit
    return peak, gain / (2.0 * sigma**2) / level


def _night_boxes(det, t, flux, night, cadence_s):  # type: ignore[no-untyped-def]
    return [
        [b.night, b.snr, b.t_start, b.t_end + cadence_s, b.n_in, b.n_out]
        for b in det._boxes(t, flux, night, cadence_s, floor=0.0)
    ]


def _recording_detector():  # type: ignore[no-untyped-def]
    from skywatch.detectors.single_transit import SingleTransitDetector

    stars: list[dict] = []

    class Recording(SingleTransitDetector):
        # spies on the star the real `_star_boxes` is reading: which gate looked, and what it said
        def __init__(self, **kw: object) -> None:
            super().__init__(**kw)  # type: ignore[arg-type]
            self.path: dict = {}
            self.inside = False

        def _pulsating(self, *a, **kw):  # type: ignore[no-untyped-def]
            r = super()._pulsating(*a, **kw)
            self.path.setdefault("guard_det" if self.inside else "guard_raw", []).append(r)
            return r

        def _residual_pulsating(self, *a, **kw):  # type: ignore[no-untyped-def]
            r = super()._residual_pulsating(*a, **kw)
            self.path.setdefault("guard_resid", []).append(r)
            return r

        def _stands_unflared(self, *a, **kw):  # type: ignore[no-untyped-def]
            r = super()._stands_unflared(*a, **kw)
            self.path.setdefault("flare_gate", []).append(r)
            return r

        def _detrended(self, *a, **kw):  # type: ignore[no-untyped-def]
            self.inside = True
            self.path["detrended"] = True
            try:
                r = super()._detrended(*a, **kw)
            finally:
                self.inside = False
            self.path["det_found"] = r is not None
            return r

        def _star_boxes(self, star_id, c, common_mode, cadence_s):  # type: ignore[no-untyped-def]
            self.path = {}
            out = super()._star_boxes(star_id, c, common_mode, cadence_s)
            path = self.path
            t, flux, night, level = self._bls._prepare(c, common_mode)
            peak, ratio = _sinusoid(self._bls, t, flux, night, cadence_s)
            if peak < MIN_PEAK:
                return out
            rec = {
                "sid": star_id,
                "peak": peak,
                "ratio": ratio,
                "raw": _night_boxes(self, t, flux, night, cadence_s),
                "reported": [[b.snr, b.t_start, b.t_end + cadence_s] for b in out[0]],
                "path": path,
                "sub": None,
            }
            first = self._first_ratio(c, t)
            rec["band"] = "3+" if ratio >= self._bls.trend_min_ratio else "under 3"
            if ratio >= first:
                t0 = time.process_time()
                resid = self._residual(c, t, flux, night, level, cadence_s, first)
                rec["cpu_residual_s"] = time.process_time() - t0
                if resid is not None:
                    t0 = time.process_time()
                    rec["sub"] = _night_boxes(self, t, resid, night, cadence_s)
                    rec["cpu_search_s"] = time.process_time() - t0
            stars.append(rec)
            return out

    return Recording(subtract_first=False), stars


def _records_one(args: tuple[str, str]) -> str:
    run, out = args
    run_dir = Path(run)
    curves = store.read_curves(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    det, stars = _recording_detector()
    t0 = time.process_time()
    dets = det.detect(curves, cadence_s)
    wall = time.process_time() - t0
    var = {s.star_id: s.variability for s in store.read_stars(run_dir)}
    for s in stars:
        s["var"] = var[s["sid"]]
    dest = Path(out) / f"{run_dir.name}.json"
    dest.write_text(json.dumps({"wall_s": wall, "n_dets": len(dets), "stars": stars}))
    return f"{run_dir.name} stars {len(stars)} detections {len(dets)} wall {wall:.1f}s"


def cmd_records(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(r), str(out)) for r in args.runs]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            for line in pool.map(_records_one, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(_records_one(job), flush=True)


# -- the table -------------------------------------------------------------------------------


def _times(run: Path, star_ids: list[int]) -> dict[int, np.ndarray]:
    if not star_ids:
        return {}
    tbl = pq.read_table(
        run / "observations.parquet",
        columns=["star_id", "t"],
        filters=[("star_id", "in", star_ids)],
    )
    sid, t = tbl["star_id"].to_numpy(), tbl["t"].to_numpy().astype(np.float64)
    return {s: np.sort(t[sid == s]) for s in star_ids}


def _best(boxes: list[list], lo: float, hi: float) -> float | None:
    s = [b[1] for b in boxes if b[2] < hi and b[3] > lo]
    return max(s, default=None)


def _cause(path: dict, raw_best: float, bar: float, near: float) -> str:
    """Where the unchanged path stopped a star whose subtracted box reads at the bar."""
    if "detrended" in path:
        if path.get("det_found"):
            return "detrended and reported other boxes (this event's night not among them)"
        if path.get("flare_gate") and not any(path["flare_gate"]):
            return "detrended: every box refused by the flare gate"
        if path.get("guard_resid") and all(path["guard_resid"]):
            return "detrended: guard (a) refused the residual"
        return "detrended: no box at the bar on the residual in the path"
    if raw_best < near:
        return f"never detrended: the star's raw best box reads under {near:g}"
    if raw_best < bar:
        return f"never detrended: raw best box {near:g} to {bar:g} and guard (a) accepts it"
    if path.get("guard_raw") and not path["guard_raw"][0]:
        return "never detrended: raw box at the bar and guard (a) accepts it"
    return "other"


def _pct(num: int, den: int) -> str:
    return f"{num} ({100 * num / den:.0f}%)" if den else f"{num}"


def cmd_table(args: argparse.Namespace) -> None:
    rec_dir = Path(args.records)
    bar, near = args.bar, args.near
    ev: list[dict] = []  # one row per observable transit label on a recorded star
    boxes_by_kind: dict[str, Counter] = {"raw": Counter(), "sub": Counter()}
    n_stars = Counter()
    cpu_res = cpu_search = wall = 0.0
    n_sig = 0
    n_band: Counter = Counter()
    rat: list[float] = []
    best_raw: list[float] = []
    best_sub: list[float] = []
    for run in map(Path, args.runs):
        data = json.loads((rec_dir / f"{run.name}.json").read_text())
        wall += data["wall_s"]
        stars = {s["sid"]: s for s in data["stars"]}
        field = "all-sinusoidal" if run.name.startswith("puls") else "standard"
        sigma = {
            s.star_id: math.sqrt(s.noise_floor**2 + 1.0 / s.flux0) for s in store.read_stars(run)
        }
        truth = store.read_truth(run)
        under: dict[int, list] = defaultdict(list)
        for lab in truth:
            if lab.star_id in stars and lab.kind != "pulsation":
                under[lab.star_id].append(lab)
        ts = _times(run, sorted({lab.star_id for lab in truth if lab.star_id in stars}))
        for sid, s in stars.items():
            rat.append(s["ratio"])
            n_band[s["band"]] += 1
            if s["sub"] is not None:
                best_raw.append(max((b[1] for b in s["raw"]), default=0.0))
                best_sub.append(max((b[1] for b in s["sub"]), default=0.0))
            n_stars[field, "peak>=1"] += 1
            if s["sub"] is not None:
                n_stars[field, "subtracted"] += 1
                cpu_res += s["cpu_residual_s"]
                cpu_search += s["cpu_search_s"]
                n_sig += 1
            for kind in ("raw", "sub"):
                for b in s[kind] or []:
                    if b[1] < bar:
                        continue
                    kinds = sorted(
                        {lab.kind for lab in under[sid] if lab.t_start < b[3] and lab.t_end > b[2]}
                    )
                    boxes_by_kind[kind][field, "+".join(kinds) or "nothing"] += 1
        for lab in truth:
            if lab.kind != "transit" or lab.star_id not in stars or not lab.observable:
                continue
            s = stars[lab.star_id]
            if s["sub"] is None:
                continue
            prm = lab.params
            if prm.get("footprint_samples") == 0:
                continue
            truth_snr = _event_snr(
                ts[lab.star_id], prm["depth"], prm["duration_s"], prm["t_mid"], sigma[lab.star_id]
            )
            ev.append(
                {
                    "band": s["band"],
                    "path": s["path"],
                    "raw_best": max((b[1] for b in s["raw"]), default=0.0),
                    "field": field,
                    "truth": truth_snr,
                    "raw": _best(s["raw"], lab.t_start, lab.t_end),
                    "sub": _best(s["sub"], lab.t_start, lab.t_end),
                    "rep": any(r[1] < lab.t_end and r[2] > lab.t_start for r in s["reported"]),
                }
            )
    print(
        f"stars whose strongest Fourier peak reaches the false-alarm level: {sum(v for (f, k), v in n_stars.items() if k == 'peak>=1')}"
    )
    print(
        f"  of them with a fitted sinusoid the first-pass subtraction takes (ratio >= its bar): {n_sig}; ratio median {np.median(rat):.1f}"
    )
    for f in ("standard", "all-sinusoidal"):
        print(f"  {f}: {n_stars[f, 'peak>=1']} / {n_stars[f, 'subtracted']}")
    print(
        f"  best-night S/N of a subtracted star, raw / subtracted: median {np.median(best_raw):.1f} / "
        f"{np.median(best_sub):.1f}; at least {bar:g}: {sum(x >= bar for x in best_raw)} / "
        f"{sum(x >= bar for x in best_sub)}; at least {near:g}: {sum(x >= near for x in best_raw)} / "
        f"{sum(x >= near for x in best_sub)}"
    )
    print()
    for lo in (5.0, 8.0, 10.0):
        for f in ("all-sinusoidal", "standard"):
            rows = [e for e in ev if e["field"] == f and e["truth"] >= lo]
            if not rows:
                continue
            raw = [e["raw"] or 0.0 for e in rows]
            sub = [e["sub"] or 0.0 for e in rows]
            ratio_raw = [r / e["truth"] for r, e in zip(raw, rows, strict=True)]
            ratio_sub = [r / e["truth"] for r, e in zip(sub, rows, strict=True)]
            n = len(rows)
            print(
                f"{f}, events with truth S/N >= {lo:g}: {n}; raw box >= {near:g}: "
                f"{_pct(sum(r >= near for r in raw), n)}, >= {bar:g}: {_pct(sum(r >= bar for r in raw), n)}"
                f"; subtracted >= {near:g}: {_pct(sum(r >= near for r in sub), n)}, "
                f">= {bar:g}: {_pct(sum(r >= bar for r in sub), n)}; "
                f"median read/truth raw {np.median(ratio_raw):.2f}, subtracted {np.median(ratio_sub):.2f}; "
                f"reported by the unchanged path {_pct(sum(e['rep'] for e in rows), n)}; "
                f"subtracted >= {bar:g} and not reported "
                f"{sum(r >= bar and not e['rep'] for r, e in zip(sub, rows, strict=True))}, "
                f"reported and subtracted < {bar:g} "
                f"{sum(r < bar and e['rep'] for r, e in zip(sub, rows, strict=True))}"
            )
    print("\nby the strongest sinusoid's band (events with truth S/N >= 8, both fields):")
    for band in ("under 3", "3+"):
        rows = [e for e in ev if e["band"] == band and e["truth"] >= 8]
        n = len(rows)
        print(
            f"  ratio {band}: stars {n_band[band]}, events {n}; "
            f"raw box >= {bar:g} {sum((e['raw'] or 0.0) >= bar for e in rows)}, subtracted >= {bar:g} "
            f"{sum((e['sub'] or 0.0) >= bar for e in rows)}, reported by the unchanged path "
            f"{sum(e['rep'] for e in rows)}"
        )
    print(
        f"\nwhy an event whose subtracted box reaches {bar:g} is not reported today (truth S/N >= 8):"
    )
    why: Counter = Counter()
    for e in ev:
        if e["truth"] < 8 or e["rep"] or not (e["sub"] or 0.0) >= bar:
            continue
        why[e["field"], _cause(e["path"], e["raw_best"], bar, near)] += 1
    for k, v in sorted(why.items(), key=lambda kv: -kv[1]):
        print(f"  {k[0]}: {k[1]}: {v}")
    print(f"\nboxes at S/N >= {bar:g} by the truth under them (star-night best boxes):")
    for kind in ("raw", "sub"):
        print(f"  {kind}: {dict(sorted(boxes_by_kind[kind].items(), key=lambda kv: -kv[1]))}")
    print(
        f"\nCPU of subtracting for every such star ({n_sig} stars): _residual {cpu_res:.1f} s "
        f"({1000 * cpu_res / max(1, n_sig):.0f} ms a star) + the search of the residual "
        f"{cpu_search:.1f} s; the runs' detect wall time {wall:.1f} s"
    )


def _quiet_stars(run: str) -> tuple[str, list[int]]:
    """The stars of `run` whose strongest sinusoid is under the guard's own level (ratio < 1)."""
    from skywatch.detectors.single_transit import SingleTransitDetector

    run_dir = Path(run)
    curves = store.read_curves(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    det = SingleTransitDetector()
    bls = det._bls
    cleaned = {}
    for sid in sorted(curves):
        c = bls._clean(curves[sid], cadence_s)
        if c is not None:
            cleaned[sid] = c
    common_mode = bls._common_mode(cleaned, cadence_s)
    quiet = []
    for sid, c in cleaned.items():
        t, flux, night, _ = bls._prepare(c, common_mode)
        if _sinusoid(bls, t, flux, night, cadence_s)[0] < MIN_PEAK:
            quiet.append(sid)
    return run_dir.name, quiet


def cmd_identical(args: argparse.Namespace) -> None:
    """Per run, the detections of two directories on the stars with no significant sinusoid.

    Writes each side's filtered detections to ``--out/{a,b}/<run>.json`` (sorted, canonical JSON)
    so that ``cmp`` can compare them, and prints whether they are equal.
    """
    out = Path(args.out)
    for side in ("a", "b"):
        (out / side).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        quiet = dict(pool.map(_quiet_stars, args.runs))
    n_equal = 0
    for run in map(Path, args.runs):
        kept = set(quiet[run.name])
        same = True
        for side, d in (("a", Path(args.a)), ("b", Path(args.b))):
            rows = json.loads((d / f"{run.name}.single_transit.json").read_text())
            rows = [r for r in rows if r["star_id"] in kept]
            (out / side / f"{run.name}.json").write_text(json.dumps(rows, sort_keys=True))
        same = (out / "a" / f"{run.name}.json").read_bytes() == (
            out / "b" / f"{run.name}.json"
        ).read_bytes()
        n_equal += same
        print(f"{run.name}: {len(kept)} stars without a significant sinusoid; identical {same}")
    print(f"identical on {n_equal} of {len(args.runs)} runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("records")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_records)
    p = sub.add_parser("identical")
    p.add_argument("--a", required=True, help="directory of <run>.single_transit.json (main)")
    p.add_argument("--b", required=True, help="the same for the branch")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_identical)
    p = sub.add_parser("table")
    p.add_argument("--records", required=True)
    p.add_argument("--bar", type=float, default=8.0)
    p.add_argument("--near", type=float, default=5.0)
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_table)
    args = ap.parse_args()
    args.fn(args)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
