"""Every star `bls` reports on a run: which path accepted it and what its gates read (issue #57).

Usage::

    uv run python scripts/bls_claims_audit.py [--dump DIR] runs/puls-pf-seed830-200s-20n [more ...]

``--dump DIR`` also writes the detections of each run as ``DIR/<run>.bls.json``, the format
``scripts/single_transit_adjacent_audit.py tally --bls DIR`` reads.

Detects each run with the shipped :class:`BlsDetector` and prints one line for each star that
carries a detection: the run, the star, its variability class, the path that accepted the box
(``first`` for the first pass, ``detrend`` for the search with the pulsation subtracted; the
second pass is off by default and is not counted), the box (period, duration, events, nights),
the S/N, the local S/N, the night-level sine ratio, the shape statistic, the repeat S/N (the box
over every night but its strongest), the number of samples of the strongest night, and the
matched truth kind of each event (the scorer's matcher; a measurement, the detector never reads
it). A final line counts the detections per run. It reads the light curves, the stars and the
truth of the run and writes nothing.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from skywatch.detectors.bls import BlsDetector, Candidate
from skywatch.io import store
from skywatch.scoring.match import match_detections

DAY_S = 86400.0


class _Audit(BlsDetector):
    """A `BlsDetector` that remembers the candidate and the path of every star it reports."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.sid = -1
        self.found: dict[int, tuple[str, Candidate]] = {}
        self._via = "first"

    def _detrended(self, a, c, cadence_s, plan_mask):  # type: ignore[no-untyped-def]
        out = super()._detrended(a, c, cadence_s, plan_mask)
        if out is not None:
            self._via = "detrend"
        return out

    def _star(self, star_id, c, common_mode, cadence_s):  # type: ignore[no-untyped-def]
        self.sid = star_id
        self._via = "first"
        return super()._star(star_id, c, common_mode, cadence_s)

    def _search(self, *args, **kw):  # type: ignore[no-untyped-def]
        cand = super()._search(*args, **kw)
        if cand is not None and cand.rejected is None and not args[3:] and not kw.get("masked"):
            self.found[self.sid] = (self._via, cand)
        return cand


def audit(run_dir: Path, dump: Path | None = None) -> list[str]:
    stars = {s.star_id: s for s in store.read_stars(run_dir)}
    truth = store.read_truth(run_dir)
    kind = {t.label_id: t.kind for t in truth}
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    curves = store.read_curves(run_dir)
    det = _Audit()
    dets = det.detect(curves, cadence_s)
    if dump is not None:  # the format of `single_transit_adjacent_audit.py detect`
        fields = ("detector", "det_id", "star_id", "t_start", "t_end", "score", "kind_guess")
        rows = [{f: getattr(d, f) for f in fields} for d in dets]
        (dump / f"{run_dir.name}.bls.json").write_text(json.dumps(rows))
    matches = {m.det_id: m for m in match_detections(dets, truth, cadence_s)}
    lines = []
    for sid, (via, cand) in sorted(det.found.items()):
        mine = [d for d in dets if d.star_id == sid]
        c = det._clean(curves[sid], cadence_s)
        assert c is not None
        night = c.night_of
        kinds = Counter(
            kind[matches[d.det_id].label_id] if matches[d.det_id].label_id else "-" for d in mine
        )
        star = stars[sid]
        strong = (
            int(np.bincount(night)[np.unique([n for _, _, n in cand.events])].min())
            if cand.events
            else 0
        )
        lines.append(
            f"{run_dir.name} star {sid:3d} {star.variability[:5]:5s} {via:7s} "
            f"P {cand.box.period_s / DAY_S:6.3f} d dur {cand.box.duration_s / 3600:3.1f} h "
            f"events {len(cand.events):2d} S/N {cand.snr:5.1f} local {cand.snr_local:5.1f} "
            f"ratio {cand.sine_ratio if cand.sine_ratio is not None else float('nan'):4.2f} "
            f"shape {cand.shape if cand.shape is not None else float('nan'):5.2f} "
            f"repeat {cand.snr_repeat:5.2f} short night {strong:4d} | "
            + " ".join(f"{k}:{n}" for k, n in sorted(kinds.items()))
        )
    lines.append(f"{run_dir.name}: {len(dets)} detections on {len(det.found)} stars")
    return lines


if __name__ == "__main__":
    args = sys.argv[1:]
    dump_dir = None
    if args[:1] == ["--dump"]:
        dump_dir = Path(args[1])
        dump_dir.mkdir(parents=True, exist_ok=True)
        args = args[2:]
    for path in args:
        for line in audit(Path(path), dump_dir):
            print(line, flush=True)
