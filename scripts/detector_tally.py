"""One detector's headline counts on stored runs, the table every `bls` change quotes.

Usage::

    uv run python scripts/detector_tally.py [--detector bls] runs/seed61-200s-20n [more runs ...]

Reads the stored detections, scores and confusion of each run (run ``skywatch detect`` and
``skywatch score`` first; a stale score is a stale table) and prints one line per run and a
total: transit true positives (``tp``, bounded and touched), false positives of every kind,
and the detector's transit guesses on eclipses, pulsations and flares, and the number of
detections. This is the instrument behind the "main against branch" tables of the `bls` shape
changes (#36, #41); it reads no light curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from skywatch.io import store

COLUMNS = ("transit_tp", "fp", "eclipse>transit", "pulsation>transit", "flare>transit", "dets")


def tally(run_dir: Path, detector: str) -> tuple[int, ...]:
    scores = {s.kind: s for s in store.read_scores(run_dir) if s.detector == detector}
    conf = store.read_confusion(run_dir)
    dets = [d for d in store.read_detections(run_dir) if d.detector == detector]
    return (
        scores["transit"].tp if "transit" in scores else 0,
        scores["all"].fp if "all" in scores else 0,
        conf.get((detector, "eclipse", "transit"), 0),
        conf.get((detector, "pulsation", "transit"), 0),
        conf.get((detector, "flare", "transit"), 0),
        len(dets),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="bls")
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    total = [0] * len(COLUMNS)
    print(f"{'run':24s} " + " ".join(f"{c:>17s}" for c in COLUMNS))
    for run in args.runs:
        row = tally(run, args.detector)
        total = [a + b for a, b in zip(total, row, strict=True)]
        print(f"{run.name:24s} " + " ".join(f"{v:17d}" for v in row))
    print(f"{'TOTAL':24s} " + " ".join(f"{v:17d}" for v in total))


if __name__ == "__main__":
    main()
