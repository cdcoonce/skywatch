"""Per-detector, per-kind scores, confusion table, and a text scoreboard."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from skywatch.scoring.match import Match
from skywatch.types import Detection, TruthLabel

UNKNOWN = "unknown"
ALL = "all"
MISSED = "missed"


@dataclass(frozen=True)
class KindScore:
    detector: str
    kind: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _row(detector: str, kind: str, tp: int, fp: int, fn: int) -> KindScore:
    p = _div(tp, tp + fp)
    r = _div(tp, tp + fn)
    return KindScore(detector, kind, tp, fp, fn, p, r, _div(2 * p * r, p + r))


def _detectors(matches: Sequence[Match], detections: Sequence[Detection]) -> list[str]:
    return sorted({m.detector for m in matches} | {d.detector for d in detections})


def score(
    matches: Sequence[Match],
    detections: Sequence[Detection],
    truth: Sequence[TruthLabel],
) -> list[KindScore]:
    guess_of = {(d.detector, d.det_id): d.kind_guess or UNKNOWN for d in detections}
    rows: list[KindScore] = []
    for detector in _detectors(matches, detections):
        mine = [m for m in matches if m.detector == detector]
        matched = {m.label_id for m in mine if m.label_id is not None}
        tp: Counter[str] = Counter()
        fn: Counter[str] = Counter()
        fp: Counter[str] = Counter()
        for t in truth:
            (tp if t.label_id in matched else fn)[t.kind] += 1
        for m in mine:
            if m.label_id is None:
                fp[guess_of.get((detector, m.det_id), UNKNOWN)] += 1
        kinds = set(tp) | set(fn) | set(fp)
        for kind in sorted(kinds):
            rows.append(_row(detector, kind, tp[kind], fp[kind], fn[kind]))
        rows.append(_row(detector, ALL, sum(tp.values()), sum(fp.values()), sum(fn.values())))
    rows.sort(key=lambda r: (r.detector, r.kind))
    return rows


def confusion(
    matches: Sequence[Match],
    detections: Sequence[Detection],
    truth: Sequence[TruthLabel],
) -> dict[tuple[str, str, str], int]:
    kind_of = {t.label_id: t.kind for t in truth}
    guess_of = {(d.detector, d.det_id): d.kind_guess or UNKNOWN for d in detections}
    out: Counter[tuple[str, str, str]] = Counter()
    for detector in _detectors(matches, detections):
        matched: set[str] = set()
        for m in matches:
            if m.detector != detector or m.label_id is None:
                continue
            matched.add(m.label_id)
            out[(detector, kind_of[m.label_id], guess_of.get((detector, m.det_id), UNKNOWN))] += 1
        for t in truth:
            if t.label_id not in matched:
                out[(detector, t.kind, MISSED)] += 1
    return dict(out)


def format_scoreboard(scores: Sequence[KindScore]) -> str:
    header = ("detector", "kind", "tp", "fp", "fn", "P", "R", "F1")
    body = [
        (
            s.detector,
            s.kind,
            str(s.tp),
            str(s.fp),
            str(s.fn),
            f"{s.precision:.3f}",
            f"{s.recall:.3f}",
            f"{s.f1:.3f}",
        )
        for s in scores
    ]
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]

    def fmt(r: tuple[str, ...]) -> str:
        cells = [c.ljust(widths[i]) if i < 2 else c.rjust(widths[i]) for i, c in enumerate(r)]
        return "  ".join(cells).rstrip()

    return "\n".join([fmt(header), "  ".join("-" * w for w in widths), *map(fmt, body)])
