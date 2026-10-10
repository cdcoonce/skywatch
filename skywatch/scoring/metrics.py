"""Per-detector, per-kind scores, confusion table, and a text scoreboard."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from skywatch.scoring.match import Match
from skywatch.types import ORIGIN_OF_KIND, Detection, TruthLabel

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
    tp_bounded: int = 0
    tp_touched: int = 0
    corroborating: int = 0
    bounded_rate: float = 0.0
    unobservable: int = 0
    """Labels whose event changed zero samples: excluded from fn and from recall."""


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _row(
    detector: str,
    kind: str,
    tp_bounded: int,
    tp_touched: int,
    corr: int,
    fp: int,
    fn: int,
    unobservable: int = 0,
) -> KindScore:
    tp = tp_bounded + tp_touched
    p = _div(tp, tp + fp)
    r = _div(tp, tp + fn)
    return KindScore(
        detector,
        kind,
        tp,
        fp,
        fn,
        p,
        r,
        _div(2 * p * r, p + r),
        tp_bounded,
        tp_touched,
        corr,
        _div(tp_bounded, tp),
        unobservable,
    )


def _detectors(
    matches: Sequence[Match],
    detections: Sequence[Detection],
    detectors: Sequence[str] | None = None,
) -> list[str]:
    found = {m.detector for m in matches} | {d.detector for d in detections}
    return sorted(found | set(detectors or ()))


def score(
    matches: Sequence[Match],
    detections: Sequence[Detection],
    truth: Sequence[TruthLabel],
    detectors: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
) -> list[KindScore]:
    """Per-detector, per-kind rows, ordered by detector, then kind, with "all" last.

    `detectors` lists detectors that must get rows even with zero detections (all
    truth counts as FN); None derives them from matches/detections. `kinds` fixes
    the kind rows; None uses every kind in ORIGIN_OF_KIND (plus "unknown" and any
    other kind actually seen) so the table is stable across runs.

    A label with ``footprint_samples == 0`` is counted only in ``unobservable``: it is not
    a TP, an FN, or part of recall's denominator, whatever the matches say.
    """
    guess_of = {(d.detector, d.det_id): d.kind_guess or UNKNOWN for d in detections}
    rows: list[KindScore] = []
    all_kinds = set(ORIGIN_OF_KIND if kinds is None else kinds)
    for detector in _detectors(matches, detections, detectors):
        mine = [m for m in matches if m.detector == detector]
        grade_of = {m.label_id: m.grade for m in mine if m.grade in ("bounded", "touched")}
        tpb: Counter[str] = Counter()
        tpt: Counter[str] = Counter()
        fn: Counter[str] = Counter()
        fp: Counter[str] = Counter()
        corr: Counter[str] = Counter()
        unobs: Counter[str] = Counter()
        kind_of = {t.label_id: t.kind for t in truth}
        for t in truth:
            if not t.observable:
                unobs[t.kind] += 1
                continue
            g = grade_of.get(t.label_id)
            if g == "bounded":
                tpb[t.kind] += 1
            elif g == "touched":
                tpt[t.kind] += 1
            else:
                fn[t.kind] += 1
        for m in mine:
            if m.label_id is None:
                fp[guess_of.get((detector, m.det_id), UNKNOWN)] += 1
            elif m.grade == "corroborating" and m.label_id in kind_of:
                corr[kind_of[m.label_id]] += 1
        seen = all_kinds | set(tpb) | set(tpt) | set(fn) | set(fp) | set(corr) | set(unobs)
        for kind in sorted(seen):
            rows.append(
                _row(
                    detector,
                    kind,
                    tpb[kind],
                    tpt[kind],
                    corr[kind],
                    fp[kind],
                    fn[kind],
                    unobs[kind],
                )
            )
        rows.append(
            _row(
                detector,
                ALL,
                sum(tpb.values()),
                sum(tpt.values()),
                sum(corr.values()),
                sum(fp.values()),
                sum(fn.values()),
                sum(unobs.values()),
            )
        )
    rows.sort(key=lambda r: (r.detector, r.kind == ALL, r.kind))
    return rows


def confusion(
    matches: Sequence[Match],
    detections: Sequence[Detection],
    truth: Sequence[TruthLabel],
    detectors: Sequence[str] | None = None,
) -> dict[tuple[str, str, str], int]:
    truth = [t for t in truth if t.observable]  # an unobservable label is not "missed"
    kind_of = {t.label_id: t.kind for t in truth}
    guess_of = {(d.detector, d.det_id): d.kind_guess or UNKNOWN for d in detections}
    out: Counter[tuple[str, str, str]] = Counter()
    for detector in _detectors(matches, detections, detectors):
        matched: set[str] = set()
        for m in matches:
            if m.detector != detector or m.label_id is None or m.grade == "corroborating":
                continue
            matched.add(m.label_id)
            out[(detector, kind_of[m.label_id], guess_of.get((detector, m.det_id), UNKNOWN))] += 1
        for t in truth:
            if t.label_id not in matched:
                out[(detector, t.kind, MISSED)] += 1
    return dict(out)


def format_scoreboard(scores: Sequence[KindScore]) -> str:
    header = (
        "detector",
        "kind",
        "tpB",
        "tpT",
        "corr",
        "fp",
        "fn",
        "unobs",
        "P",
        "R",
        "F1",
        "bounded",
    )
    body = [
        (
            s.detector,
            s.kind,
            str(s.tp_bounded),
            str(s.tp_touched),
            str(s.corroborating),
            str(s.fp),
            str(s.fn),
            str(s.unobservable),
            f"{s.precision:.3f}",
            f"{s.recall:.3f}",
            f"{s.f1:.3f}",
            f"{s.bounded_rate:.3f}",
        )
        for s in scores
    ]
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]

    def fmt(r: tuple[str, ...]) -> str:
        cells = [c.ljust(widths[i]) if i < 2 else c.rjust(widths[i]) for i, c in enumerate(r)]
        return "  ".join(cells).rstrip()

    return "\n".join([fmt(header), "  ".join("-" * w for w in widths), *map(fmt, body)])
