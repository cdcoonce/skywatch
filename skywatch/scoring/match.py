"""Greedy one-to-one matching of detections to truth labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from skywatch.types import Detection, TruthLabel

Grade = Literal["bounded", "touched", "corroborating"]


@dataclass(frozen=True)
class Match:
    detector: str
    det_id: str
    label_id: str | None
    iou: float
    grade: Grade | None = None


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection over union of two half-open integer spans."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _eligible(
    det: Detection, label: TruthLabel, cadence_s: int, iou_min: float
) -> tuple[float, Grade] | None:
    """Return (IoU, grade) if `det` may match `label`, else None.

    Eligible means same star (or field-wide label) and any span overlap, or the
    single-cadence tolerance. Grade is "bounded" at IoU >= iou_min or under the
    tolerance, else "touched".
    """
    if label.star_id is not None and label.star_id != det.star_id:
        return None
    iou = span_iou(det.t_start, det.t_end, label.t_start, label.t_end)
    if iou >= iou_min:
        return iou, "bounded"
    single = label.t_end - label.t_start == cadence_s
    if single and det.t_start < label.t_end + cadence_s and det.t_end > label.t_start - cadence_s:
        return iou, "bounded"
    if iou > 0:
        return iou, "touched"
    return None


def _edge_gap(det: Detection, label: TruthLabel) -> int | None:
    """Distance in seconds between two non-overlapping half-open spans, else None.

    0 means the spans abut: the detection starts at the label's `t_end` (the first
    sample after it) or ends at the label's `t_start`. Same star (or field-wide) only.
    """
    if label.star_id is not None and label.star_id != det.star_id:
        return None
    if det.t_start >= label.t_end:
        return det.t_start - label.t_end
    if det.t_end <= label.t_start:
        return label.t_start - det.t_end
    return None


def _overlaps_any(det: Detection, truth: Sequence[TruthLabel]) -> bool:
    """True if `det` overlaps any label on its star (or any field-wide label)."""
    return any(
        (label.star_id is None or label.star_id == det.star_id)
        and det.t_start < label.t_end
        and det.t_end > label.t_start
        for label in truth
    )


def _edge_pass(
    ordered: Sequence[Detection],
    results: list[Match],
    truth: Sequence[TruthLabel],
    taken: set[str],
    limit_s: int,
) -> None:
    """Second pass: give still-unmatched detections an edge match, in place.

    Runs only on detections the overlap pass left unmatched, so it can never change
    an overlap or single-cadence outcome or take a label an overlapping detection
    wanted. A detection that overlaps any label (claimed or not) is skipped: the overlap
    pass has already matched it or, for a claimed label, made it corroborating. A
    detection that overlaps no label but lies within `limit_s` seconds of one's edge
    takes the nearest unclaimed such label as "touched" (ties by label_id), else is
    "corroborating" for the nearest already-claimed one, else stays unmatched. Any
    unclaimed label is preferred over a nearer claimed one.
    """
    for i, (det, res) in enumerate(zip(ordered, results, strict=True)):
        # Defence in depth: the overlap pass now settles every overlapping detection, so
        # this guard cannot change an outcome and no test can make it fire; it keeps the
        # edge pass safe if a later change to the overlap pass lets an overlapper through.
        if res.label_id is not None or _overlaps_any(det, truth):
            continue
        free: tuple[int, str] | None = None
        claimed: tuple[int, str] | None = None
        for label in truth:
            gap = _edge_gap(det, label)
            if gap is None or gap >= limit_s:
                continue
            cand = (gap, label.label_id)
            if label.label_id in taken:
                if claimed is None or cand < claimed:
                    claimed = cand
            elif free is None or cand < free:
                free = cand
        if free is not None:
            taken.add(free[1])
            results[i] = Match(det.detector, det.det_id, free[1], 0.0, "touched")
        elif claimed is not None:
            results[i] = Match(det.detector, det.det_id, claimed[1], 0.0, "corroborating")


def match_detections(
    detections: Sequence[Detection],
    truth: Sequence[TruthLabel],
    cadence_s: int,
    iou_min: float = 0.3,
    edge_tolerance_cadences: int = 1,
) -> list[Match]:
    """Match each detector's detections to truth independently.

    Greedy by score desc. A detection takes the best unclaimed bounded label, else
    the best unclaimed touched label, else corroborates a field-wide label this detector
    already claimed, else corroborates the best-IoU already-claimed star-level label it
    overlaps (a fragment of an event it already found), else is unmatched. Star-level
    labels are one-to-one: only one detection earns the tp.

    Edge tolerance: a detection still unmatched after that, which overlaps nothing but
    lies within `edge_tolerance_cadences` cadences of a label's edge (its end or its
    start; see `_edge_gap`), is matched to that label by a second pass over the
    leftovers, so overlap always beats edge. It is "touched" if the label is still
    unclaimed, else "corroborating": it is evidence of an event this detector already
    counted, not a false alarm. 0 disables the pass and reproduces the overlap-only
    matcher exactly.

    Labels with ``footprint_samples == 0`` (events that changed no sample) are dropped
    before matching: they are never matched, claim nothing, and a detection over one is
    handled exactly as if the label were absent, so it can still be a false positive.
    """
    if edge_tolerance_cadences < 0:
        raise ValueError("edge_tolerance_cadences must be >= 0")
    truth = [t for t in truth if t.observable]
    by_detector: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        by_detector[d.detector].append(d)

    out: list[Match] = []
    for detector in sorted(by_detector):
        taken: set[str] = set()
        ordered = sorted(by_detector[detector], key=lambda d: (-d.score, d.det_id))
        results: list[Match] = []
        for det in ordered:
            best: dict[str, tuple[float, str]] = {}
            corr: tuple[float, str] | None = None  # claimed field-wide label
            frag: tuple[float, str] | None = None  # claimed star-level label it overlaps
            for label in truth:
                el = _eligible(det, label, cadence_s, iou_min)
                if el is None:
                    continue
                iou, grade = el
                cand = (iou, label.label_id)
                if label.label_id in taken:
                    # Already claimed: a field-wide label is corroborated by anything
                    # eligible for it; a star-level one only by a detection that overlaps
                    # it (iou > 0), a fragment of an event this detector already found.
                    # Best IoU wins, ties by label_id.
                    if label.star_id is None:
                        if corr is None or iou > corr[0] or (iou == corr[0] and cand[1] < corr[1]):
                            corr = cand
                    elif iou > 0 and (
                        frag is None or iou > frag[0] or (iou == frag[0] and cand[1] < frag[1])
                    ):
                        frag = cand
                    continue
                cur = best.get(grade)
                if cur is None or iou > cur[0] or (iou == cur[0] and cand[1] < cur[1]):
                    best[grade] = cand
            for grade in ("bounded", "touched"):
                if grade in best:
                    iou, lid = best[grade]
                    taken.add(lid)
                    results.append(Match(detector, det.det_id, lid, iou, grade))
                    break
            else:
                # A field-wide corroboration (scoring v1) keeps precedence over a star-level
                # one, so detections that already corroborated a field-wide label keep it.
                target = corr or frag
                if target is not None:
                    results.append(
                        Match(detector, det.det_id, target[1], target[0], "corroborating")
                    )
                else:
                    results.append(Match(detector, det.det_id, None, 0.0))
        if edge_tolerance_cadences:
            _edge_pass(ordered, results, truth, taken, edge_tolerance_cadences * cadence_s)
        out.extend(results)
    return out
