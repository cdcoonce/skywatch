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


def match_detections(
    detections: Sequence[Detection],
    truth: Sequence[TruthLabel],
    cadence_s: int,
    iou_min: float = 0.3,
) -> list[Match]:
    """Match each detector's detections to truth independently.

    Greedy by score desc. A detection takes the best unclaimed bounded label, else
    the best unclaimed touched label, else corroborates a field-wide label this
    detector already claimed, else is unmatched. Star-level labels are one-to-one.
    """
    by_detector: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        by_detector[d.detector].append(d)

    out: list[Match] = []
    for detector in sorted(by_detector):
        taken: set[str] = set()
        ordered = sorted(by_detector[detector], key=lambda d: (-d.score, d.det_id))
        for det in ordered:
            best: dict[str, tuple[float, str]] = {}
            corr: tuple[float, str] | None = None
            for label in truth:
                el = _eligible(det, label, cadence_s, iou_min)
                if el is None:
                    continue
                iou, grade = el
                cand = (iou, label.label_id)
                if label.label_id in taken:
                    if label.star_id is None and (
                        corr is None or iou > corr[0] or (iou == corr[0] and cand[1] < corr[1])
                    ):
                        corr = cand
                    continue
                cur = best.get(grade)
                if cur is None or iou > cur[0] or (iou == cur[0] and cand[1] < cur[1]):
                    best[grade] = cand
            for grade in ("bounded", "touched"):
                if grade in best:
                    iou, lid = best[grade]
                    taken.add(lid)
                    out.append(Match(detector, det.det_id, lid, iou, grade))
                    break
            else:
                if corr is not None:
                    out.append(Match(detector, det.det_id, corr[1], corr[0], "corroborating"))
                else:
                    out.append(Match(detector, det.det_id, None, 0.0))
    return out
