"""Greedy one-to-one matching of detections to truth labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from skywatch.types import Detection, TruthLabel


@dataclass(frozen=True)
class Match:
    detector: str
    det_id: str
    label_id: str | None
    iou: float


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection over union of two half-open integer spans."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _eligible_iou(
    det: Detection, label: TruthLabel, cadence_s: int, iou_min: float
) -> float | None:
    """Return the IoU if `det` may match `label`, else None."""
    if label.star_id is not None and label.star_id != det.star_id:
        return None
    iou = span_iou(det.t_start, det.t_end, label.t_start, label.t_end)
    if iou >= iou_min:
        return iou
    single = label.t_end - label.t_start == cadence_s
    if single and det.t_start < label.t_end + cadence_s and det.t_end > label.t_start - cadence_s:
        return iou
    return None


def match_detections(
    detections: Sequence[Detection],
    truth: Sequence[TruthLabel],
    cadence_s: int,
    iou_min: float = 0.3,
) -> list[Match]:
    """Match each detector's detections to truth independently."""
    by_detector: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        by_detector[d.detector].append(d)

    out: list[Match] = []
    for detector in sorted(by_detector):
        taken: set[str] = set()
        ordered = sorted(by_detector[detector], key=lambda d: (-d.score, d.det_id))
        for det in ordered:
            best: tuple[float, str] | None = None
            for label in truth:
                if label.label_id in taken:
                    continue
                iou = _eligible_iou(det, label, cadence_s, iou_min)
                if iou is None:
                    continue
                if best is None or iou > best[0] or (iou == best[0] and label.label_id < best[1]):
                    best = (iou, label.label_id)
            if best is None:
                out.append(Match(detector, det.det_id, None, 0.0))
            else:
                taken.add(best[1])
                out.append(Match(detector, det.det_id, best[1], best[0]))
    return out
