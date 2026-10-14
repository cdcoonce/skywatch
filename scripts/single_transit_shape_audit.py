"""The shape statistic of every box `single_transit` would report on a run, with what it is.

Usage::

    uv run python scripts/single_transit_shape_audit.py runs/seed32-200s-20n [more runs ...]

Every star is searched with the shape test raised out of reach (``shape_threshold=1e9``) so that
``NightBox.shape`` is read for each box that passes the search and the guards, and never acted
on. One line per box, highest statistic first: the star's variability, the night, S/N, the number
of samples in the box, whether the box touches the night's first or last sample, the kind of the
truth label the scorer matches the box to and the statistic. This is the instrument behind the
"Shape test" section of the `skywatch.detectors.single_transit` docstring (issue #45); it reads
the run's light curves, stars and truth tables, no detections or scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

from skywatch.detectors.single_transit import SingleTransitDetector
from skywatch.io import store
from skywatch.scoring.match import match_detections
from skywatch.types import Detection


def audit(run_dir: Path) -> list[tuple[float, str]]:
    stars = {s.star_id: s for s in store.read_stars(run_dir)}
    truth = store.read_truth(run_dir)
    kinds = {t.label_id: t.kind for t in truth}
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    detector = SingleTransitDetector(shape_threshold=1e9)  # compute the statistic, never refuse
    curves = store.read_curves(run_dir)
    bls = detector._bls
    cleaned = {}
    for star_id in sorted(curves):
        c = bls._clean(curves[star_id], cadence_s)
        if c is not None:
            cleaned[star_id] = c
    common_mode = bls._common_mode(cleaned, cadence_s)  # as `single_transit.detect` does
    found = []  # (detection, box, touches the night's edge)
    for star_id, c in cleaned.items():
        boxes, (t, _flux, night) = detector._star_boxes(star_id, c, common_mode, cadence_s)
        for k, b in enumerate(boxes):
            t_night = t[night == b.night]
            edge = b.t_start <= t_night[0] + cadence_s or b.t_end >= t_night[-1] - cadence_s
            det = Detection(
                detector="single_transit",
                det_id=f"single_transit-s{star_id}-{k}",
                star_id=star_id,
                t_start=b.t_start,
                t_end=b.t_end + cadence_s,
                score=b.snr,
                kind_guess="transit",
            )
            found.append((det, b, edge))
    matches = {m.det_id: m for m in match_detections([d for d, _, _ in found], truth, cadence_s)}
    rows = []
    for det, b, edge in found:
        m = matches[det.det_id]
        kind = kinds.get(m.label_id, "none") if m.label_id else "none"
        grade = m.grade or "-"
        line = (
            f"{run_dir.name} star {det.star_id:4d} night {b.night:2d} "
            f"{stars[det.star_id].variability:17s} S/N {b.snr:6.1f} n_in {b.n_in:4d} "
            f"{'edge ' if edge else 'inner'} {kind:14s} {grade:9s} shape {b.shape:6.2f}"
        )
        rows.append((b.shape, line))
    return sorted(rows, reverse=True)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        for _, line in audit(Path(path)):
            print(line)
