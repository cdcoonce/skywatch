"""The shape statistic of every star `bls` would report on a run, with what the star is.

Usage::

    uv run python scripts/bls_shape_audit.py runs/seed32-200s-20n [more runs ...]

Every star is searched with the shape gate raised out of reach (``shape_threshold=1e9``,
``shape_chi2=None``) so that `Candidate.shape` and `Candidate.shape_chi2` are read for each
candidate that passes every other gate, and never acted on. One line per such star, highest
statistic first: the star's variability, whether it carries a planet, S/N, events, the statistic
and the chi-square by which a V beats the box (issue #41). This is the instrument behind the "Shape guard" section of
the `skywatch.detectors.bls` docstring (issue #36); it reads no detections or scores, only the
run's light curves and its stars and planets tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

from skywatch.detectors.bls import BlsDetector
from skywatch.io import store


def audit(run_dir: Path) -> list[tuple[float, str]]:
    stars = {s.star_id: s for s in store.read_stars(run_dir)}
    hosts = {p.star_id for p in store.read_planets(run_dir)}
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    detector = BlsDetector(shape_threshold=1e9, shape_chi2=None)  # read the statistic, never reject
    curves = store.read_curves(run_dir)
    cleaned = {}
    for star_id in sorted(curves):
        c = detector._clean(curves[star_id], cadence_s)
        if c is not None:
            cleaned[star_id] = c
    common_mode = detector._common_mode(cleaned)
    rows = []
    for star_id, c in cleaned.items():
        cand = detector._search(c, common_mode, cadence_s)
        if cand is None or cand.rejected is not None or cand.shape is None:
            continue
        host = "planet host" if star_id in hosts else "no planet"
        line = (
            f"{run_dir.name} star {star_id:4d} {stars[star_id].variability:17s} {host:11s} "
            f"S/N {cand.snr:6.1f} events {len(cand.events):2d} shape {cand.shape:6.2f} "
            f"chi2 {cand.shape_chi2:7.1f}"
        )
        rows.append((cand.shape, line))
    return sorted(rows, reverse=True)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        for _, line in audit(Path(path)):
            print(line)
