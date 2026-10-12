"""What decides the two planet-free gates of `bls`, for every star a run would report.

Usage::

    uv run python scripts/bls_planet_free_audit.py runs/pf-seed44-200s-20n [more runs ...]

Every star is searched with the sinusoid-against-box gate raised out of reach
(``guard_night_ratio=1e9``) and the flare bar at the ordinary local S/N bar, so that
``Candidate.sine_ratio`` is read for each candidate that passes the other gates, including a
flare star that fails only the flare bar (listed as "flare star below its local bar"); the
flare-tail gate is left on and its rejections are kept, so that
``Candidate.snr_clean`` (the local S/N with the samples inside and within ``flare_window_h`` after
a flare-sized masked run left out) is read too; nothing is acted on. One line per candidate: the
star's variability, whether it carries a planet, S/N, local S/N and the clean S/N, events, the
ratio of the best sinusoid's chi-square gain to the box's (free level for every night), the share
of samples masked as positive outliers, and what the shipping defaults do with it. This is the
instrument behind the "Planet-free stars" section of the `skywatch.detectors.bls` docstring
(issues #39 and #43); it reads no detections or scores, only the run's light curves and its stars
and planets tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

from skywatch.detectors.bls import BlsDetector
from skywatch.io import store


def audit(run_dir: Path) -> list[str]:
    stars = {s.star_id: s for s in store.read_stars(run_dir)}
    hosts = {p.star_id for p in store.read_planets(run_dir)}
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    defaults = BlsDetector()
    # compute the ratio and leave the flare bar at the ordinary one, never reject on either; the
    # flare-tail gate rejects last and is read
    detector = BlsDetector(guard_night_ratio=1e9, flare_snr_threshold=defaults.snr_threshold)
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
        if cand is None or cand.rejected not in (None, "flare tails") or cand.sine_ratio is None:
            continue
        flare = 1.0 - len(detector._prepare(c, common_mode)[0]) / len(c.t)
        verdict = "reported"
        if cand.sine_ratio >= defaults.guard_night_ratio:
            verdict = "sinusoid beats the box"
        elif flare >= defaults.flare_fraction and cand.snr_local < defaults.flare_snr_threshold:
            verdict = "flare star below its local bar"
        elif cand.rejected == "flare tails":
            verdict = "flare tails: the clean S/N is under the bar"
        host = "planet host" if star_id in hosts else "no planet"
        rows.append(
            f"{run_dir.name} star {star_id:4d} {stars[star_id].variability:17s} {host:11s} "
            f"S/N {cand.snr:6.1f} local {cand.snr_local:6.1f} clean {cand.snr_clean:6.1f} "
            f"events {len(cand.events) if cand.events else '-':>2} "
            f"ratio {cand.sine_ratio:5.2f} flare share {100 * flare:5.2f}% {verdict}"
        )
    return rows


if __name__ == "__main__":
    for path in sys.argv[1:]:
        for line in audit(Path(path)):
            print(line)
