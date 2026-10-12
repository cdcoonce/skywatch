"""Every eclipsing binary that reaches the shape gate of `bls`, and what the gate reads on it.

Usage::

    uv run python scripts/bls_wide_eclipse_audit.py runs/seed64-200s-20n [more runs ...]
    uv run python scripts/bls_wide_eclipse_audit.py --markdown runs/...   # the table of issue #41
    uv run python scripts/bls_wide_eclipse_audit.py --all runs/...        # also the other stars

Every star is searched with the shape gate raised out of reach (``shape_threshold=1e9``,
``shape_chi2=None``) so that the statistic is read for each candidate that passes every earlier
gate, whether or not the gate would reject it. One line per binary, the world's truth beside what
the detector read: the eclipse's period, base duration (and as a share of the night), depths, the
box's period against the eclipse's and its width, S/N, the nights holding a box event and, for each
(the first six), whether the eclipse in it is the primary (P) or the secondary (S) and the share of
its base that lies inside the night; the statistic of each event night fitted alone; the joint
statistic of ``main`` before #41 (one depth for all nights, ``_shape_excess``); the final statistic
(the larger of that and the same ratio with a depth for each night) and the chi-square by which
the best V beats the best box with a depth for each night; and the verdicts of the gate before
(joint statistic at 1.5) and after (statistic at 1.5 or chi-square at 14). With ``--all`` the other
stars (planet hosts, flare stars, ...) are listed too, with the same columns where they apply.
It is the instrument behind the measurement table of issue #41 and the "Eclipses wider than the
night" notes of the `skywatch.detectors.bls` docstring; it reads the run's light curves and its
stars and planets tables only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from skywatch.detectors.bls import BlsDetector, Box
from skywatch.io import store

HOUR_S = 3600.0
OLD_THRESHOLD = 1.5  # `main` before #41: the joint statistic at 1.5, nothing else


def _event_nights(t: np.ndarray, night: np.ndarray, box: Box) -> np.ndarray:
    x = (t - box.epoch_s) - np.rint((t - box.epoch_s) / box.period_s) * box.period_s
    return np.unique(night[np.abs(x) <= box.duration_s / 2.0])


def _eclipse_in_night(params: dict, t: np.ndarray) -> tuple[str, float]:
    """(P or S, share of the eclipse's base inside [t.min, t.max]) of the eclipse the night holds best."""
    best = ("P", 0.0)
    for j in range(-2, int(t.max() // params["period_s"]) + 3):
        for which, offset in (("P", 0.0), ("S", params["period_s"] / 2.0)):
            c = params["epoch_s"] + j * params["period_s"] + offset
            lo = max(c - params["duration_s"] / 2.0, float(t.min()))
            hi = min(c + params["duration_s"] / 2.0, float(t.max()))
            share = max(0.0, hi - lo) / params["duration_s"]
            if share > best[1]:
                best = (which, share)
    return best


def audit(run_dir: Path, everything: bool) -> list[dict]:
    stars = {s.star_id: s for s in store.read_stars(run_dir)}
    hosts = {p.star_id for p in store.read_planets(run_dir)}
    manifest = store.read_manifest(run_dir)
    cadence_s = manifest["config"]["cadence_s"]
    night_h = manifest["config"]["night_length_s"] / HOUR_S
    detector = BlsDetector(shape_threshold=1e9, shape_chi2=None)  # read, never reject
    seen: list[tuple] = []
    real = detector._shape_test

    def spy(t, flux, night, sigma, box):  # type: ignore[no-untyped-def]
        seen.append((t, flux, night, sigma, box))
        return real(t, flux, night, sigma, box)

    detector._shape_test = spy  # type: ignore[method-assign]
    curves = store.read_curves(run_dir)
    cleaned = {}
    for star_id in sorted(curves):
        c = detector._clean(curves[star_id], cadence_s)
        if c is not None:
            cleaned[star_id] = c
    common_mode = detector._common_mode(cleaned)
    rows = []
    for star_id, c in cleaned.items():
        seen.clear()
        cand = detector._search(c, common_mode, cadence_s)
        if cand is None or cand.shape is None or not seen:
            continue
        star = stars[star_id]
        if star.variability != "eclipsing_binary" and not everything:
            continue
        t, flux, night, sigma, box = seen[-1]
        events = _event_nights(t, night, box)
        alone = [
            detector._shape_excess(t[night == k], flux[night == k], night[night == k], sigma, box)
            for k in events
        ]
        joint = detector._shape_excess(t, flux, night, sigma, box)
        row = {
            "run": run_dir.name,
            "star": star_id,
            "kind": star.variability + (" + planet" if star_id in hosts else ""),
            "snr": cand.snr,
            "n_events": len(events),
            "box_period_d": box.period_s / 86400.0,
            "box_h": box.duration_s / HOUR_S,
            "alone": alone,
            "joint": joint,
            "shape": cand.shape,
            "chi2": cand.shape_chi2,
            "before": "rejected" if joint >= OLD_THRESHOLD else "leaks",
            "after": "rejected" if cand.shape >= 1.5 or cand.shape_chi2 >= 14.0 else "leaks",
            "other": cand.rejected,
        }
        if star.variability == "eclipsing_binary":
            p = star.params
            in_night = [_eclipse_in_night(p, t[night == k]) for k in events]
            row.update(
                period_d=p["period_s"] / 86400.0,
                base_h=p["duration_s"] / HOUR_S,
                night_share=p["duration_s"] / HOUR_S / night_h,
                depths=(p["depth_primary"], p["depth_secondary"]),
                box_over_period=box.period_s / p["period_s"],
                kinds="".join(k for k, _ in in_night),
                v_share=float(np.mean([s for _, s in in_night])) if in_night else 0.0,
            )
        rows.append(row)
    return rows


def _line(r: dict) -> str:
    head = f"{r['run']:20s} star {r['star']:4d} {r['kind']:26s}"
    if "period_d" in r:
        head += (
            f" eclipse P {r['period_d']:5.2f} d base {r['base_h']:5.2f} h "
            f"({100 * r['night_share']:3.0f}% of night) depths {r['depths'][0]:.2f}/{r['depths'][1]:.2f}"
            f" | box/P {r['box_over_period']:.2f} {r['box_h']:.1f} h | {r['kinds'][:10]:10s} "
            f"V share {100 * r['v_share']:3.0f}%"
        )
    alone = ", ".join(f"{a:.2f}" for a in r["alone"][:6])
    return (
        f"{head} | S/N {r['snr']:5.1f} nights {r['n_events']:2d} | alone [{alone}] | joint "
        f"{r['joint']:6.2f} shape {r['shape']:6.2f} chi2 {r['chi2']:7.1f} | before {r['before']}, "
        f"after {r['after']}" + (f" (then {r['other']})" if r["other"] else "")
    )


def _markdown(rows: Sequence[dict]) -> str:
    out = [
        (
            "| run / star | eclipse P (d) | base (h) | % of night | depths | box P / eclipse P "
            "| box (h) | S/N | nights | P/S, V share | each night alone | joint (before) "
            "| statistic | chi2 | before | after |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        alone = ", ".join(f"{a:.2f}" for a in r["alone"][:6]) + (
            ", ..." if len(r["alone"]) > 6 else ""
        )
        out.append(
            f"| {r['run']} / {r['star']} | {r['period_d']:.2f} | {r['base_h']:.2f} "
            f"| {100 * r['night_share']:.0f} | {r['depths'][0]:.2f}/{r['depths'][1]:.2f} "
            f"| {r['box_over_period']:.2f} | {r['box_h']:.1f} | {r['snr']:.1f} | {r['n_events']} "
            f"| {r['kinds'][:10]}, {100 * r['v_share']:.0f}% | {alone} | {r['joint']:.2f} "
            f"| {r['shape']:.2f} | {r['chi2']:.1f} | {r['before']} | {r['after']} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--all", action="store_true", help="also the stars that are not binaries")
    ap.add_argument("--markdown", action="store_true", help="the binaries as a markdown table")
    args = ap.parse_args()
    rows = [r for run in args.runs for r in audit(run, args.all)]
    if args.markdown:
        print(_markdown([r for r in rows if "period_d" in r]))
        return
    for r in sorted(rows, key=lambda r: -r["shape"]):
        print(_line(r))


if __name__ == "__main__":
    main()
