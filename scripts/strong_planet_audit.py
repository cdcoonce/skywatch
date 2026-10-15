"""What the two detectors read, step by step, on chosen stars of a stored run (issue #66).

Usage::

    uv run python scripts/strong_planet_audit.py runs/puls-seed847-200s-20n 146 150 [--detector bls]
    uv run python scripts/strong_planet_audit.py --json out.json runs/seed861-200s-20n 4 150

Both detectors are run through their real ``detect`` (the common mode is taken over the whole
field as in a survey), with the methods that decide a star's fate wrapped so that they record
their arguments' summary and their return as they are called on the star. Nothing is replayed:
the numbers are the ones the detector's own path read. Stars not named are skipped after the
common mode, so a run costs the common mode (about a minute and a half) and the named stars.

For every named star the output lists, in call order and indented by call depth:

* ``bls`` (``BlsDetector``): ``_search`` (first look, re-search), ``_search_arrays`` (the box, its
  S/N, local S/N, the gate that rejected it, the residual-sinusoid power, the night-level sine
  ratio, the shape statistics), ``_pulsating`` (residual power over the false-alarm level and
  ``2 * sine power`` over the box's chi-square, the guard's two tests), ``_night_sine_ratio``,
  ``_pulsation_trend`` (fitted frequency, amplitude, ratio; with the box left out, whether a slower
  sinusoid won), ``_detrended`` (what the second look returned);
* ``single_transit`` (``SingleTransitDetector``): ``_boxes`` (each night's box at the bar),
  ``_pulsating`` (guard (a) on the raw flux), ``_residual``, ``_residual_pulsating``,
  ``_stands_unflared``, ``_shape`` (the shape statistic of each box) and the star's final boxes.

Truth appears only as a label beside the output (the star's injected sinusoid and the planets on
it); no detector reads it. This is the instrument behind the measurement table of issue #66.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from skywatch.detectors import bls as bls_mod
from skywatch.detectors.bls import BlsDetector
from skywatch.detectors.single_transit import SingleTransitDetector
from skywatch.io import store

DAY_S = 86400.0
HOUR_S = 3600.0


def _f(x: object) -> object:
    """JSON-friendly scalar."""
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) else round(v, 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def cand_summary(c: bls_mod.Candidate | None) -> dict | None:
    if c is None:
        return None
    b = c.box
    return {
        "P_d": _f(b.period_s / DAY_S),
        "dur_h": _f(b.duration_s / HOUR_S),
        "depth": _f(b.depth),
        "n_in": b.n_in,
        "snr": _f(c.snr),
        "local": _f(c.snr_local),
        "rejected": c.rejected,
        "resid_power": _f(c.excess),
        "sine_ratio": _f(c.sine_ratio),
        "shape": _f(c.shape),
        "shape_chi2": _f(c.shape_chi2),
        "supported": c.supported_nights,
        "events": len(c.events),
        "repeat": _f(c.snr_repeat),
        "clean": _f(c.snr_clean),
        "trend": None if c.trend is None else [_f(c.trend[0]), _f(c.trend[1])],
        "alias_of_P": None,
    }


class _Spy:
    """Shared recording for the two spy detectors."""

    def _init_spy(self, targets: set[int]) -> None:
        self.targets = targets
        self.sid: int | None = None
        self.trace: dict[int, list[dict]] = {s: [] for s in targets}
        self._level = [0]  # call depth; shared with the `bls` spy a `single_transit` spy holds

    @property
    def depth(self) -> int:
        return self._level[0]

    @depth.setter
    def depth(self, value: int) -> None:
        self._level[0] = value

    def _spy(self, name: str, info: dict, call, summ=lambda r: r):  # type: ignore[no-untyped-def]
        if self.sid not in self.targets:
            return call()
        rec = {"fn": name, "d": self.depth, **info}
        self.trace[self.sid].append(rec)  # type: ignore[index]
        self.depth += 1
        try:
            out = call()
        finally:
            self.depth -= 1
        rec["ret"] = summ(out)
        return out


class SpyBls(_Spy, BlsDetector):
    def __init__(self, targets: set[int], **kw: object) -> None:
        BlsDetector.__init__(self, **kw)  # type: ignore[arg-type]
        self._init_spy(targets)

    def _star(self, star_id, c, common_mode, cadence_s):  # type: ignore[no-untyped-def]
        if star_id not in self.targets:
            return []
        self.sid = star_id
        try:
            out = super()._star(star_id, c, common_mode, cadence_s)
            self.trace[star_id].append({"fn": "detections", "d": 0, "ret": len(out)})
            return out
        finally:
            self.sid = None

    def _search(self, c, common_mode, cadence_s, masked=(), plan_mask=False):  # type: ignore[no-untyped-def]
        return self._spy(
            "_search",
            {"masked_windows": len(masked), "plan_mask": plan_mask},
            lambda: super(SpyBls, self)._search(c, common_mode, cadence_s, masked, plan_mask),
            cand_summary,
        )

    def _search_arrays(self, a, c, cadence_s, plan_mask, *bar, **kw):  # type: ignore[no-untyped-def]
        self._night = (
            a.night
        )  # the guard is not handed the nights; its night-level ratio needs them
        return self._spy(
            "_search_arrays",
            {"samples": len(a.t), "residual_bar": bar[0] if bar else 1.0},
            lambda: super(SpyBls, self)._search_arrays(a, c, cadence_s, plan_mask, *bar, **kw),
            cand_summary,
        )

    def _detrended(self, a, c, cadence_s, plan_mask):  # type: ignore[no-untyped-def]
        return self._spy(
            "_detrended",
            {},
            lambda: super(SpyBls, self)._detrended(a, c, cadence_s, plan_mask),
            cand_summary,
        )

    def _re_search(self, star_id, c, common_mode, cadence_s, first):  # type: ignore[no-untyped-def]
        return self._spy(
            "_re_search",
            {"first_P_d": _f(first.box.period_s / DAY_S), "first_rejected": first.rejected},
            lambda: super(SpyBls, self)._re_search(star_id, c, common_mode, cadence_s, first),
            cand_summary,
        )

    def _pulsating(self, t, flux, sigma, cadence_s, box, *bar):  # type: ignore[no-untyped-def]
        def call():  # type: ignore[no-untyped-def]
            return super(SpyBls, self)._pulsating(t, flux, sigma, cadence_s, box, *bar)

        if self.sid not in self.targets:
            return call()
        out = call()
        # the guard's two readings, from the arguments it was called with
        core = bls_mod.in_box(
            t, bls_mod.replace(box, duration_s=bls_mod._GUARD_CORE_WIDTH * box.duration_s)
        )
        n, n_in = len(t), box.n_in
        box_chi2 = box.depth**2 * n_in * (n - n_in) / (n * sigma**2)
        s_power, level = self._peak_power(t, flux - 1.0, sigma, cadence_s)
        night = getattr(self, "_night", None)
        night_ratio = None
        if box.period_s < 1e8 and night is not None and len(night) == len(t):
            # the third reading of the guard, which `_pulsating` itself does not compute: shown
            # for the box the first two readings refused, from the same arrays
            night_ratio = _f(BlsDetector._night_sine_ratio(self, t, flux, night, box, cadence_s))
        self.trace[self.sid].append(  # type: ignore[index]
            {
                "fn": "_pulsating",
                "d": self.depth,
                "one_event": box.period_s > 1e8,
                "box_P_d": _f(box.period_s / DAY_S),
                "box_dur_h": _f(box.duration_s / HOUR_S),
                "core_samples": int(core.sum()),
                "resid_power": _f(out[1]),
                "false_alarm_level": _f(level),
                "resid_over_level": _f(out[1] / level),
                "sine_gain_over_box_chi2": _f(2.0 * s_power / box_chi2) if box_chi2 > 0 else None,
                "refused": bool(out[0]),
                "night_sine_ratio": night_ratio,
            }
        )
        return out

    def _night_sine_ratio(self, t, flux, night, box, cadence_s):  # type: ignore[no-untyped-def]
        return self._spy(
            "_night_sine_ratio",
            {},
            lambda: super(SpyBls, self)._night_sine_ratio(t, flux, night, box, cadence_s),
            _f,
        )

    def _pulsation_trend(self, a, cadence_s, box=None):  # type: ignore[no-untyped-def]
        info = {
            "box_left_out": None
            if box is None
            else {"P_d": _f(box.period_s / DAY_S), "dur_h": _f(box.duration_s / HOUR_S)}
        }
        return self._spy(
            "_pulsation_trend",
            info,
            lambda: super(SpyBls, self)._pulsation_trend(a, cadence_s, box),
            lambda r: (
                None
                if r is None
                else {"f_per_d": _f(r[1][0]), "amp": _f(r[1][1]), "ratio": _f(r[1][2])}
            ),
        )

    def _strongest_sinusoid(self, t, td, x, k, sigma, cadence_s, span_d):  # type: ignore[no-untyped-def]
        # the ratio of the best sinusoid to the false-alarm level, shown whether or not it clears
        # `trend_min_ratio` (`_pulsation_trend` returns None when it does not)
        def summary(r):  # type: ignore[no-untyped-def]
            if r is None:
                return None
            gain, f, sin_a, cos_a, level = r
            return {
                "f_per_d": _f(f),
                "amp": _f(math.hypot(sin_a, cos_a)),
                "ratio": _f(gain / (2.0 * sigma**2) / level),
            }

        return self._spy(
            "_strongest_sinusoid",
            {},
            lambda: super(SpyBls, self)._strongest_sinusoid(t, td, x, k, sigma, cadence_s, span_d),
            summary,
        )

    def _slower_sinusoid_wins(self, td, x, k, gain):  # type: ignore[no-untyped-def]
        return self._spy(
            "_slower_sinusoid_wins",
            {},
            lambda: super(SpyBls, self)._slower_sinusoid_wins(td, x, k, gain),
            bool,
        )

    def _shape_test(self, t, flux, night, sigma, box):  # type: ignore[no-untyped-def]
        return self._spy(
            "_shape_test",
            {},
            lambda: super(SpyBls, self)._shape_test(t, flux, night, sigma, box),
            lambda r: {"shape": _f(r[0]), "chi2_advantage": _f(r[1])},
        )


class SpySingle(_Spy, SingleTransitDetector):
    def __init__(self, targets: set[int], **kw: object) -> None:
        SingleTransitDetector.__init__(self, **kw)  # type: ignore[arg-type]
        self._init_spy(targets)
        self._bls = SpyBls(targets, trend_guard_ratio=None)  # as the detector pins its own
        self._bls.trace = self.trace  # one trace, in call order, for both detectors' methods
        self._bls._level = self._level

    def _star(self, star_id, c, common_mode, cadence_s):  # type: ignore[no-untyped-def]
        if star_id not in self.targets:
            return []
        self.sid = star_id
        self._bls.sid = star_id
        try:
            out = super()._star(star_id, c, common_mode, cadence_s)
            self.trace[star_id].append({"fn": "detections", "d": 0, "ret": len(out)})
            return out
        finally:
            self.sid = None
            self._bls.sid = None

    def _boxes_info(self, boxes):  # type: ignore[no-untyped-def]
        return [
            {
                "night": b.night,
                "snr": _f(b.snr),
                "depth": _f(b.depth),
                "len_h": _f((b.t_end - b.t_start) / HOUR_S),
                "n_in": b.n_in,
                "shape": _f(b.shape),
            }
            for b in boxes
        ]

    def _boxes(self, t, flux, night, cadence_s, floor=None):  # type: ignore[no-untyped-def]
        return self._spy(
            "_boxes",
            {"samples": len(t), "floor": floor},
            lambda: super(SpySingle, self)._boxes(t, flux, night, cadence_s, floor),
            self._boxes_info,
        )

    def _pulsating(self, t, flux, night, boxes, cadence_s, *bar):  # type: ignore[no-untyped-def]
        return self._spy(
            "_pulsating(guard a)",
            {
                "strongest_snr": _f(max((b.snr for b in boxes), default=0.0)),
                "residual_bar": bar[0] if bar else 1.0,
            },
            lambda: super(SpySingle, self)._pulsating(t, flux, night, boxes, cadence_s, *bar),
            bool,
        )

    def _residual(self, c, t, flux, night, level, cadence_s):  # type: ignore[no-untyped-def]
        return self._spy(
            "_residual",
            {},
            lambda: super(SpySingle, self)._residual(c, t, flux, night, level, cadence_s),
            lambda r: None if r is None else "residual",
        )

    def _residual_pulsating(self, t, resid, night, boxes, cadence_s, *bar):  # type: ignore[no-untyped-def]
        return self._spy(
            "_residual_pulsating",
            {"boxes": self._boxes_info(boxes), "residual_bar": bar[0] if bar else 1.0},
            lambda: super(SpySingle, self)._residual_pulsating(
                t, resid, night, boxes, cadence_s, *bar
            ),
            bool,
        )

    def _stands_unflared(self, t, flux, night, b):  # type: ignore[no-untyped-def]
        return self._spy(
            "_stands_unflared",
            {"night": b.night, "snr": _f(b.snr)},
            lambda: super(SpySingle, self)._stands_unflared(t, flux, night, b),
            bool,
        )

    def _detrended(self, c, t, flux, night, level, cadence_s):  # type: ignore[no-untyped-def]
        return self._spy(
            "_detrended",
            {},
            lambda: super(SpySingle, self)._detrended(c, t, flux, night, level, cadence_s),
            lambda r: None if r is None else self._boxes_info(r[0]),
        )

    def _shape(self, t, flux, night, b, cadence_s, *ramp):  # type: ignore[no-untyped-def]
        return self._spy(
            "_shape",
            {"night": b.night, "snr": _f(b.snr), "ramped_box": ramp[0] if ramp else None},
            lambda: super(SpySingle, self)._shape(t, flux, night, b, cadence_s, *ramp),
            _f,
        )

    def _star_boxes(self, star_id, c, common_mode, cadence_s):  # type: ignore[no-untyped-def]
        return self._spy(
            "_star_boxes",
            {},
            lambda: super(SpySingle, self)._star_boxes(star_id, c, common_mode, cadence_s),
            lambda r: self._boxes_info(r[0]),
        )


def run_audit(
    run: Path, stars: list[int], which: tuple[str, ...], single_kw: dict | None = None
) -> dict:
    curves = store.read_curves(run)
    cadence_s = store.read_manifest(run)["config"]["cadence_s"]
    info = {s.star_id: s for s in store.read_stars(run)}
    planets = [p for p in store.read_planets(run) if p.star_id in stars]
    out: dict = {
        "run": run.name,
        "stars": {
            s: {
                "variability": info[s].variability,
                "params": info[s].params,
                "planets": [
                    {
                        "planet_id": p.planet_id,
                        "P_d": p.period_s / DAY_S,
                        "depth": p.depth,
                        "dur_h": p.duration_s / HOUR_S,
                    }
                    for p in planets
                    if p.star_id == s
                ],
            }
            for s in stars
        },
    }
    for name in which:
        det = SpyBls(set(stars)) if name == "bls" else SpySingle(set(stars), **(single_kw or {}))
        dets = det.detect(curves, cadence_s)
        out[name] = {"trace": det.trace, "detections": len(dets)}
    return out


def render(res: dict) -> str:
    lines = [f"== {res['run']}"]
    for sid, info in res["stars"].items():
        prm = info["params"]
        extra = ""
        if info["variability"] == "sinusoidal":
            extra = f" injected f {DAY_S / prm['period_s']:.3f}/d amp {prm['amplitude']:.5f}"
        lines.append(f"-- star {sid} {info['variability']}{extra}")
        for p in info["planets"]:
            lines.append(
                f"   planet {p['planet_id']}: P {p['P_d']:.4f} d depth {p['depth']:.5f} "
                f"dur {p['dur_h']:.2f} h"
            )
        for det in ("bls", "single_transit"):
            if det not in res:
                continue
            lines.append(f"   [{det}]")
            for rec in res[det]["trace"][int(sid) if not isinstance(sid, int) else sid]:
                rest = {k: v for k, v in rec.items() if k not in ("fn", "d")}
                lines.append(f"   {'  ' * rec['d']}{rec['fn']} {json.dumps(rest)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detector", choices=("bls", "single_transit", "both"), default="both")
    ap.add_argument("--json", help="write the full trace as JSON to this path")
    ap.add_argument(
        "--single-kw",
        default="{}",
        help="JSON of SingleTransitDetector keywords, e.g. '{\"reread\": true}' (default: its defaults)",
    )
    ap.add_argument("run")
    ap.add_argument("stars", nargs="+", type=int)
    args = ap.parse_args()
    which = ("bls", "single_transit") if args.detector == "both" else (args.detector,)
    res = run_audit(Path(args.run), args.stars, which, json.loads(args.single_kw))
    if args.json:
        Path(args.json).write_text(json.dumps(res))
    print(render(res))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
