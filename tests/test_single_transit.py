import math
import time

import numpy as np
import pytest

import skywatch.detectors.single_transit as st
from skywatch.detectors import REGISTRY, Detector, load_all
from skywatch.detectors.bls import BlsDetector
from skywatch.detectors.single_transit import SingleTransitDetector, search_night
from skywatch.scoring.match import span_iou
from skywatch.types import LightCurve
from tests.test_bls import (
    DAY,
    NIGHT_N,
    C,
    curve,
    eclipsing,
    make_field,
    night_times,
)

HOUR = 3600


def detect(curves, **kw):
    return SingleTransitDetector(**kw).detect(curves, C)


def dip_profile(t, night, start_h, dur_h, depth):
    """A single box dip: starts `start_h` hours after the night's first sample (may be < 0)."""
    lo = night * DAY + start_h * HOUR
    return 1.0 - depth * ((t >= lo) & (t < lo + dur_h * HOUR))


def one_dip(night=4, start_h=2.5, dur_h=3.0, depth=0.01, noise=0.005, seed=0, n_nights=10):
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    return t, curve(0, t, dip_profile(t, night, start_h, dur_h, depth), noise, rng)


def clipped_window(night, start_h, dur_h):
    """The planted dip clipped to its night: what the world's truth label would span."""
    ns, ne = night * DAY, night * DAY + NIGHT_N * C
    return max(ns, ns + int(start_h * HOUR)), min(ne, ns + int((start_h + dur_h) * HOUR))


# -- registration, protocol, validation ------------------------------------------------


def test_registered_and_a_detector():
    assert load_all()["single_transit"] is SingleTransitDetector
    assert REGISTRY["single_transit"] is SingleTransitDetector
    d: Detector = SingleTransitDetector()
    assert d.name == "single_transit"


@pytest.mark.parametrize(
    "kw",
    [
        {"snr_threshold": 0},
        {"durations_h": ()},
        {"durations_h": (1.0, -1.0)},
        {"min_in": 0},
        {"min_out": 0},
        {"mask_pad_h": -1},
        {"shape_threshold": 0},
        {"shape_threshold": -1.0},
    ],
)
def test_constructor_validation(kw):
    with pytest.raises(ValueError):
        SingleTransitDetector(**kw)


def test_defaults_are_the_documented_ones():
    # the headline threshold and the search constants are results, not incidental values:
    # a default-constructed detector is what `skywatch detect` builds
    d = SingleTransitDetector()
    assert d.snr_threshold == 8.0
    assert (d.min_in, d.min_out) == (15, 30)
    assert d.durations_h == (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert (d.contrast, d.guards, d.mask_pad_h) == (True, True, 0.5)
    assert d.shape_threshold == 1.0


def test_empty_and_tiny_curves():
    t = np.arange(3, dtype=np.int64) * C
    tiny = LightCurve(1, t, np.ones(3), np.ones(3), np.zeros(3, dtype=np.uint8))
    empty = LightCurve(2, t[:0], np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.uint8))
    assert detect({1: tiny, 2: empty}) == []
    assert detect({}) == []


# -- the statistic -----------------------------------------------------------------------


def test_search_night_snr_is_the_contrast_matched_filter():
    rng = np.random.default_rng(1)
    t = np.arange(NIGHT_N, dtype=np.int64) * C
    flux = 1.0 + rng.normal(0.0, 0.005, NIGHT_N)
    flux[100:280] -= 0.02  # 180 samples = 3 h
    found = search_night(t, flux, C, (3 * HOUR,), 15, 30)
    assert found is not None
    snr, depth, n_in, n_out, sigma, first, last = found
    assert (n_in, n_out) == (180, NIGHT_N - 180) and (first, last) == (100, 279)
    assert depth == pytest.approx(0.02, abs=0.002)
    assert snr == pytest.approx(depth * math.sqrt(n_in * n_out / NIGHT_N) / sigma)
    plain = search_night(t, flux, C, (3 * HOUR,), 15, 30, contrast=False)
    assert plain[0] == pytest.approx(depth * math.sqrt(n_in) / sigma)
    assert plain[0] > snr  # the plain form assumes the reference level is known


def test_search_night_needs_min_in_and_min_out():
    rng = np.random.default_rng(6)
    t = np.arange(NIGHT_N, dtype=np.int64) * C
    flux = 1.0 + rng.normal(0.0, 0.005, NIGHT_N)
    flux[100:110] -= 0.05
    assert search_night(t, flux, C, (10 * 60.0,), 15, 30) is None  # a 10-sample box < min_in
    assert search_night(t, flux, C, (10 * 60.0,), 10, 30) is not None
    assert search_night(t, flux, C, (3 * HOUR,), 15, NIGHT_N - 14) is None  # out <= 465
    assert search_night(t[:44], flux[:44], C, (0.5 * HOUR,), 15, 30) is None  # 44 < 15 + 30


def _night(dip_slice, depth=0.05, n=NIGHT_N, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64) * C
    flux = 1.0 + rng.normal(0.0, 0.005, n)
    flux[dip_slice] -= depth
    return t, flux


def test_sigma_is_the_mad_of_the_night_outside_the_box():
    t, flux = _night(slice(100, 340))  # the dip fills half the night
    flux[5] += 0.2  # outliers outside the box: a std would be far above the MAD
    flux[400] += 0.2
    out = np.ones(NIGHT_N, dtype=bool)
    out[100:340] = False
    found = search_night(t, flux, C, (4 * HOUR,), 15, 30)
    assert found is not None and (found[5], found[6]) == (100, 339)
    ref = flux[out]
    assert found[4] == pytest.approx(1.4826 * np.median(np.abs(ref - np.median(ref))))


def test_min_out_limits_the_box_not_only_the_night():
    t, flux = _night(slice(10, 55), n=70)  # a 45-sample dip with 25 samples to hold it against
    found = search_night(t, flux, C, (45 * C,), 15, 30)
    assert found is not None and found[3] >= 30  # n_out: the full 45-sample box is not allowed
    assert found[2] <= 40


@pytest.mark.parametrize("at_start", [True, False])
def test_search_night_truncated_box_covers_the_visible_part_only(at_start):
    # 100 dip samples are visible against a 3 h (180-sample) box: only a box hanging over the
    # night's edge holds exactly them
    sl = slice(0, 100) if at_start else slice(NIGHT_N - 100, NIGHT_N)
    t, flux = _night(sl)
    found = search_night(t, flux, C, (3 * HOUR,), 15, 30)
    assert found is not None
    assert (found[5], found[6], found[2]) == (sl.start, sl.stop - 1, 100)


# -- what it finds -----------------------------------------------------------------------


def test_a_planted_one_percent_dip_in_mid_night_is_found():
    _t, cv = one_dip(night=4, start_h=2.5)
    out = detect({0: cv})
    assert len(out) == 1
    d = out[0]
    assert (d.detector, d.star_id, d.kind_guess) == ("single_transit", 0, "transit")
    assert d.det_id == "single_transit-s0-0"
    lo, hi = clipped_window(4, 2.5, 3.0)
    assert span_iou(d.t_start, d.t_end, lo, hi) >= 0.5
    assert d.score >= 6.0
    assert d.t_end % C == 0 and d.t_start % C == 0


@pytest.mark.parametrize("dur_h", [0.5, 1.0, 2.0, 5.0])
def test_other_durations_are_found(dur_h):
    _t, cv = one_dip(night=3, start_h=1.5, dur_h=dur_h, depth=0.02)
    out = detect({0: cv})
    assert len(out) == 1
    lo, hi = clipped_window(3, 1.5, dur_h)
    assert span_iou(out[0].t_start, out[0].t_end, lo, hi) >= 0.5


@pytest.mark.parametrize("start_h", [6.5, -1.5])  # past the night's end, before its start
def test_a_dip_half_past_the_night_edge_is_found_as_a_truncated_box(start_h):
    _t, cv = one_dip(night=4, start_h=start_h, dur_h=3.0, depth=0.02)
    out = detect({0: cv})
    assert len(out) == 1
    lo, hi = clipped_window(4, start_h, 3.0)
    assert hi - lo == pytest.approx(1.5 * HOUR, abs=C)
    assert span_iou(out[0].t_start, out[0].t_end, lo, hi) >= 0.5


@pytest.mark.parametrize("start_h", [-3.6, 7.6])  # 0.4 h (24 samples) of a 4 h dip is visible
def test_a_short_remnant_of_a_deep_dip_at_a_night_edge_is_found_exactly(start_h):
    # 24 samples are shorter than the shortest grid duration (0.5 h = 30): the box must hang
    # over the night's edge to hold exactly the remnant
    _t, cv = one_dip(night=4, start_h=start_h, dur_h=4.0, depth=0.1, noise=0.002)
    (d,) = detect({0: cv})
    ns = 4 * DAY
    lo, hi = (ns, ns + 24 * C) if start_h < 0 else (ns + (NIGHT_N - 24) * C, ns + NIGHT_N * C)
    assert (d.t_start, d.t_end) == (lo, hi)


def test_the_span_is_the_covered_samples_half_open():
    _t, cv = one_dip(night=4, start_h=2.5, dur_h=3.0, depth=0.1, noise=0.002)
    (d,) = detect({0: cv})
    assert (d.t_start, d.t_end) == (4 * DAY + int(2.5 * HOUR), 4 * DAY + int(5.5 * HOUR))


def test_one_detection_per_night_with_a_dip_and_det_ids_count_up():
    rng = np.random.default_rng(5)
    t = night_times(10)
    prof = dip_profile(t, 2, 2.0, 3.0, 0.015) * dip_profile(t, 7, 4.0, 2.0, 0.006)
    # guards off: with a second event this strong the residual test would read the star as a
    # sinusoid (see the module docstring), which is not what this test is about
    out = detect({0: curve(0, t, prof, 0.005, rng)}, guards=False)
    assert [d.det_id for d in out] == ["single_transit-s0-0", "single_transit-s0-1"]
    assert [d.t_start // DAY for d in out] == [2, 7]


def test_a_dip_below_the_threshold_is_not_reported():
    _t, cv = one_dip(depth=0.002)
    assert detect({0: cv}) == []
    assert detect({0: cv}, snr_threshold=2.0) != []


def test_threshold_is_the_reporting_level():
    _t, cv = one_dip()
    score = detect({0: cv})[0].score
    assert detect({0: cv}, snr_threshold=score - 0.01) != []
    assert detect({0: cv}, snr_threshold=score + 0.01) == []


def test_flagged_samples_are_not_searched():
    rng = np.random.default_rng(2)
    t = night_times(10)
    prof = dip_profile(t, 4, 2.5, 3.0, 0.02)
    q = np.zeros(len(t), dtype=np.uint8)
    q[(t >= 4 * DAY + 2 * HOUR) & (t < 4 * DAY + 6 * HOUR)] = 1
    assert detect({0: curve(0, t, prof, 0.005, rng, quality=q)}) == []


def test_a_planted_dip_on_a_field_star_survives_the_common_mode_division():
    t, rng, curves = make_field(n_stars=12)
    prof = dip_profile(t, 4, 2.5, 3.0, 0.015)
    curves[5] = curve(5, t, prof, 0.005, rng)
    out = detect(curves)
    assert [d.star_id for d in out] == [5]


def test_common_mode_offset_shared_by_the_field_is_not_a_transit():
    t, rng, _ = make_field()
    step = np.ones(len(t))
    for k in (2, 6):
        step[(t >= k * DAY + 5 * HOUR) & (t < k * DAY + 8 * HOUR)] = 0.9
    curves = {i: curve(i, t, step, 0.005, rng) for i in range(12)}
    assert detect(curves) == []


def test_a_common_mode_dip_on_one_night_is_not_a_transit():
    # one night, every star: guard (a) does not read this as a pulsation, so only the
    # division by the field's per-time median stands between it and twelve "transits"
    t, rng, _ = make_field()
    dip = np.ones(len(t))
    dip[(t >= 3 * DAY + 2 * HOUR) & (t < 3 * DAY + 5 * HOUR)] = 0.9
    curves = {i: curve(i, t, dip, 0.005, rng) for i in range(12)}
    assert detect(curves) == []


# -- what it must not report ----------------------------------------------------------------


def test_a_positive_flare_gives_nothing():
    rng = np.random.default_rng(4)
    t = night_times(10)
    x = np.clip(t - (5 * DAY + 3 * HOUR), 0, None) / (20 * 60.0)
    flare = np.where(t >= 5 * DAY + 3 * HOUR, 0.05 * np.exp(-x), 0.0)
    assert flare.max() == pytest.approx(0.05)
    assert detect({0: curve(0, t, 1.0 + flare, 0.005, rng)}) == []


@pytest.mark.parametrize("period_d", [0.3, 5.0 / 24.0, 3.0 / 24.0])
def test_a_sinusoid_gives_nothing_but_would_without_the_guard(period_d):
    rng = np.random.default_rng(3)
    t = night_times(10)
    prof = 1.0 + 0.02 * np.sin(2 * np.pi * t / (period_d * DAY) + 0.7)
    cv = curve(0, t, prof, 0.005, rng)
    assert detect({0: cv}) == []
    # ... and the shape test would refuse the rounded dips of a slow one, so it is off too
    assert detect({0: cv}, guards=False, shape_threshold=None) != []


EB = (2.0, 0.3, 5.0, 0.20, 0.08)  # P, epoch (d), eclipse duration (h), primary, secondary depth


def eb_with_dip(dip=None, seed=0, n_nights=20):
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = eclipsing(t, *EB)
    if dip is not None:
        prof = prof * dip_profile(t, *dip)
    return t, curve(0, t, prof, 0.005, rng)


def test_an_eclipsing_binary_gives_nothing_when_guard_b_is_on():
    _t, cv = eb_with_dip()
    assert detect({0: cv}) == []
    # the eclipses are 20 and 8 percent deep: without the guards every night reports one ...
    assert len(detect({0: cv}, guards=False, shape_threshold=None)) >= 10
    # ... and the shape test refuses them without the guards: they are V-shaped
    assert detect({0: cv}, guards=False) == []


def test_a_planted_dip_on_an_eclipsing_binary_is_found_beside_masked_eclipses():
    # night 6 holds the eclipse at hour 7.2 (centre) +- 2.5 h; the dip sits at hours 0.5 to 3.5
    _t, cv = eb_with_dip(dip=(6, 0.5, 3.0, 0.02))
    out = detect({0: cv})
    assert len(out) == 1
    lo, hi = clipped_window(6, 0.5, 3.0)
    assert span_iou(out[0].t_start, out[0].t_end, lo, hi) >= 0.5


def test_a_star_with_two_unequal_single_events_is_not_searched():
    # documented limit: the second event stays in the guard's residual, but the periodic search
    # rejects its best box for min_events, not as a pulsation, so there is no eclipse to mask
    # and no re-search (guard (b) reads the rejection reason)
    rng = np.random.default_rng(5)
    t = night_times(10)
    prof = dip_profile(t, 2, 2.0, 3.0, 0.015) * dip_profile(t, 7, 4.0, 2.0, 0.006)
    assert detect({0: curve(0, t, prof, 0.005, rng)}) == []


# -- the shape test ----------------------------------------------------------------------


def v_profile(t, night, centre_h, width_h, depth):
    """A V (triangle) dip, `width_h` wide at its base, centred `centre_h` after the night starts."""
    x = t - (night * DAY + centre_h * HOUR)
    return 1.0 - depth * np.clip(1.0 - 2.0 * np.abs(x) / (width_h * HOUR), 0.0, None)


def trapezoid_profile(t, night, start_h, dur_h, depth, ramp=0.1):
    """A transit with ingress and egress ramps of `ramp` of its duration, as the world draws it."""
    lo = night * DAY + start_h * HOUR
    r = ramp * dur_h * HOUR
    return 1.0 - depth * np.minimum(
        np.clip((t - lo) / r, 0, 1), np.clip((lo + dur_h * HOUR - t) / r, 0, 1)
    )


def one_event(prof, noise=0.005, seed=0, n_nights=10):
    t = night_times(n_nights)
    return curve(0, t, prof(t), noise, np.random.default_rng(seed))


def shapes(cv, **kw):
    """The statistic of every box the search reports for the star (shape test on, no refusal)."""
    det = SingleTransitDetector(shape_threshold=1e9, **kw)
    c = det._bls._clean(cv, C)
    boxes, _ = det._star_boxes(0, c, det._bls._common_mode({0: c}), C)
    return [b.shape for b in boxes]


V_MID = (4, 4.0, 3.0, 0.03)  # night, centre h, base width h, depth: S/N about 30, statistic 4.7
V_END = (4, 8.0, 4.0, 0.06)  # the same V cut in half by the night's end


def test_a_v_shaped_event_is_found_without_the_shape_test_and_refused_with_it():
    cv = one_event(lambda t: v_profile(t, *V_MID))
    assert len(detect({0: cv}, shape_threshold=None)) == 1
    assert detect({0: cv}) == []


def test_a_v_cut_by_the_night_edge_is_refused_too():
    # half a V is a ramp down to the edge; a half box is a step. The statistic tells them apart
    # when the ramp is long and the signal strong
    for centre in (8.0, 0.0):
        cv = one_event(lambda t, c=centre: v_profile(t, 4, c, 4.0, 0.06))
        assert len(detect({0: cv}, shape_threshold=None)) == 1
        assert detect({0: cv}) == []


def test_a_flat_bottomed_event_has_a_negative_statistic_and_a_v_a_positive_one():
    (flat,) = shapes(one_event(lambda t: dip_profile(t, 4, 2.5, 3.0, 0.02)))
    (vee,) = shapes(one_event(lambda t: v_profile(t, *V_MID)))
    assert flat < -1.0 and vee > 3.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_planet_with_ramps_is_not_refused(seed):
    cv = one_event(lambda t: trapezoid_profile(t, 4, 2.5, 3.0, 0.02), seed=seed)
    (stat,) = shapes(cv)
    assert stat < 0.5
    assert len(detect({0: cv})) == 1


def test_a_truncated_box_is_kept():
    # a transit half past the night's end: a step down to the edge, flat where it is seen
    cv = one_event(lambda t: dip_profile(t, 4, 6.5, 3.0, 0.02))
    assert shapes(cv)[0] < 0.5
    assert len(detect({0: cv})) == 1


def test_a_short_wing_of_an_eclipse_at_the_night_end_is_kept_the_limit_of_the_test():
    # a V of 1.6 h centred 0.4 h past the night's end leaves a 1 h ramp at 1.5 percent: the
    # statistic is 0.49 here and 0.4 to 2.4 across seeds, so the test cannot call it
    cv = one_event(lambda t: v_profile(t, 4, 8.4, 1.6, 0.05))
    assert shapes(cv) == [pytest.approx(0.49, abs=0.1)]
    assert len(detect({0: cv})) == 1


def test_one_v_night_removes_the_star_but_not_the_other_stars():
    # guards off: two strong events would read as a sinusoid. The box on night 7 is a transit by
    # itself; beside a V on night 2 it is read as another eclipse of the same star
    rng = np.random.default_rng(3)
    t = night_times(10)
    prof = v_profile(t, 2, 4.0, 3.0, 0.03) * dip_profile(t, 7, 2.5, 3.0, 0.015)
    both = curve(0, t, prof, 0.005, rng)
    lone = one_event(lambda t: dip_profile(t, 7, 2.5, 3.0, 0.015), seed=4)
    lone = LightCurve(1, lone.t, lone.flux, lone.flux_err, lone.quality)
    assert {d.star_id for d in detect({0: both, 1: lone}, guards=False, shape_threshold=None)} == {
        0,
        1,
    }
    assert [d.star_id for d in detect({0: both, 1: lone}, guards=False)] == [1]
    assert [d.star_id for d in detect({1: lone}, guards=False)] == [1]


def test_a_statistic_equal_to_the_threshold_refuses():
    cv = one_event(lambda t: v_profile(t, *V_MID))
    (stat,) = shapes(cv)
    assert detect({0: cv}, shape_threshold=stat) == []
    assert len(detect({0: cv}, shape_threshold=stat + 0.01)) == 1


def test_the_statistic_is_computed_only_when_the_test_is_on():
    cv = one_event(lambda t: dip_profile(t, 4, 2.5, 3.0, 0.02))
    det = SingleTransitDetector(shape_threshold=None)
    c = det._bls._clean(cv, C)
    boxes, _ = det._star_boxes(0, c, det._bls._common_mode({0: c}), C)
    assert [b.shape for b in boxes] == [None]
    assert shapes(cv)[0] is not None


def test_the_test_reads_the_night_of_the_box_only(monkeypatch):
    # the statistic is a single-night fit: the star's other nights (here a V on night 2 beside
    # the box on night 7) must not enter it, or the free level would be one for the whole star
    sizes = []
    real = BlsDetector._shape_excess

    def spy(self, t, flux, night, sigma, box):
        sizes.append((len(t), len(np.unique(night)), len(np.unique(t // DAY))))
        return real(self, t, flux, night, sigma, box)

    monkeypatch.setattr(BlsDetector, "_shape_excess", spy)
    t = night_times(10)
    prof = v_profile(t, 2, 4.0, 3.0, 0.03) * dip_profile(t, 7, 2.5, 3.0, 0.015)
    both = shapes(curve(0, t, prof, 0.005, np.random.default_rng(3)), guards=False)
    assert len(both) == 2
    assert sizes == [(NIGHT_N, 1, 1)] * 2


# -- determinism, order, speed ---------------------------------------------------------------


def test_determinism():
    _, cv = one_dip()
    assert detect({0: cv}) == detect({0: cv})


def test_output_is_ordered_by_star_id():
    t, rng, curves = make_field(n_stars=11)
    for sid in (9, 3):
        curves[sid] = curve(sid, t, dip_profile(t, 4, 2.5, 3.0, 0.015), 0.005, rng)
    out = detect(dict(reversed(list(curves.items()))))
    assert [d.star_id for d in out] == [3, 9]


def test_stars_are_independent_of_each_other_beyond_the_common_mode():
    _, cv = one_dip()
    t = night_times(10)
    other = curve(1, t, np.ones(len(t)), 0.005, np.random.default_rng(9))
    assert [d.det_id for d in detect({0: cv, 1: other})] == [d.det_id for d in detect({0: cv})]


def test_one_search_per_night(monkeypatch):
    calls = []
    real = st.search_night

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(st, "search_night", counting)
    t = night_times(10)
    rng = np.random.default_rng(0)
    curves = {i: curve(i, t, np.ones(len(t)), 0.005, rng) for i in range(12)}
    detect(curves)
    assert len(calls) == 12 * 10


def test_performance_60_stars_20_nights():
    # measured CPU time here: 0.55 s (slowest of three runs, host load above 70). The bound is
    # about four times that, so a 15-fold slowdown of the night search fails it
    t = night_times(20)
    rng = np.random.default_rng(0)
    curves = {}
    for i in range(60):
        prof = dip_profile(t, i % 20, 2.0, 3.0, 0.015) if i % 5 == 0 else np.ones(len(t))
        curves[i] = curve(i, t, prof, 0.005, rng)
    t0 = time.process_time()
    out = detect(curves)
    elapsed = time.process_time() - t0
    assert {d.star_id for d in out} == set(range(0, 60, 5))
    assert elapsed < 2.3, f"60 stars x 20 nights took {elapsed:.2f} s of CPU"
