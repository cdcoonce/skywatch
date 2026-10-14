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
    _pin_repeat_snr,
    _pulsator_field,
    curve,
    eclipsing,
    make_field,
    night_times,
    pulsator,
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
        {"min_night_h": -1.0},
        {"near_snr": 0},
        {"near_snr": -1.0},
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
    assert d.detrend_pulsation is True
    assert d.clip_residual is True
    assert d.min_night_h == 2.0
    assert d.leave_out_events is True
    assert d.near_snr == 5.0


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
    # the guard alone removes it: the shape test is off, or it would hide a dead guard
    assert detect({0: cv}, shape_threshold=None) == []
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
    # the guard alone removes it: the shape test is off, or it would hide a dead guard
    assert detect({0: cv}, shape_threshold=None) == []
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


def wing(seed):
    # a 1.6 h V centred 0.4 h past the night's end (the limit case above); only the noise varies
    return one_event(lambda t: v_profile(t, 4, 8.4, 1.6, 0.05), seed=seed)


def test_a_statistic_of_1_2_is_refused_and_one_of_0_9_is_kept():
    # pins the effective threshold between 0.9 and 1.2, not only the default attribute
    refused, kept = wing(14), wing(27)
    assert shapes(refused) == [pytest.approx(1.22, abs=0.05)]
    assert shapes(kept) == [pytest.approx(0.90, abs=0.05)]
    assert detect({0: refused}) == []
    assert len(detect({0: kept})) == 1


def test_sigma_is_the_nights_scatter_outside_the_box_not_the_whole_nights(monkeypatch):
    seen = []
    real = BlsDetector._shape_excess

    def spy(self, t, flux, night, sigma, box):
        seen.append(sigma)
        return real(self, t, flux, night, sigma, box)

    monkeypatch.setattr(BlsDetector, "_shape_excess", spy)
    cv = one_event(lambda t: v_profile(t, 4, 4.0, 5.0, 0.02))  # a V filling 5 h of the 8 h night
    shapes(cv)
    # the white noise is 0.005; the whole night's MAD (dip included) reads about 0.0075
    assert seen == [pytest.approx(0.005, rel=0.15)]


def test_a_weaker_v_night_still_removes_the_star():
    # guards off (two strong events would read as a sinusoid). The V on night 2 is the weaker box
    # (1.2 percent against 3 percent): the rule is "any box", not "the strongest box"
    t = night_times(10)
    prof = v_profile(t, 2, 4.0, 3.0, 0.012) * dip_profile(t, 7, 2.5, 3.0, 0.03)
    cv = curve(0, t, prof, 0.005, np.random.default_rng(3))
    assert len(detect({0: cv}, guards=False, shape_threshold=None)) == 2
    assert detect({0: cv}, guards=False) == []


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


def _boxes_calls(bls_detector, cv):
    """Sample counts of every ``_boxes`` call `single_transit` makes on one star's arrays."""
    det = SingleTransitDetector(detrend_pulsation=False)  # the eclipse mask is what is counted
    det._bls = bls_detector
    calls: list[int] = []
    real = det._boxes

    def spy(t, flux, night, cadence_s):
        calls.append(len(t))
        return real(t, flux, night, cadence_s)

    det._boxes = spy
    c = det._bls._clean(cv, C)
    det._star_boxes(0, c, det._bls._common_mode({0: c}), C)
    return calls


def test_a_detrended_box_the_repeat_test_refuses_leaves_the_star_to_the_eclipse_mask(monkeypatch):
    # guard (a) refuses this fast pulsator's night box. `bls` first accepts a detrended box and
    # the star is not searched (guard (b)); with the repeat test on and the box refused it keeps
    # its original "pulsation guard" candidate, so its dip windows are masked and the nights are
    # searched again, on fewer samples. A refusal that did not restore that candidate would be
    # read as "not a pulsation guard rejection" and the star dropped, with one search.
    _, cv = pulsator(0, period_d=0.5)
    _pin_repeat_snr(monkeypatch, 0.0)
    off = _boxes_calls(BlsDetector(), cv)
    assert len(off) == 1
    on = _boxes_calls(BlsDetector(trend_repeat_snr=5.0), cv)
    assert len(on) == 2 and on[1] < on[0] == off[0]


# -- pulsating hosts (issue #51) ---------------------------------------------------------


def pulsating_host(seed=0, dip=(4, 2.5, 3.0, 0.01), amp=0.015, period_d=0.9, n_nights=20):
    """A sinusoidal star (guard (a) refuses its night boxes) with one dip on one night."""
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = 1.0 + amp * np.sin(2 * np.pi * t / (period_d * DAY) + 0.7)
    if dip is not None:
        prof = prof * dip_profile(t, *dip)
    return t, curve(0, t, prof, 0.005, rng)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_single_transit_on_a_pulsating_host_is_found_after_the_detrend(seed):
    # a 1 percent, 3 h dip on a 1.5 percent pulsator: the raw nights are all refused by guard
    # (a); with the sinusoid subtracted the dip night is the only one that reads
    _t, cv = pulsating_host(seed)
    out = detect({0: cv})
    assert len(out) == 1
    lo, hi = clipped_window(4, 2.5, 3.0)
    assert span_iou(out[0].t_start, out[0].t_end, lo, hi) >= 0.6
    assert out[0].score >= 8.0 and out[0].kind_guess == "transit"
    assert detect({0: cv}, detrend_pulsation=False) == []


def test_a_pulsating_host_without_a_dip_gives_nothing():
    for seed in range(4):
        _t, cv = pulsating_host(seed, dip=None)
        assert detect({0: cv}) == []


def test_the_detrend_does_not_search_a_star_the_guard_accepts():
    # a dip on a quiet star never reaches the detrend: the sinusoid fit is not even tried
    _t, cv = one_dip()
    tried: list[int] = []
    real = BlsDetector._pulsation_trend

    def spy(self, a, cadence_s, box=None):
        tried.append(1)
        return real(self, a, cadence_s, box)

    st_det = SingleTransitDetector()
    st_det._bls._pulsation_trend = spy.__get__(st_det._bls)
    assert len(st_det.detect({0: cv}, C)) == 1
    assert tried == []


def test_the_sinusoid_is_fitted_again_around_the_box():
    # the first fit has the dip in it; the box is then left out of a second fit
    _t, cv = pulsating_host(0)
    calls: list = []
    real = BlsDetector._pulsation_trend

    def spy(self, a, cadence_s, box=None):
        calls.append(box)
        return real(self, a, cadence_s, box)

    det = SingleTransitDetector()
    det._bls._pulsation_trend = spy.__get__(det._bls)
    assert len(det.detect({0: cv}, C)) == 1
    assert calls[0] is None and calls[1] is not None
    lo, hi = clipped_window(4, 2.5, 3.0)
    assert lo <= calls[1].epoch_s <= hi
    assert calls[1].period_s > 1e8  # one event


def test_a_v_shaped_event_on_a_pulsating_host_is_refused_by_the_shape_test():
    # the same night, an eclipse-like V instead of a flat dip: the detrend finds it, the shape
    # test (run on the residual) refuses it
    rng = np.random.default_rng(0)
    t = night_times(20)
    prof = 1.0 + 0.015 * np.sin(2 * np.pi * t / (0.9 * DAY) + 0.7)
    prof = prof * v_profile(t, 4, 4.0, 3.0, 0.03)
    cv = curve(0, t, prof, 0.005, rng)
    assert len(detect({0: cv}, shape_threshold=None)) == 1
    assert detect({0: cv}) == []


def test_a_planet_with_several_events_on_a_pulsating_host_is_left_to_bls():
    # its other events stand out of the residual, so guard (a) still refuses it: the periodic
    # search is the one that finds this planet (see the bls tests of pulsating hosts)
    _t, cv = pulsator(0)
    assert detect({0: cv}) == []


# -- the other events of a planet on a pulsating host (issue #61) ------------------------------


def two_event_host(dips, seed=0, mode2=None, amp=0.015, period_d=0.9):
    """A sinusoidal star with one dip on each of several nights (`dips`: night, start_h, dur_h,
    depth), and optionally a second sinusoid `mode2` = (amplitude, period_d)."""
    rng = np.random.default_rng(seed)
    t = night_times(20)
    prof = 1.0 + amp * np.sin(2 * np.pi * t / (period_d * DAY) + 0.7)
    if mode2 is not None:
        prof = prof + mode2[0] * np.sin(2 * np.pi * t / (mode2[1] * DAY) + 1.0)
    for dip in dips:
        prof = prof * dip_profile(t, *dip)
    return t, curve(0, t, prof, 0.005, rng)


TWO_DIPS = [(4, 2.5, 3.0, 0.01), (11, 3.0, 3.0, 0.01)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_events_on_a_pulsating_host_are_both_found(seed):
    # the residual holds both dips: the guard read the second as a sinusoid and refused the
    # star (it was left to bls); with the other box's window left out each is a one-event box
    _t, cv = two_event_host(TWO_DIPS, seed)
    out = detect({0: cv})
    assert [d.t_start // DAY for d in out] == [4, 11]
    for d, (night, start_h, dur_h, _) in zip(out, TWO_DIPS, strict=True):
        lo, hi = clipped_window(night, start_h, dur_h)
        assert span_iou(d.t_start, d.t_end, lo, hi) >= 0.6
        assert d.score >= 8.0 and d.kind_guess == "transit"
    assert detect({0: cv}, leave_out_events=False) == []


def test_three_events_on_a_pulsating_host_are_found():
    _t, cv = two_event_host(TWO_DIPS + [(15, 2.0, 3.0, 0.012)])
    assert [d.t_start // DAY for d in detect({0: cv})] == [4, 11, 15]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_second_pulsation_mode_in_the_residual_still_refuses_the_star(seed):
    # leaving the events out does not blind the guard: the second mode is in what is left
    _t, cv = two_event_host(TWO_DIPS, seed, mode2=(0.004, 0.37))
    assert detect({0: cv}) == []


@pytest.mark.parametrize("seed", [0, 1])
def test_two_v_shaped_events_on_a_pulsating_host_are_refused_by_the_shape_test(seed):
    rng = np.random.default_rng(seed)
    t = night_times(20)
    prof = 1.0 + 0.015 * np.sin(2 * np.pi * t / (0.9 * DAY) + 0.7)
    prof = prof * v_profile(t, 4, 4.0, 3.0, 0.03) * v_profile(t, 11, 4.0, 3.0, 0.03)
    cv = curve(0, t, prof, 0.005, rng)
    assert len(detect({0: cv}, shape_threshold=None)) == 2
    assert detect({0: cv}) == []


def _residual_guard_calls(det, cv):
    """(samples, boxes) of every `_pulsating` call made on the detrended path's residual."""
    calls: list[tuple[int, int]] = []
    real = SingleTransitDetector._pulsating

    def spy(t, flux, night, boxes, cadence_s):
        calls.append((len(t), len(boxes)))
        return real(det, t, flux, night, boxes, cadence_s)

    det._pulsating = spy
    det.detect({0: cv}, C)
    return calls


def test_a_residual_the_guard_accepts_is_read_once():
    # one box: nothing to leave out, and a star the first read accepts is not read again
    _t, cv = pulsating_host(0)
    calls = _residual_guard_calls(SingleTransitDetector(), cv)
    assert len(calls) == 2  # the raw star, then its residual
    assert calls[1][1] == 1


def test_the_other_events_are_left_out_of_the_second_read_only():
    # two boxes whose first read refuses: the second read holds the strongest box alone, with
    # the other box's 1.5-times window (4.5 h of 60 s samples) cut out
    _t, cv = two_event_host(TWO_DIPS)
    kept = SingleTransitDetector()
    refused = SingleTransitDetector(leave_out_events=False)
    full = _residual_guard_calls(refused, cv)
    cut = _residual_guard_calls(kept, cv)
    assert full[1][1] == 2 and cut[1][1] == 2  # the first read sees both boxes
    assert cut[2][1] == 1  # the second one the strongest alone
    assert cut[2][0] == cut[1][0] - 270  # 1.5 x (3 h + 60 s) of samples: 270 of 271 cut
    assert len(full) == 2


def test_the_night_of_a_cut_box_is_levelled_again():
    # a dip over most of a night is that night's median; with its window cut out the rest of the
    # night sits above 1, a step the guard would read as a sinusoid, so the night is divided by
    # its median again; a night nothing was cut from is left as it is
    det = SingleTransitDetector()
    n = 480
    t = (np.arange(n * 3, dtype=np.int64) * C).reshape(3, n) + (np.arange(3) * DAY)[:, None]
    t, night = t.ravel(), np.repeat(np.arange(3), n)
    flux = np.ones((3, n))
    flux[1, 100:200] = 0.97  # night 1: the strongest box
    flux[2, 100:400] = 0.95  # night 2: a dip over 300 of 480 samples
    flux /= np.median(flux, axis=1)[:, None]
    flux[1] *= 1.01  # an uncut night at a level other than 1: dividing it again would show
    flux = flux.ravel()
    assert float(np.median(flux[night == 2][:60])) > 1.04  # the rest of night 2, before the cut
    boxes = [
        st.NightBox(1, int(t[n + 100]), int(t[n + 199]), 0.03, 100, 380, 0.005, 20.0),
        st.NightBox(2, int(t[2 * n + 100]), int(t[2 * n + 399]), 0.05, 300, 180, 0.005, 9.0),
    ]
    reads: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def spy(t_, flux_, night_, boxes_, cadence_s):
        reads.append((t_, flux_, night_))
        return True

    det._pulsating = spy
    assert det._residual_pulsating(t, flux, night, boxes, C) is True
    (_, _, _), (t2, f2, n2) = reads
    assert len(t2) == 3 * n - (450 - 0)  # night 2 loses 1.5 x 300 samples
    assert float(np.median(f2[n2 == 2])) == pytest.approx(1.0, abs=1e-12)
    assert np.array_equal(f2[n2 == 1], flux[night == 1])  # nothing cut from night 1
    assert np.array_equal(f2[n2 == 0], flux[night == 0])


def test_a_two_box_residual_the_plain_read_accepts_is_read_once():
    # a star the guard accepted is accepted as before: the plain read decides, and the second
    # read (the other box left out) is made only for a residual the plain read refuses
    det = SingleTransitDetector()
    n = 480
    t = (np.arange(n * 3, dtype=np.int64) * C).reshape(3, n) + (np.arange(3) * DAY)[:, None]
    t, night = t.ravel(), np.repeat(np.arange(3), n)
    flux = np.ones(3 * n)
    boxes = [
        st.NightBox(1, int(t[n + 100]), int(t[n + 199]), 0.03, 100, 380, 0.005, 20.0),
        st.NightBox(2, int(t[2 * n + 100]), int(t[2 * n + 199]), 0.03, 100, 380, 0.005, 9.0),
    ]
    reads: list[int] = []

    def spy(t_, flux_, night_, boxes_, cadence_s):
        reads.append(len(boxes_))
        return False

    det._pulsating = spy
    assert det._residual_pulsating(t, flux, night, boxes, C) is False
    assert reads == [2]  # one read, of both boxes
    # ... and a refused one is read again, with the strongest box alone
    reads.clear()
    det._pulsating = lambda *a: reads.append(len(a[3])) is None
    assert det._residual_pulsating(t, flux, night, boxes, C) is True
    assert reads == [2, 1]


def test_a_binary_with_several_planted_events_is_left_to_bls_after_the_eclipse_mask():
    # the leave-out is for the residual of a pulsator only: the guard that follows the eclipse
    # mask reads the star as it is, or an eclipsing binary with a few dips is claimed as a planet
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        t = night_times(20)
        prof = eclipsing(t, *EB)
        for night in (3, 9, 15):
            prof = prof * dip_profile(t, night, 0.3, 2.5, 0.02)
        assert detect({0: curve(0, t, prof, 0.005, rng)}) == []


def test_an_eclipsing_binary_still_gives_nothing_with_the_detrend():
    _t, cv = eb_with_dip()
    assert detect({0: cv}) == []
    assert detect({0: cv}, shape_threshold=None) == []


def _trend_boxes(cv):
    """The `box` argument of every `_pulsation_trend` call a full detect makes on one star."""
    calls: list = []
    real = BlsDetector._pulsation_trend

    def spy(self, a, cadence_s, box=None):
        calls.append(box)
        return real(self, a, cadence_s, box)

    det = SingleTransitDetector()
    det._bls._pulsation_trend = spy.__get__(det._bls)
    det.detect({0: cv}, C)
    return calls


def test_the_detrended_path_keeps_the_default_threshold():
    # a dip at a residual S/N of 7.67 is under the default bar of 8 on the detrended path too:
    # nothing at the default, found when the bar is lowered (the dip is really there)
    _t, cv = pulsating_host(0, dip=(4, 2.5, 3.0, 0.0036))
    assert detect({0: cv}) == []
    low = detect({0: cv}, snr_threshold=7.0)
    assert len(low) == 1
    assert 7.5 < low[0].score < 8.0


def test_the_refit_box_is_the_dip_in_epoch_and_duration():
    # the second fit leaves out a window the size of the dip, centred on it: not three times
    # the dip, not a window hanging off its left edge
    _t, cv = pulsating_host(0)
    calls = _trend_boxes(cv)
    assert calls[1] is not None
    lo, hi = clipped_window(4, 2.5, 3.0)
    assert abs(calls[1].epoch_s - (lo + hi) / 2) < 0.5 * HOUR
    assert abs(calls[1].duration_s - (hi - lo)) < 0.5 * HOUR


def test_the_refit_leaves_out_the_strongest_box_of_two():
    # two dips on one pulsator, night 4 at 1.2 percent and night 12 at 0.8: the refit sits on
    # the stronger one
    rng = np.random.default_rng(0)
    t = night_times(20)
    prof = 1.0 + 0.015 * np.sin(2 * np.pi * t / (0.9 * DAY) + 0.7)
    prof = prof * dip_profile(t, 4, 2.5, 3.0, 0.012) * dip_profile(t, 12, 2.5, 3.0, 0.008)
    calls = _trend_boxes(curve(0, t, prof, 0.005, rng))
    lo, hi = clipped_window(4, 2.5, 3.0)
    assert calls[1] is not None and lo - HOUR <= calls[1].epoch_s <= hi + HOUR


def test_a_dip_that_pulls_a_fast_sinusoid_is_still_found_after_the_refit():
    # a 0.4 d, 4 h, 1.2 percent dip drags a single fit of a short-period sinusoid (S/N about
    # 23.5 once the refit leaves the dip out); a single fit or a refit with no window misses it
    _t, cv = pulsating_host(0, dip=(4, 1.0, 4.0, 0.012), period_d=0.4)
    out = detect({0: cv})
    assert len(out) == 1 and out[0].kind_guess == "transit"


# -- flares on a pulsator's residual (issue #54) -----------------------------------------


def flared_host(seed=0, flare=(8, 4.0, 0.03, 60.0), dip=None, amp=0.015, period_d=0.9):
    """A pulsator with an exponential flare (night, start h, amplitude, decay minutes)."""
    rng = np.random.default_rng(seed)
    t = night_times(20)
    prof = 1.0 + amp * np.sin(2 * np.pi * t / (period_d * DAY) + 0.7)
    if dip is not None:
        prof = prof * dip_profile(t, *dip)
    night, start_h, height, decay_min = flare
    dt = t - (night * DAY + start_h * HOUR)
    prof = prof + np.where(dt >= 0, height * np.exp(-np.clip(dt, 0, None) / (decay_min * 60.0)), 0)
    return t, curve(0, t, prof, 0.005, rng)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_flare_the_pulsation_hid_from_the_first_clip_is_not_a_transit(seed):
    # a 3 percent flare with a 1 h decay on a 1.5 percent pulsator is under the first clip's bar
    # (the pulsation inflates its scale); once the sinusoid is subtracted the night holds a box
    # round it. Without the second clip that box is reported (S/N 9 to 12), with it the box
    # does not reach the bar once the flare is left out (about 4 to 6)
    _t, cv = flared_host(seed)
    off = detect({0: cv}, clip_residual=False)
    assert len(off) == 1 and off[0].score >= 8.0 and off[0].t_start // DAY == 8
    assert detect({0: cv}) == []


def test_the_bar_of_the_second_clip_is_the_detectors_threshold():
    _t, cv = flared_host(0)
    assert detect({0: cv}) == []
    kept = detect({0: cv}, snr_threshold=4.0)
    assert len(kept) == 1 and kept[0].t_start // DAY == 8
    assert (
        kept[0].score == detect({0: cv}, clip_residual=False)[0].score
    )  # the score is not changed


@pytest.mark.parametrize("night", [8, 4])
def test_a_short_flare_beside_a_real_dip_does_not_remove_it(night):
    # a 3 percent, 10 minute flare is flagged by the second clip but leaves no box: the transit
    # stays, with the score it has without the gate, on the flare's night and on another
    _t, cv = flared_host(0, flare=(8, 6.5, 0.03, 10.0), dip=(night, 0.2, 3.0, 0.012))
    out = detect({0: cv})
    assert len(out) == 1 and out[0].t_start // DAY == night and out[0].score > 20
    assert out == detect({0: cv}, clip_residual=False)


def test_the_second_clip_runs_only_on_the_detrended_path(monkeypatch):
    # a quiet star (guard (a) accepts it) never reaches the detrend, so the gate is not asked
    def refuse(self, *a, **k):
        raise AssertionError("the second clip ran on a star that was not detrended")

    monkeypatch.setattr(SingleTransitDetector, "_stands_unflared", refuse)
    _t, cv = one_dip()
    assert len(detect({0: cv})) == 1


def _night_with_box(n=200, lo=100, flare_at=None, flare_len=10):
    """`n` samples of white noise with a 20-sample, 5 percent dip at lo:lo+20; optional flare."""
    rng = np.random.default_rng(0)
    t = np.arange(n) * C
    x = 1.0 + 0.005 * rng.normal(size=n)
    x[lo : lo + 20] -= 0.05
    if flare_at is not None:
        x[flare_at : flare_at + flare_len] += 0.3
    box = st.NightBox(0, int(t[lo]), int(t[lo + 19]), 0.05, 20, n - 20, 0.005, 20.0)
    return t, x, np.zeros(n, dtype=np.int64), box


def test_a_box_the_second_clip_leaves_too_little_of_is_refused():
    det = SingleTransitDetector()
    t, x, night, box = _night_with_box()
    assert det._stands_unflared(t, x, night, box)  # nothing flagged: unchanged
    t, x, night, box = _night_with_box(flare_at=20)
    assert det._stands_unflared(t, x, night, box)  # a flare far from the box: it stands
    t, x, night, box = _night_with_box(flare_at=105)
    assert not det._stands_unflared(t, x, night, box)  # 10 of its 20 samples are the flare


def test_a_box_with_too_little_reference_left_is_refused():
    # 62 samples: the box holds 20, the reference 42; the 15-sample flare and its tail take it to 23 (min_out 30)
    det = SingleTransitDetector()
    t, x, night, box = _night_with_box(n=62, lo=20, flare_at=45, flare_len=15)
    assert not det._stands_unflared(t, x, night, box)
    assert SingleTransitDetector(min_out=20)._stands_unflared(t, x, night, box)


def _one_night_box(depth, spike_in_box=None, flare=None, n=200, lo=100, w=20, seed=0, noise=0.005):
    """White noise, a `w`-sample dip of `depth` at lo; optional +0.25 spike in the box, flare run."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * C
    x = 1.0 + noise * rng.normal(size=n)
    x[lo : lo + w] -= depth
    if spike_in_box is not None:
        x[lo + spike_in_box[0]] += spike_in_box[1]
    if flare is not None:
        x[flare[0] : flare[0] + flare[1]] += 0.3
    box = st.NightBox(0, int(t[lo]), int(t[lo + w - 1]), 0.05, w, n - w, 0.005, 20.0)
    return t, x, np.zeros(n, dtype=np.int64), box


def test_a_spike_inside_the_box_is_left_out_of_the_box_too():
    # a 30-sample box: clean S/N 13.4 (25 samples left, the spike and its tail out); with the
    # spike (+0.25) counted in the box it reads 6.0 and would be refused
    t, x, night, box = _one_night_box(0.0135, spike_in_box=(10, 0.25), w=30)
    assert SingleTransitDetector()._stands_unflared(t, x, night, box)


def test_the_clip_reads_the_night_of_the_box_only():
    # night 0 holds the box (clean S/N about 10) and a flare far from it; night 1 is far noisier.
    # The reference is night 0's own: the noisy night must not enter the box's scatter
    t, x, night, box = _one_night_box(0.0135, flare=(20, 10))
    rng = np.random.default_rng(1)
    x2 = 1.0 + 0.03 * rng.normal(size=len(t))
    tt = np.concatenate([t, t + 86400])
    xx = np.concatenate([x, x2])
    nn = np.concatenate([night, night + 1])
    assert SingleTransitDetector()._stands_unflared(tt, xx, nn, box)


def test_an_unflagged_night_is_not_cut_by_a_stricter_clip():
    # a pure-noise night with nothing above 4 sigma and a dip at S/N 8.13 (seed 23, depth 0.0095):
    # it stands. A clip at 2 sigma flags the noise's own upper tail and reads 7.9. Thin by
    # construction: it pins the scale of the clip, so the seed is part of the test
    t, x, night, box = _one_night_box(0.0095, seed=23)
    assert SingleTransitDetector()._stands_unflared(t, x, night, box)


def test_a_box_of_exactly_min_in_and_a_reference_of_exactly_min_out_stands():
    # 54 samples, a 15-sample box (min_in); the clip flags 9 (the 5 of the flare and its tail), so
    # 30 are left outside (min_out): both limits are met with no sample to spare
    rng = np.random.default_rng(0)
    n = 54
    t = np.arange(n) * C
    x = 1.0 + 0.002 * rng.normal(size=n)
    x[30:45] -= 0.05
    x[2:7] += 0.3
    box = st.NightBox(0, int(t[30]), int(t[44]), 0.05, 15, 39, 0.002, 30.0)
    night = np.zeros(n, dtype=np.int64)
    det = SingleTransitDetector()
    assert (det.min_in, det.min_out) == (15, 30)
    assert det._stands_unflared(t, x, night, box)


# -- nights too short to have a reference (issue #57) ------------------------------------


def _night_cut_short(n_short, depth=0.03, noise=0.002, seed=0, night=4):
    """Ten nights, night `night` cut to its first `n_short` samples, a 40-sample dip in it.

    The dip sits at samples 20 to 59 of the short night, so it is a clean dip with 30 or more
    samples to spare outside it from 90 samples of night on: only the length of the night varies.
    """
    rng = np.random.default_rng(seed)
    t = night_times(10)
    t = t[~((t >= night * DAY + n_short * C) & (t < (night + 1) * DAY))]
    prof = 1.0 - depth * ((t >= night * DAY + 20 * C) & (t < night * DAY + 60 * C))
    return curve(0, t, prof, noise, rng)


def test_a_plateau_that_fills_a_short_night_is_not_a_transit():
    # a hot pixel (flux stuck at 2.2 times the star) covers the last 59 of 96 samples of a night:
    # the night's median is the plateau, so the 37 normal samples before it read as a 55 percent
    # "dip" that nothing above the level can flag (seed 712 star 75 of the development runs)
    rng = np.random.default_rng(0)
    t = night_times(10)
    t = t[~((t >= 4 * DAY + 96 * C) & (t < 5 * DAY))]
    prof = np.where((t >= 4 * DAY + 37 * C) & (t < 5 * DAY), 2.2, 1.0)
    cv = curve(0, t, prof, 0.005, rng)
    found = detect({0: cv}, min_night_h=0.0)
    assert [(d.t_start, d.t_end) for d in found] == [(4 * DAY, 4 * DAY + 37 * C)]
    assert found[0].score > 100
    assert detect({0: cv}) == []


def test_a_night_is_searched_from_two_hours_of_samples_on():
    # the bar is the samples a night holds, not a clock span: 120 samples at 60 s is two hours
    assert len(detect({0: _night_cut_short(120)})) == 1
    assert detect({0: _night_cut_short(119)}) == []
    # ... and the same dip on the same short night is found when the bar is switched off
    assert len(detect({0: _night_cut_short(119)}, min_night_h=0.0)) == 1


def test_the_short_night_bar_is_in_hours_at_any_cadence():
    # at 120 s a night needs 60 samples for the same two hours
    rng = np.random.default_rng(0)
    cadence = 2 * C

    def short(n_short):
        t = np.concatenate(
            [
                k * DAY + np.arange(NIGHT_N // 2 if k != 4 else n_short, dtype=np.int64) * cadence
                for k in range(10)
            ]
        )
        prof = 1.0 - 0.03 * ((t >= 4 * DAY + 10 * cadence) & (t < 4 * DAY + 30 * cadence))
        return curve(0, t, prof, 0.002, rng)

    assert len(SingleTransitDetector().detect({0: short(60)}, cadence)) == 1
    assert SingleTransitDetector().detect({0: short(59)}, cadence) == []


def test_a_full_night_is_untouched_by_the_short_night_bar():
    _, cv = one_dip()
    assert detect({0: cv}) == detect({0: cv}, min_night_h=0.0) != []


# -- the night bar acts in every search of a star, not only the first (issue #57) ------------


def _pulsating_host_with_a_fragment(seed, depth=0.02):
    """`pulsating_host` plus a night (9) cut to 96 samples whose first 40 hold a 2 percent dip.

    The dip is a box on a night under the 120 samples the bar asks for. It is only a way to put
    a box on a short night: the pulsator's raw nights are all refused by guard (a), so the box on
    night 4 is found in the search that follows the sinusoid's subtraction.
    """
    rng = np.random.default_rng(seed)
    t = night_times(20)
    t = t[~((t >= 9 * DAY + 96 * C) & (t < 10 * DAY))]
    prof = 1.0 + 0.015 * np.sin(2 * np.pi * t / (0.9 * DAY) + 0.7)
    prof = prof * dip_profile(t, 4, 2.5, 3.0, 0.01)
    prof = prof * (1.0 - depth * ((t >= 9 * DAY + 20 * C) & (t < 9 * DAY + 60 * C)))
    return curve(0, t, prof, 0.005, rng)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_bar_is_also_in_the_search_that_follows_the_pulsation_subtraction(seed):
    # seeds 375 star 80, 376 star 37, 712 star 75 and 466 star 104 of the development runs were
    # pulsator hosts whose short-night plateau only this second search saw
    cv = _pulsating_host_with_a_fragment(seed)
    out = detect({0: cv})
    assert [d.t_start // DAY for d in out] == [4]
    # the fragment is a box that the same search reports when the bar is off
    assert sorted(d.t_start // DAY for d in detect({0: cv}, min_night_h=0.0)) == [4, 9]


def _eclipsing_binary_with_a_fragment(seed):
    """An eclipsing binary (eclipse centred at hour 2 of every night) with night 6 cut to 360
    samples and a 3 percent, 40-sample dip at samples 290 to 329 of it.

    Night 6 holds 360 samples, so the first search reads it. The eclipse mask removes the first
    4 to 5 hours of the night and leaves under two hours: the dip sits in what is left.
    """
    rng = np.random.default_rng(seed)
    t = night_times(20)
    t = t[~((t >= 6 * DAY + 360 * C) & (t < 7 * DAY))]
    prof = eclipsing(t, 2.0, 2.0 / 24.0, 5.0, 0.20, 0.08)
    prof = prof * (1.0 - 0.03 * ((t >= 6 * DAY + 290 * C) & (t < 6 * DAY + 330 * C)))
    return curve(0, t, prof, 0.005, rng)


@pytest.mark.parametrize("seed", [0, 1])
def test_the_bar_is_also_in_the_search_that_follows_the_eclipse_mask(seed, caplog):
    # a night the mask leaves under two hours is not searched (documented: puls-seed466 star 14,
    # a planet host, lost 94 samples left; the same placement keeps seed 804 star 78, a hot pixel
    # at S/N 135, and four false boxes out)
    cv = _eclipsing_binary_with_a_fragment(seed)
    with caplog.at_level("INFO", logger="skywatch.detectors.single_transit"):
        assert detect({0: cv}) == []
    assert not any("eclipses masked" in r.getMessage() for r in caplog.records)  # no box stood
    # ... and without the bar the fragment is searched (after the mask) and reports the dip
    with caplog.at_level("INFO", logger="skywatch.detectors.single_transit"):
        found = detect({0: cv}, min_night_h=0.0)
    assert [d.t_start // DAY for d in found] == [6]
    assert any("eclipses masked" in r.getMessage() for r in caplog.records)


def test_single_transit_passes_the_cadence_so_the_common_mode_is_the_pulsation_free_one(
    monkeypatch,
):
    # issue #59: `single_transit` takes the same common mode as `bls` (it passed no cadence in
    # #57, and so always the plain median); the share gate keeps a standard field on the plain one
    seen = []
    real = BlsDetector._common_mode

    def spy(self, cleaned, cadence_s=None):
        seen.append(cadence_s)
        return real(self, cleaned, cadence_s)

    monkeypatch.setattr(BlsDetector, "_common_mode", spy)
    curves = make_field(n_stars=11)[2]
    detect(curves)
    assert seen == [C]


def test_in_a_field_of_pulsators_single_transit_divides_by_the_median_of_the_freed_stars(
    monkeypatch,
):
    # the field handed to every star is `bls`'s, not the plain median (a wobble of the 40 pulsators)
    curves = _pulsator_field()
    got = []
    real = SingleTransitDetector._star

    def spy(self, star_id, c, common_mode, cadence_s):
        got.append(common_mode)
        return real(self, star_id, c, common_mode, cadence_s)

    monkeypatch.setattr(SingleTransitDetector, "_star", spy)
    detect(curves)
    bls = BlsDetector()
    cleaned = {k: bls._clean(v, C) for k, v in curves.items()}
    freed = bls._common_mode(cleaned, C)
    plain = bls._common_mode(cleaned)
    assert got and all(g is not None for g in got)
    np.testing.assert_array_equal(got[0][1], freed[1])
    assert not np.array_equal(freed[1], plain[1])


def _mixed_field(n_pulsators, n_flat, n_nights=10, seed=3):
    """`n_pulsators` stars with a sinusoid (the pulsator field's) and `n_flat` flat ones."""
    curves = dict(_pulsator_field(n_stars=n_pulsators, n_nights=n_nights, seed=seed))
    t = night_times(n_nights)
    rng = np.random.default_rng(7)
    for j in range(n_flat):
        i = n_pulsators + j
        curves[i] = curve(i, t, np.ones(len(t)), 0.001, rng)
    return curves


def _handed_common_modes(curves, monkeypatch):
    got = []
    real = SingleTransitDetector._star

    def spy(self, star_id, c, common_mode, cadence_s):
        got.append(common_mode)
        return real(self, star_id, c, common_mode, cadence_s)

    monkeypatch.setattr(SingleTransitDetector, "_star", spy)
    detect(curves)
    return got


def _plain_and_freed(curves):
    bls = BlsDetector()
    cleaned = {k: bls._clean(v, C) for k, v in curves.items()}
    return bls._common_mode(cleaned), bls._common_mode(cleaned, C)


def test_a_minority_of_pulsators_leaves_the_plain_median(monkeypatch):
    # 10 of 40 stars carry a sinusoid (0.25, the standard-field range): below the share of `bls`,
    # so a standard field keeps the plain median and `single_transit` stays byte-identical to #57
    curves = _mixed_field(10, 30)
    plain, freed = _plain_and_freed(curves)
    got = _handed_common_modes(curves, monkeypatch)
    np.testing.assert_array_equal(got[0][1], plain[1])
    np.testing.assert_array_equal(freed[1], plain[1])


def test_a_clear_majority_of_pulsators_below_nine_tenths_takes_the_freed_median(monkeypatch):
    # 16 pulsators of 40 put 35 stars past the fit (0.875): above the share of 0.5, below 0.9; the
    # freed median differs from the plain one, and it is the one handed to every star
    curves = _mixed_field(16, 24)
    plain, freed = _plain_and_freed(curves)
    assert not np.array_equal(freed[1], plain[1])
    got = _handed_common_modes(curves, monkeypatch)
    np.testing.assert_array_equal(got[0][1], freed[1])


def test_the_cadence_handed_on_is_the_one_detect_was_given(monkeypatch):
    seen = []
    real = BlsDetector._common_mode

    def spy(self, cleaned, cadence_s=None):
        seen.append(cadence_s)
        return real(self, cleaned, cadence_s)

    monkeypatch.setattr(BlsDetector, "_common_mode", spy)
    SingleTransitDetector().detect(_mixed_field(0, 12), 120)
    assert seen == [120]


# -- a pulsator whose raw box is near the bar (issue #63) -----------------------------------


def fast_pulsator(seed=0, amp=0.03, dip=(4, 2.5, 3.0, 0.01)):
    """An 86 minute pulsator: its pulsation sits in the raw scatter, so a 1 percent dip reads low."""
    return pulsating_host(seed, dip=dip, amp=amp, period_d=0.06)


def _raw_best(cv):
    """The strongest raw box of one star at any S/N, as the first search of the detector sees it."""
    det = SingleTransitDetector()
    t, flux, night, _ = det._bls._prepare(det._bls._clean(cv, C), None)
    return max(b.snr for b in det._boxes(t, flux, night, C, floor=1.0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_pulsator_whose_raw_box_is_under_the_bar_is_still_detrended(seed):
    # the raw box reads 6.3 (the pulsation is in the scatter the S/N is measured against), under
    # the bar of 8 and over the floor of 5; the residual reads 20 or more
    _t, cv = fast_pulsator(seed)
    assert 5.0 < _raw_best(cv) < 8.0
    assert detect({0: cv}, near_snr=None) == []
    out = detect({0: cv})
    assert len(out) == 1 and out[0].score >= 8.0 and out[0].kind_guess == "transit"
    lo, hi = clipped_window(4, 2.5, 3.0)
    assert span_iou(out[0].t_start, out[0].t_end, lo, hi) >= 0.6


def test_the_floor_is_the_raw_s_n_a_star_needs_to_be_given_to_the_detrend():
    _t, cv = fast_pulsator(0)
    assert 6.0 < _raw_best(cv) < 7.0
    assert len(detect({0: cv}, near_snr=6.0)) == 1
    assert detect({0: cv}, near_snr=7.0) == []  # the raw box is under the floor: not tried
    assert detect({0: cv}, near_snr=8.0) == []  # the floor at the bar is the old behaviour
    # a floor over the bar is the bar: a dip of 8.7 on a quiet star is found as before
    _t, quiet = one_dip(depth=0.0042)
    assert 8.0 < _raw_best(quiet) < 9.0
    assert len(detect({0: quiet}, near_snr=9.0)) == 1
    assert detect({0: quiet}, near_snr=9.0) == detect({0: quiet}, near_snr=None)


def test_the_floor_cannot_lower_the_bar_the_residual_has_to_reach():
    # a 0.36 percent dip reads 7.7 on the residual of a pulsator: under the bar however low the
    # floor, and found only when the bar itself is lowered
    _t, cv = fast_pulsator(0, dip=(4, 2.5, 3.0, 0.0036))
    assert detect({0: cv}, near_snr=1.0) == []
    assert len(detect({0: cv}, near_snr=1.0, snr_threshold=7.0)) == 1


def test_a_quiet_star_with_a_box_near_the_bar_is_not_detrended():
    # guard (a) accepts a quiet star's box: there is nothing to subtract and the fit is not tried
    _t, cv = one_dip(depth=0.0034)
    assert 5.0 < _raw_best(cv) < 8.0
    tried: list[int] = []
    real = BlsDetector._pulsation_trend

    def spy(self, a, cadence_s, box=None):
        tried.append(1)
        return real(self, a, cadence_s, box)

    det = SingleTransitDetector()
    det._bls._pulsation_trend = spy.__get__(det._bls)
    assert det.detect({0: cv}, C) == []
    assert tried == []


def test_a_star_with_a_box_at_the_bar_reads_as_before():
    # the floor only matters for a star with no box at the bar: the detections of a host whose raw
    # box reaches 8 do not depend on it
    _t, cv = pulsating_host(0)
    assert _raw_best(cv) >= 8.0
    assert detect({0: cv}) == detect({0: cv}, near_snr=None)
    _t, cv = one_dip()
    assert detect({0: cv}) == detect({0: cv}, near_snr=None) and len(detect({0: cv})) == 1


def test_the_floor_is_off_without_the_guards_or_the_detrend():
    _t, cv = fast_pulsator(0)
    assert detect({0: cv}, guards=False) == detect({0: cv}, guards=False, near_snr=None)
    assert detect({0: cv}, detrend_pulsation=False) == []


def test_the_near_floor_costs_no_second_search_of_a_night(monkeypatch):
    # the first search reads every night once, at the floor, and the bar is applied to its result
    _t, cv = fast_pulsator(0)
    calls: list[int] = []
    real = st.search_night

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(st, "search_night", spy)
    detect({0: cv}, near_snr=None, detrend_pulsation=False)
    plain = len(calls)
    calls.clear()
    detect({0: cv}, detrend_pulsation=False)
    assert len(calls) == plain
