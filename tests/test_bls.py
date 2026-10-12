import math
import time
import warnings

import numpy as np
import pytest

from skywatch.detectors import REGISTRY, Detector, bls, load_all
from skywatch.detectors.bls import (
    BlsDetector,
    Box,
    Candidate,
    _circular_runs,
    _is_alias,
    _renormalise_nights,
    flare_windows,
    frequency_grid,
    in_box,
    in_windows,
    night_contrasts,
    positive_outliers,
    refine,
)
from skywatch.scoring.match import span_iou
from skywatch.types import LightCurve

C = 60
DAY = 86400
NIGHT_N = 480  # 8 h at 60 s


def night_times(n_nights, cadence=C, n=NIGHT_N, skip=()):
    """One global int64 axis: night k covers [k * DAY, k * DAY + n * cadence)."""
    parts = [
        k * DAY + np.arange(n, dtype=np.int64) * cadence for k in range(n_nights) if k not in skip
    ]
    return np.concatenate(parts)


def box_profile(t, period_s, epoch_s, duration_s, depth):
    """Multiplicative box: 1 - depth inside |phase - epoch| <= duration / 2."""
    ph = np.mod(t - epoch_s + period_s / 2.0, period_s) - period_s / 2.0
    return 1.0 - depth * (np.abs(ph) <= duration_s / 2.0)


def curve(star_id, t, profile, noise, rng, quality=None):
    flux = 1000.0 * profile * (1.0 + rng.normal(0.0, noise, len(t)))
    q = np.zeros(len(t), dtype=np.uint8) if quality is None else quality
    return LightCurve(star_id, t, flux, np.full(len(t), 5.0), q)


def planted(n_nights=10, seed=0, period_d=2.3, epoch_d=0.14, depth=0.01, dur_h=3.0, noise=0.005):
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = box_profile(t, period_d * DAY, epoch_d * DAY, dur_h * 3600.0, depth)
    return t, prof, curve(0, t, prof, noise, rng)


def planted_windows(t, period_d, epoch_d, dur_h):
    """Planted events clipped to the nights, as the world's truth labels are: half-open."""
    period, epoch, half = period_d * DAY, epoch_d * DAY, dur_h * 1800.0
    out = []
    for n in range(-1, int(t[-1] // period) + 2):
        mid = epoch + n * period
        for k in range(math.floor((mid - half) / DAY), math.floor((mid + half) / DAY) + 1):
            ns, ne = k * DAY, k * DAY + NIGHT_N * C
            lo, hi = max(mid - half, ns), min(mid + half, ne)
            if hi > lo and np.any((t >= lo) & (t < hi)):
                out.append((int(lo), int(hi)))
    return out


def detect(curves, **kw):
    return BlsDetector(**kw).detect(curves, C)


# -- registration, protocol, validation ------------------------------------------------


def test_registered_and_a_detector():
    assert load_all()["bls"] is BlsDetector
    assert REGISTRY["bls"] is BlsDetector
    d: Detector = BlsDetector()
    assert d.name == "bls"


@pytest.mark.parametrize(
    "kw",
    [
        {"min_period_d": 0},
        {"max_period_d": 0.4},
        {"durations_h": ()},
        {"durations_h": (1.0, -1.0)},
        {"oversample": 0},
        {"snr_threshold": 0},
        {"min_events": 0},
        {"clip_sigma": 0},
        {"guard_fap": 0},
        {"guard_fap": 1},
        {"guard_ratio": 0},
        {"guard_f_min_per_d": 0},
        {"guard_f_min_per_d": 5, "guard_f_max_per_d": 5},
        {"tail_sigma": -1},
    ],
)
def test_constructor_validation(kw):
    with pytest.raises(ValueError):
        BlsDetector(**kw)


def test_empty_and_tiny_curves():
    t = np.arange(3, dtype=np.int64) * C
    tiny = LightCurve(1, t, np.ones(3), np.ones(3), np.zeros(3, dtype=np.uint8))
    empty = LightCurve(2, t[:0], np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.uint8))
    assert detect({1: tiny, 2: empty}) == []
    assert detect({}) == []


# -- the statistic ---------------------------------------------------------------------


def test_snr_formula_on_a_noise_free_box():
    # 3 events of 3 h (181 samples each: both window ends are samples) depth 0.02
    period, epoch, dur, depth = 2.0 * DAY, 4.0 * 3600.0, 3 * 3600.0, 0.02
    # nights 0..5: mids at 4 h, 2 d + 4 h and 4 d + 4 h fall in nights 0, 2 and 4
    t = night_times(6)
    flux = box_profile(t, period, epoch, dur, depth)
    guess = Box(period, epoch, dur, 0.0, 0, len(t), 0.0)
    box = refine(t, flux, guess)
    assert box.n_in == 3 * 181
    assert box.depth == pytest.approx(depth, rel=1e-12)
    sigma = 0.004
    assert box.snr(sigma) == pytest.approx(0.02 * math.sqrt(3 * 181) / 0.004, rel=1e-12)
    assert box.snr(sigma) == pytest.approx(5.0 * math.sqrt(543), rel=1e-12)


def test_reported_snr_is_depth_sqrt_nin_over_sigma():
    _, _, cv = planted()
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected is None
    assert cand.snr == pytest.approx(cand.box.depth * math.sqrt(cand.box.n_in) / cand.sigma)
    out = detect({0: cv})
    assert {d.score for d in out} == {cand.snr}


def test_frequency_grid_is_log_uniform_with_the_documented_step():
    base, dur = 10 * DAY, 1800.0
    p = frequency_grid(base, 0.5 * DAY, 5 * DAY, dur, 3.0)
    assert p[0] == 0.5 * DAY and p[-1] == 5 * DAY
    step = (dur / base) / 3.0
    assert np.diff(np.log(p))[:-1] == pytest.approx(step, rel=1e-9)


# -- finding a planted transit ---------------------------------------------------------


def test_finds_planted_transit_one_detection_per_in_night_event():
    t, _, cv = planted()
    cand = BlsDetector().search(cv, C)
    assert cand is not None
    assert cand.box.period_s == pytest.approx(2.3 * DAY, rel=0.01)
    out = detect({0: cv})
    wins = planted_windows(t, 2.3, 0.14, 3.0)
    assert len(wins) == 3
    assert len(out) == len(wins)
    assert [d.det_id for d in out] == [f"bls-s0-{k}" for k in range(len(out))]
    for d in out:
        assert d.detector == "bls" and d.star_id == 0 and d.kind_guess == "transit"
        assert d.score >= 7.0
    for lo, hi in wins:
        ious = [span_iou(d.t_start, d.t_end, lo, hi) for d in out]
        assert sum(i >= 0.5 for i in ious) == 1
    assert [d.t_start for d in out] == sorted(d.t_start for d in out)


def test_spans_are_half_open_on_the_sample_grid():
    _, _, cv = planted()
    for d in detect({0: cv}):
        assert d.t_start in cv.t
        assert d.t_end - C in cv.t


def test_works_at_another_cadence():
    cad = 120
    rng = np.random.default_rng(3)
    t = np.concatenate([k * DAY + np.arange(240, dtype=np.int64) * cad for k in range(10)])
    prof = box_profile(t, 2.3 * DAY, 0.14 * DAY, 3 * 3600.0, 0.01)
    cv = curve(0, t, prof, 0.005, rng)
    out = BlsDetector().detect({0: cv}, cad)
    assert len(out) == 3
    assert all(d.t_end - d.t_start >= cad for d in out)


def test_single_in_night_event_at_three_nights_yields_nothing():
    rng = np.random.default_rng(1)
    t = night_times(3)
    # period 1.7 d, epoch 0.14 d: one event (night 0); the next lands in the day gap
    prof = box_profile(t, 1.7 * DAY, 0.14 * DAY, 3 * 3600.0, 0.01)
    cv = curve(0, t, prof, 0.005, rng)
    assert int(((t >= 0) & (prof < 1)).sum()) > 100  # the dip is really in the data
    assert detect({0: cv}) == []
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.snr > 7.0 and cand.rejected == "min_events"
    # one event is enough when the user asks for one
    assert len(detect({0: cv}, min_events=1)) == 1


def test_night_contrasts_by_hand():
    # nights 0, 1, 2 of 4 samples; the box covers samples 1-2 of every night; night 1 is flat
    t = np.array([0, 60, 120, 180, 86400, 86460, 86520, 86580, 172800, 172860, 172920, 172980])
    flux = np.array([1.0, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 1.0])
    night = np.repeat([0, 1, 2], 4)
    box = Box(86400.0, 90.0, 60.0, 0.0, 0, 12, 0.0)  # |phase - 90 s| <= 30 s: samples at 60, 120
    depth, weight = night_contrasts(t.astype(float), flux, night, box)
    assert depth == pytest.approx([0.1, 0.0, 0.2])
    assert weight == pytest.approx([2 * 2 / 4] * 3)
    # a night with nothing outside the box (or nothing inside) has no contrast and is left out
    t2 = np.concatenate([t, [3 * 86400 + 60, 3 * 86400 + 120]]).astype(float)
    f2 = np.concatenate([flux, [0.5, 0.5]])
    n2 = np.concatenate([night, [3, 3]])
    depth2, _ = night_contrasts(t2, f2, n2, box)
    assert len(depth2) == 3


def test_events_are_exact_half_open_spans():
    # a 1 h box every day at 04:00: samples 03:30..04:30 inclusive are in it
    t = night_times(3).astype(float)
    night = np.repeat(np.arange(3), NIGHT_N)
    box = Box(float(DAY), 4 * 3600.0, 3600.0, 0.01, 61, len(t), 1.0)
    ev = BlsDetector._events(t, night, box, C)
    assert ev == [(k * DAY + 12600, k * DAY + 16200 + C, k) for k in range(3)]


def test_local_snr_gates_a_long_box_that_snr_alone_would_pass():
    # a 6 h event in 8 h nights leaves a short reference: snr_local is well below snr
    rng = np.random.default_rng(7)
    t = night_times(8)
    prof = box_profile(t, 2.0 * DAY, 0.1 * DAY, 6 * 3600.0, 0.004)
    cv = curve(0, t, prof, 0.01, rng)
    cand = BlsDetector(snr_threshold=1.0).search(cv, C)
    assert cand.box.duration_s > 5 * 3600 and cand.snr_local < 0.75 * cand.snr
    mid = (cand.snr + cand.snr_local) / 2
    assert BlsDetector(snr_threshold=mid).search(cv, C).rejected == "local snr"
    assert BlsDetector(snr_threshold=cand.snr_local * 0.9).search(cv, C).rejected != "local snr"


def test_short_box_has_local_snr_close_to_snr():
    _, _, cv = planted(dur_h=1.0, depth=0.02)
    cand = BlsDetector().search(cv, C)
    assert cand.rejected is None and cand.snr_local > 0.9 * cand.snr


def test_each_event_night_must_show_the_dip_on_its_own():
    _, _, cv = planted()
    cand = BlsDetector().search(cv, C)
    assert cand.rejected is None and cand.supported_nights == 3
    # at default support_snr every night qualifies; demand an impossible per-night S/N
    assert BlsDetector(support_snr=1e9).search(cv, C).rejected == "support"
    assert detect({0: cv}, support_snr=1e9) == []


def test_min_events_counts_nights_not_events():
    _, _, cv = planted()
    assert detect({0: cv}, min_events=4) == []
    assert len(detect({0: cv}, min_events=3)) == 3


def test_pure_noise_yields_nothing():
    rng = np.random.default_rng(5)
    t = night_times(10)
    assert detect({0: curve(0, t, np.ones(len(t)), 0.005, rng)}) == []


# -- pulsation guard -------------------------------------------------------------------


@pytest.mark.parametrize("period_d,amp", [(1.3, 0.02), (5 / 24, 0.01), (0.07, 0.02)])
def test_sinusoid_is_not_a_transit(period_d, amp):
    rng = np.random.default_rng(2)
    t = night_times(10)
    prof = 1.0 + amp * np.sin(2 * np.pi * t / (period_d * DAY) + 0.7)
    out = detect({0: curve(0, t, prof, 0.005, rng)})
    assert out == []


def test_guard_is_what_rejects_the_sinusoid():
    rng = np.random.default_rng(2)
    t = night_times(10)
    prof = 1.0 + 0.02 * np.sin(2 * np.pi * t / (1.3 * DAY) + 0.7)
    cv = curve(0, t, prof, 0.005, rng)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.snr > 7.0
    assert cand.rejected == "pulsation guard" and cand.excess > 100.0
    # without the guard it is a "transit" (the shape test would also refuse it: it is no box)
    assert len(detect({0: cv}, guard_fap=None, shape_threshold=None)) > 0
    assert detect({0: cv}, guard_fap=None) == []


@pytest.mark.parametrize("kw", [{"guard_ratio": 1e9}, {"guard_fap": 1e-300}])
def test_either_half_of_the_guard_rejects_a_strong_sinusoid_alone(kw):
    # kw switches one test off (impossible ratio / impossible false-alarm level)
    rng = np.random.default_rng(2)
    t = night_times(10)
    prof = 1.0 + 0.02 * np.sin(2 * np.pi * t / (1.3 * DAY) + 0.7)
    cand = BlsDetector(**kw).search(curve(0, t, prof, 0.005, rng), C)
    assert cand.rejected == "pulsation guard"


def test_planted_transit_passes_the_guard_with_white_residuals():
    _, _, cv = planted()
    cand = BlsDetector().search(cv, C)
    assert cand.excess is not None and cand.excess < 10.0  # false-alarm level is ~13


# -- outliers ---------------------------------------------------------------------------


def test_positive_outliers_follow_a_flare_into_its_tail_and_never_mask_dips():
    rng = np.random.default_rng(0)
    n, peak, tau = 480, 200, 10.0
    x = 1.0 + rng.normal(0.0, 0.005, n)
    k = np.arange(n) - peak
    x += np.where(k >= 0, 0.2 * np.exp(-np.maximum(k, 0) / tau), 0.0)  # 40 sigma flare
    x[300:360] -= 0.05  # a 10 sigma dip
    mask = positive_outliers(x, clip_sigma=4.0, tail_sigma=1.0)
    assert mask[peak : peak + 35].all()  # 0.2 * exp(-3.5) is still 1.2 sigma, below the seed cut
    assert not mask[: peak - 3].any() and not mask[peak + 70 :].any()
    assert not mask[300:360].any()
    seeds_only = positive_outliers(x, clip_sigma=4.0, tail_sigma=1e9)
    assert seeds_only.sum() < mask.sum() - 10


def test_positive_outliers_of_pure_noise_are_few():
    rng = np.random.default_rng(1)
    x = 1.0 + rng.normal(0.0, 0.005, 480)
    assert positive_outliers(x, 4.0, 1.0).sum() <= 3


# -- dropouts and flags -----------------------------------------------------------------


def test_positive_flare_on_the_transit_changes_nothing():
    t, prof, clean = planted()
    base = detect({0: clean})
    assert len(base) == 3
    rng = np.random.default_rng(0)
    flared_prof = prof.copy()
    # +5% for 8 samples in the middle of the first event (t ~ 0.14 d = 12096 s)
    i0 = int(np.searchsorted(t, 0.14 * DAY))
    flared_prof[i0 : i0 + 8] += 0.05
    flared = curve(0, t, flared_prof, 0.005, rng)
    out = detect({0: flared})
    assert [(d.t_start, d.t_end) for d in out] == [(d.t_start, d.t_end) for d in base]


def test_flare_is_masked_but_still_counts_as_observed():
    # a flare on the leading edge of the first event: its samples are masked from the
    # fit but are observed, so the event's span still starts at the window's first sample
    t, prof, clean = planted()
    base = detect({0: clean})
    rng = np.random.default_rng(0)
    p2 = prof.copy()
    i0 = int(np.searchsorted(t, 0.14 * DAY - 5400))  # first sample of the first window
    p2[i0 : i0 + 12] += 0.2
    out = detect({0: curve(0, t, p2, 0.005, rng)})
    assert [(d.t_start, d.t_end) for d in out] == [(d.t_start, d.t_end) for d in base]


def test_dropouts_do_not_shift_predicted_spans():
    t, prof, clean = planted()
    base = detect({0: clean})
    keep = np.ones(len(t), bool)
    i0 = int(np.searchsorted(t, 0.14 * DAY))
    keep[i0 - 10 : i0 + 10] = False  # a hole in the middle of the first event
    keep[2000:2010] = False
    keep[3000] = False
    rng = np.random.default_rng(0)
    holey = curve(0, t[keep], prof[keep], 0.005, rng)
    out = detect({0: holey})
    assert len(out) == len(base)
    assert [(d.t_start, d.t_end) for d in out] == [(d.t_start, d.t_end) for d in base]


def test_a_sample_missing_at_an_event_edge_shrinks_the_span_to_observed_samples():
    t, prof, clean = planted()
    base = detect({0: clean})
    first = min(base, key=lambda d: d.t_start)
    keep = t != first.t_start
    rng = np.random.default_rng(0)
    out = detect({0: curve(0, t[keep], prof[keep], 0.005, rng)})
    f2 = min(out, key=lambda d: d.t_start)
    assert f2.t_start == first.t_start + C and f2.t_end == first.t_end


def test_quality_flagged_samples_are_dropped():
    rng = np.random.default_rng(4)
    t = night_times(10)
    prof = np.ones(len(t))
    flagged = np.zeros(len(t), dtype=np.uint8)
    # a deep periodic "dip" that is entirely inside instrument-flagged samples
    inside = box_profile(t, 2.0 * DAY, 0.15 * DAY, 3 * 3600.0, 0.3) < 1
    prof[inside] = 0.7
    flagged[inside] = 1
    assert detect({0: curve(0, t, prof, 0.005, rng, quality=flagged)}) == []
    # the same dip unflagged is found
    assert len(detect({0: curve(0, t, prof, 0.005, np.random.default_rng(4))})) > 0


def test_quality_bit_other_than_zero_is_not_dropped():
    rng = np.random.default_rng(0)
    t, prof, _ = planted()
    q = np.full(len(t), 2, dtype=np.uint8)
    assert len(detect({0: curve(0, t, prof, 0.005, rng, quality=q)})) == 3


# -- the field -------------------------------------------------------------------------


def make_field(n_stars=12, n_nights=10, seed=0, noise=0.005):
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    return t, rng, {i: curve(i, t, np.ones(len(t)), noise, rng) for i in range(n_stars)}


def test_common_mode_offset_shared_by_the_field_is_not_a_transit():
    # a held step the quality flag does not cover (last 3 h of two nights, every star)
    t, rng, _ = make_field()
    step = np.ones(len(t))
    for k in (2, 6):
        step[(t >= k * DAY + 5 * 3600) & (t < k * DAY + 8 * 3600)] = 0.9
    curves = {i: curve(i, t, step, 0.005, rng) for i in range(12)}
    assert detect(curves) == []
    # with the correction off the step is the best "transit" on every star
    assert len(detect(curves, common_mode_min_stars=0)) > 0


def test_common_mode_does_not_hide_a_planet_on_one_star():
    t, rng, curves = make_field()
    prof = box_profile(t, 2.3 * DAY, 0.14 * DAY, 3 * 3600.0, 0.01)
    curves[5] = curve(5, t, prof, 0.005, rng)
    out = detect(curves)
    assert {d.star_id for d in out} == {5} and len(out) == 3


def test_stars_are_independent_of_each_other_in_a_small_field():
    _, _, cv = planted()
    rng = np.random.default_rng(9)
    t = night_times(10)
    other = curve(1, t, np.ones(len(t)), 0.005, rng)
    both = detect({0: cv, 1: other})
    assert [d.det_id for d in both] == [d.det_id for d in detect({0: cv})]


def test_determinism():
    _, _, cv = planted()
    assert detect({0: cv}) == detect({0: cv})


def test_iteration_order_is_by_star_id():
    t, rng, curves = make_field(n_stars=11)
    prof = box_profile(t, 2.3 * DAY, 0.14 * DAY, 3 * 3600.0, 0.01)
    for sid in (9, 3):
        curves[sid] = curve(sid, t, prof, 0.005, rng)
    out = detect(dict(reversed(list(curves.items()))))
    assert [d.star_id for d in out] == sorted(d.star_id for d in out)


def test_performance_20_stars_10_nights():
    t = night_times(10)
    rng = np.random.default_rng(0)
    curves = {}
    for i in range(20):
        # four planets, each on its own period and phase (a field never shares one transit)
        j = i // 5
        prof = box_profile(t, (2.3 + 0.37 * j) * DAY, (0.14 + 0.02 * j) * DAY, 3 * 3600.0, 0.01)
        curves[i] = curve(i, t, prof if i % 5 == 0 else np.ones(len(t)), 0.005, rng)
    t0 = time.perf_counter()
    out = detect(curves)
    elapsed = time.perf_counter() - t0
    assert {d.star_id for d in out} == {0, 5, 10, 15}
    # 2.0-2.9 s measured (load average up to 140); four times the slowest of three runs
    assert elapsed < 12.0, f"20 stars x 10 nights took {elapsed:.1f} s"


# -- review follow-up: pins for the defaults and the gates the first suite let through -----


@pytest.mark.parametrize("period_d", [1.7 / 24, 0.05])
def test_fast_sinusoids_are_rejected_by_the_pulsation_guard_itself(period_d):
    # 14.1 and 20 cycles/day, inside the guard band (1-30 c/d) but outside 1-5: the
    # guard names itself as the reason, so no other gate can stand in for it
    rng = np.random.default_rng(3)
    t = night_times(10)
    prof = 1.0 + 0.0035 * np.sin(2 * np.pi * t / (period_d * DAY) + 0.7)
    cv = curve(0, t, prof, 0.005, rng)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.snr > 7.0 and cand.snr_local > 7.0
    assert cand.rejected == "pulsation guard"
    assert cand.excess is not None and cand.excess > 100.0
    assert detect({0: cv}) == []
    # it is the guard that holds it back: without the guard the sinusoid is a "transit"
    assert BlsDetector(guard_fap=None).search(cv, C).rejected is None


def test_sigma_is_the_mad_not_the_standard_deviation():
    # 15 percent of the samples carry 8x the noise: the standard deviation of the
    # out-of-box flux is about 2.5x the noise of the other 85 percent, the MAD barely moves
    t, prof, _ = planted(depth=0.0035)
    rng = np.random.default_rng(3)
    wide = rng.random(len(t)) < 0.15
    g = rng.normal(0.0, 1.0, len(t))
    flux = 1000.0 * prof * (1.0 + g * np.where(wide, 0.005 * 8, 0.005))
    cv = LightCurve(0, t, flux, np.full(len(t), 5.0), np.zeros(len(t), dtype=np.uint8))
    # guard off: the heavy tails are not what is under test and would trip the residual test
    cand = BlsDetector(guard_fap=None).search(cv, C)
    assert cand is not None and cand.rejected is None
    assert cand.box.period_s == pytest.approx(2.3 * DAY, rel=0.01)
    assert 0.005 <= cand.sigma <= 0.0065  # the core noise, not the ~0.013 of the whole
    assert cand.snr > 12.0
    assert len(detect({0: cv}, guard_fap=None)) == 3


def test_default_threshold_finds_a_marginal_planet_and_a_higher_one_loses_it():
    _, _, cv = planted(depth=0.0021, seed=0)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected is None
    assert 8.0 < cand.snr < 11.0  # marginal: well under any round-number "strong" cut
    assert cand.box.period_s == pytest.approx(2.3 * DAY, rel=0.01)
    assert len(detect({0: cv})) == 3
    assert BlsDetector(snr_threshold=20.0).search(cv, C).rejected == "snr"
    assert detect({0: cv}, snr_threshold=20.0) == []


def test_default_parameters():
    d = BlsDetector()
    assert d.min_period_d == 0.5 and d.max_period_d is None
    assert d.durations_h == (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert d.oversample == 3.0
    assert d.snr_threshold == 7.0 and d.min_events == 2 and d.support_snr == 2.0
    assert d.clip_sigma == 4.0 and d.tail_sigma == 1.0
    assert d.guard_fap == 1e-3 and d.guard_ratio == 1.0
    assert (d.guard_f_min_per_d, d.guard_f_max_per_d) == (1.0, 30.0)
    assert d.night_gap_cadences == 100 and d.common_mode_min_stars == 10
    assert d.second_pass is False  # opt-in: held-out seeds gain false positives, no transits
    assert d.shape_threshold == 1.5 and d.shape_chi2 == 14.0


def test_short_transit_at_a_long_period_keeps_every_event_placed():
    # 0.5 h at 3 percent every 9.1 d over 20 nights: three in-night events, the last 18 d
    # after the first, so the period grid must not smear it by more than the duration
    rng = np.random.default_rng(0)
    t = night_times(20)
    prof = box_profile(t, 9.1 * DAY, 0.05 * DAY, 0.5 * 3600.0, 0.03)
    cv = curve(0, t, prof, 0.005, rng)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected is None
    assert cand.box.duration_s == pytest.approx(0.5 * 3600.0, rel=0.1)
    out = detect({0: cv})
    wins = planted_windows(t, 9.1, 0.05, 0.5)
    assert len(wins) == 3 and len(out) == len(wins)
    for lo, hi in wins:
        assert max(span_iou(d.t_start, d.t_end, lo, hi) for d in out) >= 0.5


def test_durations_are_sorted_so_the_bin_comes_from_the_shortest():
    _, _, cv = planted()
    odd = BlsDetector(durations_h=(6.0, 0.5, 3.0))
    assert odd.durations_h == (0.5, 3.0, 6.0)
    assert odd.detect({0: cv}, C) == detect({0: cv}, durations_h=(0.5, 3.0, 6.0))
    assert len(odd.detect({0: cv}, C)) == 3


def test_a_planet_below_the_two_day_period_is_found_at_its_own_period():
    # P = 0.8 d: five in-night events in 10 nights; periods under 1 d are in the grid
    rng = np.random.default_rng(0)
    t = night_times(10)
    prof = box_profile(t, 0.8 * DAY, 0.1 * DAY, 1.5 * 3600.0, 0.01)
    cv = curve(0, t, prof, 0.005, rng)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected is None
    assert cand.box.period_s == pytest.approx(0.8 * DAY, rel=0.01)
    assert len(detect({0: cv})) == len(planted_windows(t, 0.8, 0.1, 1.5)) == 5


# -- second pass (#32): one re-search after a rejected best box ------------------------------


def eclipsing(t, period_d, epoch_d, dur_h, depth_primary, depth_secondary):
    """V-shaped primary and secondary eclipses (half a period apart), as the world draws them."""
    period, epoch, dur = period_d * DAY, epoch_d * DAY, dur_h * 3600.0
    out = np.ones(len(t))
    for offset, depth in ((0.0, depth_primary), (period / 2.0, depth_secondary)):
        ph = np.mod(t - epoch - offset + period / 2.0, period) - period / 2.0
        out *= 1.0 - depth * np.clip(1.0 - np.abs(ph) / (dur / 2.0), 0.0, 1.0)
    return out


# The box lands on half the binary's period (both eclipses fall in it), as on seed7 star 138.
EB_HALF_PERIOD = (4.0, 1.4, 11.4, 0.28, 0.126)
# A weak secondary: the box lands on the binary's own period and the secondary is left behind
# (as on seed14-200s-20n star 70, in a stored run of another seed).
EB_PRIMARY_ONLY = (4.34, 1.3, 5.4, 0.30, 0.07)
# (binary, planet period d, planet epoch d): the planet's in-night events sit away from eclipses
EB_WITH_PLANET = [(EB_HALF_PERIOD, 6.1, 0.14), (EB_PRIMARY_ONLY, 5.3, 0.14)]


def eb_curve(eb, planet=None, n_nights=20, seed=0, depth=0.015, dur_h=3.0):
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = eclipsing(t, *eb)
    if planet is not None:
        prof = prof * box_profile(t, planet[0] * DAY, planet[1] * DAY, dur_h * 3600.0, depth)
    return t, curve(0, t, prof, 0.005, rng)


def dip_curve(dip_night, depth=0.0021, dip_depth=0.03, seed=0):
    """A planet (P 2.3 d, S/N about 9 alone) and one deep 3 h dip on a single night."""
    rng = np.random.default_rng(seed)
    t = night_times(10)
    prof = box_profile(t, 2.3 * DAY, 0.14 * DAY, 3 * 3600.0, depth)
    dip = (t >= dip_night * DAY + 2 * 3600) & (t < dip_night * DAY + 5 * 3600)
    return t, curve(0, t, prof * (1.0 - dip_depth * dip), 0.005, rng)


def assert_planet_events(out, wins, star=0):
    """Each planted window is bounded by exactly one detection and nothing else is reported."""
    assert len(out) == len(wins)
    assert {d.kind_guess for d in out} == {"transit"} and {d.star_id for d in out} == {star}
    assert len({d.score for d in out}) == 1 and out[0].score >= 7.0
    assert [d.det_id for d in out] == [f"bls-s{star}-{k}" for k in range(len(out))]
    for lo, hi in wins:
        assert sum(span_iou(d.t_start, d.t_end, lo, hi) >= 0.5 for d in out) == 1


@pytest.mark.parametrize("eb,period_d,epoch_d", EB_WITH_PLANET)
def test_second_pass_recovers_a_planet_behind_an_eclipsing_binary(eb, period_d, epoch_d):
    t, cv = eb_curve(eb, (period_d, epoch_d))
    wins = planted_windows(t, period_d, epoch_d, 3.0)
    assert len(wins) >= 2
    # the binary is the star's best box and the guard rejects it: the planet is never searched
    first = BlsDetector().search(cv, C)
    assert first.rejected == "pulsation guard" and first.snr > 100
    assert detect({0: cv}, second_pass=False) == []
    out = detect({0: cv}, second_pass=True)
    assert_planet_events(out, wins)


@pytest.mark.parametrize("eb", [EB_HALF_PERIOD, EB_PRIMARY_ONLY])
def test_second_pass_reports_nothing_on_a_bare_eclipsing_binary(eb):
    _, cv = eb_curve(eb)
    assert BlsDetector().search(cv, C).rejected == "pulsation guard"
    assert detect({0: cv}, second_pass=True) == []


@pytest.mark.parametrize("dip_night", [3, 8])
def test_second_pass_finds_the_planet_a_deep_one_off_dip_hid(dip_night):
    t, cv = dip_curve(dip_night)
    wins = planted_windows(t, 2.3, 0.14, 3.0)
    first = BlsDetector().search(cv, C)
    # the dip wins the search with S/N near 80 and is refused for lack of events
    assert first.snr > 50 and first.rejected in ("min_events", "support", "local snr")
    assert detect({0: cv}, second_pass=False) == []
    out = detect({0: cv}, second_pass=True)
    assert_planet_events(out, wins)


def test_second_pass_changes_nothing_for_a_star_that_already_has_a_candidate():
    _, _, cv = planted()
    assert detect({0: cv}, second_pass=True) == detect({0: cv}, second_pass=False)
    assert len(detect({0: cv}, second_pass=True)) == 3


def test_second_pass_changes_nothing_for_a_quiet_star_or_a_weak_best_box():
    rng = np.random.default_rng(5)
    t = night_times(10)
    quiet = curve(0, t, np.ones(len(t)), 0.005, rng)
    assert BlsDetector().search(quiet, C).rejected == "snr"
    assert detect({0: quiet}, second_pass=True) == []


def test_second_pass_is_deterministic():
    _, cv = eb_curve(EB_HALF_PERIOD, (6.1, 0.14))
    a = detect({0: cv}, second_pass=True)
    assert a and a == detect({0: cv}, second_pass=True)


def test_second_pass_finds_the_planet_in_a_field_with_common_mode_removal():
    # a 12-star field of white noise over 20 nights is not quiet once the (noisy) common mode is
    # divided out, so compare against the same field without the second pass
    _, _, curves = make_field(n_stars=12, n_nights=20)
    _, eb = eb_curve(EB_HALF_PERIOD, (6.1, 0.14))
    curves[5] = LightCurve(5, eb.t, eb.flux, eb.flux_err, eb.quality)
    off = detect(curves, second_pass=False)
    on = detect(curves, second_pass=True)
    assert not [d for d in off if d.star_id == 5]
    assert [d for d in on if d.star_id != 5] == [d for d in off if d.star_id != 5]
    wins = planted_windows(night_times(20), 6.1, 0.14, 3.0)
    assert_planet_events([d for d in on if d.star_id == 5], wins, star=5)


def test_second_pass_logs_the_re_search_outcome_at_info(caplog):
    _, cv = eb_curve(EB_HALF_PERIOD, (6.1, 0.14))
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        detect({0: cv}, second_pass=True)
    lines = [r.getMessage() for r in caplog.records if "re-search" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].startswith("bls star 0: re-search found P ") and lines[0].endswith("accepted")
    caplog.clear()
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        detect({0: cv}, second_pass=False)
    assert not [r for r in caplog.records if "re-search" in r.getMessage()]


def test_second_pass_logs_when_the_re_search_is_rejected_too(caplog):
    _, cv = eb_curve(EB_HALF_PERIOD)
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        assert detect({0: cv}, second_pass=True) == []
    lines = [r.getMessage() for r in caplog.records if "re-search" in r.getMessage()]
    assert len(lines) == 1 and "re-search" in lines[0] and not lines[0].endswith("accepted")


# -- second pass: the pieces -----------------------------------------------------------------


def test_circular_runs_wrap_around_the_end():
    f = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    assert sorted(_circular_runs(f)) == [(4, 1), (6, 4)]  # 6, 7, 0, 1 is one run
    assert _circular_runs(np.ones(5, dtype=bool)) == [(0, 5)]
    assert _circular_runs(np.zeros(5, dtype=bool)) == []
    assert _circular_runs(np.array([0, 1, 1, 0], dtype=bool)) == [(1, 2)]


@pytest.mark.parametrize("ratio", [1.0, 0.5, 2.0, 1.019, 0.981, 0.5 * 1.019, 2 * 0.981])
def test_aliases_of_the_first_period_are_the_period_half_and_double(ratio):
    assert _is_alias(ratio * 3.0 * DAY, 3.0 * DAY)


@pytest.mark.parametrize("ratio", [1.03, 0.97, 0.33, 1.5, 3.0, 4.009, 0.25, 0.51 * 1.03])
def test_other_periods_are_not_aliases(ratio):
    # 4.009 is star 138: its planet (8.016 d) against the binary's half period (2.000 d)
    assert not _is_alias(ratio * 3.0 * DAY, 3.0 * DAY)


def test_renormalise_nights_divides_each_night_by_its_median():
    flux = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 9.0, 9.0])
    night = np.array([0] * 5 + [1] * 5 + [2] * 2)
    out = _renormalise_nights(flux, night)
    assert out[:10] == pytest.approx(1.0)
    assert out[10:] == pytest.approx([9.0, 9.0])  # fewer than 5 samples: left alone
    assert flux[0] == 2.0  # the input is not modified


def _eb_mask(eb, planet=None, seed=0):
    t, cv = eb_curve(eb, planet, seed=seed)
    det = BlsDetector(second_pass=True)
    cand = det._search(det._clean(cv, C), None, C, plan_mask=True)
    return t, cand


def _covered(t, mask):
    ok = np.zeros(len(t), dtype=bool)
    for m in mask:
        ok |= in_box(t, m)
    return ok


@pytest.mark.parametrize("eb", [EB_HALF_PERIOD, EB_PRIMARY_ONLY])
def test_the_mask_covers_every_eclipse_wing_and_leaves_the_planet(eb):
    planet = (6.1, 0.14) if eb is EB_HALF_PERIOD else (5.3, 0.14)
    t, cand = _eb_mask(eb, planet)
    assert cand.rejected == "pulsation guard" and cand.mask
    covered = _covered(t, cand.mask)
    depth = 1.0 - eclipsing(t, *eb)
    assert depth.max() > 0.2
    assert covered[depth > 0.005].all()  # the wings, not just the 3 h box
    assert covered.mean() < 0.5  # and not the whole record
    planet_samples = box_profile(t, planet[0] * DAY, planet[1] * DAY, 3 * 3600.0, 1.0) < 1.0
    assert planet_samples.sum() > 300 and not covered[planet_samples].any()
    for m in cand.mask:
        assert m.period_s == cand.box.period_s


def test_the_mask_follows_the_eclipse_not_the_box_width():
    _, cand = _eb_mask(EB_HALF_PERIOD)
    widest = max(m.duration_s for m in cand.mask)
    assert cand.box.duration_s <= 6 * 3600.0 < widest  # wider than any box the grid has


def _two_dips(second_dur_h, second_depth):
    """The mask for a 20 percent, 3 h dip every 2 d with a second dip half a period later."""
    t = night_times(20).astype(float)
    prof = box_profile(t, 2.0 * DAY, 0.1 * DAY, 3 * 3600.0, 0.2) * box_profile(
        t, 2.0 * DAY, 1.1 * DAY, second_dur_h * 3600.0, second_depth
    )
    flux = prof * (1.0 + np.random.default_rng(0).normal(0.0, 0.003, len(t)))
    box = Box(2.0 * DAY, 0.1 * DAY, 3 * 3600.0, 0.2, 500, len(t), 1.0)
    covered = _covered(t, BlsDetector(second_pass=True)._dip_mask(t, flux, box, 0.003))
    deep = box_profile(t, 2.0 * DAY, 0.1 * DAY, 3 * 3600.0, 1.0) < 1.0
    second = box_profile(t, 2.0 * DAY, 1.1 * DAY, second_dur_h * 3600.0, 1.0) < 1.0
    assert deep.sum() > 500 and second.sum() > 200 and covered[deep].all()
    return covered, second


def test_a_short_dip_at_half_phase_is_not_taken_for_a_secondary_eclipse():
    # shorter than 0.7 of the primary: a planet that happens to sit there, kept for the re-search
    covered, second = _two_dips(1.2, 0.03)
    assert not covered[second].any()


def test_an_equally_long_dip_at_half_phase_is_masked_as_the_secondary():
    covered, second = _two_dips(3.0, 0.05)
    assert covered[second].all()


def _clean_mask(prof, t, period_d, epoch_d, sigma, dur_h=3.0, **kw):
    """Samples covered by the mask of a noise-free profile whose box is at the given phase."""
    box = Box(period_d * DAY, epoch_d * DAY, dur_h * 3600.0, 0.2, 500, len(t), 1.0)
    det = BlsDetector(second_pass=True, **kw)
    return _covered(t, det._dip_mask(t, prof, box, sigma))


def test_a_box_that_leaves_nothing_outside_it_is_refused_by_the_guard():
    # after a mask the record can be short enough for the box and its margin to fill it; the
    # residual test then has nothing to measure (an empty mean) and must not wave it through
    t = night_times(3)
    cv = curve(0, t, np.ones(len(t)), 0.005, np.random.default_rng(0))
    det = BlsDetector(second_pass=False)  # a first-pass fix: it holds whatever the flag says
    c = det._clean(cv, C)
    box = Box(float(DAY), 4 * 3600.0, 2 * 24 * 3600.0, 0.01, len(t), len(t), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pulsating, power = det._pulsating(c.t.astype(float), c.flux, 0.005, C, box)
    assert pulsating and power == math.inf


def test_an_undefined_local_snr_does_not_pass_the_gate(monkeypatch):
    # a record in which no night holds both box and out-of-box samples has no local contrast
    monkeypatch.setattr(bls, "night_contrasts", lambda *a: (np.zeros(0), np.zeros(0)))
    _, _, cv = planted()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # and says so without a numpy warning
        cand = BlsDetector(second_pass=False).search(cv, C)
    assert cand.rejected == "local snr" and math.isnan(cand.snr_local)


def test_a_dip_grows_through_wings_below_the_seed_level():
    # a 3 h core at 20 percent with 1.3 per mille wings out to 6 h either side: with sigma 0.01 a
    # 15-minute bin of the fold sees the wings at 1.6 standard errors, under support_snr (2) and
    # over tail_sigma (1). The margin alone (3 h) stops short of the night's last half hour
    t = night_times(20).astype(float)
    core = box_profile(t, 2.0 * DAY, 0.1 * DAY, 3 * 3600.0, 0.2)
    wings = box_profile(t, 2.0 * DAY, 0.1 * DAY, 12 * 3600.0, 0.0013)
    prof = core * wings
    far = (wings < 1.0) & (core == 1.0) & (np.mod(t, DAY) > 7.3 * 3600.0)
    assert far.sum() > 100
    assert _clean_mask(prof, t, 2.0, 0.1, 0.01)[far].all()
    # without the growth (tail_sigma at the seed level) the wings are left behind
    assert not _clean_mask(prof, t, 2.0, 0.1, 0.01, tail_sigma=2.0)[far].any()


def test_the_mask_is_the_dip_plus_one_box_duration_each_side():
    # a 1 h dip at 1.9-2.9 h into every other night: the window is the dip (to the 15 min bin)
    # and one more hour each side, so 0.9-3.9 h, and nothing beyond 15 min of that
    t = night_times(20).astype(float)
    prof = box_profile(t, 2.0 * DAY, 0.1 * DAY, 3600.0, 0.2)
    covered = _clean_mask(prof, t, 2.0, 0.1, 0.003, dur_h=1.0)
    hour = np.mod(t, DAY) / 3600.0
    on_night = np.isin(np.floor(t / DAY) % 2, [0])
    assert covered[on_night & (hour >= 1.0) & (hour <= 3.8)].all()
    assert not covered[on_night & ((hour < 0.6) | (hour > 4.2))].any()
    assert not covered[~on_night].any()


def test_a_box_wider_than_its_dip_is_masked_whole():
    # the dip is 1 h but the box is 3 h: the window is the box plus a box duration each side
    t = night_times(20).astype(float)
    prof = box_profile(t, 2.0 * DAY, 0.1 * DAY, 3600.0, 0.2)
    covered = _clean_mask(prof, t, 2.0, 0.1, 0.003, dur_h=3.0)
    hour = np.mod(t, DAY) / 3600.0
    on_night = np.floor(t / DAY) % 2 == 0
    centre = 0.1 * 24.0
    assert covered[on_night & (np.abs(hour - centre) <= 4.3)].all()
    assert not covered[on_night & (np.abs(hour - centre) >= 4.9)].any()


def test_the_masked_nights_are_divided_by_the_median_of_what_is_left(monkeypatch):
    calls = []
    real = bls._renormalise_nights

    def spy(flux, night, *a, **k):
        calls.append(len(flux))
        return real(flux, night, *a, **k)

    monkeypatch.setattr(bls, "_renormalise_nights", spy)
    _, _, cv = planted()
    det = BlsDetector()
    c = det._clean(cv, C)
    det._search(c, None, C)
    assert calls == []
    window = Box(2.0 * DAY, 0.1 * DAY, 4 * 3600.0, 0.0, 0, 0, 0.0)
    det._search(c, None, C, masked=[window])
    assert len(calls) == 1 and calls[0] < len(cv.t)


def test_a_dip_that_is_not_half_a_period_from_the_primary_is_not_a_secondary():
    # P = 3 d: nights fall on phases 0, 1 and 2 d; the primary is at 0.1 d, half a period is 1.6 d
    # (no data) and an equally deep, equally long dip at 2.1 d is simply another signal
    t = night_times(30).astype(float)
    prof = box_profile(t, 3.0 * DAY, 0.1 * DAY, 3 * 3600.0, 0.2) * box_profile(
        t, 3.0 * DAY, 2.1 * DAY, 3 * 3600.0, 0.2
    )
    covered = _clean_mask(prof, t, 3.0, 0.1, 0.003)
    primary = box_profile(t, 3.0 * DAY, 0.1 * DAY, 3 * 3600.0, 1.0) < 1.0
    other = box_profile(t, 3.0 * DAY, 2.1 * DAY, 3 * 3600.0, 1.0) < 1.0
    assert primary.sum() > 500 and other.sum() > 500
    assert covered[primary].all() and not covered[other].any()


@pytest.mark.parametrize("depth,masked", [(0.0013, False), (0.0020, True)])
def test_a_secondary_must_reach_the_detection_threshold_on_its_own(depth, masked):
    # sigma 0.01 and 150 samples a bin: 0.0013 is 1.6 standard errors a bin (S/N 5.5 over the
    # 3 h run), 0.0020 is 2.4 (S/N 8.4); one bin of the first holds a seed above support_snr
    t = night_times(20).astype(float)
    second = box_profile(t, 2.0 * DAY, 1.1 * DAY, 3 * 3600.0, depth)
    seed = box_profile(t, 2.0 * DAY, 1.1 * DAY, 900.0, 0.0008)
    prof = box_profile(t, 2.0 * DAY, 0.1 * DAY, 3 * 3600.0, 0.2) * second * seed
    covered = _clean_mask(prof, t, 2.0, 0.1, 0.01)
    flat = box_profile(t, 2.0 * DAY, 1.1 * DAY, 3 * 3600.0, 1.0) < 1.0
    assert flat.sum() > 500 and covered[flat].all() == masked and covered[flat].any() == masked


def test_a_rejection_on_snr_or_without_plan_mask_carries_no_mask():
    det = BlsDetector(second_pass=True)
    rng = np.random.default_rng(5)
    t = night_times(10)
    quiet = curve(0, t, np.ones(len(t)), 0.005, rng)
    cand = det._search(det._clean(quiet, C), None, C, plan_mask=True)
    assert cand.rejected == "snr" and cand.mask == ()
    _, eb = eb_curve(EB_HALF_PERIOD)
    cand = det._search(det._clean(eb, C), None, C)
    assert cand.rejected == "pulsation guard" and cand.mask == ()


def test_masked_windows_are_not_observed_for_the_search_or_the_events():
    _, _, cv = planted()
    det = BlsDetector()
    c = det._clean(cv, C)
    full = det._search(c, None, C)
    assert len(full.events) == 3
    first = full.events[0]
    # mask the first event (a window around its middle, as wide as the event)
    mid = (first[0] + first[1]) / 2.0
    window = Box(full.box.period_s * 7.0, mid, 4 * 3600.0, 0.0, 0, 0, 0.0)
    out = det._search(c, None, C, masked=[window])
    assert out is not None and out.rejected is None
    assert [e[:2] for e in out.events] == [e[:2] for e in full.events[1:]]
    # and no sample of the window took part: the box was fitted on the rest
    assert out.box.n_total < full.box.n_total


class _Stub(BlsDetector):
    """A detector whose `_search` is scripted: first call, then the re-search."""

    def __init__(self, first, second, **kw):
        super().__init__(**kw)
        self.script = [first, second]
        self.calls = []

    def _search(self, c, common_mode, cadence_s, masked=(), plan_mask=False):
        self.calls.append((tuple(masked), plan_mask))
        return self.script[len(self.calls) - 1] if len(self.calls) <= len(self.script) else None


def _cand(period_d, rejected, mask=(), snr=12.0, epoch_d=0.14):
    box = Box(period_d * DAY, epoch_d * DAY, 3 * 3600.0, 0.01, 100, 5000, 1.0)
    events = [(0, 3600, 0), (DAY, DAY + 3600, 1)] if rejected is None else []
    return Candidate(box, 0.005, snr, events, rejected, None, None, snr, 2, tuple(mask))


def _run_stub(first, second, **kw):
    _, _, cv = planted()
    det = _Stub(first, second, **kw)
    return det, det.detect({0: cv}, C)


MASK = (Box(2.0 * DAY, 0.5 * DAY, 9 * 3600.0, 0.0, 0, 0, 0.0),)


@pytest.mark.parametrize("why", ["local snr", "min_events", "support", "pulsation guard"])
def test_a_rejection_other_than_snr_is_searched_again_once_with_the_mask(why):
    first = _cand(2.0, why, MASK)
    det, out = _run_stub(first, _cand(5.3, None), second_pass=True)
    assert len(det.calls) == 2
    assert det.calls[0] == ((), True) and det.calls[1] == (MASK, False)
    assert [d.det_id for d in out] == ["bls-s0-0", "bls-s0-1"] and {d.score for d in out} == {12.0}


def test_a_rejection_on_snr_is_not_searched_again():
    det, out = _run_stub(_cand(2.0, "snr"), _cand(5.3, None), second_pass=True)
    assert len(det.calls) == 1 and out == []


def test_nothing_is_searched_again_when_the_pass_is_off_or_the_first_box_stands():
    det, out = _run_stub(_cand(2.0, "pulsation guard", MASK), _cand(5.3, None), second_pass=False)
    assert det.calls == [((), False)] and out == []  # and no mask was planned
    det, out = _run_stub(_cand(2.0, None), _cand(5.3, None), second_pass=True)
    assert len(det.calls) == 1 and len(out) == 2 and out[0].score == 12.0


@pytest.mark.parametrize("why", ["local snr", "min_events", "support", "snr", "pulsation guard"])
def test_a_second_candidate_that_fails_any_gate_gives_nothing_and_is_not_followed_up(why):
    det, out = _run_stub(_cand(2.0, "min_events", MASK), _cand(5.3, why), second_pass=True)
    assert out == [] and len(det.calls) == 2
    det, out = _run_stub(_cand(2.0, "min_events", MASK), None, second_pass=True)
    assert out == [] and len(det.calls) == 2


@pytest.mark.parametrize("ratio", [1.0, 0.5, 2.0])
def test_after_a_pulsation_rejection_the_first_period_and_its_halves_are_refused(ratio):
    first = _cand(2.0, "pulsation guard", MASK)
    det, out = _run_stub(first, _cand(2.0 * ratio, None), second_pass=True)
    assert out == [] and len(det.calls) == 2
    det, out = _run_stub(first, _cand(2.0 * ratio * 1.1, None), second_pass=True)
    assert len(out) == 2


@pytest.mark.parametrize("why", ["local snr", "min_events", "support"])
def test_the_period_of_a_box_refused_for_lack_of_events_means_nothing(why):
    # a one-off dip fits many periods; a planet at half or twice the arbitrary one is no alias
    _, out = _run_stub(_cand(4.52, why, MASK), _cand(4.52 / 2.0, None), second_pass=True)
    assert len(out) == 2


def test_without_the_secondary_rule_a_weak_secondary_hides_the_planet(monkeypatch):
    # the box sits on the binary's own period; the secondary half a period away is left in the
    # record unless it is masked too, and then the re-search finds only its residue
    _, cv = eb_curve(EB_PRIMARY_ONLY, (5.3, 0.14))
    assert len(detect({0: cv}, second_pass=True)) == 2
    monkeypatch.setattr(bls, "_SECONDARY_WIDTH", 99.0)
    assert detect({0: cv}, second_pass=True) == []


def test_the_second_pass_is_off_by_default():
    assert BlsDetector().second_pass is False
    _, cv = eb_curve(EB_HALF_PERIOD, (6.1, 0.14))
    assert detect({0: cv}) == []  # the first pass leaves the planet behind the binary
    assert len(detect({0: cv}, second_pass=True)) == 3


# -- second pass: gates and inputs the stubbed tests cannot see (review of #34) --------------------
# The _Stub tests replace `_search`, so they say nothing about what the real search does to a
# second candidate. These run the real thing.

# A mask window no sample of a 20-night record falls in: a masked search that changes nothing.
HARMLESS_MASK = Box(1000.0 * DAY, 500.0 * DAY, 3600.0, 0.0, 0, 0, 0.0)


def _real_search(cv, masked=(), plan_mask=False, **kw):
    det = BlsDetector(second_pass=True, **kw)
    return det._search(det._clean(cv, C), None, C, masked=masked, plan_mask=plan_mask)


def _long_box_star():
    """A 6 h box in 8 h nights: its S/N is far above its local S/N."""
    t = night_times(8)
    prof = box_profile(t, 2.0 * DAY, 0.1 * DAY, 6 * 3600.0, 0.004)
    return curve(0, t, prof, 0.01, np.random.default_rng(7))


def _gate_stars():
    """One star and detector settings per gate: the real search names that gate as the reason."""
    _, _, planet = planted()
    _, binary = eb_curve(EB_HALF_PERIOD)
    _, one_off = dip_curve(3, depth=0.0)
    t = night_times(10)
    quiet = curve(0, t, np.ones(len(t)), 0.005, np.random.default_rng(5))
    long_box = _long_box_star()
    probe = BlsDetector(snr_threshold=1.0).search(long_box, C)
    return {
        "snr": (quiet, {}),
        "local snr": (long_box, {"snr_threshold": (probe.snr + probe.snr_local) / 2}),
        "min_events": (one_off, {}),
        "support": (planet, {"support_snr": 1e9}),
        "pulsation guard": (binary, {}),
    }


@pytest.mark.parametrize("why", ["snr", "local snr", "min_events", "support", "pulsation guard"])
def test_the_second_candidate_faces_every_gate_in_the_real_search(why):
    cv, kw = _gate_stars()[why]
    assert _real_search(cv, **kw).rejected == why  # the star's own search is refused by the gate
    # and so is a candidate of the masked search: nothing is waved through because it is second
    assert _real_search(cv, masked=(HARMLESS_MASK,), **kw).rejected == why


@pytest.mark.parametrize("why", ["local snr", "min_events", "support", "pulsation guard"])
def test_a_real_rejection_plans_a_mask_for_every_reason_but_snr(why):
    cv, kw = _gate_stars()[why]
    cand = _real_search(cv, plan_mask=True, **kw)
    assert cand.rejected == why and cand.mask
    cv, kw = _gate_stars()["snr"]
    assert _real_search(cv, plan_mask=True, **kw).mask == ()


@pytest.mark.parametrize("why", ["local snr", "min_events", "support", "pulsation guard"])
def test_the_real_search_is_searched_again_after_each_rejection_but_snr(why, caplog):
    cv, kw = _gate_stars()[why]
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        BlsDetector(second_pass=True, **kw).detect({0: cv}, C)
    lines = [r.getMessage() for r in caplog.records if "re-search" in r.getMessage()]
    assert len(lines) == 1  # once: a re-search that finds nothing or fails a gate ends the star
    caplog.clear()
    cv, kw = _gate_stars()["snr"]
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        BlsDetector(second_pass=True, **kw).detect({0: cv}, C)
    assert not [r for r in caplog.records if "re-search" in r.getMessage()]


def test_a_star_whose_first_box_stands_is_never_searched_again(monkeypatch):
    calls = []
    monkeypatch.setattr(BlsDetector, "_re_search", lambda self, *a: calls.append(a))
    _, _, cv = planted()
    out = detect({0: cv}, second_pass=True)
    assert len(out) == 3 and calls == []
    # not even when the accepted candidate happens to carry a mask
    det, out = _run_stub(_cand(2.0, None, MASK), _cand(5.3, None), second_pass=True)
    assert len(det.calls) == 1 and len(out) == 2 and out[0].score == 12.0


def test_a_search_with_no_box_left_is_logged(caplog):
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        _run_stub(_cand(2.0, "pulsation guard", MASK), None, second_pass=True)
    lines = [r.getMessage() for r in caplog.records if "re-search" in r.getMessage()]
    assert lines == ["bls star 0: re-search found no box"]


@pytest.mark.parametrize(
    "eb", [(3.0, 1.1, 40.0, 0.3, 0.1), (5.0, 1.0, 60.0, 0.3, 0.1)], ids=["40h", "60h"]
)
def test_the_dip_is_found_in_star_level_flux_when_an_eclipse_fills_the_nights(eb):
    # an eclipse that fills most of every night is flat once each night is divided by its own
    # median: only flux in the star's own units shows it all, and the mask must cover it
    t, cand = _eb_mask(eb)
    assert cand.rejected is not None and cand.mask
    deep = 1.0 - eclipsing(t, *eb) > 0.02
    assert deep.sum() > 5000 and _covered(t, cand.mask)[deep].all()


def test_a_secondary_run_with_no_bin_at_the_seed_level_is_not_a_dip():
    # 1.9 standard errors a bin over 4 h: S/N 7.6 over the run, yet no bin reaches support_snr
    # (2), so it is not a dip and not the primary's secondary however long and deep it is
    t = night_times(20).astype(float)
    prof = box_profile(t, 2.0 * DAY, 0.1 * DAY, 3 * 3600.0, 0.2) * box_profile(
        t, 2.0 * DAY, 1.1 * DAY, 4 * 3600.0, 0.00154
    )
    flat = box_profile(t, 2.0 * DAY, 1.1 * DAY, 4 * 3600.0, 1.0) < 1.0
    covered = _clean_mask(prof, t, 2.0, 0.1, 0.01)
    assert flat.sum() > 500 and not covered[flat].any()


def test_a_secondary_is_as_long_as_the_whole_primary_dip_not_just_the_box():
    # primary: a 3 h core at 20 percent inside 12 h wings. A clean 3 h, 5 percent dip half a
    # period on is as long as the core but shorter than 0.7 of the whole dip: a planet that
    # happens to sit there, not the mirror image of the primary
    t = night_times(20).astype(float)
    prof = (
        box_profile(t, 2.0 * DAY, 0.1 * DAY, 3 * 3600.0, 0.2)
        * box_profile(t, 2.0 * DAY, 0.1 * DAY, 12 * 3600.0, 0.0013)
        * box_profile(t, 2.0 * DAY, 1.1 * DAY, 3 * 3600.0, 0.05)
    )
    second = box_profile(t, 2.0 * DAY, 1.1 * DAY, 3 * 3600.0, 1.0) < 1.0
    covered = _clean_mask(prof, t, 2.0, 0.1, 0.01)
    assert second.sum() > 500 and not covered[second].any()


def test_the_level_follows_the_flux_through_the_common_mode_correction():
    # flux * level is the star's flux in its own units up to the common mode, whatever the
    # per-night medians and the common mode divided out on the way
    _, _, curves = make_field(n_stars=12, n_nights=10)
    det = BlsDetector()
    cleaned = {k: det._clean(v, C) for k, v in curves.items()}
    common_mode = det._common_mode(cleaned)
    assert common_mode is not None
    grid, cm = common_mode
    c = cleaned[3]
    t, flux, _, level = det._prepare(c, common_mode)
    in_own_units = c.flux * c.level / cm[np.searchsorted(grid, c.t)]
    np.testing.assert_allclose(flux * level, in_own_units[np.isin(c.t, t)], rtol=1e-9)


def test_performance_with_re_searches_20_stars_10_nights(caplog):
    # the guard above never reaches the second pass (its noise stars are refused on S/N): here
    # four stars carry a deep one-off dip that wins the search and is refused for lack of events,
    # so each is searched a second time (the planet behind it is not there to find)
    t = night_times(10)
    rng = np.random.default_rng(0)
    curves = {}
    for i in range(20):
        j = i // 5
        prof = np.ones(len(t))
        if i % 5 == 0:  # four planets, each on its own period and phase
            prof = box_profile(t, (2.3 + 0.37 * j) * DAY, (0.14 + 0.02 * j) * DAY, 3 * 3600.0, 0.01)
        elif i % 5 == 1:  # one deep dip on a single night
            night = 2 + j
            prof = 1.0 - 0.03 * ((t >= night * DAY + 2 * 3600) & (t < night * DAY + 5 * 3600))
        curves[i] = curve(i, t, prof, 0.005, rng)
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        t0 = time.perf_counter()
        out = detect(curves, second_pass=True)
        elapsed = time.perf_counter() - t0
    searched = [r for r in caplog.records if "re-search" in r.getMessage()]
    assert len(searched) == 4, "the fixture must fire re-searches or it guards nothing"
    assert {d.star_id for d in out} == {0, 5, 10, 15}
    # 2.1-2.7 s measured (three runs); four times the slowest, as above
    assert elapsed < 11.0, (
        f"20 stars x 10 nights with {len(searched)} re-searches took {elapsed:.1f} s"
    )


# -- pulsating hosts: the sinusoid is subtracted before the box search (#35) -------------------


def pulsator(seed, period_d=0.9, amp=0.015, depth=0.01, noise=0.005, n_nights=20, transit=True):
    """A star with a sinusoid and (optionally) a 3 h transit at P 3.1 d, epoch 0.05 d.

    The epoch puts four events inside the nights of a 20-night record (days 0, 3, 6 and 9).
    """
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = 1.0 + amp * np.sin(2 * np.pi * t / (period_d * DAY) + 0.7)
    if transit:
        prof = prof * box_profile(t, 3.1 * DAY, 0.05 * DAY, 3 * 3600.0, depth)
    return t, curve(0, t, prof, noise, rng)


def _arrays(det, cv):
    """The prepared samples of one star, as the search sees them."""
    c = det._clean(cv, C)
    t, flux, night, level = det._prepare(c, None)
    return bls._Arrays(t, flux, night, level, c.t, c.night_of)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_planet_on_a_pulsating_host_is_found_with_every_event_bounded(seed):
    t, cv = pulsator(seed)
    truth = planted_windows(t, 3.1, 0.05, 3.0)
    assert len(truth) == 4
    out = detect({0: cv})
    for lo, hi in truth:
        assert max((span_iou(d.t_start, d.t_end, lo, hi) for d in out), default=0.0) >= 0.5
    for d in out:  # and nothing else is reported
        assert max(span_iou(d.t_start, d.t_end, lo, hi) for lo, hi in truth) >= 0.5
    assert len(out) == len(truth)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_same_pulsating_star_without_the_transit_gives_nothing(seed):
    _, cv = pulsator(seed, transit=False)
    assert detect({0: cv}) == []
    assert BlsDetector().search(cv, C).rejected == "pulsation guard"


def test_without_the_detrending_the_guard_rejects_the_whole_star():
    _, cv = pulsator(0)
    off = BlsDetector(detrend_pulsation=False)
    assert off.detect({0: cv}, C) == []
    assert off.search(cv, C).rejected == "pulsation guard"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_sinusoid_near_the_transit_period_is_not_a_transit(seed):
    # 0.33 cycles/day is below the guard band; in nights divided by their median it is a ramp
    # that folds into a box at 1.33 cycles/day. Subtracting that alias once left the ramp's other
    # alias as a dip in every night (S/N 14, 20 events at 0.6 d in two of these three seeds)
    _, cv = pulsator(seed, period_d=3.0, amp=0.015, transit=False)
    assert detect({0: cv}) == []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_planet_beside_a_sinusoid_at_a_nearby_period_adds_no_false_event(seed):
    t, cv = pulsator(seed, period_d=3.0, amp=0.015)
    truth = planted_windows(t, 3.1, 0.05, 3.0)
    for d in detect({0: cv}):
        assert max(span_iou(d.t_start, d.t_end, lo, hi) for lo, hi in truth) >= 0.5


def test_a_slower_sinusoid_than_the_band_is_told_from_one_inside_it():
    det = BlsDetector()
    for period_d, slower in [(3.0, True), (0.9, False)]:
        _, cv = pulsator(0, period_d=period_d, transit=False)
        a = _arrays(det, cv)
        td = (a.t - float(a.t[0])) / DAY
        k = np.unique(a.night, return_inverse=True)[1]
        x = bls._demean_by_night(a.flux - 1.0, k)
        # the gain of the best in-band sinusoid, as the detrending finds it
        f_band, power, _, df = det._band_power(a.t, x, det._white_sigma(a.flux, a.night), C)
        peaks = bls._top_peaks(f_band, power, bls._TREND_PEAKS, 1.0 / float(td[-1]))
        gain = max(
            bls._sine_fit(td, x, k, bls._refine_frequency(td, x, k, float(f_band[i]), df))[2]
            for i in peaks
        )
        assert det._slower_sinusoid_wins(td, x, k, gain) is slower


def _events_found(out, truth):
    """How many of the `truth` windows some detection bounds (IoU >= 0.5)."""
    return sum(
        1
        for lo, hi in truth
        if max((span_iou(d.t_start, d.t_end, lo, hi) for d in out), default=0.0) >= 0.5
    )


def test_a_box_below_the_local_bar_is_refused_though_it_is_real():
    # a 0.17 percent transit on the 1.5 percent pulsator: detrended S/N 9.3 but local S/N 8.0,
    # so only the local bar (and not the S/N one) refuses it at the bar of 9
    t, cv = pulsator(1, depth=0.0017)
    truth = planted_windows(t, 3.1, 0.05, 3.0)
    assert detect({0: cv}) == []
    assert BlsDetector().search(cv, C).rejected == "pulsation guard"
    cand = BlsDetector(trend_snr_threshold=7.9).search(cv, C)
    assert cand.rejected is None and cand.snr > 9.0 > cand.snr_local > 7.9
    assert _events_found(detect({0: cv}, trend_snr_threshold=7.9), truth) == 4


def test_a_box_above_the_bar_of_nine_is_accepted():
    # a 0.22 percent transit: S/N 11.7 and local S/N 10.0, between a bar of 9 and one of 12
    t, cv = pulsator(1, depth=0.0022)
    assert _events_found(detect({0: cv}), planted_windows(t, 3.1, 0.05, 3.0)) == 4


def test_the_ratio_floor_is_three_in_the_units_of_the_guards_level():
    # a 0.4 d pulsator: amplitude 0.05 percent gives a ratio of 2.6 (refused), 0.07 percent
    # about 4.7 (accepted); the ratio is gain / (2 sigma**2) over the guard's level
    det = BlsDetector()
    _, weak = pulsator(3, period_d=0.4, amp=0.0005, transit=False)
    _, mid = pulsator(3, period_d=0.4, amp=0.0007, transit=False)
    assert det._pulsation_trend(_arrays(det, weak), C) is None
    got = det._pulsation_trend(_arrays(det, mid), C)
    assert got is not None and 3.0 < got[1][2] < 6.0


def test_a_fast_pulsator_is_detrended():
    # 10 cycles/day, well above a band that stopped at 5
    t, cv = pulsator(0, period_d=0.1, amp=0.015)
    assert _events_found(detect({0: cv}), planted_windows(t, 3.1, 0.05, 3.0)) == 4


def test_the_refit_leaves_out_one_and_a_half_boxes(monkeypatch):
    det = BlsDetector()
    _, cv = pulsator(0)
    a = _arrays(det, cv)
    box = Box(3.1 * DAY, 0.05 * DAY, 2 * 3600.0, 0.03, 0, len(a.t), 0.0)
    seen = []
    real = bls.in_box
    monkeypatch.setattr(bls, "in_box", lambda t, b: (seen.append(b.duration_s), real(t, b))[1])
    det._pulsation_trend(a, C, box)
    assert seen == [pytest.approx(1.5 * box.duration_s)]


def test_a_short_box_still_leaves_the_transit_out_of_the_fit():
    # a 2 h box on a 3 h, 5 percent transit: the wings of the transit stay in the fit unless
    # the window is wider than the box (fit error 0.64 of the amplitude at 1.0 x, 0.03 at 1.5 x)
    det = BlsDetector()
    _, cv = pulsator(0, amp=0.003, depth=0.05, noise=0.003)
    a = _arrays(det, cv)
    truth = 0.003 * np.sin(2 * np.pi * a.t / (0.9 * DAY) + 0.7)
    k = np.unique(a.night, return_inverse=True)[1]
    box = Box(3.1 * DAY, 0.05 * DAY, 2 * 3600.0, 0.05, 0, len(a.t), 0.0)
    err = float(bls._demean_by_night(det._pulsation_trend(a, C, box)[0] - truth, k).std())
    assert err < 0.1 * 0.003


def test_performance_with_pulsating_hosts_20_stars_10_nights(caplog):
    # two of the 20 stars pulsate (1.5 percent, 0.9 and 0.6 d) and carry the 3 h transit of
    # `pulsator`: the guard rejects each, so each is detrended and searched twice more
    t = night_times(10)
    rng = np.random.default_rng(0)
    curves = {}
    for i in range(20):
        if i in (0, 10):
            curves[i] = pulsator(i, period_d={0: 0.9, 10: 0.6}[i], n_nights=10)[1]
        else:
            curves[i] = curve(i, t, np.ones(len(t)), 0.005, rng)
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        t0 = time.perf_counter()
        out = detect(curves)
        elapsed = time.perf_counter() - t0
    subtracted = [r for r in caplog.records if "subtracted" in r.getMessage()]
    assert len(subtracted) == 2, "the fixture must run the detrending or it guards nothing"
    assert {d.star_id for d in out} == {0, 10}
    # 3.1-6.0 s measured (several runs, host load average 30-60); four times the slowest
    assert elapsed < 24.0, f"20 stars x 10 nights with 2 pulsators took {elapsed:.1f} s"


def test_performance_of_the_pulsation_fit_on_a_20_night_record():
    # the fit is a small part of a detrended star's search (about 0.15 s of 1.5 s), so the test
    # above cannot see it grow: raising _TREND_REFINE_STEPS from 14 to 600 left that one under
    # its bound (3.1 s to 5.7 s) and turned this one red (3.2 s). A fit without and one with a box.
    det = BlsDetector()
    a = _arrays(det, pulsator(0, n_nights=20)[1])
    box = Box(3.1 * DAY, 0.05 * DAY, 3 * 3600.0, 0.03, 0, len(a.t), 0.0)
    t0 = time.perf_counter()
    first = det._pulsation_trend(a, C)
    refit = det._pulsation_trend(a, C, box)
    elapsed = time.perf_counter() - t0
    assert first is not None and refit is not None, "the fixture must fit a sinusoid"
    assert first[1][0] == pytest.approx(1 / 0.9, abs=0.01)
    # 0.10-0.18 s measured (several runs, host load average 30-60); four times the slowest
    assert elapsed < 0.75, f"two pulsation fits on 20 nights took {elapsed:.2f} s"


def test_detrending_is_deterministic():
    _, cv = pulsator(1)
    a, b = detect({0: cv}), detect({0: cv})
    assert a == b and len(a) == 4
    assert BlsDetector().search(cv, C) == BlsDetector().search(cv, C)


def test_a_star_with_no_pulsation_is_searched_as_before():
    _, _, cv = planted()
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected is None and cand.trend is None
    assert detect({0: cv}) == detect({0: cv}, detrend_pulsation=False)
    rng = np.random.default_rng(5)
    t = night_times(20)
    noise = curve(0, t, np.ones(len(t)), 0.005, rng)
    assert detect({0: noise}) == []


def test_the_candidate_records_the_sinusoid_that_was_subtracted(caplog):
    _, cv = pulsator(0)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected is None
    assert cand.trend is not None
    assert cand.trend[0] == pytest.approx(1 / 0.9, abs=0.003)
    assert cand.trend[1] == pytest.approx(0.015, rel=0.05)
    with caplog.at_level("INFO", logger="skywatch.detectors.bls"):
        detect({0: cv})
    lines = [r.getMessage() for r in caplog.records if "subtracted" in r.getMessage()]
    assert len(lines) == 1 and "bls star 0: pulsation 1.1" in lines[0]


def test_the_sinusoid_is_found_at_its_frequency_with_its_amplitude():
    # 0.9 d is 1.11 cycles/day: the Fourier peak of a record of 8 h nights is not at its
    # frequency (the nights' own intercepts take most of it); the least-squares refinement of
    # the six strongest peaks is
    det = BlsDetector()
    for period_d, amp in [(0.9, 0.015), (5 / 24, 0.004), (0.4, 0.002)]:
        _, cv = pulsator(3, period_d=period_d, amp=amp, transit=False)
        a = _arrays(det, cv)
        (f, got, ratio) = det._pulsation_trend(a, C)[1]
        assert f == pytest.approx(1 / period_d, abs=0.002)
        assert got == pytest.approx(amp, rel=0.1)
        assert ratio > 3.0


def test_noise_has_no_sinusoid_and_a_weak_one_needs_the_ratio():
    det = BlsDetector()
    rng = np.random.default_rng(7)
    t = night_times(20)
    a = _arrays(det, curve(0, t, np.ones(len(t)), 0.005, rng))
    assert det._pulsation_trend(a, C) is None
    _, cv = pulsator(3, period_d=0.4, amp=0.002, transit=False)
    a = _arrays(det, cv)
    assert det._pulsation_trend(a, C) is not None
    assert BlsDetector(trend_min_ratio=1e6)._pulsation_trend(a, C) is None


def test_the_fit_leaves_out_the_box_so_a_transit_does_not_pull_it():
    # a deep transit on a weakly pulsating star: the fit with the transit in it is further
    # from the truth than the fit that leaves the transit's samples out
    det = BlsDetector()
    _, cv = pulsator(0, amp=0.004, depth=0.03, noise=0.004)
    a = _arrays(det, cv)
    box = Box(3.1 * DAY, 0.05 * DAY, 3 * 3600.0, 0.03, 0, len(a.t), 0.0)
    truth = 0.004 * np.sin(2 * np.pi * a.t / (0.9 * DAY) + 0.7)
    k = np.unique(a.night, return_inverse=True)[1]

    def error(fit):
        return float(bls._demean_by_night(fit[0] - truth, k).std())

    assert error(det._pulsation_trend(a, C, box)) < 0.7 * error(det._pulsation_trend(a, C))


def test_a_detrended_candidate_must_clear_its_own_snr_bar():
    _, cv = pulsator(1, depth=0.004)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected is None and min(cand.snr, cand.snr_local) > 9.0
    assert detect({0: cv}, trend_snr_threshold=100.0) == []
    assert BlsDetector(trend_snr_threshold=100.0).search(cv, C).rejected == "pulsation guard"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_an_eclipsing_binary_is_still_refused(seed):
    # 8 percent primary and 5 percent secondary eclipses of 3 h every 1.4 d: harmonics of the
    # binary are sinusoids too, and subtracting them must not make it a transit
    rng = np.random.default_rng(seed)
    t = night_times(20)
    prof = box_profile(t, 1.4 * DAY, 0.05 * DAY, 3 * 3600.0, 0.08) * box_profile(
        t, 1.4 * DAY, 0.75 * DAY, 3 * 3600.0, 0.05
    )
    assert detect({0: curve(0, t, prof, 0.005, rng)}) == []


def test_detrending_needs_the_guard_and_validates_its_parameters():
    _, cv = pulsator(0)
    # without the guard there is nothing to be rejected by and nothing to subtract
    cand = BlsDetector(guard_fap=None).search(cv, C)
    assert cand is not None and cand.trend is None
    for kw in [{"trend_min_ratio": 0.5}, {"trend_snr_threshold": 0}]:
        with pytest.raises(ValueError):
            BlsDetector(**kw)


def test_only_a_star_the_pulsation_guard_rejected_is_searched_again(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the pulsation was fitted for a star the guard did not reject")

    monkeypatch.setattr(BlsDetector, "_pulsation_trend", refuse)
    rng = np.random.default_rng(5)
    t = night_times(20)
    stars = {
        0: curve(0, t, np.ones(len(t)), 0.005, rng),  # no box at the S/N gate
        1: planted(n_nights=20)[2],  # a box that stands
        2: curve(
            2, t, 1.0 - 0.03 * ((t >= 3 * DAY + 3600) & (t < 3 * DAY + 4 * 3600)), 0.005, rng
        ),  # a one-night dip: rejected for lack of events
    }
    assert BlsDetector().search(stars[2], C).rejected in {"min_events", "support"}
    assert {d.star_id for d in detect(stars)} == {1}


def test_the_white_noise_scatter_does_not_contain_the_pulsation():
    det = BlsDetector()
    _, cv = pulsator(0, amp=0.02, transit=False)
    a = _arrays(det, cv)
    assert det._white_sigma(a.flux, a.night) == pytest.approx(0.005, rel=0.05)
    # the scatter about the median, which the search uses for its S/N, is the pulsation's
    assert 1.4826 * np.median(np.abs(a.flux - np.median(a.flux))) > 0.01
    assert det._white_sigma(a.flux[:1], a.night[:1]) == 0.0


# -- eclipsing binaries: a V is not a box (#36) ----------------------------------------------------
# Binaries as the world draws them (V-shaped eclipses, the secondary 0.2 to 0.8 of the primary) that
# the first pass reported as transits on the development runs, with the noise of the star they came
# from. Each is (period d, epoch d, eclipse duration h, primary depth, secondary depth), noise.

EB_LEAKS = {
    # seed35-200s-20n star 184: every reported event is the same eclipse, the other one is never
    # inside a predicted window (an odd-even or half-phase test has nothing to compare)
    "one-sided": ((2.9881, 0.8595, 6.7766, 0.0609, 0.0406), 0.0183),
    # seed40-200s-20n star 179: a 13.5 h eclipse in 8 h nights, fit by a 3 h box
    "wide": ((4.8555, 2.6003, 13.5156, 0.0511, 0.0103), 0.023),
    # seed48-200s-20n star 185: both eclipses seen, the box on half the period, 13 events
    "both seen": ((1.1505, 0.2102, 1.5208, 0.0594, 0.0308), 0.0272),
    # the hard case: equal depths, so the box on P / 2 is the whole signal and the eclipses of
    # odd and even events are the same (odd-even cannot tell it from a planet at P / 2)
    "equal depth": ((2.1, 0.14, 2.6, 0.06, 0.06), 0.02),
}


def eb_leak(key, seed=0, n_nights=20):
    eb, noise = EB_LEAKS[key]
    t = night_times(n_nights)
    return t, curve(0, t, eclipsing(t, *eb), noise, np.random.default_rng(seed))


def ramped(t, period_d, epoch_d, dur_h, depth, ramp=0.1):
    """A transit as the world draws it: a flat bottom with `ramp` of the duration as each ramp."""
    period, dur = period_d * DAY, dur_h * 3600.0
    ph = np.mod(t - epoch_d * DAY + period / 2.0, period) - period / 2.0
    edge = np.minimum(ph + dur / 2.0, dur / 2.0 - ph) / (ramp * dur)
    return 1.0 - depth * np.clip(edge, 0.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("key", list(EB_LEAKS))
def test_these_binaries_are_reported_as_transits_without_the_shape_test(key, seed):
    # the premise: every guard of the first pass lets them through
    _, cv = eb_leak(key, seed)
    cand = BlsDetector(shape_threshold=None).search(cv, C)
    assert cand is not None and cand.rejected is None and len(cand.events) >= 2
    assert len(detect({0: cv}, shape_threshold=None)) == len(cand.events)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("key", list(EB_LEAKS))
def test_a_v_shaped_eclipse_train_is_not_a_transit(key, seed):
    _, cv = eb_leak(key, seed)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected == "shape" and cand.events == []
    assert cand.shape >= BlsDetector().shape_threshold
    assert detect({0: cv}) == []


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dur_h", [0.8, 1.4, 2.5, 3.3, 5.5])
def test_a_planet_with_ramps_is_not_taken_for_an_eclipse(dur_h, seed):
    # durations between the box grid's values (a 2.5 h transit is fit by a 2 or 3 h box)
    rng = np.random.default_rng(seed)
    t = night_times(20)
    cv = curve(0, t, ramped(t, 2.3, 0.14, dur_h, 0.015), 0.005, rng)
    cand = BlsDetector().search(cv, C)
    assert cand is not None and cand.rejected is None
    assert cand.shape < 0  # the box wins: that is the shape of the dip
    out = detect({0: cv})
    assert len(out) == len(cand.events) >= 2


def test_the_shape_statistic_is_positive_for_a_v_and_negative_for_a_box():
    det = BlsDetector()
    t = night_times(20)
    night = t // DAY
    box = Box(2.0 * DAY, 0.14 * DAY, 3 * 3600.0, 0.04, 0, len(t), 0.0)
    v = eclipsing(t, 2.0, 0.14, 4.0, 0.05, 0.0)  # a triangle of 4 h, every 2 d
    flat = ramped(t, 2.0, 0.14, 3.0, 0.05)  # a flat-bottomed 3 h transit, every 2 d
    assert det._shape_excess(t, v, night, 0.005, box) > 1.5
    assert det._shape_excess(t, flat, night, 0.005, box) < -1.5


def test_the_shape_statistic_gives_no_verdict_on_a_handful_of_samples():
    det = BlsDetector()
    box = Box(2.0 * DAY, 0.14 * DAY, 3 * 3600.0, 0.05, 5, 9, 0.0)
    t = 0.14 * DAY + np.array(
        [-5400.0, -3600.0, -1800.0, 0.0, 1800.0, 3600.0, 5400.0, 7200.0, -7200.0]
    )
    flux = 1.0 - 0.05 * np.clip(1.0 - np.abs(t - 0.14 * DAY) / 5400.0, 0.0, None)
    assert len(t) < 10
    assert det._shape_excess(t, flux, np.zeros(len(t), dtype=np.int64), 0.005, box) == 0.0
    assert det._shape_test(t, flux, np.zeros(len(t), dtype=np.int64), 0.005, box) == (0.0, 0.0)
    assert det._shape_gains(t, flux, np.zeros(len(t), dtype=np.int64), box) is None
    t = np.concatenate([t, t + 60.0 * 40])  # the same V sampled twice as densely: now a verdict
    flux = np.concatenate([flux, flux])
    assert det._shape_excess(t, flux, np.zeros(len(t), dtype=np.int64), 0.005, box) != 0.0


def test_the_shape_statistic_is_zero_without_samples_in_the_box():
    det = BlsDetector()
    t = night_times(4)
    box = Box(2.0 * DAY, 0.9 * DAY, 3600.0, 0.01, 0, len(t), 0.0)  # inside the day gap
    assert det._shape_excess(t, np.ones(len(t)), t // DAY, 0.005, box) == 0.0


def test_the_shape_threshold_is_validated_and_off_with_none():
    assert BlsDetector().shape_threshold == 1.5
    assert BlsDetector(shape_threshold=None).shape_threshold is None
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            BlsDetector(shape_threshold=bad)
    _, cv = eb_leak("one-sided")
    assert BlsDetector(shape_threshold=None).search(cv, C).shape is None


def test_a_threshold_above_the_statistic_lets_the_eclipse_through():
    _, cv = eb_leak("one-sided")
    cand = BlsDetector().search(cv, C)
    kept = BlsDetector(shape_threshold=cand.shape + 1.0, shape_chi2=None).search(cv, C)
    assert kept.rejected is None and kept.shape == pytest.approx(cand.shape)


def test_an_accepted_planet_records_its_shape_statistic():
    _, _, cv = planted()
    cand = BlsDetector().search(cv, C)
    assert cand.rejected is None and cand.shape is not None and cand.shape < 0


def test_a_shape_rejection_plans_the_mask_of_a_second_pass():
    _, cv = eb_leak("one-sided")
    cand = _real_search(cv, plan_mask=True)
    assert cand.rejected == "shape" and cand.mask
    assert _real_search(cv, masked=(HARMLESS_MASK,)).rejected == "shape"


def test_after_a_shape_rejection_the_first_period_and_its_halves_are_refused():
    # the box of a rejected binary is its eclipse train, whose residue repeats like a pulsator's
    first = _cand(2.0, "shape", MASK)
    for ratio in (1.0, 0.5, 2.0):
        det, out = _run_stub(first, _cand(2.0 * ratio, None), second_pass=True)
        assert out == [] and len(det.calls) == 2
    _, out = _run_stub(first, _cand(2.0 * 1.1, None), second_pass=True)
    assert len(out) == 2


def test_a_planet_on_a_pulsating_host_still_clears_the_shape_test():
    for seed in range(3):
        _, cv = pulsator(seed)
        cand = BlsDetector().search(cv, C)
        assert cand.rejected is None and cand.trend is not None and cand.shape < 0


# -- the shape gate's boundary, period cap and amplitude sign (#40 review) -------------------------


def test_a_statistic_equal_to_the_threshold_rejects():
    # the gate is `shape >= threshold`: a threshold set to the candidate's own statistic rejects it
    _, cv = eb_leak("one-sided")
    cand = BlsDetector().search(cv, C)
    at = BlsDetector(shape_threshold=cand.shape, shape_chi2=None).search(cv, C)
    assert at.rejected == "shape" and at.shape == cand.shape


def test_a_triangle_wider_than_the_period_is_not_a_template():
    # A V train whose eclipse fills the whole period (0.7 d): the only template a wider-than-period
    # triangle adds is the bare |x| cusp, which fits it ~3 percent better than any capped width.
    # A tuned pin (no realistic fixture separates the cap): the capped statistic reads 1.45 to 1.5.
    t = night_times(20)
    night = t // DAY
    P, ep, noise = 0.7 * DAY, 0.14 * DAY, 0.012
    ph = np.mod(t - ep + P / 2, P) - P / 2
    flux = (1 - 0.02 * np.clip(1 - np.abs(ph) / (P / 2), 0, 1)) * (
        1 + np.random.default_rng(1).normal(0, noise, len(t))
    )
    box = Box(P, ep, 3 * 3600.0, 0.01, 0, len(t), 0.0)
    # sigma is passed (not measured): scaled so the capped statistic sits just under the 1.5 gate
    sigma = noise * 14.15 / 1.5
    stat = BlsDetector()._shape_excess(t, flux, night, sigma, box)
    assert 1.45 < stat < 1.5


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dur_h", [1.4, 3.0])
def test_a_brightening_on_some_event_nights_is_not_fitted_with_a_negative_depth(dur_h, seed):
    # The fits take positive amplitudes only: a night that shows a brightening where the others
    # show the dip is left out, not fitted with a negative depth (which read this planet, with
    # every other event night mirrored about 1, as a V: +6.7, +4.3 and +2.1 against -5 to -7.5).
    # With an amplitude for each night such a night adds nothing: the gains with those nights
    # mirrored are the gains with those nights absent. With one amplitude it takes the first
    # centre's numerator, which is at most 0 (the `pick = 0` quirk, kept: see the next test).
    det = BlsDetector()
    t = night_times(20)
    cv = curve(0, t, ramped(t, 2.3, 0.14, dur_h, 0.015), 0.005, np.random.default_rng(seed))
    cand = det.search(cv, C)
    a = _arrays(det, cv)
    mirrored = np.where(a.night % 2 == 1, 2.0 - a.flux, a.flux)
    even = a.night % 2 == 0
    with_nights = det._shape_gains(a.t, mirrored, a.night, cand.box)
    without = det._shape_gains(a.t[even], mirrored[even], a.night[even], cand.box)
    assert with_nights.g_box_night == pytest.approx(without.g_box_night, rel=1e-9)
    assert with_nights.g_triangle_night == pytest.approx(without.g_triangle_night, rel=1e-9)
    assert det._shape_excess(a.t, mirrored, a.night, cand.sigma, cand.box) < -1.5
    shape, advantage = det._shape_test(a.t, mirrored, a.night, cand.sigma, cand.box)
    assert shape < det.shape_threshold and advantage < det.shape_chi2


@pytest.mark.parametrize("dur_h", [1.4, 3.0])
def test_a_brightening_night_still_lowers_the_one_amplitude_gains(dur_h):
    # The `pick = 0` quirk is kept (issue #41): fixing it (a night with no positive score adding
    # nothing) read a planet of an unseen seed, 3 events at S/N 9.7, at 1.503 instead of 0.82, over
    # the gate, and cost no binary a verdict on the development runs. This pins the behaviour so
    # that changing it is a decision, not a refactor: the brightening nights subtract.
    det = BlsDetector()
    t = night_times(20)
    cv = curve(0, t, ramped(t, 2.3, 0.14, dur_h, 0.015), 0.005, np.random.default_rng(0))
    cand = det.search(cv, C)
    a = _arrays(det, cv)
    mirrored = np.where(a.night % 2 == 1, 2.0 - a.flux, a.flux)
    even = a.night % 2 == 0
    with_nights = det._shape_gains(a.t, mirrored, a.night, cand.box)
    without = det._shape_gains(a.t[even], mirrored[even], a.night[even], cand.box)
    assert with_nights.g_box < without.g_box
    assert with_nights.g_triangle < without.g_triangle


@pytest.mark.parametrize("offset_h", [0.8, 1.0, 1.2])
def test_the_centring_window_is_half_an_hour_not_two(offset_h):
    # A 2 h transit with a shallow 2.4 h V dip centred `offset_h` off its middle on every event
    # night. Each night may slide the box and the triangle within _SHAPE_SHIFT_H of the box: at
    # 0.5 h the triangle is the only one that can reach the V and the dip reads +2.3 to +2.9
    # (rejected); with a 2 h window a box slides onto the V's flank too and it reads +0.3 to +0.9
    # (accepted). The lower side of the window is pinned by the planets with ramps above.
    t = night_times(20)
    period, epoch = 2.3 * DAY, 0.14 * DAY
    ph = np.mod(t - epoch - offset_h * 3600.0 + period / 2.0, period) - period / 2.0
    v = 1.0 - 0.01 * np.clip(1.0 - np.abs(ph) / (2.4 * 1800.0), 0.0, 1.0)
    cv = curve(0, t, ramped(t, 2.3, 0.14, 2.0, 0.015) * v, 0.005, np.random.default_rng(1))
    cand = BlsDetector().search(cv, C)
    assert cand.rejected == "shape" and cand.shape > 2.0
    assert detect({0: cv}) == []


# -- eclipses the joint reading misses (#41): a depth for each night, and the chi-square a V wins by --

# seed42-200s-3n star 0: P 1.123 d, a 2.38 h eclipse, 50 and 14 percent deep, 3 nights. The box (at
# half the period) lands on a whole secondary on night 0 and on the last hour of a primary on
# night 2 (the primary's centre is before the night starts: a steep egress ramp). Alone each night
# is a V; jointly the box needs one depth (0.087 and 0.107) and the triangle cannot (0.133 and 0.72).
EB_UNLIKE = ((1.1229, 0.8445, 2.3807, 0.4985, 0.1447), 0.0184)
# A V that wins by 24 in chi-square (about 5 sigma) at S/N 19 and reads 1.1 and 1.3: the chi-square
# divided by the S/N cannot see it. A 3.55 h, 3.7 percent eclipse (a draw of seed 19 of a scan of
# binaries as the world draws them), the box on half the period.
EB_MODERATE = ((2.261, 0.926, 3.55, 0.037, 0.018), 0.0244)


def eb_fixture(eb, n_nights, seed):
    (period, epoch, dur, d1, d2), noise = eb
    t = night_times(n_nights)
    return curve(0, t, eclipsing(t, period, epoch, dur, d1, d2), noise, np.random.default_rng(seed))


def test_events_that_are_unlike_slices_of_one_eclipse_are_read_with_a_depth_for_each_night():
    cv = eb_fixture(EB_UNLIKE, 3, 3)
    det = BlsDetector()
    bare = BlsDetector(shape_threshold=None).search(cv, C)
    assert bare.rejected is None and len(bare.events) == 2  # the premise: a "planet"
    a = _arrays(det, cv)
    assert det._shape_excess(a.t, a.flux, a.night, bare.sigma, bare.box) < 1.0  # one depth: a box
    cand = BlsDetector(shape_chi2=None).search(cv, C)  # the ratio of the depth-per-night reading
    assert cand.rejected == "shape" and cand.shape >= 1.5 and cand.events == []
    assert detect({0: cv}) == []
    # the chi-square is the depth-per-night advantage too (the one depth reads the box 64 ahead)
    by_chi = BlsDetector(shape_threshold=1e9).search(cv, C)
    assert by_chi.rejected == "shape" and by_chi.shape_chi2 > 100.0


def test_a_v_that_the_ratio_cannot_see_is_refused_by_the_chi_square_it_wins_by():
    cv = eb_fixture(EB_MODERATE, 20, 1019)
    ratio_only = BlsDetector(shape_chi2=None).search(cv, C)
    assert ratio_only.rejected is None and ratio_only.shape < 1.5  # the premise: a "planet"
    assert len(ratio_only.events) >= 2 and ratio_only.shape_chi2 >= 14.0
    cand = BlsDetector().search(cv, C)
    assert cand.rejected == "shape" and cand.events == [] and cand.shape == ratio_only.shape
    assert cand.shape_chi2 == ratio_only.shape_chi2 >= BlsDetector().shape_chi2
    assert detect({0: cv}) == []


def test_a_chi_square_equal_to_the_threshold_rejects():
    cv = eb_fixture(EB_MODERATE, 20, 1019)
    advantage = BlsDetector().search(cv, C).shape_chi2
    # the ratio is out of reach: only the chi-square decides
    at = BlsDetector(shape_threshold=1e9, shape_chi2=advantage).search(cv, C)
    assert at.rejected == "shape" and at.shape_chi2 == advantage
    above = BlsDetector(shape_threshold=1e9, shape_chi2=advantage + 0.01).search(cv, C)
    assert above.rejected is None


def test_the_chi_square_is_in_units_of_the_larger_of_sigma_and_the_boxs_own_scatter():
    # Noise that is not white (a heavy tail, red noise) is more than the MAD scale says, and an
    # amplitude for each night absorbs some of it; the box's residual scatter on the event nights
    # is the floor. Told a sigma 3x too small, the fits still read the V's advantage at the noise
    # the data have (the same fixture with the true sigma reads the same).
    cv = eb_fixture(EB_MODERATE, 20, 1019)
    det = BlsDetector()
    cand = BlsDetector(shape_chi2=None).search(cv, C)
    a = _arrays(det, cv)
    fit = det._shape_gains(a.t, a.flux, a.night, cand.box)
    assert fit.scatter == pytest.approx(cand.sigma, rel=0.05)
    true = det._shape_test(a.t, a.flux, a.night, cand.sigma, cand.box)[1]
    small = det._shape_test(a.t, a.flux, a.night, cand.sigma / 3.0, cand.box)[1]
    assert small == pytest.approx(true, rel=0.1)


def test_on_one_night_the_two_amplitudes_are_the_same_fit():
    # what `single_transit` relies on: a one-event box has one night, so a depth for each night is
    # the depth of the only night, and `_shape_excess` is unchanged by the second reading
    det = BlsDetector()
    t = night_times(1)
    box = Box(2.0 * DAY, 0.14 * DAY, 3 * 3600.0, 0.04, 0, len(t), 0.0)
    for flux in (
        eclipsing(t, 2.0, 0.14, 4.0, 0.05, 0.0),
        ramped(t, 2.0, 0.14, 3.0, 0.05),
    ):
        flux = flux * (1.0 + np.random.default_rng(0).normal(0.0, 0.004, len(t)))
        fit = det._shape_gains(t, flux, np.zeros(len(t), dtype=np.int64), box)
        assert fit.g_box == fit.g_box_night > 0 and fit.g_triangle == fit.g_triangle_night > 0


def test_the_chi_square_threshold_is_validated_and_off_with_none():
    assert BlsDetector().shape_chi2 == 14.0
    assert BlsDetector(shape_chi2=None).shape_chi2 is None
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            BlsDetector(shape_chi2=bad)
    _, cv = eb_leak("one-sided")
    assert BlsDetector(shape_threshold=None).search(cv, C).shape_chi2 is None  # the gate is off
    assert BlsDetector().search(cv, C).shape_chi2 is not None


def test_an_accepted_planet_records_a_negative_chi_square_advantage():
    _, _, cv = planted()
    cand = BlsDetector().search(cv, C)
    assert cand.rejected is None and cand.shape_chi2 is not None and cand.shape_chi2 < 0


# -- planet-free stars (#39): a sinusoid is told from a box with the nights' levels free -------


def night_pulsator(seed, period_h, amp=0.0025, noise=0.02, n_nights=20):
    """A pure sinusoid of `period_h` hours in 8 h nights, white noise on top (no planet)."""
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = 1.0 + amp * np.sin(2 * np.pi * t / (period_h * 3600.0) + 0.7)
    return curve(0, t, prof, noise, rng)


@pytest.mark.parametrize("period_h,seed", [(6.03, 1), (9.3, 0)])
def test_a_pulsator_the_plain_sine_test_lets_through_is_refused_with_the_nights_levels_free(
    period_h, seed
):
    # 0.25 percent in 2 percent noise: the nights' own medians absorb part of a 6-9 h sinusoid, so
    # the guard's plain Fourier power does not reach the box's chi-square (and its residual stays
    # under the false-alarm level) and a box of 3-6 h on the same phase every night passes
    cv = night_pulsator(seed, period_h)
    before = BlsDetector(guard_night_ratio=None, detrend_pulsation=False).search(cv, C)
    assert before.rejected is None and before.snr_local > 7.0 and len(before.events) >= 15
    cand = BlsDetector(detrend_pulsation=False).search(cv, C)
    assert cand.rejected == "pulsation guard" and cand.sine_ratio >= 0.81
    assert detect({0: cv}) == []  # nor does subtracting the sinusoid leave a box


def test_the_ratio_is_the_sinusoids_chi_square_gain_over_the_boxs_with_free_nightly_levels():
    # a noise-free sinusoid with the box over its lowest half-cycle: the box keeps 8 / pi**2 = 81
    # percent of the sinusoid's variance, so the ratio is about 1 / 0.81 = 1.23; a square wave
    # is the other way round, a sinusoid keeps at most 81 percent of its variance
    det = BlsDetector()
    t = night_times(20)
    night = (t // DAY).astype(np.int64)
    jitter = np.random.default_rng(0).normal(0.0, 0.0003, len(t))  # the scatter must be defined
    sine = 1.0 + 0.01 * np.sin(2 * np.pi * t / (0.25 * DAY) - np.pi / 2.0) + jitter
    low = Box(0.25 * DAY, 0.25 * DAY * 0.5, 0.125 * DAY, 0.01, 0, len(t), 0.0)
    ratio = det._night_sine_ratio(t, sine, night, low, C)
    assert 1.1 < ratio < 1.4
    period, dur = 0.4 * DAY, 0.2 * DAY  # a 50 percent duty square wave over 9.6 h cycles
    square = box_profile(t, period, 0.1 * DAY, dur, 0.01) + jitter
    ratio = det._night_sine_ratio(
        t, square, night, Box(period, 0.1 * DAY, dur, 0.01, 0, len(t), 0.0), C
    )
    assert 0.6 < ratio < 0.85


def test_a_planet_stays_under_the_ratio_even_when_it_is_short_and_deep():
    det = BlsDetector()
    _, _, plain = planted()
    t = night_times(20)
    prof = box_profile(t, 0.51 * DAY, 0.07 * DAY, 3 * 3600.0, 0.012)
    hot = curve(0, t, prof, 0.005, np.random.default_rng(0))
    for cv in (plain, hot):
        cand = det.search(cv, C)
        assert cand.rejected is None and 0.0 < cand.sine_ratio < 0.81
    assert det.search(hot, C).sine_ratio > 0.4  # the hot one sits near the bound (0.73 on seed 38)


def test_the_ratio_is_recorded_and_the_test_can_be_switched_off():
    _, _, cv = planted()
    assert BlsDetector().search(cv, C).sine_ratio is not None
    assert BlsDetector(guard_fap=None).search(cv, C).sine_ratio is None
    assert BlsDetector(guard_night_ratio=None).search(cv, C).sine_ratio is None
    # a bar at zero... is refused, an unreachable one lets the sinusoid through
    with pytest.raises(ValueError):
        BlsDetector(guard_night_ratio=0)
    sine = night_pulsator(1, 6.03)
    assert (
        BlsDetector(guard_night_ratio=1e9, detrend_pulsation=False).search(sine, C).rejected is None
    )


# -- planet-free stars (#39): flare stars are held to a higher local S/N ---------------------------


def flaring(seed, per_night=1, amp=0.05, tau=6.0, depth=0.012, n_nights=12):
    """A planet on a star with `per_night` flares in every night (positions drawn at random)."""
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = box_profile(t, 2.3 * DAY, 0.14 * DAY, 3 * 3600.0, depth)
    for k in range(n_nights):
        for _ in range(per_night):
            at = k * DAY + rng.integers(20, NIGHT_N - 120) * C
            dt = (t - at) / C
            prof = prof + np.where(dt >= 0, amp * np.exp(-np.maximum(dt, 0.0) / tau), 0.0)
    return curve(0, t, prof, 0.005, rng)


def masked_share(det, cv):
    c = det._clean(cv, C)
    return 1.0 - len(det._prepare(c, None)[0]) / len(c.t)


def test_a_star_with_many_flares_must_clear_the_higher_local_bar():
    det = BlsDetector()
    cv = flaring(0)
    assert masked_share(det, cv) >= 0.03  # a flare star by the rule: 3 percent of its samples
    _, _, quiet = planted(n_nights=12, depth=0.012)
    assert masked_share(det, quiet) < 0.03
    base = det.search(cv, C)
    assert base.rejected is None and base.snr_local > 7.0  # it passes the ordinary bar
    bar = base.snr_local + 0.1
    flare = BlsDetector(flare_snr_threshold=bar)
    assert flare.search(cv, C).rejected == "local snr"
    assert flare.detect({0: cv}, C) == []
    # a quiet star with a planet at least as clear is not held to it, nor is the flare star when
    # the share that makes a flare star is raised above its own
    assert BlsDetector(flare_snr_threshold=1e3).search(quiet, C).rejected is None
    assert BlsDetector(flare_snr_threshold=bar, flare_fraction=0.99).search(cv, C).rejected is None


def test_the_flare_bar_never_lowers_the_ordinary_one():
    cv = flaring(0)  # S/N 45, local S/N 37: a bar of 40 passes the first and fails the second
    low = BlsDetector(snr_threshold=40.0, flare_snr_threshold=1.0)
    assert low.search(cv, C).rejected == "local snr"
    assert BlsDetector(snr_threshold=50.0, flare_snr_threshold=7.2).search(cv, C).rejected == "snr"
    d = BlsDetector()
    assert d.flare_fraction == 0.03 and d.flare_snr_threshold == 7.2
    assert d.guard_night_ratio == 0.81


@pytest.mark.parametrize(
    "kw",
    [
        {"flare_fraction": 0},
        {"flare_fraction": 1.5},
        {"flare_snr_threshold": 0},
        {"guard_night_ratio": -1},
    ],
)
def test_the_new_parameters_are_validated(kw):
    with pytest.raises(ValueError):
        BlsDetector(**kw)


# -- flare tails (#43): the box that sits where a flare's decay is not -----------------------------


def flare_only(seed, n_nights=20, rate=1.5, noise=0.005):
    """A star with flares and no planet: Poisson(`rate`) per night, amplitude log-uniform 1 to 50
    percent, decay 2 to 30 minutes (the world's flare law), starting at a random sample."""
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = np.ones(len(t))
    for k in range(n_nights):
        for _ in range(rng.poisson(rate)):
            at = k * DAY + rng.integers(0, NIGHT_N) * C
            amp = math.exp(rng.uniform(math.log(0.01), math.log(0.5)))
            dt = (t - at) / C
            prof = prof + np.where(
                dt >= 0, amp * np.exp(-np.maximum(dt, 0.0) / rng.uniform(2, 30)), 0
            )
    return curve(0, t, prof, noise, rng)


def dip_after_flare(gap_min, dur_min=30, flare_at=60, n_nights=12, depth=0.012):
    """A planet (P 1.0 d, so one event in every night) whose box starts `gap_min` minutes after
    the end of a 40 percent flare that peaks at sample `flare_at` of every night. The flare's
    masked run ends 29 to 38 minutes after its peak (the test asserts it), the box is 30 minutes."""
    rng = np.random.default_rng(4)
    t = night_times(n_nights)
    run_end = flare_at + 33  # samples; the runs measure 29 to 38, and the test asserts it
    start = (run_end + gap_min) * C
    prof = box_profile(t, DAY, start + dur_min * 30, dur_min * 60.0, depth)
    for k in range(n_nights):
        dt = (t - (k * DAY + flare_at * C)) / C
        prof = prof + np.where(dt >= 0, 0.4 * np.exp(-np.maximum(dt, 0.0) / 6.0), 0.0)
    return curve(0, t, prof, 0.005, rng)


def masked_runs(det, cv):
    """Lengths of the runs of samples the one-sided clip masks, per night, in order."""
    c = det._clean(cv, C)
    kept = np.isin(c.t, det._prepare(c, None)[0])
    out = []
    for k in np.unique(c.night_of):
        m = ~kept[c.night_of == k]
        e = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))))
        out += [int(b - a) for a, b in zip(e[::2], e[1::2], strict=True)]
    return out


def test_flare_windows_span_each_flare_sized_run_and_the_decay_after_it():
    t = night_times(2)
    night = (t // DAY).astype(np.int64)
    kept = np.ones(len(t), dtype=bool)
    kept[100:110] = False  # 10 masked samples: exactly the bar
    kept[200:209] = False  # 9: a cosmic ray with a long tail
    kept[NIGHT_N + 50 : NIGHT_N + 70] = False  # 20 in the second night
    w = flare_windows(t, night, kept, 10, 3600.0)
    assert w.shape == (2, 2)
    # each window ends `window_s` after the run's last sample and starts within the run, after the
    # last sample before it (the samples of a run are out of the search's arrays, so where inside
    # the run the window starts is not observable)
    assert w[0][1] == t[109] + 3600.0 and t[99] < w[0][0] <= t[109]
    assert w[1][1] == t[NIGHT_N + 69] + 3600.0 and t[NIGHT_N + 49] < w[1][0] <= t[NIGHT_N + 69]
    covered = in_windows(t, w)
    assert covered[109:170].all() and not covered[99] and not covered[170:NIGHT_N].any()
    assert flare_windows(t, night, kept, 9, 3600.0).shape == (3, 2)
    assert flare_windows(t, night, np.ones(len(t), dtype=bool), 10, 3600.0).shape == (0, 2)
    # a run is a run within one night: five masked samples at the end of a night and five at the
    # start of the next are two short runs, not one of ten
    edge = np.ones(len(t), dtype=bool)
    edge[NIGHT_N - 5 : NIGHT_N + 5] = False
    assert flare_windows(t, night, edge, 10, 3600.0).shape == (0, 2)


def test_in_windows_includes_both_ends_and_nothing_else():
    t = np.arange(10, dtype=np.float64) * 60.0
    hit = in_windows(t, np.array([[120.0, 240.0], [480.0, 480.0]]))
    assert hit.tolist() == [False, False, True, True, True, False, False, False, True, False]
    assert not in_windows(t, np.zeros((0, 2))).any()


def test_a_cosmic_ray_rarely_and_a_small_flare_nearly_always_opens_a_window():
    # the clip masks a cosmic ray's 5 samples and, by chance, a few more (the running mean stays
    # above its threshold in noise): 4 to 6 percent of 400 spikes reach 10 samples. A flare of
    # 6 sigma and a 4 minute decay reaches 10 in 92 percent of 400, one of 6 sigma and 10 minutes
    # always
    rng = np.random.default_rng(11)
    k = np.arange(NIGHT_N) - 200

    def share(add):
        n = 0
        for _ in range(400):
            x = 1.0 + add + rng.normal(0.0, 0.005, NIGHT_N)
            m = positive_outliers(x / np.median(x), 4.0, 1.0)
            n += int(
                m.any()
                and flare_windows(
                    np.arange(NIGHT_N) * float(C), np.zeros(NIGHT_N, dtype=np.int64), ~m, 10, 1.0
                ).shape[0]
                > 0
            )
        return n / 400

    spike = np.zeros(NIGHT_N)
    spike[200] = 1.0
    assert 0.02 < share(spike) < 0.08
    assert share(np.where(k >= 0, 0.03 * np.exp(-np.maximum(k, 0) / 4.0), 0.0)) > 0.85
    assert share(np.where(k >= 0, 0.03 * np.exp(-np.maximum(k, 0) / 10.0), 0.0)) == 1.0


@pytest.mark.parametrize("seed", [49, 66])
def test_a_box_that_exists_only_between_the_tails_of_flares_is_refused(seed):
    # no planet: 20 nights of flares at the world's rate, noise 0.5 percent. The box sits on the
    # quiet stretches (events at 2 or 3 nights, local S/N 7.2 to 7.3) and its contrast is the
    # decay the clip left behind in the rest of every event night. Left out the samples within
    # 3 hours of a flare-sized masked run there is no contrast
    cv = flare_only(seed)
    off = BlsDetector(flare_window_h=None).search(cv, C)
    assert off.rejected is None and 7.2 < off.snr_local < 7.3 and len(off.events) >= 2
    assert off.snr_clean == off.snr_local
    assert detect({0: cv}, flare_window_h=None) != []
    cand = BlsDetector().search(cv, C)
    assert cand.rejected == "flare tails"
    assert not cand.snr_clean >= 7.0 and cand.snr_local == off.snr_local
    assert cand.shape is not None and cand.sine_ratio is not None  # it is the last gate
    assert detect({0: cv}) == []


def test_the_tail_gate_plans_a_second_pass_mask_like_any_other_rejection():
    det = BlsDetector()
    c = det._clean(flare_only(49), C)
    cand = det._search(c, None, C, plan_mask=True)
    assert cand.rejected == "flare tails" and len(cand.mask) > 0


def test_a_planet_on_a_flare_star_keeps_its_signal_when_the_tails_are_left_out():
    cv = flaring(0)
    cand = BlsDetector().search(cv, C)
    assert cand.rejected is None and 7.0 < cand.snr_clean < cand.snr_local
    assert cand.snr_local > 30 and len(detect({0: cv})) == len(cand.events) > 0


def test_the_clean_snr_is_the_local_snr_when_no_run_is_flare_sized():
    _, _, quiet = planted(n_nights=12, depth=0.012)
    det = BlsDetector()
    assert masked_runs(det, quiet) == []
    cand = det.search(quiet, C)
    assert cand.rejected is None and cand.snr_clean == cand.snr_local


@pytest.mark.parametrize(
    "gap_min,accepted,other_h",
    [(135, False, 2.0), (200, True, 5.0)],
)
def test_the_window_is_three_hours_long(gap_min, accepted, other_h):
    # a 30 minute dip in every night that begins 135 minutes (130 to 140) after the end of the
    # flare's masked run lies inside the 3 hour window (nothing is left of it), one that begins 200
    # minutes after lies outside. A window of 2 hours accepts the first, one of 5 hours refuses the
    # second
    cv = dip_after_flare(gap_min)
    det = BlsDetector()
    runs = masked_runs(det, cv)
    assert len(runs) == 12 and 28 < min(runs) <= max(runs) < 40  # runs of 29 to 38 minutes
    cand = det.search(cv, C)
    assert (cand.rejected is None) is accepted
    assert cand.rejected in (None, "flare tails")
    other = BlsDetector(flare_window_h=other_h).search(cv, C)
    assert (other.rejected is None) is not accepted
    assert BlsDetector().flare_window_h == 3.0


@pytest.mark.parametrize(
    "depth,accepted,clean",
    [(0.0059, False, (6.6, 6.95)), (0.0062, True, (7.05, 7.2))],
)
def test_the_clean_snr_is_held_to_the_ordinary_bar_not_a_higher_or_lower_one(
    depth, accepted, clean
):
    # the same dip as above, 150 minutes after the run ends: half of every event lies in the window.
    # The local S/N is 18 or 19, far above any bar, and what is left of it reads 6.8 or 7.1: a bar
    # of 7.2 (the flare bar) refuses the second, one of 6.5 accepts the first
    cv = dip_after_flare(150, depth=depth)
    cand = BlsDetector().search(cv, C)
    assert clean[0] < cand.snr_clean < clean[1] and cand.snr_local > 17
    assert (cand.rejected is None) is accepted
    assert cand.rejected in (None, "flare tails")
    assert BlsDetector().snr_threshold == 7.0


def test_the_reference_a_flare_may_lift_is_left_out_as_well_as_the_box():
    # a 30 minute dip in every night begins 260 minutes after the flare's masked run ends: no
    # sample of the box is inside a window. The windows cover 3 hours of the reference before it,
    # and with that left out the contrast has less weight and no flare residue (local S/N 7.4,
    # clean 6.8). A gate that left out the samples of the box that lie in a window, and nothing
    # else, would leave this one untouched
    cv = dip_after_flare(260, depth=0.0021)
    det = BlsDetector()
    c = det._clean(cv, C)
    kept = np.isin(c.t, det._prepare(c, None)[0])
    windows = flare_windows(c.t, c.night_of, kept, 10, 3 * 3600.0)
    cand = det.search(cv, C)
    assert abs(cand.box.period_s / DAY - 1.0) < 0.01
    assert len(windows) == 12 and in_windows(c.t, windows).sum() > 12 * 150
    assert not (in_windows(c.t, windows) & in_box(c.t, cand.box)).any()
    assert cand.snr_local > 7.2 and cand.snr_clean < 7.0 and cand.snr_clean < cand.snr_local - 0.3
    assert cand.rejected == "flare tails"
    off = BlsDetector(flare_window_h=None).search(cv, C)
    assert off.rejected is None and off.snr_clean == off.snr_local == cand.snr_local


@pytest.mark.parametrize("seed,first_gate", [(140, "local snr"), (223, "min_events")])
def test_the_box_of_the_second_pass_meets_the_flare_tail_gate_too(seed, first_gate):
    # no planet. The best box is refused for another reason, the record is searched once more
    # without its windows, and the second box sits between the tails of flares (local S/N 7.3 and
    # 8.3, clean 5.3 and 2.2). With the tail gate off the second box is reported
    cv = flare_only(seed)
    det = BlsDetector(second_pass=True)
    c = det._clean(cv, C)
    first = det._search(c, None, C, plan_mask=True)
    assert first.rejected == first_gate and len(first.mask) > 0
    second = det._search(c, None, C, masked=first.mask)
    assert second.rejected == "flare tails"
    assert second.snr_local > 7.2 and second.snr_clean < 7.0
    assert det.detect({0: cv}, C) == []
    assert len(BlsDetector(second_pass=True, flare_window_h=None).detect({0: cv}, C)) >= 2


def clipped_plateaus(plateau, seed=3, n_nights=12):
    """A planet (P 1.0 d, a 30 minute event at sample 100 of every night, depth 1.2 percent) and a
    plateau of `plateau` samples at +10 percent at sample 60 of every night, which the one-sided
    clip masks with two samples of tail on either side (`plateau` + 4, and now and then a few more
    by chance). The plateau is not a flare and the box is inside the 3 hours after it"""
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = box_profile(t, DAY, 100 * C + 900, 1800.0, 0.012)
    for k in range(n_nights):
        prof[k * NIGHT_N + 60 : k * NIGHT_N + 60 + plateau] += 0.1
    return curve(0, t, prof, 0.005, rng)


def test_a_masked_run_of_eight_opens_no_window_and_one_of_twelve_does():
    # the same event, 20 to 30 minutes after a masked run. Runs of 8 or 9 samples (a cosmic ray
    # that the clip followed a little way, a hot pixel) leave the contrast alone; runs of 12 to 14
    # leave nothing of an event inside the 3 hours that follow. A run length of 5 refuses the first,
    # one of 20 accepts the second
    short, long_ = clipped_plateaus(4), clipped_plateaus(8)
    det = BlsDetector()
    assert set(masked_runs(det, short)) <= {8, 9} and set(masked_runs(det, long_)) <= {12, 13, 14}
    cand = det.search(short, C)
    assert cand.rejected is None and cand.snr_clean == cand.snr_local > 30
    assert len(det.detect({0: short}, C)) == 12
    cand = det.search(long_, C)
    assert cand.rejected == "flare tails" and cand.snr_local > 30 and math.isnan(cand.snr_clean)
    assert det.detect({0: long_}, C) == []
    assert BlsDetector(flare_window_h=None).search(long_, C).rejected is None


def test_runs_do_not_merge_across_the_end_of_a_night():
    # 6 to 8 masked samples at the end of every night and as many at the start of the next, where
    # the two are adjacent in the record: 12 to 16 in a row. They are two short runs, and a 30
    # minute event 40 minutes into a night, which the "run" of the night before would reach, is
    # not left out
    rng = np.random.default_rng(3)
    t = night_times(12)
    prof = box_profile(t, DAY, 40 * C + 900, 1800.0, 0.012)
    for k in range(12):
        if k > 0:
            prof[k * NIGHT_N : k * NIGHT_N + 4] += 0.1
        if k < 11:
            prof[(k + 1) * NIGHT_N - 4 : (k + 1) * NIGHT_N] += 0.1
    cv = curve(0, t, prof, 0.005, rng)
    det = BlsDetector()
    assert set(masked_runs(det, cv)) <= {6, 7, 8}
    cand = det.search(cv, C)
    assert cand.rejected is None and cand.snr_clean == cand.snr_local > 30
    assert len(det.detect({0: cv}, C)) == 12


def test_the_flare_bar_still_refuses_what_the_tail_gate_passes():
    # 30 hot pixels (a flare star by the 3 percent rule) and a box of local S/N 7.11 whose events
    # no window touches: the tail gate has nothing to take away from it (clean 7.11), and the
    # flare bar of 7.2 is what refuses it, as it does with the gate off. At 7.0 it is accepted.
    # (Seed 96 star 10 of the held-out runs is this star: local 7.13, clean 7.93)
    n, flagged, depth, _, _ = FLARE_CASES["just over the share, local S/N 7.1"]
    cv = spiked(depth, n, flagged=flagged)
    det = BlsDetector()
    assert masked_share(det, cv) >= 0.03
    cand = det.search(cv, C)
    assert cand.rejected == "local snr" and 7.05 < cand.snr_local < 7.19
    assert cand.snr_clean >= 7.0
    assert det.detect({0: cv}, C) == []
    assert BlsDetector(flare_snr_threshold=7.0).search(cv, C).rejected is None
    assert BlsDetector(flare_window_h=None).search(cv, C).rejected == "local snr"


@pytest.mark.parametrize(
    "kw", [{"flare_window_h": 0}, {"flare_window_h": -1}, {"flare_min_run": 0}]
)
def test_the_flare_parameters_are_validated(kw):
    with pytest.raises(ValueError):
        BlsDetector(**kw)
    assert BlsDetector(flare_window_h=None).flare_window_h is None


# -- the constants of #39 pinned by what they decide, not by their values (review of #42) ----------
# The three bars (ratio 0.81, flare local S/N 7.2, flare share 3 percent) were guarded only by an
# assertion of the defaults, which a changed constant and its test edited together pass. These
# stars sit on both sides of each bar, so that a bar moved either way changes a verdict.


def _bare_planet(period_d, dur_h, epoch_d, depth=0.012):
    """A planet on a quiet star, 20 nights at 0.5 percent noise (the hot-planet recipe)."""
    t = night_times(20)
    prof = box_profile(t, period_d * DAY, epoch_d * DAY, dur_h * 3600.0, depth)
    return curve(0, t, prof, 0.005, np.random.default_rng(0))


@pytest.mark.parametrize("period_d,dur_h,epoch_d", [(0.6, 4.0, 0.07), (0.55, 4.5, 0.03)])
def test_a_planet_whose_ratio_is_between_seven_tenths_and_the_bar_is_accepted(
    period_d, dur_h, epoch_d
):
    # short periods and long boxes put a real planet at 0.74-0.78 (two real ones read 0.73 on the
    # stored runs): a bar of 0.70 would refuse them
    cv = _bare_planet(period_d, dur_h, epoch_d)
    cand = BlsDetector().search(cv, C)
    assert cand.rejected is None and 0.71 < cand.sine_ratio < 0.80
    assert len(detect({0: cv})) == len(cand.events) > 0


@pytest.mark.parametrize(
    "period_h,seed,amp,lo,hi",
    [(9.3, 5, 0.0020, 0.83, 0.90), (9.6, 7, 0.0016, 0.88, 0.93), (9.6, 7, 0.0024, 0.95, 1.02)],
)
def test_a_pulsator_the_ratio_alone_catches_is_refused_from_the_bar_up(period_h, seed, amp, lo, hi):
    # 0.83 to 1.0: above 0.81 and so refused, though the older tests let the box through. A bar
    # of 0.95 would accept the first two, and one above 1 the third
    cv = night_pulsator(seed, period_h, amp=amp)
    before = BlsDetector(guard_night_ratio=None, detrend_pulsation=False).search(cv, C)
    assert before.rejected is None and len(before.events) >= 10
    cand = BlsDetector(detrend_pulsation=False).search(cv, C)
    assert cand.rejected == "pulsation guard" and lo < cand.sine_ratio < hi
    assert detect({0: cv}) == []  # and subtracting the sinusoid leaves no box


def spiked(depth, n_spikes, seed=1, n_nights=12, flagged=0.0):
    """A planet (P 2.3 d, 3 h) on a quiet star with `n_spikes` hot pixels at 20 sigma, each of which
    the one-sided clip masks with its tail (about 6 samples), and `flagged` of the samples
    quality-flagged. The spikes are drawn among the clean samples. The noise is the same for every
    spike count, so that the local S/N moves with `depth` alone."""
    rng = np.random.default_rng(seed)
    t = night_times(n_nights)
    prof = box_profile(t, 2.3 * DAY, 0.14 * DAY, 3 * 3600.0, depth)
    quality = (np.random.default_rng(5).random(len(t)) < flagged).astype(np.uint8)
    hot = np.random.default_rng(99).choice(np.flatnonzero(quality == 0), n_spikes, replace=False)
    prof[hot] += 0.1
    return curve(0, t, prof, 0.005, rng, quality=quality)


# (spikes, flagged, depth): the local S/N of the box and the share of clean samples that is
# masked as positive outliers are asserted too, so that a drift of either shows up here
FLARE_CASES = {
    "just under the share, local S/N 7.1": (28, 0.0, 0.00192, (7.05, 7.19), (0.027, 0.0299)),
    "just over the share, local S/N 7.1": (30, 0.0, 0.00198, (7.05, 7.19), (0.0301, 0.034)),
    "just over the share, local S/N 7.36": (30, 0.0, 0.00206, (7.25, 7.45), (0.0301, 0.034)),
}


@pytest.mark.parametrize(
    "case,accepted",
    [
        ("just under the share, local S/N 7.1", True),
        ("just over the share, local S/N 7.1", False),
        ("just over the share, local S/N 7.36", True),
    ],
)
def test_the_flare_bar_and_the_flare_share_decide_stars_just_either_side_of_them(case, accepted):
    # a share under 3 percent is judged at 7.0 and one over at 7.2: the same box at a local S/N
    # of 7.1 is accepted on the first and refused on the second, and at 7.36 accepted on both.
    # A bar of 7.0 accepts the second, 7.5 refuses the third; a share of 1 percent refuses the
    # first, one of 10 percent accepts the second. The tail gate is off here (its windows would
    # change the verdict of a star with 28 spikes): it is tested on its own below
    n, flagged, depth, local, share = FLARE_CASES[case]
    cv = spiked(depth, n, flagged=flagged)
    det = BlsDetector(flare_window_h=None)
    assert share[0] < masked_share(det, cv) < share[1]
    ordinary = BlsDetector(flare_fraction=1.0, flare_window_h=None).search(
        cv, C
    )  # no star is a flare star
    assert ordinary.rejected is None and local[0] < ordinary.snr_local < local[1]
    cand = det.search(cv, C)
    assert (cand.rejected is None) is accepted
    assert cand.rejected in (None, "local snr")
    assert (len(det.detect({0: cv}, C)) > 0) is accepted


def test_the_flare_share_is_over_the_clean_samples_not_over_all_of_them():
    # a quarter of the samples are quality-flagged and dropped before anything else. The spikes
    # mask 3.35 percent of what is left (a flare star) and 2.5 percent of all the samples. Judged
    # against all of them the star would be held to 7.0, not to 7.2; and the flagged samples are
    # not masked samples either: with no spike at all the star is not a flare star
    det = BlsDetector(flare_window_h=None)
    cv = spiked(0.0022, 26, flagged=0.25)
    share = masked_share(det, cv)
    assert 0.031 < share < 0.036
    assert share * len(det._clean(cv, C).t) / len(cv.t) < 0.0299
    ordinary = BlsDetector(flare_fraction=1.0, flare_window_h=None).search(cv, C)
    assert ordinary.rejected is None and 7.05 < ordinary.snr_local < 7.19
    assert det.search(cv, C).rejected == "local snr"
    quiet = spiked(0.00222, 0, flagged=0.25)
    assert masked_share(det, quiet) == 0.0
    cand = det.search(quiet, C)
    assert cand.rejected is None and 7.05 < cand.snr_local < 7.19


# -- the band of the ratio's sinusoid, and the level it reports (review of #42) ------------------


def _sinusoid_and_its_box(f_per_d):
    """A noise-free-ish sinusoid at `f_per_d` cycles/day over 20 nights, a box on its lowest half
    cycle, and the nights; the ratio of such a pair is about 1 / 0.81 = 1.23."""
    t = night_times(20)
    night = (t // DAY).astype(np.int64)
    jitter = np.random.default_rng(0).normal(0.0, 0.0003, len(t))  # the scatter must be defined
    sine = 1.0 + 0.01 * np.sin(2 * np.pi * f_per_d * t / DAY + 0.3) + jitter
    period = DAY / f_per_d
    at_minimum = (-np.pi / 2 - 0.3) / (2 * np.pi * f_per_d) * DAY
    return t, night, sine, Box(period, at_minimum % period, period / 2, 0.01, 0, len(t), 0.0)


def _expected_ratio(t, night, sine, box, f_per_d):
    """The chi-square gain of the sinusoid at its own frequency over the box's, by hand."""
    k = np.unique(night, return_inverse=True)[1]
    x = bls._demean_by_night(sine - 1.0, k)
    gain = bls._sine_fit((t - t[0]) / DAY, x, k, f_per_d)[2]
    depth_k, weight_k = night_contrasts(t, sine, night, box)
    return gain / (float(depth_k @ weight_k) ** 2 / float(weight_k.sum()))


@pytest.mark.parametrize("f_per_d", [3.0, 5.5, 12.0, 29.0])
def test_the_ratio_sees_a_sinusoid_anywhere_in_the_band_of_one_to_thirty_cycles_a_day(f_per_d):
    # a band that stopped at 5 sees 12 and 29 as noise (0.03), one that started at 3 sees nothing
    # below it as itself
    t, night, sine, box = _sinusoid_and_its_box(f_per_d)
    expected = _expected_ratio(t, night, sine, box, f_per_d)
    assert 1.1 < expected < 1.4
    assert BlsDetector()._night_sine_ratio(t, sine, night, box, C) == pytest.approx(
        expected, rel=0.02
    )


@pytest.mark.parametrize("f_per_d", [35.0, 60.0])
def test_the_ratio_does_not_see_a_sinusoid_above_thirty_cycles_a_day(f_per_d):
    t, night, sine, box = _sinusoid_and_its_box(f_per_d)
    assert _expected_ratio(t, night, sine, box, f_per_d) > 1.1
    assert BlsDetector()._night_sine_ratio(t, sine, night, box, C) < 0.1


def _fit_of(det, f_per_d):
    """`_strongest_sinusoid` on a sinusoid at `f_per_d` (20 nights, 0.01 amplitude, 0.03 percent
    jitter), and the gain of the sinusoid at its own frequency."""
    t, night, sine, _ = _sinusoid_and_its_box(f_per_d)
    td = (t - t[0]) / DAY
    k = np.unique(night, return_inverse=True)[1]
    x = bls._demean_by_night(sine - 1.0, k)
    fit = det._strongest_sinusoid(t, td, x, k, 0.0003, C, float(td[-1]))
    return fit, bls._sine_fit(td, x, k, f_per_d)[2], float(td[-1])


@pytest.mark.parametrize("f_per_d", [0.7, 0.9])
def test_a_sinusoid_slower_than_the_band_is_left_to_the_guard_and_its_alias_is_fitted(f_per_d):
    # the nights are a day apart, so the best sinusoid the band holds is the alias one cycle/day
    # above (it keeps 90 percent of the gain) and the slow one itself is not fitted; a band that
    # started at 0.3 would fit it, in the ratio and in the guard's own fit
    fit, true_gain, _ = _fit_of(BlsDetector(), f_per_d)
    assert fit is not None and fit[1] == pytest.approx(f_per_d + 1.0, abs=0.02)
    assert fit[0] < 0.95 * true_gain
    t, night, sine, box = _sinusoid_and_its_box(f_per_d)
    expected = _expected_ratio(t, night, sine, box, f_per_d)
    ratio = BlsDetector()._night_sine_ratio(t, sine, night, box, C)
    assert 0.85 * expected < ratio < 0.95 * expected


def test_the_strongest_sinusoid_reports_the_planted_one_and_the_guards_false_alarm_level():
    # the level is ln(M / guard_fap) with M = (30 - 1) cycles/day times the record in days
    det = BlsDetector()
    fit, true_gain, span_d = _fit_of(det, 12.0)
    gain, f, sin_a, cos_a, level = fit
    assert f == pytest.approx(12.0, abs=1e-3)
    assert math.hypot(sin_a, cos_a) == pytest.approx(0.01, rel=0.02)
    assert gain == pytest.approx(true_gain, rel=1e-3)
    assert level == pytest.approx(math.log(29.0 * span_d / 1e-3), rel=1e-12)
    assert _fit_of(BlsDetector(guard_fap=1e-6), 12.0)[0][4] == pytest.approx(
        math.log(29.0 * span_d / 1e-6), rel=1e-12
    )
    assert _fit_of(BlsDetector(guard_f_max_per_d=20.0), 12.0)[0][4] == pytest.approx(
        math.log(19.0 * span_d / 1e-3), rel=1e-12
    )


# -- a host only the ratio refuses (review of #42) -------------------------------------------------


def ratio_host():
    """A 6.03 h pulsator (0.3 percent, 2 percent noise) with a 3 h, 1 percent transit at P 3.1 d.

    Its best box is the fold of the pulsation (P 0.503 d, 20 events), which the box-versus-sine and
    the residual tests let through and only the ratio refuses; the planet is behind it.
    """
    rng = np.random.default_rng(1)
    t = night_times(20)
    prof = 1.0 + 0.003 * np.sin(2 * np.pi * t / (6.03 * 3600.0) + 0.7)
    prof = prof * box_profile(t, 3.1 * DAY, 0.05 * DAY, 3 * 3600.0, 0.01)
    return t, curve(0, t, prof, 0.02, rng)


def test_a_ratio_rejected_host_is_still_detrended_and_its_planet_found():
    # the ratio rejects as "pulsation guard" and nothing else: only that label reaches the
    # subtract-and-search of #38
    t, cv = ratio_host()
    older = BlsDetector(guard_night_ratio=None, detrend_pulsation=False).search(cv, C)
    assert older.rejected is None and older.box.period_s < 0.6 * DAY and len(older.events) >= 15
    first = BlsDetector(detrend_pulsation=False).search(cv, C)
    assert first.rejected == "pulsation guard" and first.sine_ratio >= 0.81
    assert first.box.period_s == older.box.period_s
    cand = BlsDetector().search(cv, C)
    assert cand.rejected is None and cand.trend is not None
    assert cand.box.period_s == pytest.approx(3.1 * DAY, rel=0.01)
    truth = planted_windows(t, 3.1, 0.05, 3.0)
    assert _events_found(detect({0: cv}), truth) == len(truth) == 4
    assert len(detect({0: cv})) == 4


class _FirstSearchReal(BlsDetector):
    """The real first search; the re-search returns an accepted candidate at `factor` times the
    first period."""

    def __init__(self, factor, **kw):
        super().__init__(second_pass=True, **kw)
        self.factor = factor
        self.calls = []
        self.first = None

    def _search(self, c, common_mode, cadence_s, masked=(), plan_mask=False):
        self.calls.append((tuple(masked), plan_mask))
        if len(self.calls) == 1:
            self.first = super()._search(c, common_mode, cadence_s, masked, plan_mask)
            return self.first
        return _cand(self.first.box.period_s / DAY * self.factor, None)


def test_a_ratio_rejection_carries_a_mask_and_the_re_search_refuses_its_aliases():
    # the pure 6.03 h pulsator of the ratio test: the ratio is what refuses it, as "pulsation
    # guard", so the second pass plans its mask, searches again without it and refuses a box at the
    # first period, half of it or twice it
    cv = night_pulsator(1, 6.03)
    for factor in (1.0, 0.5, 2.0):
        det = _FirstSearchReal(factor, detrend_pulsation=False)
        assert det.detect({0: cv}, C) == []
        assert det.first.rejected == "pulsation guard" and det.first.sine_ratio >= 0.81
        assert len(det.first.mask) >= 1
        assert len(det.calls) == 2 and det.calls[0] == ((), True)
        assert det.calls[1] == (det.first.mask, False)
    det = _FirstSearchReal(1.1, detrend_pulsation=False)  # a period that is not an alias stands
    assert len(det.detect({0: cv}, C)) == 2
    assert det.first.rejected == "pulsation guard" and len(det.calls) == 2
