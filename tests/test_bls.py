import math
import time

import numpy as np
import pytest

from skywatch.detectors import REGISTRY, Detector, load_all
from skywatch.detectors.bls import (
    BlsDetector,
    Box,
    frequency_grid,
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
    assert len(detect({0: cv}, guard_fap=None)) > 0  # without the guard it is a "transit"


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
    assert elapsed < 30.0, f"20 stars x 10 nights took {elapsed:.1f} s"


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
