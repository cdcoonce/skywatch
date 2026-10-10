import numpy as np
import pytest

from skywatch.detectors import REGISTRY
from skywatch.detectors.rolling_z import RollingZDetector, rolling_z_scores
from skywatch.types import LightCurve

C = 60


def curve(flux, star_id=1, t=None):
    n = len(flux)
    t = np.arange(n, dtype=np.int64) * C if t is None else t
    return LightCurve(star_id, t, np.asarray(flux, float), np.ones(n), np.zeros(n, dtype=np.int64))


def noisy(n=200, seed=0):
    return 100 + np.random.default_rng(seed).normal(0, 1, n)


def test_registered():
    assert REGISTRY["rolling_z"] is RollingZDetector


def test_single_spike():
    f = noisy()
    f[100] += 10
    out = RollingZDetector().detect({1: curve(f)}, C)
    assert len(out) == 1
    d = out[0]
    assert (d.detector, d.star_id, d.t_start, d.t_end) == ("rolling_z", 1, 6000, 6060)
    assert d.det_id == "rolling_z-s1-0" and d.kind_guess is None
    assert d.score >= 4.0


def test_ramp_below_threshold():
    f = 100 + np.arange(200) * 0.01
    assert RollingZDetector().detect({1: curve(f)}, C) == []


def test_plateau_merged():
    f = noisy()
    f[100:103] += 40
    out = RollingZDetector(z_threshold=3.0, min_run=1).detect({1: curve(f)}, C)
    assert len(out) == 1
    assert out[0].t_start == 6000 and out[0].t_end == 6000 + 3 * C


def test_min_run_drops_short():
    f = noisy()
    f[100] += 10
    assert RollingZDetector(min_run=2).detect({1: curve(f)}, C) == []


def test_warmup_and_short_curve():
    # warmup >= window restores the pre-growing-window rule: a curve with n <= window is blind
    f = noisy(20)
    f[5] += 50
    assert RollingZDetector(warmup=30).detect({1: curve(f)}, C) == []


def test_gaps_treated_by_index():
    f = noisy()
    f[100] += 10
    t = np.arange(200, dtype=np.int64) * C
    t[100:] += 5000
    out = RollingZDetector().detect({1: curve(f, t=t)}, C)
    assert len(out) == 1 and out[0].t_start == int(t[100]) and out[0].t_end == int(t[100]) + C


def test_multi_star_and_determinism():
    f1, f2 = noisy(seed=1), noisy(seed=2)
    f1[50] += 12
    f2[150] += 12
    curves = {1: curve(f1, 1), 2: curve(f2, 2)}
    a = RollingZDetector().detect(curves, C)
    b = RollingZDetector().detect(curves, C)
    assert a == b
    assert [(d.star_id, d.det_id) for d in a] == [(1, "rolling_z-s1-0"), (2, "rolling_z-s2-0")]


def test_flux_err_floors_zero_std():
    f = np.full(200, 1000.0)
    f[100:] += 10
    f[101:] -= 10
    out = RollingZDetector(window=30, z_threshold=4.0).detect({1: curve(f)}, C)
    assert len(out) == 1
    assert out[0].score == pytest.approx(10.0)


def test_negative_dip_flagged():
    f = noisy()
    f[100] -= 10
    out = RollingZDetector().detect({1: curve(f)}, C)
    assert len(out) == 1 and out[0].t_start == int(curve(f).t[100])


def test_trailing_window_excludes_current_sample():
    f = np.array([1.0, 2.0, 3.0, 10.0, 4.0, 2.0])
    err = np.full(6, 0.1)
    z = rolling_z_scores(f, err, 3)
    assert z[:3].tolist() == [0.0, 0.0, 0.0]
    # i=3: window [1,2,3], mean 2, population std sqrt(2/3)
    assert z[3] == pytest.approx((10 - 2) / np.sqrt(2 / 3))
    # i=4: window [2,3,10], mean 5, std = sqrt(((3)^2+(2)^2+(5)^2)/3)
    assert z[4] == pytest.approx((4 - 5) / np.sqrt(38 / 3))
    # i=5: window [3,10,4], mean 17/3
    m = 17 / 3
    sd = np.sqrt(((3 - m) ** 2 + (10 - m) ** 2 + (4 - m) ** 2) / 3)
    assert z[5] == pytest.approx((2 - m) / sd)


# --- per-night windows (issue 10) ---

NIGHT = 480


def two_night_t():
    return np.concatenate(
        [np.arange(NIGHT, dtype=np.int64) * C, 86400 + np.arange(NIGHT, dtype=np.int64) * C]
    )


def held_step_flux(seed=0):
    """Night 1 drops to 0.7x for its second half and stays there; night 2 is flat at 1.0."""
    rng = np.random.default_rng(seed)
    f = 100 + rng.normal(0, 1, 2 * NIGHT)
    f[NIGHT // 2 : NIGHT] *= 0.7
    return f


def test_night_boundary_is_not_a_step():
    t = two_night_t()
    out = RollingZDetector().detect({1: curve(held_step_flux(), t=t)}, C)
    night2 = [d for d in out if d.t_start >= 86400]
    assert night2 == []


def test_night_2_spike_still_detected_after_held_step():
    t = two_night_t()
    f = held_step_flux()
    f[NIGHT + 240] += 12
    out = RollingZDetector().detect({1: curve(f, t=t)}, C)
    night2 = [d for d in out if d.t_start >= 86400]
    assert [(d.t_start, d.t_end) for d in night2] == [
        (int(t[NIGHT + 240]), int(t[NIGHT + 240]) + C)
    ]


def test_warmup_prefix_of_each_night_is_unflaggable():
    t = two_night_t()
    f = held_step_flux()
    f[NIGHT + 2] += 50  # inside night 2's warmup prefix: treated like the start of a curve
    out = RollingZDetector().detect({1: curve(f, t=t)}, C)
    assert [d for d in out if d.t_start >= 86400] == []


def test_det_ids_continue_across_nights():
    # No-merge is true by construction for rolling_z: the first sample of every night has
    # z = 0 (warm-up is at least one sample), so a run cannot span a night boundary. This
    # pins only that det_ids keep counting across nights.
    t = two_night_t()
    f = 100 + np.random.default_rng(3).normal(0, 1, 2 * NIGHT)
    f[NIGHT - 1] += 30
    f[NIGHT + 40] += 30
    out = RollingZDetector(window=30).detect({1: curve(f, t=t)}, C)
    assert [(d.t_start, d.det_id) for d in out] == [
        (int(t[NIGHT - 1]), "rolling_z-s1-0"),
        (int(t[NIGHT + 40]), "rolling_z-s1-1"),
    ]


def _legacy_detect(det, c, cadence_s):
    """The pre-per-night whole-curve loop, kept as a regression oracle."""
    absz = np.abs(rolling_z_scores(c.flux, c.flux_err, det.window))
    flagged = absz >= det.z_threshold
    out, k, i, n = [], 0, 0, len(c)
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flagged[j + 1]:
            j += 1
        if j - i + 1 >= det.min_run:
            out.append(
                (
                    f"rolling_z-s{c.star_id}-{k}",
                    int(c.t[i]),
                    int(c.t[j]) + cadence_s,
                    float(absz[i : j + 1].max()),
                )
            )
            k += 1
        i = j + 1
    return out


@pytest.mark.parametrize("seed", range(5))
def test_single_night_identical_to_whole_curve_behaviour(seed):
    f = noisy(seed=seed)
    rng = np.random.default_rng(seed + 100)
    f[rng.integers(40, 190, 6)] += 15
    f[120:123] += 20
    c = curve(f)
    det = RollingZDetector(min_run=1)
    got = [(d.det_id, d.t_start, d.t_end, d.score) for d in det.detect({1: c}, C)]
    assert got == _legacy_detect(det, c, C)
    assert got


def test_night_gap_cadences_param():
    assert RollingZDetector().night_gap_cadences == 100
    # tests above pass t with a 16 h hole; a huge threshold restores window-across-gap behaviour
    t = two_night_t()
    out = RollingZDetector(night_gap_cadences=10_000).detect({1: curve(held_step_flux(), t=t)}, C)
    assert any(d.t_start == 86400 for d in out)


# --- growing window at the start of each night (issue 16) ---


def brute_old_z(flux, err, window):
    """The pre-issue-16 rule, written out sample by sample: the first `window` samples are 0."""
    flux = np.asarray(flux, float)
    z = np.zeros(len(flux))
    for i in range(window, len(flux)):
        w = flux[i - window : i]
        sd = max(w.std(), err[i])
        z[i] = (flux[i] - w.mean()) / sd
    return z


def night_two_spike(sample, amp=30.0, seed=0):
    t = two_night_t()
    f = 100 + np.random.default_rng(seed).normal(0, 1, 2 * NIGHT)
    f[NIGHT + sample] += amp
    return t, f


@pytest.mark.parametrize("sample", [5, 6, 12, 29])
def test_cosmic_ray_just_after_warmup_is_flagged_in_night_2(sample):
    t, f = night_two_spike(sample)
    out = RollingZDetector().detect({1: curve(f, t=t)}, C)
    night2 = [d for d in out if d.t_start >= 86400]
    assert [(d.t_start, d.t_end) for d in night2] == [
        (int(t[NIGHT + sample]), int(t[NIGHT + sample]) + C)
    ]


@pytest.mark.parametrize("sample", [0, 1, 2, 3, 4])
def test_cosmic_ray_inside_warmup_is_not_flagged_in_night_2(sample):
    t, f = night_two_spike(sample, amp=1000.0)
    out = RollingZDetector().detect({1: curve(f, t=t)}, C)
    assert [d for d in out if d.t_start >= 86400] == []


def test_warmup_parameter_moves_the_blind_edge():
    t, f = night_two_spike(3)
    assert [
        d for d in RollingZDetector(warmup=5).detect({1: curve(f, t=t)}, C) if d.t_start >= 86400
    ] == []
    out = RollingZDetector(warmup=3).detect({1: curve(f, t=t)}, C)
    assert [d.t_start for d in out if d.t_start >= 86400] == [int(t[NIGHT + 3])]
    f2 = f.copy()
    f2[NIGHT + 3] -= 30  # undo; put the spike at sample 2 instead
    f2[NIGHT + 2] += 30
    assert [
        d for d in RollingZDetector(warmup=3).detect({1: curve(f2, t=t)}, C) if d.t_start >= 86400
    ] == []


def test_default_warmup_is_five():
    assert RollingZDetector().warmup == 5


def test_negative_warmup_rejected():
    with pytest.raises(ValueError, match="warmup"):
        RollingZDetector(warmup=-1)
    with pytest.raises(ValueError, match="warmup"):
        rolling_z_scores(np.ones(10), np.ones(10), 5, warmup=-1)


@pytest.mark.parametrize("warmup", [0, 30, 31, 500])
def test_warmup_zero_or_at_least_window_reproduces_old_scores(warmup):
    f = noisy(120, seed=4)
    f[10] += 40
    f[60] += 40
    err = np.full(120, 1.0)
    got = rolling_z_scores(f, err, 30, warmup=warmup)
    assert got == pytest.approx(brute_old_z(f, err, 30), abs=1e-12)
    assert not got[:30].any()


@pytest.mark.parametrize("warmup", [0, 30, 99])
@pytest.mark.parametrize("seed", range(3))
def test_warmup_disabled_detector_matches_old_per_night_behaviour(warmup, seed):
    t = two_night_t()
    rng = np.random.default_rng(seed)
    f = 100 + rng.normal(0, 1, 2 * NIGHT)
    f[rng.integers(5, 29, 3)] += 40  # first-window spikes: blind when warmup is off
    f[NIGHT + rng.integers(5, 29, 3)] += 40
    f[rng.integers(40, NIGHT - 1, 5)] += 15
    f[NIGHT + rng.integers(40, NIGHT - 1, 5)] += 15
    det = RollingZDetector(warmup=warmup)
    got = [(d.t_start, d.t_end, d.score) for d in det.detect({1: curve(f, t=t)}, C)]
    want = []
    for lo, hi in ((0, NIGHT), (NIGHT, 2 * NIGHT)):
        z = np.abs(brute_old_z(f[lo:hi], np.ones(hi - lo), 30))
        want += [
            (int(t[lo + i]), int(t[lo + i]) + C, float(z[i])) for i in np.flatnonzero(z >= 4.0)
        ]
    assert got == pytest.approx(want)
    assert got


def test_growing_window_hand_computed_and_threshold_scaled():
    f = np.array([1.0, 2.0, 3.0, 10.0, 9.0, 2.0, 5.0, 1.0])
    err = np.full(8, 0.1)
    z = rolling_z_scores(f, err, 4, warmup=2)
    assert z[:2].tolist() == [0.0, 0.0]
    # i=2: window [1,2] (mean 1.5, std 0.5), scaled by sqrt(2/4)
    assert z[2] == pytest.approx((3 - 1.5) / 0.5 * np.sqrt(2 / 4))
    # i=3: window [1,2,3] (mean 2, std sqrt(2/3)), scaled by sqrt(3/4)
    assert z[3] == pytest.approx((10 - 2) / np.sqrt(2 / 3) * np.sqrt(3 / 4))
    # i=4: first full window [1,2,3,10], no scaling
    m, sd = 4.0, np.sqrt(((3) ** 2 + (2) ** 2 + (1) ** 2 + (6) ** 2) / 4)
    assert z[4] == pytest.approx((9 - m) / sd)


def test_growing_window_nan_gives_zero():
    f = np.array([1.0, np.nan, 3.0, 10.0, 4.0, 2.0, 5.0, 1.0])
    z = rolling_z_scores(f, np.full(8, 0.1), 5, warmup=2)
    assert z[2] == 0.0 and z[3] == 0.0 and z[4] == 0.0
    assert z[5] == 0.0  # window [nan,3,10,4,2]... still poisoned


@pytest.mark.parametrize("seed", range(4))
def test_night_boundary_is_not_a_step_with_growing_window(seed):
    # The key risk: a short window right after the boundary must not bring the night-start
    # alarm back. The window never crosses a night, so a held step across the gap stays silent.
    t = two_night_t()
    for factor in (0.7, 1.4):
        f = 100 + np.random.default_rng(seed).normal(0, 1, 2 * NIGHT)
        f[NIGHT // 2 : NIGHT] *= factor
        out = RollingZDetector(warmup=2).detect({1: curve(f, t=t)}, C)
        assert [d for d in out if d.t_start >= 86400 and d.t_start < 86400 + 40 * C] == []


@pytest.mark.parametrize("warmup", [1, 5, 12])
def test_growing_window_uses_every_sample_seen_so_far(warmup):
    # window_i = i (all samples so far, not a fixed small tail), scaled by sqrt(i / window)
    w = 30
    f = noisy(80, seed=9)
    err = np.full(80, 0.5)
    z = rolling_z_scores(f, err, w, warmup=warmup)
    for i in range(warmup, w):
        head = f[:i]
        want = (f[i] - head.mean()) / max(head.std(), 0.5) * np.sqrt(i / w)
        assert z[i] == pytest.approx(want)
    assert not z[:warmup].any()


def test_short_night_under_default_warmup_is_detected_without_error():
    # warmup <= n < window with the default warmup: the growing loop must stop at n
    f = noisy(20, seed=3)
    f[5] += 30
    out = RollingZDetector().detect({1: curve(f)}, C)
    assert [(d.t_start, d.t_end) for d in out] == [(5 * C, 6 * C)]
    assert out[0].score > 4.0


def test_run_starting_in_growing_region_records_the_scaled_score():
    f = noisy(80, seed=5)
    f[10] += 30
    z = rolling_z_scores(f, np.ones(80), 30, warmup=5)
    head = f[:10]
    by_hand = abs(f[10] - head.mean()) / max(head.std(), 1.0) * np.sqrt(10 / 30)
    assert abs(z[10]) == pytest.approx(by_hand, abs=1e-12)
    out = RollingZDetector().detect({1: curve(f)}, C)
    assert [(d.t_start, d.t_end) for d in out] == [(10 * C, 11 * C)]
    assert out[0].score == pytest.approx(by_hand, abs=1e-12)


def test_threshold_equal_to_growing_region_score_flags():
    f = noisy(80, seed=5)
    f[10] += 30
    z_exact = float(abs(rolling_z_scores(f, np.ones(80), 30, warmup=5)[10]))
    out = RollingZDetector(z_threshold=z_exact).detect({1: curve(f)}, C)
    assert [d.t_start for d in out] == [10 * C]


def test_growing_window_nan_current_sample_gives_zero():
    f = np.array([1.0, 2.0, 1.5, 2.5, np.nan, 2.0, 1.0, 2.0])
    z = rolling_z_scores(f, np.full(8, 0.1), 5, warmup=2)
    assert z[4] == 0.0


def test_steady_state_nan_current_sample_gives_zero_and_never_flags():
    f = noisy(60, seed=1)
    f[40] = np.nan
    z = rolling_z_scores(f, np.ones(60), 30, warmup=5)
    assert z[40] == 0.0
    assert not np.isnan(z[:40]).any()
    assert RollingZDetector().detect({1: curve(f)}, C) == []


def test_nan_flux_err_gives_zero_in_both_regions():
    f = np.array([1.0, 2.0, 1.5, 2.5, 9.0, 2.0, 1.0, 2.0, 30.0])
    err = np.full(9, 0.1)
    err[4] = np.nan  # growing region (window 6, warmup 2)
    err[8] = np.nan  # steady state
    z = rolling_z_scores(f, err, 6, warmup=2)
    assert z[4] == 0.0 and z[8] == 0.0
