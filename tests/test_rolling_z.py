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
    f = noisy(20)
    f[5] += 50
    assert RollingZDetector().detect({1: curve(f)}, C) == []


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


def test_first_window_of_each_night_is_unflaggable():
    t = two_night_t()
    f = held_step_flux()
    f[NIGHT + 5] += 50  # inside night 2's warm-up: treated like the start of a curve
    out = RollingZDetector().detect({1: curve(f, t=t)}, C)
    assert [d for d in out if d.t_start >= 86400] == []


def test_run_does_not_merge_across_nights():
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
