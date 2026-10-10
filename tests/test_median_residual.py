import numpy as np
import pytest

from skywatch.detectors import REGISTRY, load_all
from skywatch.detectors.median_residual import (
    MedianResidualDetector,
    _running_median,
    median_residual_scores,
)
from skywatch.types import LightCurve

C = 60


def curve(flux, star_id=1, err=1.0):
    n = len(flux)
    return LightCurve(
        star_id,
        np.arange(n, dtype=np.int64) * C,
        np.asarray(flux, float),
        np.full(n, err),
        np.zeros(n, dtype=np.int64),
    )


def noisy(n=300, seed=0):
    return 100 + np.random.default_rng(seed).normal(0, 1, n)


def run(f, **kw):
    return MedianResidualDetector(**kw).detect({1: curve(f)}, C)


def test_window_must_be_odd():
    with pytest.raises(ValueError):
        MedianResidualDetector(window=60)


def test_registered_after_load_all():
    load_all()
    assert REGISTRY["median_residual"] is MedianResidualDetector


def test_single_spike_is_cosmic_ray():
    f = noisy()
    f[150] += 10
    (d,) = run(f)
    assert (d.detector, d.star_id, d.t_start, d.t_end) == ("median_residual", 1, 150 * C, 151 * C)
    assert d.det_id == "median_residual-s1-0"
    assert d.kind_guess == "cosmic_ray"
    assert d.score == pytest.approx(10, rel=0.35)


def test_flare_shape_is_flare():
    f = noisy()
    f[150:158] += 12 * np.exp(-np.arange(8) / 3.0)
    (d,) = run(f)
    assert d.t_start == 150 * C
    assert d.kind_guess == "flare"


def test_negative_dip_has_no_guess():
    f = noisy()
    f[150:155] -= 10
    (d,) = run(f)
    assert d.kind_guess is None
    assert (d.t_start, d.t_end) == (150 * C, 155 * C)


def test_slow_ramp_not_flagged():
    f = 100 + np.arange(300) * 0.01
    assert run(f) == []


def test_min_run():
    f = noisy()
    f[150:152] += 10
    assert run(f, min_run=3) == []
    assert len(run(f, min_run=2)) == 1


def test_helper_hand_computed():
    flux = np.array([1.0, 2.0, 10.0, 3.0, 4.0])
    z = median_residual_scores(flux, np.full(5, 0.1), 3)
    k = 1.4826
    assert z[0] == pytest.approx(-0.5 / (k * 0.25))  # edge: window [1, 2]
    assert z[1] == pytest.approx(0.0)
    assert z[2] == pytest.approx(7.0 / (k * 1.0))
    assert z[4] == pytest.approx(0.5 / (k * 0.75))  # edge: window [3, 4]


def test_flux_err_floor():
    f = np.full(300, 1000.0)
    f[150:] += 10
    f[151:] -= 10
    (d,) = run(f)
    assert d.score == pytest.approx(10.0)


def test_zero_sigma_gives_zero():
    z = median_residual_scores(np.full(20, 5.0), np.zeros(20), 5)
    assert np.all(z == 0)


def test_determinism_and_multi_star():
    f1, f2 = noisy(seed=1), noisy(seed=2)
    f1[50] += 12
    f2[200] += 12
    curves = {2: curve(f2, 2), 1: curve(f1, 1)}
    a = MedianResidualDetector().detect(curves, C)
    assert a == MedianResidualDetector().detect(curves, C)
    assert [d.det_id for d in a] == ["median_residual-s1-0", "median_residual-s2-0"]


def test_event_shorter_than_half_window_caught_at_onset():
    f = noisy(400)
    f[150:170] += 8
    out = run(f)
    assert out and out[0].t_start == 150 * C


def test_event_longer_than_window_is_absorbed():
    # Documented limitation: a plateau longer than ~window/2 becomes the baseline,
    # so its onset is not flagged (the spec's "caught at onset" expectation is false
    # for a median baseline; the median flips to the plateau level before the onset).
    f = noisy(400)
    f[150:250] += 5
    assert all(not (d.t_start <= 150 * C < d.t_end) for d in run(f))


@pytest.mark.parametrize("n", [1, 3, 9, 10, 11, 12, 40])
def test_running_median_matches_brute_force(n):
    window, half = 11, 5
    x = np.random.default_rng(n).normal(0, 1, n)
    expected = [np.median(x[max(0, i - half) : i + half + 1]) for i in range(n)]
    np.testing.assert_allclose(_running_median(x, window), expected)


def test_score_is_peak_abs_z():
    f = noisy()
    f[149:152] += np.array([6.0, 12.0, 6.0])
    (d,) = run(f)
    z = median_residual_scores(f, np.full(len(f), 1.0), 61)
    assert d.score == pytest.approx(np.max(np.abs(z)))


def test_two_spikes_get_sequential_ids():
    f = noisy()
    f[80] += 12
    f[220] += 12
    out = run(f)
    assert [d.det_id for d in out] == ["median_residual-s1-0", "median_residual-s1-1"]


def test_mixed_sign_run_guess():
    # Rule: positive guess only if run.max() >= threshold and run.mean() > 0.
    f = np.full(300, 100.0)
    f[100:103] += [8, -9, 8]
    f[200:203] += [-9, 4.5, -9]
    pos, neg = run(f + np.random.default_rng(5).normal(0, 0.01, 300), threshold=2.0)[:2]
    assert pos.t_start == 100 * C and pos.kind_guess == "flare"
    assert neg.t_start == 200 * C and neg.kind_guess is None


def test_nan_blinds_neighbourhood():
    f = noisy(400)
    f[100] = np.nan
    f[101] += 15  # spike adjacent to NaN: inside blinded windows
    f[300] += 15  # far from NaN
    out = run(f, window=21)
    assert all(not (d.t_start <= 101 * C < d.t_end) for d in out)
    assert all(abs(d.t_start // C - 100) > 10 for d in out)
    assert any(d.t_start <= 300 * C < d.t_end for d in out)


@pytest.mark.parametrize(
    "kw", [{"window": 1}, {"window": -3}, {"threshold": 0}, {"threshold": -1.0}, {"min_run": 0}]
)
def test_bad_args_rejected(kw):
    with pytest.raises(ValueError):
        MedianResidualDetector(**kw)


# --- per-night windows (issue 10) ---

NIGHT = 480


def two_night_curve(f, star_id=1):
    n = len(f)
    t = np.concatenate(
        [np.arange(n // 2, dtype=np.int64) * C, 86400 + np.arange(n - n // 2, dtype=np.int64) * C]
    )
    return LightCurve(
        star_id, t, np.asarray(f, float), np.full(n, 1.0), np.zeros(n, dtype=np.int64)
    )


def held_step_flux(seed=0):
    """Night 1 drops to 0.7x for its second half and stays there; night 2 is flat at 1.0."""
    f = 100 + np.random.default_rng(seed).normal(0, 1, 2 * NIGHT)
    f[NIGHT // 2 : NIGHT] *= 0.7
    return f


@pytest.mark.parametrize("seed", [13, 54, 0, 1])  # 13 and 54 trip the whole-curve window
def test_night_boundary_is_not_a_step(seed):
    c = two_night_curve(held_step_flux(seed))
    out = MedianResidualDetector().detect({1: c}, C)
    near_boundary = [d for d in out if int(c.t[NIGHT - 3]) <= d.t_start <= int(c.t[NIGHT + 2])]
    assert near_boundary == []


def test_night_2_spike_still_detected_after_held_step():
    f = held_step_flux()
    f[NIGHT + 240] += 12
    c = two_night_curve(f)
    out = MedianResidualDetector().detect({1: c}, C)
    near = [d for d in out if int(c.t[NIGHT - 3]) <= d.t_start <= int(c.t[NIGHT + 240])]
    assert [(d.t_start, d.kind_guess) for d in near] == [(int(c.t[NIGHT + 240]), "cosmic_ray")]


def test_run_does_not_merge_across_nights():
    f = noisy(2 * NIGHT, seed=3)
    f[NIGHT - 1] += 40
    f[NIGHT] += 40
    c = two_night_curve(f)
    out = MedianResidualDetector().detect({1: c}, C)
    assert [(d.t_start, d.t_end, d.det_id) for d in out] == [
        (int(c.t[NIGHT - 1]), int(c.t[NIGHT - 1]) + C, "median_residual-s1-0"),
        (int(c.t[NIGHT]), int(c.t[NIGHT]) + C, "median_residual-s1-1"),
    ]


def _legacy_detect(det, c, cadence_s):
    """The pre-per-night whole-curve loop, kept as a regression oracle."""
    z = median_residual_scores(c.flux, c.flux_err, det.window)
    flagged = np.abs(z) >= det.threshold
    out, k, i, n = [], 0, 0, len(c)
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flagged[j + 1]:
            j += 1
        length = j - i + 1
        if length >= det.min_run:
            run = z[i : j + 1]
            guess = None
            if run.max() >= det.threshold and run.mean() > 0:
                guess = "cosmic_ray" if length == 1 else "flare"
            out.append(
                (
                    f"median_residual-s{c.star_id}-{k}",
                    int(c.t[i]),
                    int(c.t[j]) + cadence_s,
                    float(np.abs(run).max()),
                    guess,
                )
            )
            k += 1
        i = j + 1
    return out


@pytest.mark.parametrize("seed", range(5))
def test_single_night_identical_to_whole_curve_behaviour(seed):
    f = noisy(seed=seed)
    rng = np.random.default_rng(seed + 100)
    f[rng.integers(40, 260, 6)] += 15
    f[120:124] += 20
    c = curve(f)
    det = MedianResidualDetector()
    got = [(d.det_id, d.t_start, d.t_end, d.score, d.kind_guess) for d in det.detect({1: c}, C)]
    assert got == _legacy_detect(det, c, C)
    assert got


def test_night_gap_cadences_param():
    assert MedianResidualDetector().night_gap_cadences == 100
    c = two_night_curve(held_step_flux(54))
    out = MedianResidualDetector(night_gap_cadences=10_000).detect({1: c}, C)
    assert any(int(c.t[NIGHT - 3]) <= d.t_start <= int(c.t[NIGHT + 2]) for d in out)
