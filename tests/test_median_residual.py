import numpy as np
import pytest

from skywatch.detectors import REGISTRY, load_all
from skywatch.detectors.median_residual import MedianResidualDetector, median_residual_scores
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
