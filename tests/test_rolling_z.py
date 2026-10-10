import numpy as np

from skywatch.detectors import REGISTRY
from skywatch.detectors.rolling_z import RollingZDetector
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
