import time

import numpy as np

from skywatch.detectors import REGISTRY, Detector, load_all
from skywatch.detectors.field_common_mode import FieldCommonModeDetector
from skywatch.types import LightCurve

C = 60


def make_field(n_stars=8, n=240, seed=0, noise=0.01, t0=0):
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n, dtype=np.int64) * C
    flux = 1000.0 * (1 + rng.normal(0, noise, (n_stars, n)))
    return t, flux


def to_curves(t, flux, keep=None):
    """keep: optional bool (n_stars, n) mask of observed samples."""
    out = {}
    for i in range(flux.shape[0]):
        m = np.ones(len(t), bool) if keep is None else keep[i]
        out[i] = LightCurve(
            i, t[m], flux[i, m], np.ones(m.sum()), np.zeros(m.sum(), dtype=np.int64)
        )
    return out


def cosine_dip(n, lo, hi, depth):
    f = np.ones(n)
    x = np.arange(lo, hi)
    f[lo:hi] = 1 - depth * 0.5 * (1 + np.cos(np.pi * (x - (lo + hi) / 2) / ((hi - lo) / 2)))
    return f


def overlap_frac(det, s, e):
    inter = max(0, min(det.t_end, e) - max(det.t_start, s))
    return inter / (e - s)


def test_cloud_all_stars():
    t, flux = make_field()
    flux = flux * cosine_dip(240, 100, 130, 0.05)
    out = FieldCommonModeDetector().detect(to_curves(t, flux), C)
    assert len(out) == 1
    d = out[0]
    assert d.star_id is None and d.kind_guess == "cloud" and d.detector == "field_common_mode"
    assert d.score > 1
    # the 2%-threshold crossing of a 5% raised cosine covers the core of the span
    s, e = t[100], t[130]
    assert d.t_start >= s and d.t_end <= e
    assert overlap_frac(d, s, e) >= 0.5


def test_cloud_one_star_only():
    t, flux = make_field()
    flux[0] = flux[0] * cosine_dip(240, 100, 130, 0.05)
    assert FieldCommonModeDetector().detect(to_curves(t, flux), C) == []


def test_drift_held_to_end():
    t, flux = make_field(n=300)
    f = np.ones(300)
    f[150:200] = 1 - 0.05 * (np.arange(150, 200) - 150) / 50
    f[200:] = 0.95
    # a window longer than the held tail, so the baseline cannot absorb the step
    out = FieldCommonModeDetector(baseline_window=301).detect(to_curves(t, flux * f), C)
    assert len(out) == 1
    d = out[0]
    assert d.star_id is None and d.kind_guess == "tracking_drift"
    assert d.t_end == t[-1] + C


def test_default_baseline_absorbs_held_drift():
    # documented limitation: the shrinking centered median follows a held step
    t, flux = make_field()
    f = np.ones(240)
    f[150:200] = 1 - 0.05 * (np.arange(150, 200) - 150) / 50
    f[200:] = 0.95
    out = FieldCommonModeDetector().detect(to_curves(t, flux * f), C)
    assert all(d.t_end < t[-1] + C for d in out)


def test_dropout_all_stars():
    t, flux = make_field()
    keep = np.ones(flux.shape, bool)
    keep[:, 60:70] = False
    out = FieldCommonModeDetector().detect(to_curves(t, flux, keep), C)
    assert len(out) == 1
    d = out[0]
    assert d.kind_guess == "dropout" and d.star_id is None
    assert (d.t_start, d.t_end) == (int(t[60]), int(t[70]))
    assert d.score == 11  # gap from t59 to t70 is 11 cadences


def test_dropout_one_star_only():
    t, flux = make_field()
    keep = np.ones(flux.shape, bool)
    keep[0, 60:70] = False
    assert FieldCommonModeDetector().detect(to_curves(t, flux, keep), C) == []


def test_two_nights():
    n = 240
    t1, f1 = make_field(n=n, seed=1)
    t2, f2 = make_field(n=n, seed=2, t0=int(t1[-1]) + C + 16 * 3600)
    f1 = f1 * cosine_dip(n, 80, 110, 0.06)
    f2 = f2 * cosine_dip(n, 120, 150, 0.06)
    t = np.concatenate([t1, t2])
    flux = np.concatenate([f1, f2], axis=1)
    out = FieldCommonModeDetector().detect(to_curves(t, flux), C)
    assert sorted(d.det_id for d in out) == [
        "field_common_mode-n0-0",
        "field_common_mode-n1-0",
    ]
    assert all(d.kind_guess == "cloud" for d in out)
    assert out[0].t_end <= t1[-1] + C <= out[1].t_start


def test_too_few_stars():
    t, flux = make_field(n_stars=4)
    flux = flux * cosine_dip(240, 100, 130, 0.05)
    assert FieldCommonModeDetector().detect(to_curves(t, flux), C) == []
    keep = np.ones(flux.shape, bool)
    keep[:, 60:70] = False
    out = FieldCommonModeDetector().detect(to_curves(t, flux, keep), C)
    assert [d.kind_guess for d in out] == ["dropout"]


def test_threshold_respected():
    t, flux = make_field(noise=0.002)
    flux = flux * cosine_dip(240, 100, 130, 0.01)
    assert FieldCommonModeDetector().detect(to_curves(t, flux), C) == []
    assert len(FieldCommonModeDetector(threshold=0.005).detect(to_curves(t, flux), C)) == 1


def test_empty_input():
    assert FieldCommonModeDetector().detect({}, C) == []


def test_deterministic():
    t, flux = make_field()
    flux = flux * cosine_dip(240, 100, 130, 0.05)
    curves = to_curves(t, flux)
    d = FieldCommonModeDetector()
    assert d.detect(curves, C) == d.detect(curves, C)


def test_registered_and_protocol():
    assert load_all()["field_common_mode"] is FieldCommonModeDetector
    assert REGISTRY["field_common_mode"] is FieldCommonModeDetector
    det: Detector = FieldCommonModeDetector()
    assert det.name == "field_common_mode"


def test_performance_guard():
    t, flux = make_field(n_stars=200, n=1440)
    curves = to_curves(t, flux)
    t0 = time.perf_counter()
    FieldCommonModeDetector().detect(curves, C)
    assert time.perf_counter() - t0 < 3.0
