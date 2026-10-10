import time

import numpy as np
import pytest

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


def test_centered_baseline_absorbs_held_drift():
    # documented limitation of the centered baseline: the shrinking median follows a held step
    t, flux = make_field()
    f = np.ones(240)
    f[150:200] = 1 - 0.05 * (np.arange(150, 200) - 150) / 50
    f[200:] = 0.95
    out = FieldCommonModeDetector(baseline="centered").detect(to_curves(t, flux * f), C)
    assert all(d.t_end < t[-1] + C for d in out)


def test_causal_baseline_does_not_absorb_the_same_held_drift():
    # the same field under the default (causal) baseline: the held tail is still a
    # departure from the past, so the run reaches the night end and is a drift
    t, flux = make_field()
    f = np.ones(240)
    f[150:200] = 1 - 0.05 * (np.arange(150, 200) - 150) / 50
    f[200:] = 0.95
    out = FieldCommonModeDetector().detect(to_curves(t, flux * f), C)
    assert [d.kind_guess for d in out] == ["tracking_drift"]
    assert out[0].t_end == t[-1] + C


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
    # Deliberately loose: this guards against an algorithmic blow-up (an O(n * window)
    # Python loop or quadratic work), which costs minutes at this size. A tight bound
    # (it was 3.0 s) flaked once on a loaded machine, so it is 30 s, ~10x the typical run.
    assert time.perf_counter() - t0 < 30.0


def test_baseline_does_not_cross_night_gap():
    n = 240
    t1, f1 = make_field(n=n, seed=1, noise=0.02)
    t2, f2 = make_field(n=n, seed=2, noise=0.02, t0=int(t1[-1]) + C + 16 * 3600)
    hold = np.ones(n)
    hold[-60:] = 0.95  # drift hold at the end of night 1
    t = np.concatenate([t1, t2])
    flux = np.concatenate([f1 * hold, f2], axis=1)
    out = FieldCommonModeDetector().detect(to_curves(t, flux), C)
    assert [d for d in out if d.t_start >= t2[0]] == []


def test_single_star_outlier_does_not_move_common_mode():
    t, flux = make_field()
    dip = flux.copy()
    dip[0] = dip[0] * cosine_dip(240, 100, 130, 0.40)
    assert FieldCommonModeDetector().detect(to_curves(t, dip), C) == []
    hot = flux.copy()
    hot[0, 100:130] *= 3.0  # +200% hot pixel
    assert FieldCommonModeDetector().detect(to_curves(t, hot), C) == []


def test_min_run_binds_when_unsmoothed():
    t, flux = make_field(noise=0.0)
    flux[:, 120] *= 0.95  # single-sample field-wide dip
    curves = to_curves(t, flux)
    assert FieldCommonModeDetector(smooth=1).detect(curves, C) == []
    assert len(FieldCommonModeDetector(smooth=1, min_run=1).detect(curves, C)) == 1


def test_smoothing_is_centered():
    t, flux = make_field(noise=0.0)
    flux = flux * cosine_dip(240, 100, 140, 0.06)
    out = FieldCommonModeDetector().detect(to_curves(t, flux), C)
    assert len(out) == 1
    t_mid = (t[100] + t[140]) / 2
    assert abs((out[0].t_start + out[0].t_end) / 2 - t_mid) <= C


# --- night_gap_cadences wiring -------------------------------------------------------
# A "hole of G cadences" means consecutive observed times G * C apart. split_nights
# splits only when the spacing is strictly greater than night_gap_cadences * C, so with
# night_gap_cadences=10 a spacing of 10 stays one night and 11 starts a new one. The
# detector reads the setting at two sites (_baseline and detect); one test per site per
# side of the boundary, each blind to the other site's value.

GAP = 10
N_NIGHT = 40


def two_night_field(spacing, night2_dip):
    """Two noise-free 40-sample nights `spacing` cadences apart; night 2 dipped to 0.9
    over the slice `night2_dip`. Returns (t, flux, t2_first_index)."""
    t1 = np.arange(N_NIGHT, dtype=np.int64) * C
    t2 = t1[-1] + spacing * C + np.arange(N_NIGHT, dtype=np.int64) * C
    f2 = np.ones(N_NIGHT)
    f2[night2_dip] = 0.9
    t = np.concatenate([t1, t2])
    f = np.concatenate([np.ones(N_NIGHT), f2])
    return t, np.tile(1000.0 * f, (8, 1)), N_NIGHT


def small_gap_detector(baseline="causal"):
    return FieldCommonModeDetector(
        night_gap_cadences=GAP, baseline_window=21, smooth=1, min_run=1, baseline=baseline
    )


# The dip is placed where the two baselines each tell "crossed the gap" from "split":
# centered absorbs a night's first samples through its shrunken edge window, so the dip
# starts at the edge; causal absorbs whatever it has already seen, so the dip starts 4
# samples in and the baseline's only memory of "flat" is the pre-dip samples. Offsets are
# (first, one-past-last) in samples from night 2's first sample.
EDGE_CASES = [
    ("centered", slice(0, 8), (5, 8), (0, 8)),
    ("causal", slice(4, 12), (4, 8), (4, 12)),
]


def cloud_offsets(det, t, flux, i2):
    out = det.detect(to_curves(t, flux), C)
    return [
        ((d.t_start - int(t[i2])) // C, (d.t_end - int(t[i2])) // C)
        for d in out
        if d.kind_guess == "cloud"
    ]


@pytest.mark.parametrize(("baseline", "dip", "split_expected", "_"), EDGE_CASES)
def test_baseline_splits_when_spacing_exceeds_night_gap(baseline, dip, split_expected, _):
    # Split: the baseline never sees night 1's flat samples. A baseline that crossed the
    # gap would see 10 flat samples from night 1 and flag the whole dip.
    t, flux, i2 = two_night_field(GAP + 1, dip)
    det = small_gap_detector(baseline)
    assert cloud_offsets(det, t, flux, i2) == [split_expected]


@pytest.mark.parametrize(("baseline", "dip", "_", "joined_expected"), EDGE_CASES)
def test_baseline_window_crosses_spacing_of_exactly_night_gap(baseline, dip, _, joined_expected):
    t, flux, i2 = two_night_field(GAP, dip)
    det = small_gap_detector(baseline)
    assert cloud_offsets(det, t, flux, i2) == [joined_expected]


def test_detect_splits_when_spacing_exceeds_night_gap():
    # Mid-night dip, far from any edge, so the baseline site cannot matter. The 11-cadence
    # spacing is between nights, not a field dropout, and the dip is in the second night (n1).
    t, flux, _ = two_night_field(GAP + 1, slice(20, 26))
    out = small_gap_detector().detect(to_curves(t, flux), C)
    assert [d.det_id for d in out] == ["field_common_mode-n1-0"]
    assert out[0].kind_guess == "cloud"


def test_detect_treats_spacing_of_exactly_night_gap_as_dropout():
    t, flux, i2 = two_night_field(GAP, slice(20, 26))
    out = small_gap_detector().detect(to_curves(t, flux), C)
    assert [(d.det_id, d.kind_guess) for d in out] == [
        ("field_common_mode-n0-0", "dropout"),
        ("field_common_mode-n0-1", "cloud"),
    ]
    assert out[0].score == GAP
    assert (out[0].t_start, out[0].t_end) == (int(t[N_NIGHT - 1]) + C, int(t[i2]))


# --- causal baseline and the long-negative-run kind rule (issue #13) -------------------


def spans(out):
    return [(d.det_id, d.t_start, d.t_end, d.score, d.kind_guess) for d in out]


def iou(det, s, e):
    inter = max(0, min(det.t_end, e) - max(det.t_start, s))
    return inter / (max(det.t_end, e) - min(det.t_start, s))


def test_centered_baseline_reproduces_pre_change_output():
    # Golden values captured on main (e08cdbe), before the baseline option existed.
    det = FieldCommonModeDetector(baseline="centered")

    t, flux = make_field()
    out = det.detect(to_curves(t, flux * cosine_dip(240, 100, 130, 0.05)), C)
    assert spans(out) == [("field_common_mode-n0-0", 6420, 7440, 2.5332160585244954, "cloud")]

    t, flux = make_field(n=300)
    f = np.ones(300)
    f[150:200] = 1 - 0.05 * (np.arange(150, 200) - 150) / 50
    f[200:] = 0.95
    wide = FieldCommonModeDetector(baseline="centered", baseline_window=301)
    out = wide.detect(to_curves(t, flux * f), C)
    assert spans(out) == [
        ("field_common_mode-n0-0", 10800, 18000, 2.2061733948392113, "tracking_drift")
    ]

    t, flux = make_field()
    f = np.ones(240)
    f[150:200] = 1 - 0.05 * (np.arange(150, 200) - 150) / 50
    f[200:] = 0.95
    assert det.detect(to_curves(t, flux * f), C) == []

    n = 240
    t1, f1 = make_field(n=n, seed=1)
    t2, f2 = make_field(n=n, seed=2, t0=int(t1[-1]) + C + 16 * 3600)
    f1 = f1 * cosine_dip(n, 80, 110, 0.06)
    f2 = f2 * cosine_dip(n, 120, 150, 0.06)
    t = np.concatenate([t1, t2])
    out = det.detect(to_curves(t, np.concatenate([f1, f2], axis=1)), C)
    assert spans(out) == [
        ("field_common_mode-n0-0", 5220, 6240, 2.9480525514203824, "cloud"),
        ("field_common_mode-n1-0", 79620, 80640, 2.8514107607497508, "cloud"),
    ]

    t, flux = make_field()
    keep = np.ones(flux.shape, bool)
    keep[:, 60:70] = False
    out = det.detect(to_curves(t, flux * cosine_dip(240, 100, 130, 0.05), keep), C)
    assert spans(out) == [
        ("field_common_mode-n0-0", 3600, 4200, 11.0, "dropout"),
        ("field_common_mode-n0-1", 6420, 7440, 2.5310068925800953, "cloud"),
    ]


def test_invalid_baseline_raises():
    with pytest.raises(ValueError, match="baseline"):
        FieldCommonModeDetector(baseline="trailing")


def test_causal_median_never_reads_the_future():
    from skywatch.detectors.field_common_mode import _trailing_median

    x = np.arange(10.0)
    out = _trailing_median(x, 5)
    # shrinking at the start, trailing 5 samples afterwards
    assert out.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    y = x.copy()
    y[7:] = 1e6  # changing the future leaves earlier baselines untouched
    assert _trailing_median(y, 5)[:7].tolist() == out[:7].tolist()


def held_drift_field(n=480, start=200, ramp=90, depth=0.15, **kw):
    t, flux = make_field(n=n, **kw)
    f = np.ones(n)
    f[start : start + ramp] = 1 - depth * (np.arange(ramp)) / ramp
    f[start + ramp :] = 1 - depth
    return t, flux * f, start


def test_causal_baseline_catches_held_drift_as_tracking_drift():
    t, flux, start = held_drift_field()
    out = FieldCommonModeDetector(baseline="causal").detect(to_curves(t, flux), C)
    assert len(out) == 1
    d = out[0]
    assert d.kind_guess == "tracking_drift"
    assert d.t_start >= t[start]
    # the run ends before the night does: the baseline absorbed the held step
    assert d.t_end < t[-1] + C
    # while the centered baseline, on the same field, still follows the step and misses it
    centered = FieldCommonModeDetector(baseline="centered").detect(to_curves(t, flux), C)
    assert all(x.kind_guess != "tracking_drift" for x in centered)


def test_causal_baseline_keeps_a_half_hour_cloud_bounded():
    t, flux = make_field()
    flux = flux * cosine_dip(240, 100, 130, 0.06)
    out = FieldCommonModeDetector(baseline="causal").detect(to_curves(t, flux), C)
    assert len(out) == 1
    assert out[0].kind_guess == "cloud"
    assert iou(out[0], int(t[100]), int(t[130])) >= 0.3


def test_negative_run_longer_than_max_cloud_is_tracking_drift_even_mid_night():
    t, flux = make_field(noise=0.002)
    f = np.ones(240)
    f[100:180] = 0.93  # 80 samples = 4800 s, ends well before the night does
    curves = to_curves(t, flux * f)
    # a window longer than the night makes the baseline a constant, so the run is whole
    kw = {"baseline_window": 1001, "baseline": "centered"}
    long_run = FieldCommonModeDetector(**kw).detect(curves, C)
    assert [d.kind_guess for d in long_run] == ["tracking_drift"]
    assert long_run[0].t_end < t[-1] + C
    assert long_run[0].t_end - long_run[0].t_start > 3600
    # the cutoff is the parameter, not a literal: raise it and the same run is a cloud
    raised = FieldCommonModeDetector(max_cloud_s=7200, **kw).detect(curves, C)
    assert [d.kind_guess for d in raised] == ["cloud"]


def test_negative_run_at_exactly_max_cloud_is_still_a_cloud():
    t, flux = make_field(noise=0.002)
    f = np.ones(240)
    f[100:160] = 0.93  # 60 samples = 3600 s
    curves = to_curves(t, flux * f)
    det = FieldCommonModeDetector(baseline_window=1001, baseline="centered", smooth=1)
    out = det.detect(curves, C)
    assert [(d.t_end - d.t_start, d.kind_guess) for d in out] == [(3600, "cloud")]


def test_short_negative_run_reaching_night_end_is_tracking_drift():
    # 30 samples = 1800 s, well under max_cloud_s, so only the night-end rule can call it drift
    t, flux = make_field(noise=0.002)
    f = np.ones(240)
    f[210:] = 0.93
    det = FieldCommonModeDetector(baseline_window=1001, smooth=1)
    out = det.detect(to_curves(t, flux * f), C)
    assert [(d.t_start, d.t_end, d.kind_guess) for d in out] == [
        (int(t[210]), int(t[-1]) + C, "tracking_drift")
    ]
    assert out[0].t_end - out[0].t_start <= 3600
    # the same run ending one sample early is just a short cloud
    f2 = np.ones(240)
    f2[210:239] = 0.93
    early = det.detect(to_curves(t, flux * f2), C)
    assert [d.kind_guess for d in early] == ["cloud"]


@pytest.mark.parametrize("cadence", [30, 120])
def test_max_cloud_rule_is_in_seconds_at_any_cadence(cadence):
    n = 600  # long enough that the dip is a minority of the night at every cadence
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.int64) * cadence
    flux = 1000.0 * (1 + rng.normal(0, 0.002, (8, n)))
    kw = {"baseline_window": 1001, "baseline": "centered", "smooth": 1}

    def run_kinds(samples):
        f = np.ones(n)
        f[200 : 200 + samples] = 0.93
        out = FieldCommonModeDetector(**kw).detect(to_curves(t, flux * f), cadence)
        return [(d.t_end - d.t_start, d.kind_guess) for d in out]

    # 3600 s at this cadence: still a cloud (a samples-vs-seconds mix-up would flag it)
    assert run_kinds(3600 // cadence) == [(3600, "cloud")]
    # one sample past 3600 s: tracking_drift (a hardcoded 60 s cadence would call it a cloud)
    one_over = 3600 // cadence + 1
    assert run_kinds(one_over) == [(one_over * cadence, "tracking_drift")]
