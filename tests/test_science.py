import numpy as np
import pytest

from skywatch.types import Planet, Star, TruthLabel
from skywatch.world.scene import Injector, Scene
from skywatch.world.science import (
    EclipseInjector,
    FlareInjector,
    PulsationInjector,
    TransitInjector,
)

N = 600  # 10 h at 60 s
CAD = 60


def _star(i, variability="constant", **params):
    return Star(i, 1000.0, 0.0, variability, 0.0, 0.0, dict(params))


def _scene(stars, t_start=0, n=N, night=0):
    return Scene.quiet(stars, t_start, n, CAD, night)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_protocol_satisfied():
    for obj in (TransitInjector([]), EclipseInjector(), PulsationInjector(), FlareInjector()):
        assert isinstance(obj, Injector)


def test_transit_depth_shape_and_untouched_outside():
    sc = _scene([_star(0), _star(1)])
    p = Planet(0, 0, period_s=1e6, epoch_s=18000.0, depth=0.01, duration_s=3000.0)
    labels = TransitInjector([p]).inject(sc, _rng())
    assert len(labels) == 1
    lab = labels[0]
    assert lab.origin == "science" and lab.kind == "transit" and lab.star_id == 0
    assert lab.label_id == "transit-n0-s0-0"
    assert (lab.t_start, lab.t_end) == (16500, 19500)
    assert lab.params["depth"] == 0.01 and lab.params["planet_id"] == 0
    assert lab.params["t_mid"] == 18000.0 and "clipped" not in lab.params
    row = sc.flux[0]
    assert row.min() == pytest.approx(1000.0 * 0.99)
    assert sc.t[row.argmin()] in range(16800, 19200 + 1)
    # ingress: 300 s long, starting 16500; sample at 16680 -> 3/5 of the way
    i = int(np.where(sc.t == 16680)[0][0])
    assert row[i] == pytest.approx(1000.0 * (1 - 0.01 * 0.6))
    outside = (sc.t <= 16500) | (sc.t >= 19500)
    assert np.all(row[outside] == 1000.0)
    assert np.all(sc.flux[1] == 1000.0)


def test_transit_is_multiplicative():
    sc = _scene([_star(0)])
    sc.flux[0] *= 2.0
    p = Planet(0, 0, 1e6, 18000.0, 0.01, 3000.0)
    TransitInjector([p]).inject(sc, _rng())
    assert sc.flux[0].min() == pytest.approx(2000.0 * 0.99)
    assert sc.flux[0].max() == pytest.approx(2000.0)


def test_transit_no_event_in_night_no_label():
    sc = _scene([_star(0)], night=1)
    # transits at 100000 + n*1e6: none in [0, 36000]
    p = Planet(0, 0, 1e6, 100000.0, 0.01, 3000.0)
    assert TransitInjector([p]).inject(sc, _rng()) == []
    assert np.all(sc.flux == 1000.0)


def test_transit_skips_planets_of_absent_stars():
    sc = _scene([_star(0)])
    p = Planet(0, 7, 1e6, 18000.0, 0.01, 3000.0)
    assert TransitInjector([p]).inject(sc, _rng()) == []


def test_transit_clipped_at_night_end():
    sc = _scene([_star(0)])
    end = int(sc.t[-1]) + CAD  # 36000
    p = Planet(0, 0, 1e6, float(end - 600), 0.02, 3000.0)
    (lab,) = TransitInjector([p]).inject(sc, _rng())
    assert lab.params["clipped"] is True
    assert (lab.t_start, lab.t_end) == (end - 600 - 1500, end)
    assert sc.flux[0, -1] < 1000.0


def test_transit_multiple_events_get_distinct_ids():
    sc = _scene([_star(0)])
    p = Planet(0, 0, 10000.0, 5000.0, 0.01, 1000.0)
    labels = TransitInjector([p]).inject(sc, _rng())
    assert len(labels) == 4  # 5000, 15000, 25000, 35000
    assert [lab.label_id for lab in labels] == [f"transit-n0-s0-{k}" for k in range(4)]


def test_transit_on_nonzero_night_grid():
    sc = _scene([_star(0)], t_start=86400, night=1)
    p = Planet(0, 0, 86400.0, 0.0, 0.01, 3000.0)
    (lab,) = TransitInjector([p]).inject(sc, _rng())
    assert lab.label_id == "transit-n1-s0-0"
    assert lab.params["clipped"] is True
    assert (lab.t_start, lab.t_end) == (86400, 86400 + 1500)


def test_eclipse_v_shape_primary_and_secondary():
    star = _star(
        0,
        "eclipsing_binary",
        period_s=18000.0,
        epoch_s=6000.0,
        depth_primary=0.2,
        depth_secondary=0.1,
        duration_s=2000.0,
    )
    sc = _scene([star, _star(1)])
    labels = EclipseInjector().inject(sc, _rng())
    got = {(lab.params["which"], lab.params["t_mid"]) for lab in labels}
    assert got == {
        ("primary", 6000.0),
        ("secondary", 15000.0),
        ("primary", 24000.0),
        ("secondary", 33000.0),
    }
    assert all(lab.origin == "science" and lab.star_id == 0 for lab in labels)
    ids = [lab.label_id for lab in labels]
    assert len(set(ids)) == 4 and all(i.startswith("eclipse-n0-s0-") for i in ids)
    row = sc.flux[0]
    assert row[np.where(sc.t == 6000)[0][0]] == pytest.approx(800.0)
    assert row[np.where(sc.t == 15000)[0][0]] == pytest.approx(900.0)
    # halfway down the V: 480 s from mid, 52% of primary depth
    assert row[np.where(sc.t == 6480)[0][0]] == pytest.approx(1000.0 * (1 - 0.2 * 0.52))
    assert row[np.where(sc.t == 7080)[0][0]] == 1000.0
    assert np.all(sc.flux[1] == 1000.0)
    sec = next(lab for lab in labels if lab.params["which"] == "secondary")
    assert sec.params["depth"] == 0.1 and sec.params["duration_s"] == 2000.0


def test_eclipse_multiplicative_and_clipped():
    star = _star(
        0,
        "eclipsing_binary",
        period_s=1e6,
        epoch_s=36000.0,
        depth_primary=0.2,
        depth_secondary=0.1,
        duration_s=2000.0,
    )
    sc = _scene([star])
    sc.flux[0] *= 3.0
    (lab,) = EclipseInjector().inject(sc, _rng())
    assert lab.params["clipped"] is True and lab.t_end == 36000
    assert sc.flux[0, -1] == pytest.approx(3000.0 * (1 - 0.2 * (1 - 60 / 1000)))


def test_eclipse_none_in_night():
    star = _star(
        0,
        "eclipsing_binary",
        period_s=1e6,
        epoch_s=400000.0,
        depth_primary=0.2,
        depth_secondary=0.1,
        duration_s=2000.0,
    )
    assert EclipseInjector().inject(_scene([star]), _rng()) == []


def test_pulsation_values_and_one_label_per_star():
    stars = [
        _star(0, "sinusoidal", period_s=7200.0, amplitude=0.01, phase=0.5),
        _star(1),
        _star(2, "sinusoidal", period_s=3600.0, amplitude=0.02, phase=0.0),
    ]
    sc = _scene(stars)
    labels = PulsationInjector().inject(sc, _rng())
    assert len(labels) == 2
    assert {lab.star_id for lab in labels} == {0, 2}
    lab0 = next(lab for lab in labels if lab.star_id == 0)
    assert lab0.label_id == "pulsation-n0-s0-0" and lab0.origin == "science"
    assert (lab0.t_start, lab0.t_end) == (0, N * CAD)
    assert lab0.params == {"period_s": 7200.0, "amplitude": 0.01}
    expect = 1000.0 * (1 + 0.01 * np.sin(2 * np.pi * sc.t / 7200.0 + 0.5))
    assert np.allclose(sc.flux[0], expect)
    assert np.all(sc.flux[1] == 1000.0)


def test_pulsation_multiplicative():
    sc = _scene([_star(0, "sinusoidal", period_s=7200.0, amplitude=0.01, phase=0.0)])
    sc.flux[0] *= 2.0
    PulsationInjector().inject(sc, _rng())
    assert sc.flux[0].max() <= 2000.0 * 1.01 + 1e-9
    assert sc.flux[0].max() > 2000.0 * 1.0099


def test_transit_dips_pulsating_curve():
    star = _star(0, "sinusoidal", period_s=7200.0, amplitude=0.01, phase=0.0)
    sc = _scene([star])
    PulsationInjector().inject(sc, _rng())
    base = sc.flux[0].copy()
    TransitInjector([Planet(0, 0, 1e6, 18000.0, 0.01, 3000.0)]).inject(sc, _rng())
    i = int(np.where(sc.t == 18000)[0][0])
    assert sc.flux[0, i] == pytest.approx(base[i] * 0.99)


def test_flare_shape_and_label():
    sc = _scene([_star(0, "flare_star", flare_rate_per_day=40.0), _star(1)])
    sc.flux[1] *= 1.0
    labels = FlareInjector(default_rate_per_day=0.0).inject(sc, _rng(3))
    assert labels, "rate 40/day over 10 h should flare"
    assert all(lab.star_id == 0 and lab.origin == "science" for lab in labels)
    assert np.all(sc.flux[1] == 1000.0)
    assert len({lab.label_id for lab in labels}) == len(labels)
    assert labels[0].label_id == "flare-n0-s0-0"
    # single-flare check on a fresh scene via a rate that yields exactly one
    for seed in range(200):
        sc = _scene([_star(0)])
        labs = FlareInjector(default_rate_per_day=2.0).inject(sc, _rng(seed))
        if len(labs) == 1:
            break
    else:
        pytest.fail("no single-flare seed found")
    (lab,) = labs
    amp, decay, tp = lab.params["amplitude"], lab.params["decay_s"], lab.params["t_peak"]
    assert 0.01 <= amp <= 0.5 and 120 <= decay <= 1800
    k = int(np.searchsorted(sc.t, tp))  # first sample >= t_peak
    if k < N:
        assert sc.flux[0, k] == pytest.approx(1000.0 * (1 + amp * np.exp(-(sc.t[k] - tp) / decay)))
        assert np.all(sc.flux[0, :k] == 1000.0)
    end = tp + 3 * decay
    assert lab.t_start == int(np.floor(tp))
    assert lab.t_end == min(int(np.ceil(end)), N * CAD)
    if end > N * CAD:
        assert lab.params["clipped"] is True


def test_flare_multiplicative():
    for seed in range(200):
        sc = _scene([_star(0)])
        sc.flux[0] *= 2.0
        labs = FlareInjector(default_rate_per_day=2.0).inject(sc, _rng(seed))
        if len(labs) == 1 and labs[0].params["t_peak"] < 30000:
            break
    else:
        pytest.fail("no seed")
    amp = labs[0].params["amplitude"]
    assert sc.flux[0].max() <= 2000.0 * (1 + amp) + 1e-9
    assert sc.flux[0].max() > 2000.0


def test_flare_deterministic_and_rate_zero():
    stars = [_star(i, "flare_star", flare_rate_per_day=20.0) for i in range(5)]
    a = FlareInjector().inject(_scene(stars), _rng(11))
    b = FlareInjector().inject(_scene(stars), _rng(11))
    assert a == b and a
    sc = _scene([_star(0)])
    assert FlareInjector(default_rate_per_day=0.0).inject(sc, _rng(11)) == []
    assert np.all(sc.flux == 1000.0)
    sc = _scene([_star(0, "flare_star", flare_rate_per_day=0.0)])
    assert FlareInjector(default_rate_per_day=50.0).inject(sc, _rng(11)) == []


def test_flare_wholly_after_peak_clipped_flag():
    # huge rate: many flares; those near the end must be flagged clipped
    sc = _scene([_star(0, "flare_star", flare_rate_per_day=500.0)])
    labels = FlareInjector().inject(sc, _rng(5))
    assert labels
    for lab in labels:
        assert 0 <= lab.t_start < lab.t_end <= N * CAD
        spans_end = lab.params["t_peak"] + 3 * lab.params["decay_s"] > N * CAD
        assert lab.params.get("clipped", False) == spans_end


def test_labels_validate_and_ids_unique_across_injectors():
    star = _star(
        0,
        "eclipsing_binary",
        period_s=18000.0,
        epoch_s=6000.0,
        depth_primary=0.2,
        depth_secondary=0.1,
        duration_s=2000.0,
    )
    sc = _scene([star])
    labels = []
    for inj in (
        TransitInjector([Planet(0, 0, 10000.0, 1000.0, 0.01, 500.0)]),
        EclipseInjector(),
        PulsationInjector(),
        FlareInjector(default_rate_per_day=30.0),
    ):
        labels += inj.inject(sc, _rng(1))
    assert all(isinstance(lab, TruthLabel) and lab.origin == "science" for lab in labels)
    assert len({lab.label_id for lab in labels}) == len(labels)
