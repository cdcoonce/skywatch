import numpy as np
import pytest

from skywatch.world.field import FieldConfig, generate_field

DAY = 86400
KEYS = {
    "constant": set(),
    "sinusoidal": {"period_s", "amplitude", "phase"},
    "eclipsing_binary": {"period_s", "epoch_s", "depth_primary", "depth_secondary", "duration_s"},
    "flare_star": {"flare_rate_per_day"},
}


def _gen(seed=1, **kw):
    return generate_field(
        FieldConfig(**kw), np.random.default_rng(seed), np.random.default_rng(seed + 1000)
    )


def test_determinism():
    assert _gen(1) == _gen(1)
    assert _gen(1) != _gen(2)


def test_ids_sequential_and_counts():
    stars, planets = _gen(3, n_stars=50, planet_rate=1.0)
    assert [s.star_id for s in stars] == list(range(50))
    assert [p.planet_id for p in planets] == list(range(len(planets)))
    assert len(planets) > 0
    assert {p.star_id for p in planets} <= set(range(50))


def test_star_ranges():
    cfg = FieldConfig(n_stars=400)
    stars, _ = generate_field(cfg, np.random.default_rng(5), np.random.default_rng(6))
    for s in stars:
        assert 10**3.0 <= s.flux0 <= 10**6.0
        assert 0.002 <= s.noise_floor <= 0.02
        assert s.ra**2 + s.dec**2 <= cfg.field_radius_deg**2 + 1e-9


def test_params_keys_and_ranges():
    stars, _ = _gen(7, n_stars=500)
    seen = set()
    for s in stars:
        seen.add(s.variability)
        assert set(s.params) == KEYS[s.variability]
        p = s.params
        if s.variability == "eclipsing_binary":
            assert 0.3 * DAY <= p["period_s"] <= 5 * DAY
            assert 0 <= p["epoch_s"] < p["period_s"]
            assert 0.05 <= p["depth_primary"] <= 0.5
            assert 0.2 * p["depth_primary"] <= p["depth_secondary"] <= 0.8 * p["depth_primary"]
            assert 0.05 * p["period_s"] <= p["duration_s"] <= 0.15 * p["period_s"]
        elif s.variability == "sinusoidal":
            assert 3600 <= p["period_s"] <= 12 * 3600
            assert 0.001 <= p["amplitude"] <= 0.02
            assert 0 <= p["phase"] <= 2 * np.pi
        elif s.variability == "flare_star":
            assert 0.5 <= p["flare_rate_per_day"] <= 5
    assert seen == set(KEYS)


def test_planet_ranges():
    _, planets = _gen(9, n_stars=300, planet_rate=1.0)
    for p in planets:
        assert 0.5 * DAY <= p.period_s <= 20 * DAY
        assert 0 <= p.epoch_s < p.period_s
        assert 1e-4 <= p.depth <= 3e-2
        assert 0.5 * 3600 <= p.duration_s <= 6 * 3600


@pytest.mark.parametrize("cls", list(KEYS))
def test_single_class_weights(cls):
    w = {k: (1.0 if k == cls else 0.0) for k in KEYS}
    stars, _ = _gen(2, n_stars=60, variability_weights=w)
    assert {s.variability for s in stars} == {cls}


def test_planet_rate_zero():
    _, planets = _gen(4, planet_rate=0.0)
    assert planets == []


def test_planet_rate_does_not_perturb_stars():
    a, _ = _gen(4, n_stars=20, planet_rate=0.0)
    b, planets = _gen(4, n_stars=20, planet_rate=2.0)
    assert planets and a == b


def test_growing_n_stars_keeps_earlier_stars():
    a, _ = _gen(4, n_stars=10)
    b, _ = _gen(4, n_stars=11)
    assert a == b[:10]


@pytest.mark.parametrize(
    ("kw", "name"),
    [
        ({"n_stars": 0}, "n_stars"),
        ({"planet_rate": -0.1}, "planet_rate"),
        ({"variability_weights": {"constant": 0.0, "sinusoidal": 0.0}}, "variability_weights"),
        ({"variability_weights": {"constant": 1.0, "sinusoidal": -0.5}}, "variability_weights"),
    ],
)
def test_field_config_validation(kw, name):
    with pytest.raises(ValueError, match=name):
        FieldConfig(**kw)
