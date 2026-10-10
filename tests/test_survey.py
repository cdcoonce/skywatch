import numpy as np
import pytest

from skywatch.types import TruthLabel
from skywatch.world.field import FieldConfig
from skywatch.world.survey import SurveyConfig, _injector_rng, simulate

SMALL = FieldConfig(n_stars=6, planet_rate=0.0)


def _cfg(**kw):
    params = {
        "seed": 7,
        "n_nights": 3,
        "cadence_s": 60,
        "night_length_s": 3600,
        "night_gap_s": 7200,
        "field": SMALL,
    }
    params.update(kw)
    return SurveyConfig(**params)


class DimStar0:
    kind = "transit"

    def inject(self, scene, rng):
        sel = slice(5, 15)
        scene.flux[0, sel] *= 0.5
        return [
            TruthLabel(
                f"dim-{scene.night_index}",
                "transit",
                "science",
                0,
                int(scene.t[5]),
                int(scene.t[14]),
                {"u": float(rng.uniform())},
            )
        ]


class FieldWide:
    kind = "cloud"

    def inject(self, scene, rng):
        rng.uniform()
        return [
            TruthLabel(
                f"cloud-{scene.night_index}",
                "cloud",
                "instrument",
                None,
                int(scene.t[20]),
                int(scene.t[29]),
                {},
            )
        ]


class Dummy:
    kind = "flare"

    def inject(self, scene, rng):
        rng.uniform(size=100)
        return []


class DropSpan:
    kind = "dropout"

    def inject(self, scene, rng):
        scene.mask[:, 10:20] = False
        return []


def test_sample_counts_and_time_grid():
    cfg = _cfg()
    res = simulate(cfg)
    n = cfg.night_length_s // cfg.cadence_s
    assert res.night_starts == [0, 10800, 21600]
    assert len(res.curves) == 6 and len(res.stars) == 6
    for lc in res.curves.values():
        assert len(lc) == 3 * n
        assert np.all(np.diff(lc.t) > 0)
        assert lc.t[n] - lc.t[n - 1] == cfg.night_gap_s + cfg.cadence_s
        assert lc.t[0] == 0 and lc.quality.sum() == 0


def test_determinism():
    a, b = simulate(_cfg()), simulate(_cfg())
    c = simulate(_cfg(seed=8))
    assert np.array_equal(a.curves[0].flux, b.curves[0].flux)
    assert not np.array_equal(a.curves[0].flux, c.curves[0].flux)


def test_injector_independence():
    a = simulate(_cfg(), science=[DimStar0()]).truth
    ab = simulate(_cfg(), science=[DimStar0(), Dummy()]).truth
    assert a == ab and len(a) == 3
    assert len({lab.params["u"] for lab in a}) == 3


def test_field_wide_label_sets_quality_bit0():
    cfg = _cfg()
    res = simulate(cfg, instrument=[FieldWide()])
    n = cfg.night_length_s // cfg.cadence_s
    assert len(res.truth) == 3
    for lc in res.curves.values():
        q = lc.quality.reshape(3, n)
        assert (q[:, 20:30] & 1).all()
        assert not (q[:, :20] & 1).any() and not (q[:, 30:] & 1).any()


def test_dim_is_visible_and_masked_absent():
    cfg = _cfg(field=FieldConfig(n_stars=3, planet_rate=0.0, flux0_log10_range=(6.0, 6.0)))
    res = simulate(cfg, science=[DimStar0()], instrument=[DropSpan()])
    n = cfg.night_length_s // cfg.cadence_s
    for lc in res.curves.values():
        assert len(lc) == 3 * (n - 10)
    lc0 = res.curves[0]
    assert (
        len(set(lc0.t.tolist()) & {int(res.night_starts[0] + 60 * k) for k in range(10, 20)}) == 0
    )
    assert lc0.flux[:5].mean() > 1.5 * lc0.flux[5:10].mean()


class Bright0:
    kind = "bright"

    def inject(self, scene, rng):
        scene.flux[0, :] = 100.0
        return []


class Recorder:
    kind = "recorder"

    def __init__(self):
        self.seen = []

    def inject(self, scene, rng):
        self.seen.append(float(scene.flux[0, 0]))
        return []


class Zero0:
    kind = "zero"

    def inject(self, scene, rng):
        scene.flux[0, :] = 0.0
        return []


def test_science_runs_before_instrument():
    rec = Recorder()
    simulate(_cfg(), science=[Bright0()], instrument=[rec])
    assert rec.seen == [100.0] * 3


def test_injector_rng_keyed_by_kind():
    alone = simulate(_cfg(), science=[DimStar0()]).truth
    second = simulate(_cfg(), science=[Dummy(), DimStar0()]).truth
    assert alone == second

    a = _injector_rng(7, 1, "transit").uniform()
    assert a != _injector_rng(7, 1, "flare").uniform()
    assert a == _injector_rng(7, 1, "transit").uniform()


def test_noise_applied_after_injectors():
    field = FieldConfig(n_stars=2, planet_rate=0.0, noise_floor_range=(0.01, 0.01))
    res = simulate(_cfg(field=field), science=[Zero0()])
    lc = res.curves[0]
    assert len(lc) > 0
    assert np.all(lc.flux == 0.0)
    assert np.all(lc.flux_err == 0.0)
    assert np.any(res.curves[1].flux_err > 0.0)


@pytest.mark.parametrize(
    ("kw", "name"),
    [
        ({"cadence_s": 0}, "cadence_s"),
        ({"n_nights": 0}, "n_nights"),
        ({"seed": -1}, "seed"),
        ({"night_gap_s": -1}, "night_gap_s"),
        ({"night_length_s": 30}, "night_length_s"),
    ],
)
def test_survey_config_validation(kw, name):
    with pytest.raises(ValueError, match=name):
        SurveyConfig(**kw)
