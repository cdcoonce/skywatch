"""Every truth label records how many samples its event actually changed."""

import numpy as np
import pytest

from skywatch.types import ORIGIN_OF_KIND, Planet, Star
from skywatch.world.glitches import (
    CloudInjector,
    CosmicRayInjector,
    DropoutInjector,
    HotPixelInjector,
    TrackingDriftInjector,
)
from skywatch.world.scene import Scene
from skywatch.world.science import (
    EclipseInjector,
    FlareInjector,
    PulsationInjector,
    TransitInjector,
)

N = 600  # 10 h at 60 s: samples at 0, 60, ..., 35940; night is [0, 36000)
CAD = 60
KEY = "footprint_samples"


def _star(i, variability="constant", **params):
    return Star(i, 1000.0, 0.0, variability, 0.0, 0.0, dict(params))


def _scene(stars, n=N, cadence=CAD):
    return Scene.quiet(stars, 0, n, cadence, 0)


class _Scripted:
    """A stand-in generator that replays chosen draws.

    ``poisson`` always returns ``count``; ``uniform`` returns the next scripted value, then
    the midpoint of its bounds; ``integers`` returns 0. Enough to place one event exactly.
    """

    def __init__(self, uniforms=(), count=1):
        self._u = list(uniforms)
        self._count = count

    def poisson(self, lam):
        return self._count

    def uniform(self, lo=0.0, hi=1.0):
        return self._u.pop(0) if self._u else (lo + hi) / 2

    def integers(self, n):
        return 0


# --- the issue's cases -------------------------------------------------------------


def test_flare_peaking_after_the_last_sample_has_zero_footprint():
    sc = _scene([_star(0, "flare_star", flare_rate_per_day=1.0)])
    assert sc.t[-1] == 35940
    (lab,) = FlareInjector().inject(sc, _Scripted([35970.0]))
    assert lab.params["t_peak"] == 35970.0
    assert lab.params[KEY] == 0
    assert np.all(sc.flux == 1000.0)  # and the scene really is untouched


def test_flare_at_mid_night_has_positive_footprint():
    sc = _scene([_star(0, "flare_star", flare_rate_per_day=1.0)])
    (lab,) = FlareInjector().inject(sc, _Scripted([18000.0]))
    assert lab.params[KEY] > 0
    assert lab.params[KEY] == int(np.count_nonzero(sc.flux[0] != 1000.0))


def test_flare_peaking_exactly_on_the_last_sample_has_footprint_one():
    sc = _scene([_star(0, "flare_star", flare_rate_per_day=1.0)])
    (lab,) = FlareInjector().inject(sc, _Scripted([35940.0]))
    assert lab.params[KEY] == 1


def test_sub_cadence_dropout_between_samples_has_zero_footprint():
    # 600 s cadence: samples at 0, 600, 1200, ...; a 60 s dropout over [100, 160) hits none.
    for scope_draw, kind in ((0.0, "field"), (0.9, "star")):
        sc = _scene([_star(0), _star(1)], n=60, cadence=600)
        (lab,) = DropoutInjector().inject(sc, _Scripted([scope_draw, 100.0, 60.0]))
        assert lab.params["scope"] == kind
        assert (lab.t_start, lab.t_end) == (100, 160)
        assert lab.params[KEY] == 0
        assert sc.mask.all()


def test_dropout_covering_a_sample_counts_it_once_per_scope():
    # [590, 650) covers the sample at 600 only.
    sc = _scene([_star(0), _star(1)], n=60, cadence=600)
    (field,) = DropoutInjector().inject(sc, _Scripted([0.0, 590.0, 60.0]))
    assert field.params[KEY] == 1  # one column of the grid, not one per star
    assert not sc.mask[:, 1].any()
    sc = _scene([_star(0), _star(1)], n=60, cadence=600)
    (one,) = DropoutInjector().inject(sc, _Scripted([0.9, 590.0, 60.0]))
    assert one.params[KEY] == 1
    assert not sc.mask[0, 1] and sc.mask[1, 1]


def test_dropout_over_already_missing_samples_changes_nothing():
    sc = _scene([_star(0)], n=60, cadence=600)
    sc.mask[:, 1] = False
    (lab,) = DropoutInjector().inject(sc, _Scripted([0.9, 590.0, 60.0]))
    assert lab.params[KEY] == 0


def test_drift_starting_after_the_last_sample_has_zero_footprint():
    sc = _scene([_star(0)])
    (lab,) = TrackingDriftInjector().inject(sc, _Scripted([35970.0, 3600.0, 0.2]))
    assert lab.params[KEY] == 0
    assert np.all(sc.flux == 1000.0)


def test_drift_counts_the_persistent_tail_not_just_the_ramp():
    sc = _scene([_star(0)])
    (lab,) = TrackingDriftInjector().inject(sc, _Scripted([3600.0, 1800.0, 0.2]))
    # The ramp starts at 3600 (factor exactly 1 there, so that sample is unchanged), and the
    # factor reached at the end of the ramp stands to the end of the night.
    assert lab.params[KEY] == N - 3600 // CAD - 1


def test_cosmic_ray_footprint_is_one():
    sc = _scene([_star(0), _star(1)])
    labels = CosmicRayInjector(5.0).inject(sc, np.random.default_rng(1))
    assert labels
    assert all(lab.params[KEY] == 1 for lab in labels)


def test_hot_pixel_clipped_to_a_single_sample_has_footprint_one():
    sc = _scene([_star(0)])
    # starts one sample before the night's end: [35940, 36000) holds only the sample at 35940
    (lab,) = HotPixelInjector().inject(sc, _Scripted([35940.0, 600.0, 2.0]))
    assert lab.params[KEY] == 1


def test_footprint_ignores_cells_already_masked_out():
    sc = _scene([_star(0, "flare_star", flare_rate_per_day=1.0)])
    sc.mask[0, :] = False
    (lab,) = FlareInjector().inject(sc, _Scripted([18000.0]))
    assert lab.params[KEY] == 0


# --- every kind writes the field ---------------------------------------------------


def _eb():
    return _star(
        0,
        "eclipsing_binary",
        period_s=1e6,
        epoch_s=18000.0,
        depth_primary=0.2,
        depth_secondary=0.1,
        duration_s=2000.0,
    )


def _pair():
    return [_star(0), _star(1)]


def _planet():
    return Planet(0, 0, period_s=1e6, epoch_s=18000.0, depth=0.01, duration_s=1000.0)


def _puls():
    return _star(0, "sinusoidal", period_s=7200.0, amplitude=0.01, phase=0.5)


CASES = {
    "transit": lambda: (TransitInjector([_planet()]), _pair(), np.random.default_rng(0)),
    "eclipse": lambda: (EclipseInjector(), [_eb(), _star(1)], np.random.default_rng(0)),
    "pulsation": lambda: (PulsationInjector(), [_puls(), _star(1)], np.random.default_rng(0)),
    "flare": lambda: (FlareInjector(2.0), _pair(), np.random.default_rng(0)),
    "cosmic_ray": lambda: (CosmicRayInjector(0.5), _pair(), np.random.default_rng(4)),
    "cloud": lambda: (CloudInjector(3.0), _pair(), np.random.default_rng(4)),
    "tracking_drift": lambda: (TrackingDriftInjector(3.0), _pair(), np.random.default_rng(5)),
    "hot_pixel": lambda: (HotPixelInjector(0.7), _pair(), np.random.default_rng(6)),
    "dropout": lambda: (DropoutInjector(4.0), _pair(), np.random.default_rng(7)),
}


def test_cases_cover_every_kind():
    assert set(CASES) == set(ORIGIN_OF_KIND)


@pytest.mark.parametrize("kind", sorted(CASES))
def test_every_label_records_a_footprint(kind):
    inj, stars, rng = CASES[kind]()
    assert inj.kind == kind
    labels = inj.inject(_scene(stars), rng)
    assert labels
    for lab in labels:
        assert isinstance(lab.params[KEY], int) and not isinstance(lab.params[KEY], bool)
        assert 0 <= lab.params[KEY] <= N


@pytest.mark.parametrize("kind", sorted(CASES))
def test_footprint_matches_an_independent_diff_of_the_scene(kind):
    """A label that alone accounts for a star's (or the grid's) changes has that footprint."""
    inj, stars, rng = CASES[kind]()
    sc = _scene(stars)
    flux0, mask0 = sc.flux.copy(), sc.mask.copy()
    labels = inj.inject(sc, rng)
    changed = (sc.flux != flux0) | (mask0 & ~sc.mask)
    field = [lab for lab in labels if lab.star_id is None]
    checked = 0
    if field and len(labels) == 1:
        assert field[0].params[KEY] == int(np.count_nonzero(changed.any(axis=0)))
        checked += 1
    if not field:
        for row, star in enumerate(sc.stars):
            mine = [lab for lab in labels if lab.star_id == star.star_id]
            if len(mine) == 1:
                assert mine[0].params[KEY] == int(np.count_nonzero(changed[row]))
                checked += 1
    if kind in ("transit", "eclipse", "pulsation", "cosmic_ray", "hot_pixel", "flare"):
        assert checked  # these cases are built so at least one label stands alone


@pytest.mark.parametrize("kind", ["cloud", "tracking_drift"])
def test_field_wide_footprint_is_a_count_of_grid_columns(kind):
    inj, stars, _ = CASES[kind]()
    sc = _scene(stars)
    flux0 = sc.flux.copy()
    (lab,) = inj.inject(sc, _Scripted([18000.0, 1800.0, 0.2]))
    assert lab.params[KEY] == int(np.count_nonzero((sc.flux != flux0).any(axis=0)))
    assert lab.params[KEY] > 0


# --- dropout and drift against an independent recount ------------------------------


def _recount_columns(flux0, mask0, sc):
    """Grid columns where any star was observed before and changed (flux or mask)."""
    changed = ((sc.flux != flux0) | (mask0 & ~sc.mask)) & mask0
    return int(np.count_nonzero(changed.any(axis=0)))


@pytest.mark.parametrize("scope_draw", [0.0, 0.9], ids=["field", "star"])
def test_dropout_footprint_is_an_independent_recount_of_masked_samples(scope_draw):
    sc = _scene([_star(0), _star(1)], n=60, cadence=60)  # samples every 60 s
    sc.mask[1, 12:14] = False  # already missing for star 1: must not be counted for it
    mask0 = sc.mask.copy()
    # [600, 1500) covers samples 10..24 (15 columns)
    (lab,) = DropoutInjector().inject(sc, _Scripted([scope_draw, 600.0, 900.0]))
    newly = mask0 & ~sc.mask
    if lab.params["scope"] == "field":
        # star 0 loses all 15 samples, star 1 loses 13: columns where anything was lost
        expected = int(np.count_nonzero(newly.any(axis=0)))
        assert expected == 15
    else:
        expected = int(np.count_nonzero(newly[0]))  # _Scripted.integers picks row 0
        assert expected == 15
    assert lab.params[KEY] == expected


def test_dropout_footprint_counts_only_newly_masked_samples():
    sc = _scene([_star(0)], n=60, cadence=60)
    sc.mask[0, 12:14] = False
    mask0 = sc.mask.copy()
    (lab,) = DropoutInjector().inject(sc, _Scripted([0.9, 600.0, 900.0]))
    assert lab.params[KEY] == int(np.count_nonzero(mask0 & ~sc.mask)) == 13


def test_drift_footprint_is_an_independent_recount_of_changed_columns():
    sc = _scene(_pair())
    flux0, mask0 = sc.flux.copy(), sc.mask.copy()
    (lab,) = TrackingDriftInjector().inject(sc, _Scripted([3600.0, 1800.0, 0.2]))
    # recount from scratch: every sample whose flux is no longer its starting value
    expected = sum(1 for j in range(N) if (sc.flux[:, j] != flux0[:, j]).any())
    assert expected == _recount_columns(flux0, mask0, sc) == N - 3600 // CAD - 1
    assert lab.params[KEY] == expected


def test_drift_footprint_ignores_samples_masked_before_it():
    sc = _scene(_pair())
    sc.mask[:, 100:110] = False
    flux0, mask0 = sc.flux.copy(), sc.mask.copy()
    (lab,) = TrackingDriftInjector().inject(sc, _Scripted([3600.0, 1800.0, 0.2]))
    assert lab.params[KEY] == _recount_columns(flux0, mask0, sc) == N - 3600 // CAD - 1 - 10
