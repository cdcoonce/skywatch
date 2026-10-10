import numpy as np
import pytest

from skywatch.types import Star, TruthLabel
from skywatch.world.glitches import (
    CLOUD_DURATION_S,
    HOT_PIXEL_DURATION_S,
    CloudInjector,
    CosmicRayInjector,
    DropoutInjector,
    HotPixelInjector,
    TrackingDriftInjector,
)
from skywatch.world.scene import Injector, Scene

N_SAMPLES = 600  # 10 h at 60 s
CADENCE = 60


def _stars(n: int = 4) -> list[Star]:
    return [
        Star(
            star_id=i + 10,
            flux0=1000.0 * (i + 1),
            noise_floor=0.01,
            variability="constant",
            ra=0.0,
            dec=0.0,
        )
        for i in range(n)
    ]


def _scene(night: int = 0, n_samples: int = N_SAMPLES) -> Scene:
    return Scene.quiet(
        _stars(), t_start=1000, n_samples=n_samples, cadence_s=CADENCE, night_index=night
    )


def _rng(seed: int = 1) -> np.random.Generator:
    return np.random.default_rng(seed)


ALL = [
    lambda **kw: CosmicRayInjector(**kw),
    lambda **kw: CloudInjector(**kw),
    lambda **kw: TrackingDriftInjector(**kw),
    lambda **kw: HotPixelInjector(**kw),
    lambda **kw: DropoutInjector(**kw),
]
ZERO = [
    CosmicRayInjector(0.0),
    CloudInjector(0.0),
    TrackingDriftInjector(0.0),
    HotPixelInjector(0.0),
    DropoutInjector(0.0),
]
HIGH = [
    CosmicRayInjector(5.0),
    CloudInjector(4.0),
    TrackingDriftInjector(3.0),
    HotPixelInjector(2.0),
    DropoutInjector(6.0),
]


def _span_idx(sc: Scene, lab: TruthLabel) -> np.ndarray:
    return (sc.t >= lab.t_start) & (sc.t < lab.t_end)


@pytest.mark.parametrize("inj", ZERO + HIGH)
def test_satisfies_injector_protocol(inj):
    assert isinstance(inj, Injector)


@pytest.mark.parametrize("inj", HIGH)
def test_labels_are_instrument_valid_and_unique(inj):
    sc = _scene()
    labels = inj.inject(sc, _rng())
    assert labels
    ids = [lab.label_id for lab in labels]
    assert len(ids) == len(set(ids))
    for lab in labels:
        assert isinstance(lab, TruthLabel)
        assert lab.origin == "instrument" and lab.kind == inj.kind
        assert lab.label_id.startswith(f"{inj.kind}-n0-s")


@pytest.mark.parametrize("inj", ZERO)
def test_zero_rate_is_noop(inj):
    sc = _scene()
    flux, mask = sc.flux.copy(), sc.mask.copy()
    assert inj.inject(sc, _rng()) == []
    assert np.array_equal(sc.flux, flux) and np.array_equal(sc.mask, mask)


@pytest.mark.parametrize("inj", HIGH)
def test_deterministic_for_seed(inj):
    a, b = _scene(), _scene()
    la, lb = inj.inject(a, _rng(7)), inj.inject(b, _rng(7))
    assert la == lb
    assert np.array_equal(a.flux, b.flux) and np.array_equal(a.mask, b.mask)


@pytest.mark.parametrize("inj", HIGH)
def test_spans_inside_night_and_ordered(inj):
    sc = _scene()
    night_end = int(sc.t[-1]) + CADENCE
    for seed in range(5):
        for lab in inj.inject(_scene(), _rng(seed)):
            assert int(sc.t[0]) <= lab.t_start < lab.t_end <= night_end


def test_cosmic_ray_changes_exactly_one_sample_by_amplitude():
    sc = _scene()
    base = sc.flux.copy()
    labels = CosmicRayInjector(5.0).inject(sc, _rng(3))
    expected = base.copy()
    for lab in labels:
        assert lab.star_id is not None
        r, c = sc.row(lab.star_id), int(np.flatnonzero(sc.t == lab.t_start)[0])
        assert lab.t_end - lab.t_start == CADENCE
        assert 0.2 <= lab.params["amplitude"] <= 5.0
        expected[r, c] += lab.params["amplitude"] * base[r, c]
    assert np.allclose(sc.flux, expected)
    # single hit: exactly one changed sample
    sc1 = _scene()
    one = CosmicRayInjector(0.0)
    assert one.inject(sc1, _rng()) == []
    for seed in range(50):
        sc2 = _scene()
        labs = CosmicRayInjector(0.3).inject(sc2, _rng(seed))
        changed = np.argwhere(sc2.flux != _scene().flux)
        if len(labs) == 1:
            assert len(changed) == 1
            assert sc2.stars[changed[0][0]].star_id == labs[0].star_id
            break
    else:
        pytest.fail("no single-hit seed found")


def test_cloud_dims_all_rows_min_at_tmid():
    sc = _scene()
    base = sc.flux.copy()
    labels = CloudInjector(4.0).inject(sc, _rng(2))
    ratio = sc.flux / base
    assert np.allclose(ratio, ratio[0])  # shared across stars
    lab = labels[0]
    assert lab.star_id is None
    if len(labels) == 1:
        inside = _span_idx(sc, lab)
        assert np.all(ratio[:, ~inside] == 1.0)
        d, tm, dur = lab.params["depth"], lab.params["t_mid"], lab.params["duration_s"]
        assert 0.05 <= d <= 0.6 and 300 <= dur <= 3600
        k = int(np.argmin(np.abs(sc.t - tm)))
        w = 0.5 * (1 + np.cos(np.pi * (sc.t[k] - tm) / (dur / 2)))
        assert ratio[0, k] == pytest.approx(1 - d * w)
        assert ratio[0].min() >= 1 - d - 1e-12


def _durations(inj, n_scenes=100):
    out = []
    for seed in range(n_scenes):
        out += [lab.params["duration_s"] for lab in inj.inject(_scene(), _rng(seed))]
    return out


def test_cloud_durations_stay_inside_the_exported_law():
    lo, hi = CLOUD_DURATION_S
    durs = _durations(CloudInjector(20.0))
    assert len(durs) >= 2000
    assert lo <= min(durs) and max(durs) <= hi
    # the draws actually span the law: a narrowed literal would also be a drift
    assert min(durs) < lo + 100 and max(durs) > hi - 100


def test_hot_pixel_durations_stay_inside_their_own_law():
    lo, hi = HOT_PIXEL_DURATION_S
    durs = _durations(HotPixelInjector(6.0), n_scenes=100)
    assert len(durs) >= 2000
    assert lo <= min(durs) and max(durs) <= hi
    assert min(durs) < lo + 100 and max(durs) > hi - 100


def test_cloud_single_exact():
    for seed in range(100):
        sc = _scene()
        labs = CloudInjector(0.8).inject(sc, _rng(seed))
        if len(labs) == 1 and not labs[0].params.get("clipped"):
            lab = labs[0]
            ratio = sc.flux[0] / 1000.0
            assert np.all(ratio[~_span_idx(sc, lab)] == 1.0)
            tm, dur, d = lab.params["t_mid"], lab.params["duration_s"], lab.params["depth"]
            w = np.where(
                _span_idx(sc, lab), 0.5 * (1 + np.cos(np.pi * (sc.t - tm) / (dur / 2))), 0.0
            )
            assert np.allclose(ratio, 1 - d * w)
            return
    pytest.fail("no unclipped single cloud found")


def test_drift_factor_at_start_end_and_after():
    for seed in range(200):
        sc = _scene()
        labs = TrackingDriftInjector(0.3).inject(sc, _rng(seed))
        if len(labs) == 1 and not labs[0].params.get("clipped"):
            lab = labs[0]
            slope = lab.params["slope"]
            ratio = sc.flux / _scene().flux
            assert np.allclose(ratio, ratio[0])
            assert lab.star_id is None
            assert np.all(ratio[0, sc.t < lab.t_start] == 1.0)
            at_start = np.flatnonzero(sc.t >= lab.t_start)[0]
            frac = (sc.t[at_start] - lab.t_start) / lab.params["duration_s"]
            assert ratio[0, at_start] == pytest.approx(1 + slope * frac)
            after = sc.t >= lab.t_end
            assert after.any()
            assert np.allclose(ratio[0, after], 1 + slope)
            assert lab.t_end - lab.t_start == lab.params["duration_s"]
            return
    pytest.fail("no unclipped single drift found")


def test_hot_pixel_overwrites_with_value():
    sc = _scene()
    labels = HotPixelInjector(2.0).inject(sc, _rng(4))
    assert labels
    covered = np.zeros_like(sc.mask)
    for lab in labels:
        r = sc.row(lab.star_id)
        idx = _span_idx(sc, lab)
        flux0 = sc.stars[r].flux0
        assert 1.5 * flux0 <= lab.params["value"] <= 4 * flux0
        covered[r] |= idx
    # rows untouched where no span covers
    assert np.all(
        sc.flux[~covered]
        == np.repeat([s.flux0 for s in sc.stars], N_SAMPLES).reshape(4, -1)[~covered]
    )
    # an event on a star with a single label is exactly `value` over its span
    for _ in range(1):
        sc = _scene()
        labs = HotPixelInjector(0.6).inject(sc, _rng(11))
        by_star: dict[int, list[TruthLabel]] = {}
        for lab in labs:
            by_star.setdefault(lab.star_id, []).append(lab)
        singles = [v[0] for v in by_star.values() if len(v) == 1]
        assert singles
        for lab in singles:
            assert np.all(sc.flux[sc.row(lab.star_id), _span_idx(sc, lab)] == lab.params["value"])


def test_dropout_masks_span_and_leaves_flux():
    for seed in range(100):
        sc = _scene()
        base = sc.flux.copy()
        labs = DropoutInjector(1.0, field_wide_fraction=0.5).inject(sc, _rng(seed))
        assert np.array_equal(sc.flux, base)
        if len(labs) == 1:
            lab = labs[0]
            idx = _span_idx(sc, lab)
            if lab.params["scope"] == "field":
                assert lab.star_id is None
                assert not sc.mask[:, idx].any() and sc.mask[:, ~idx].all()
            else:
                r = sc.row(lab.star_id)
                assert not sc.mask[r, idx].any()
                assert sc.mask.sum() == sc.mask.size - idx.sum()
            assert 60 <= lab.params["duration_s"] <= 1800


def test_dropout_scope_extremes():
    sc = _scene()
    labs = DropoutInjector(8.0, field_wide_fraction=1.0).inject(sc, _rng(1))
    assert labs and all(lab.star_id is None and lab.params["scope"] == "field" for lab in labs)
    sc = _scene()
    labs = DropoutInjector(8.0, field_wide_fraction=0.0).inject(sc, _rng(1))
    assert labs and all(lab.star_id is not None and lab.params["scope"] == "star" for lab in labs)


@pytest.mark.parametrize(
    "inj", [CloudInjector(40.0), TrackingDriftInjector(40.0), DropoutInjector(40.0)]
)
def test_clipping_flagged_at_night_end(inj):
    sc = _scene(n_samples=60)  # 1 h night, spans long enough to cross
    night_end = int(sc.t[-1]) + CADENCE
    labs = inj.inject(sc, _rng(0))
    clipped = [lab for lab in labs if lab.params.get("clipped")]
    assert clipped
    assert any(lab.t_end == night_end for lab in clipped)
    assert all(lab.t_end <= night_end for lab in labs)


def test_label_id_format_and_per_star_counter():
    sc = _scene(night=3)
    labs = HotPixelInjector(3.0).inject(sc, _rng(2))
    per: dict[int, list[str]] = {}
    for lab in labs:
        per.setdefault(lab.star_id, []).append(lab.label_id)
    for sid, ids in per.items():
        assert ids == [f"hot_pixel-n3-s{sid}-{k}" for k in range(len(ids))]
    labs = CloudInjector(5.0).inject(_scene(night=3), _rng(2))
    assert [lab.label_id for lab in labs] == [f"cloud-n3-sfield-{k}" for k in range(len(labs))]
