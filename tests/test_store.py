import numpy as np
import pytest

from skywatch.io import store
from skywatch.scoring.match import Match
from skywatch.scoring.metrics import KindScore
from skywatch.types import Detection
from skywatch.world.defaults import default_injectors
from skywatch.world.field import FieldConfig
from skywatch.world.survey import SurveyConfig, simulate


@pytest.fixture(scope="module")
def result():
    cfg = SurveyConfig(seed=3, n_nights=1, cadence_s=300, field=FieldConfig(n_stars=5))
    # planets are generated inside simulate; build injectors from a first pass
    first = simulate(cfg)
    sci, ins = default_injectors(first.planets)
    return simulate(cfg, sci, ins)


def test_round_trip(tmp_path, result):
    run = store.write_run(tmp_path, result)
    assert run.name == "seed3-5s-1n"
    curves = store.read_curves(run)
    assert set(curves) == set(result.curves)
    for sid, c in result.curves.items():
        np.testing.assert_array_equal(curves[sid].t, c.t)
        np.testing.assert_array_equal(curves[sid].flux, c.flux)
        np.testing.assert_array_equal(curves[sid].flux_err, c.flux_err)
        np.testing.assert_array_equal(curves[sid].quality, c.quality)
    assert store.read_truth(run) == result.truth
    assert store.read_stars(run) == result.stars
    assert store.read_planets(run) == result.planets


def test_manifest(tmp_path, result):
    run = store.write_run(tmp_path, result, run_id="x")
    m = store.read_manifest(run)
    for key in (
        "seed",
        "config",
        "skywatch_version",
        "created_at",
        "n_stars",
        "n_nights",
        "n_samples",
        "n_truth",
        "truth_counts",
    ):
        assert key in m
    assert m["seed"] == 3
    assert m["config"]["cadence_s"] == 300
    assert m["n_truth"] == len(result.truth)
    assert sum(m["truth_counts"].values()) == len(result.truth)


def test_overwrite_refused(tmp_path, result):
    store.write_run(tmp_path, result, run_id="x")
    with pytest.raises(FileExistsError):
        store.write_run(tmp_path, result, run_id="x")
    store.write_run(tmp_path, result, run_id="x", overwrite=True)


def test_detections_round_trip(tmp_path, result):
    run = store.write_run(tmp_path, result)
    a = Detection("a", "a-0", 1, 0, 10, 2.5, "flare")
    b = Detection("b", "b-0", None, 5, 9, 1.0, None)
    store.write_detections(run, [a, b])
    assert store.read_detections(run, "a") == [a]
    assert sorted(store.read_detections(run), key=lambda d: d.detector) == [a, b]
    a2 = Detection("a", "a-1", 2, 0, 10, 3.0)
    store.write_detections(run, [a2])
    assert store.read_detections(run, "a") == [a2]
    assert store.read_detections(run, "b") == [b]
    store.write_detections(run, [])
    assert len(store.read_detections(run)) == 2


def test_scores_round_trip(tmp_path, result):
    run = store.write_run(tmp_path, result)
    scores = [KindScore("a", "flare", 3, 2, 3, 0.6, 0.5, 0.55, 2, 1, 4, 2 / 3)]
    matches = [
        Match("a", "a-0", None, 0.0),
        Match("a", "a-1", "l", 0.5, "bounded"),
        Match("a", "a-2", "w", 0.1, "corroborating"),
    ]
    store.write_scores(
        run,
        scores,
        matches,
        {("a", "flare", "missed"): 3},
    )
    assert store.read_scores(run) == scores
    import pyarrow.parquet as pq

    assert pq.read_table(run / "matches.parquet")["grade"].to_pylist() == [
        None,
        "bounded",
        "corroborating",
    ]
    assert (run / "confusion.parquet").exists()
    assert (run / "matches.parquet").exists()


def test_read_curves_sorts_interleaved_rows(tmp_path, result):
    import pyarrow as pa
    import pyarrow.parquet as pq

    run = store.write_run(tmp_path, result)
    rows = [  # (star, t, flux), two stars interleaved, t shuffled
        (1, 300, 13.0),
        (2, 100, 21.0),
        (1, 100, 11.0),
        (2, 300, 23.0),
        (1, 200, 12.0),
        (2, 200, 22.0),
    ]
    tbl = pa.table(
        {
            "star_id": pa.array([r[0] for r in rows], pa.int64()),
            "t": pa.array([r[1] for r in rows], pa.int64()),
            "flux": pa.array([r[2] for r in rows], pa.float64()),
            "flux_err": pa.array([0.1] * 6, pa.float64()),
            "quality": pa.array([0] * 6, pa.uint8()),
        }
    )
    pq.write_table(tbl, run / "observations.parquet")
    curves = store.read_curves(run)
    np.testing.assert_array_equal(curves[1].t, [100, 200, 300])
    np.testing.assert_array_equal(curves[1].flux, [11.0, 12.0, 13.0])
    np.testing.assert_array_equal(curves[2].t, [100, 200, 300])
    np.testing.assert_array_equal(curves[2].flux, [21.0, 22.0, 23.0])


def test_params_numpy_round_trip(tmp_path, result):
    from skywatch.types import TruthLabel

    label = TruthLabel(
        "l0",
        "flare",
        "science",
        1,
        0,
        10,
        {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, 2.0]), "d": np.bool_(True)},
    )
    from dataclasses import replace

    run = store.write_run(tmp_path, replace(result, truth=[label]))
    (back,) = store.read_truth(run)
    assert back.params == {"a": 1.5, "b": 3, "c": [1.0, 2.0], "d": True}
    assert type(back.params["a"]) is float
    assert type(back.params["b"]) is int
    assert type(back.params["d"]) is bool
