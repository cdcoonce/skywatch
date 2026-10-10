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
    scores = [KindScore("a", "flare", 1, 2, 3, 0.25, 0.5, 0.3)]
    store.write_scores(
        run,
        scores,
        [Match("a", "a-0", None, 0.0), Match("a", "a-1", "l", 0.5)],
        {("a", "flare", "missed"): 3},
    )
    assert store.read_scores(run) == scores
    assert (run / "confusion.parquet").exists()
    assert (run / "matches.parquet").exists()
