import pytest

from skywatch import run
from skywatch.io import store
from skywatch.world.field import FieldConfig
from skywatch.world.survey import SurveyConfig


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    cfg = SurveyConfig(seed=3, n_nights=1, cadence_s=120, field=FieldConfig(n_stars=8))
    return run.simulate_run(cfg, tmp_path_factory.mktemp("runs"))


def test_simulate_creates_files(run_dir):
    for name in ("manifest.json", "stars.parquet", "observations.parquet", "truth.parquet"):
        assert (run_dir / name).exists()


def test_detect_score_report(run_dir):
    n = run.detect_run(run_dir, "rolling_z")
    assert n >= 0
    assert (run_dir / "detections" / "rolling_z.parquet").exists()
    scores = run.score_run(run_dir)
    kinds = {s.kind for s in scores if s.detector == "rolling_z"}
    assert "all" in kinds
    text = run.report_run(run_dir)
    assert "rolling_z" in text and "all" in text
    assert "confusion" in text.lower()
    assert store.read_scores(run_dir) == scores


def test_detect_kwargs(run_dir):
    assert run.detect_run(run_dir, "rolling_z", window=10, z_threshold=3.0) >= 0


def test_unknown_detector(run_dir):
    with pytest.raises(ValueError, match="rolling_z"):
        run.detect_run(run_dir, "nope")
