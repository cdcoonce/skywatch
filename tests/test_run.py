import pytest

from skywatch import run
from skywatch.cli import main
from skywatch.detectors import REGISTRY, register
from skywatch.io import store
from skywatch.types import Detection
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


# --- guards added after adversarial review ---


@pytest.fixture
def fresh_run(tmp_path):
    cfg = SurveyConfig(seed=3, n_nights=1, cadence_s=120, field=FieldConfig(n_stars=40))
    return run.simulate_run(cfg, tmp_path)


@pytest.fixture
def registry_guard():
    saved = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(saved)


def _make_fake(name, detections_fn=lambda curves, cadence_s: [], seen=None):
    class Fake:
        def __init__(self, **kwargs):
            pass

        def detect(self, curves, cadence_s):
            if seen is not None:
                seen.append(cadence_s)
            return detections_fn(curves, cadence_s)

    Fake.name = name
    return Fake


def test_detect_kwargs_change_count(fresh_run):
    default = run.detect_run(fresh_run, "rolling_z")
    tuned = run.detect_run(fresh_run, "rolling_z", window=10, z_threshold=3.0)
    assert default != tuned


def test_detect_passes_manifest_cadence(fresh_run, registry_guard):
    seen: list[int] = []
    register(_make_fake("fake_cad", seen=seen))
    run.detect_run(fresh_run, "fake_cad")
    assert seen == [120]


def test_zero_detection_detector_gets_rows(fresh_run, registry_guard):
    register(_make_fake("fake_empty"))
    run.detect_run(fresh_run, "fake_empty")
    rows = [s for s in run.score_run(fresh_run) if s.detector == "fake_empty"]
    assert rows
    (allrow,) = [s for s in rows if s.kind == "all"]
    assert allrow.tp == 0
    assert allrow.fn == len(store.read_truth(fresh_run))


def test_detectors_matched_independently(fresh_run, registry_guard):
    label = next(t for t in store.read_truth(fresh_run) if t.star_id is not None)

    def one(name):
        det = Detection(name, "shared-0", label.star_id, label.t_start, label.t_end, 1.0)
        return lambda curves, cadence_s: [det]

    for name in ("fake_a", "fake_b"):
        register(_make_fake(name, one(name)))
        run.detect_run(fresh_run, name)
    scores = run.score_run(fresh_run)
    for name in ("fake_a", "fake_b"):
        (row,) = [s for s in scores if s.detector == name and s.kind == label.kind]
        assert row.tp == 1


def test_simulate_run_has_truth_and_planets(fresh_run):
    assert store.read_truth(fresh_run)
    assert store.read_manifest(fresh_run)["n_truth"] > 0
    assert store.read_planets(fresh_run)


def test_report_before_score_names_the_step(fresh_run):
    with pytest.raises(FileNotFoundError, match="skywatch score"):
        run.report_run(fresh_run)
    assert main(["report", "--run", str(fresh_run)]) == 2


def test_score_before_detect_names_the_step(fresh_run):
    with pytest.raises(FileNotFoundError, match="skywatch detect"):
        run.score_run(fresh_run)
    assert main(["score", "--run", str(fresh_run)]) == 2


# --- zero-footprint labels end to end ------------------------------------------------


def _rewrite_truth(run_dir, edit):
    """Apply ``edit(params: dict) -> dict`` to every label's params in truth.parquet."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    tbl = pq.read_table(run_dir / "truth.parquet")
    params = [json.dumps(edit(json.loads(p)), sort_keys=True) for p in tbl["params"].to_pylist()]
    idx = tbl.schema.get_field_index("params")
    pq.write_table(
        tbl.set_column(idx, "params", pa.array(params, pa.string())), run_dir / "truth.parquet"
    )


def test_simulated_truth_carries_a_footprint(fresh_run):
    truth = store.read_truth(fresh_run)
    assert truth and all(isinstance(t.params["footprint_samples"], int) for t in truth)


def test_report_shows_unobservable_labels_and_keeps_them_out_of_fn(fresh_run, registry_guard):
    register(_make_fake("fake_empty"))
    run.detect_run(fresh_run, "fake_empty")
    truth = store.read_truth(fresh_run)
    target = next(t for t in truth if t.kind == "cosmic_ray")
    _rewrite_truth(
        fresh_run,
        lambda p: {**p, "footprint_samples": 0} if p == target.params else p,
    )
    n_zero = sum(1 for t in store.read_truth(fresh_run) if t.params["footprint_samples"] == 0)
    assert n_zero >= 1
    rows = {s.kind: s for s in run.score_run(fresh_run) if s.detector == "fake_empty"}
    assert rows["all"].unobservable == n_zero
    assert rows["all"].fn == len(truth) - n_zero
    text = run.report_run(fresh_run)
    head = text.splitlines()[0].split()
    row = next(ln.split() for ln in text.splitlines() if ln.split()[:2] == ["fake_empty", "all"])
    assert row[head.index("unobs")] == str(n_zero)
    assert "unobservable" in text  # the legend says what the column means


def test_old_run_without_footprints_scores_as_before_with_one_warning(
    fresh_run, registry_guard, capsys
):
    register(_make_fake("fake_empty"))
    run.detect_run(fresh_run, "fake_empty")
    n = len(store.read_truth(fresh_run))
    _rewrite_truth(fresh_run, lambda p: {k: v for k, v in p.items() if k != "footprint_samples"})
    capsys.readouterr()
    rows = {s.kind: s for s in run.score_run(fresh_run) if s.detector == "fake_empty"}
    assert (rows["all"].fn, rows["all"].unobservable) == (n, 0)
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "footprint_samples" in err and str(n) in err


def test_run_with_footprints_scores_without_a_warning(fresh_run, registry_guard, capsys):
    register(_make_fake("fake_empty"))
    run.detect_run(fresh_run, "fake_empty")
    capsys.readouterr()
    run.score_run(fresh_run)
    assert capsys.readouterr().err == ""
