"""The data layer behind `skywatch view`: no matplotlib in this file."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from skywatch import view
from skywatch.detectors.nights import split_nights
from skywatch.io import store
from skywatch.scoring.match import Match
from skywatch.types import Detection, LightCurve, TruthLabel
from skywatch.world.field import FieldConfig
from skywatch.world.survey import SurveyConfig, simulate

NIGHT = 28800
PERIOD = 86400  # night_length_s + night_gap_s


def _label(kind, star, t0, t1, k=0, night=0, footprint=5):
    origin = (
        "instrument"
        if kind in ("cloud", "cosmic_ray", "tracking_drift", "hot_pixel")
        else "science"
    )
    sid = "field" if star is None else star
    return TruthLabel(
        f"{kind}-n{night}-s{sid}-{k}", kind, origin, star, t0, t1, {"footprint_samples": footprint}
    )


TRUTH = [
    _label("flare", 1, 1000, 2000),
    _label("transit", 1, 28000, 30000, k=1),  # straddles the end of night 0
    _label("transit", 1, 40000, 41000, k=2),  # in the gap: neither night
    _label("cloud", None, 5000, 6000),  # field-wide, night 0
    _label("cosmic_ray", 2, PERIOD + 600, PERIOD + 1200, night=1),
    _label("flare", 3, PERIOD + 4000, PERIOD + 5000, night=1),
    _label("hot_pixel", 3, PERIOD + 20000, PERIOD + 21000, night=1),
    _label("tracking_drift", None, PERIOD + 9000, PERIOD + 10000, night=1),
    _label("flare", 0, PERIOD - 2000, PERIOD - 1000, k=9, footprint=0),  # gap, star 0
]

DETS = [
    Detection("alpha", "alpha-0", 1, 900, 1900, 5.0, "flare"),
    Detection("alpha", "alpha-1", 1, 28500, 29000, 4.0, None),
    Detection("alpha", "alpha-2", 1, 3000, 3600, 3.0, None),  # overlaps nothing
    Detection("alpha", "alpha-3", 2, 3000, 3600, 3.0, None),  # another star
    Detection("alpha", "alpha-4", 3, PERIOD + 4100, PERIOD + 4300, 6.0, "flare"),
    Detection("alpha", "alpha-5", 3, PERIOD + 20500, PERIOD + 20600, 2.0, "cosmic_ray"),
    Detection("beta", "beta-0", 1, 5200, 5300, 2.0, None),
    Detection("beta", "beta-1", 1, 5400, 5500, 2.0, None),
    Detection("field", "field-0", None, 5000, 5900, 9.0, "cloud"),
    Detection("field", "field-1", None, PERIOD + 9100, PERIOD + 9900, 9.0, "tracking_drift"),
]

MATCHES = [
    Match("alpha", "alpha-0", "flare-n0-s1-0", 0.9, "bounded"),
    Match("alpha", "alpha-1", "transit-n0-s1-1", 0.2, "touched"),
    Match("alpha", "alpha-2", None, 0.0, None),
    Match("alpha", "alpha-3", None, 0.0, None),
    Match("alpha", "alpha-4", "flare-n1-s3-0", 0.2, "touched"),
    Match("alpha", "alpha-5", None, 0.0, None),
    Match("beta", "beta-0", "cloud-n0-sfield-0", 0.1, "touched"),
    Match("beta", "beta-1", "cloud-n0-sfield-0", 0.1, "corroborating"),
    Match("field", "field-0", "cloud-n0-sfield-0", 0.8, "bounded"),
    Match("field", "field-1", "tracking_drift-n1-sfield-0", 0.8, "bounded"),
]


def _curves(cfg, stars):
    cadence = cfg.cadence_s
    out = {}
    for s in stars:
        ts = []
        for k in range(cfg.n_nights):
            ts.append(k * PERIOD + np.arange(0, NIGHT, cadence, dtype=np.int64))
        t = np.concatenate(ts)
        n = len(t)
        out[s.star_id] = LightCurve(
            s.star_id,
            t,
            np.full(n, 100.0 + s.star_id),
            np.full(n, 1.0),
            np.zeros(n, dtype=np.uint8),
        )
    return out


@pytest.fixture(scope="module")
def result():
    cfg = SurveyConfig(seed=1, n_nights=2, cadence_s=600, field=FieldConfig(n_stars=4))
    base = simulate(cfg)
    return replace(base, curves=_curves(cfg, base.stars), truth=list(TRUTH))


@pytest.fixture
def detected(tmp_path, result):
    run = store.write_run(tmp_path, result, run_id="tiny")
    store.write_detections(run, DETS)
    return run


@pytest.fixture
def scored(detected):
    store.write_scores(detected, [], MATCHES, {})
    return detected


def _ids(items):
    return [x.label_id for x in items]


def test_observations_are_the_nights_samples(scored):
    d = view.assemble_view(scored, 1, 1)
    assert d.night_start == PERIOD
    assert d.t.min() == PERIOD and d.t.max() < PERIOD + NIGHT
    assert len(d.t) == NIGHT // 600
    assert len(d.t) == len(d.flux) == len(d.flux_err) == len(d.quality)
    assert (d.flux == 101.0).all()


def test_night_bounds_agree_with_split_nights(scored, result):
    t = result.curves[1].t
    for k, sl in enumerate(split_nights(t, 600, gap_cadences=10)):
        d = view.assemble_view(scored, 1, k)
        np.testing.assert_array_equal(d.t, t[sl])


def test_truth_night_filter_includes_straddler_and_excludes_gap(scored):
    d = view.assemble_view(scored, 1, 0)
    ids = _ids(d.truth)
    assert "transit-n0-s1-1" in ids  # straddles the night end
    assert "transit-n0-s1-2" not in ids  # in the gap
    assert "flare-n0-s1-0" in ids
    assert not any(i.startswith(("cosmic_ray", "tracking_drift")) for i in ids)


def test_truth_straddling_the_night_start_is_included(tmp_path, result):
    early = _label("flare", 1, PERIOD - 500, PERIOD + 600, k=7, night=1)
    run = store.write_run(tmp_path, replace(result, truth=[*TRUTH, early]), run_id="edge")
    assert early.label_id in _ids(view.assemble_view(run, 1, 1).truth)
    assert early.label_id not in _ids(view.assemble_view(run, 1, 0).truth)  # starts in the gap


def test_label_ending_exactly_at_night_start_is_outside(tmp_path, result):
    touching = _label("flare", 1, PERIOD - 500, PERIOD, k=8, night=1)
    run = store.write_run(tmp_path, replace(result, truth=[*TRUTH, touching]), run_id="edge")
    assert touching.label_id not in _ids(view.assemble_view(run, 1, 1).truth)


def test_field_wide_labels_appear_for_every_star(scored):
    for sid in range(4):
        d = view.assemble_view(scored, sid, 0)
        assert "cloud-n0-sfield-0" in _ids(d.truth)
        d1 = view.assemble_view(scored, sid, 1)
        assert "tracking_drift-n1-sfield-0" in _ids(d1.truth)


def test_truth_is_sorted_and_carries_observability(scored):
    d = view.assemble_view(scored, 1, 0)
    assert [(x.t_start, x.label_id) for x in d.truth] == sorted(
        (x.t_start, x.label_id) for x in d.truth
    )
    d0 = view.assemble_view(scored, 0, 0)
    assert all(x.observable for x in d0.truth)


def test_lanes_default_to_every_detector_sorted(scored):
    d = view.assemble_view(scored, 1, 0)
    assert [lane.detector for lane in d.lanes] == ["alpha", "beta", "field"]


def test_detections_filtered_by_star_night_and_field_wide(scored):
    d = view.assemble_view(scored, 1, 0)
    lanes = {lane.detector: lane for lane in d.lanes}
    assert [x.det_id for x in lanes["alpha"].detections] == ["alpha-0", "alpha-2", "alpha-1"]
    assert [x.det_id for x in lanes["field"].detections] == ["field-0"]  # field-1 is night 1
    d2 = view.assemble_view(scored, 2, 0)
    assert [x.det_id for x in {lane.detector: lane for lane in d2.lanes}["alpha"].detections] == [
        "alpha-3"
    ]


def test_grades_join_from_matches(scored):
    d = view.assemble_view(scored, 1, 0)
    assert d.scored
    by = {x.det_id: x for lane in d.lanes for x in lane.detections}
    assert by["alpha-0"].grade == "bounded" and by["alpha-0"].label_id == "flare-n0-s1-0"
    assert by["alpha-1"].grade == "touched" and by["alpha-1"].label_id == "transit-n0-s1-1"
    assert by["beta-1"].grade == "corroborating"
    assert by["alpha-2"].grade == "false_alarm" and by["alpha-2"].label_id is None
    assert by["field-0"].grade == "bounded"


def test_lane_counts_and_summary(scored):
    d = view.assemble_view(scored, 1, 0)
    lanes = {lane.detector: lane for lane in d.lanes}
    assert lanes["alpha"].counts == {
        "bounded": 1,
        "touched": 1,
        "corroborating": 0,
        "false_alarm": 1,
    }
    assert lanes["alpha"].summary(scored=True) == "bounded 1 · touched 1 · corr 0 · fp 1"
    assert lanes["alpha"].summary(scored=False) == "unscored 3"


def test_unscored_run_assembles_with_grade_none(detected):
    d = view.assemble_view(detected, 1, 0)
    assert not d.scored
    grades = {x.grade for lane in d.lanes for x in lane.detections}
    assert grades == {None}
    assert all(x.label_id is None for lane in d.lanes for x in lane.detections)


def test_detector_scored_after_the_matches_were_written_is_unscored(scored):
    store.write_detections(scored, [Detection("late", "late-0", 1, 100, 200, 1.0)])
    d = view.assemble_view(scored, 1, 0)
    by = {lane.detector: lane for lane in d.lanes}
    assert by["late"].detections[0].grade is None
    assert by["alpha"].detections[0].grade is not None


def test_detector_subset_and_unknown_detector(scored):
    d = view.assemble_view(scored, 1, 0, detectors=["field", "alpha"])
    assert [lane.detector for lane in d.lanes] == ["field", "alpha"]  # caller's order
    with pytest.raises(ValueError, match="nope"):
        view.assemble_view(scored, 1, 0, detectors=["nope"])


def test_lane_without_match_rows_summarises_as_unscored(scored):
    # beta's match rows are gone (it was detected after scoring): its lane is grey, and its
    # title must not claim "fp 0"
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = scored / "matches.parquet"
    table = pq.read_table(path)
    pq.write_table(table.filter(pc.not_equal(table["detector"], "beta")), path)
    d = view.assemble_view(scored, 1, 0)
    lanes = {lane.detector: lane for lane in d.lanes}
    assert d.scored
    assert all(x.grade is None for x in lanes["beta"].detections)
    assert lanes["beta"].summary(d.scored) == "unscored 2"
    assert lanes["alpha"].summary(d.scored) == "bounded 1 · touched 1 · corr 0 · fp 1"
    assert lanes["field"].summary(d.scored) == "bounded 1 · touched 0 · corr 0 · fp 0"


def test_scored_lane_with_no_detections_keeps_zero_counts(scored):
    store.write_empty_detections(scored, "quiet")
    d = view.assemble_view(scored, 1, 0)
    quiet = next(lane for lane in d.lanes if lane.detector == "quiet")
    assert quiet.summary(d.scored) == "bounded 0 · touched 0 · corr 0 · fp 0"


def test_run_without_detections_has_no_lanes(tmp_path, result):
    run = store.write_run(tmp_path, result, run_id="bare")
    d = view.assemble_view(run, 1, 0)
    assert d.lanes == () and not d.scored


def test_bad_star_and_night(scored):
    with pytest.raises(ValueError, match="star"):
        view.assemble_view(scored, 99, 0)
    with pytest.raises(ValueError, match="night"):
        view.assemble_view(scored, 1, 2)
    with pytest.raises(ValueError, match="night"):
        view.assemble_view(scored, 1, -1)


def test_assemble_and_pick_do_not_import_matplotlib(scored):
    # a fresh interpreter: the module under test must not pull matplotlib in at import,
    # assembly or pick time (only render_view may)
    code = (
        "import sys\n"
        "from skywatch import view, cli, run\n"
        f"d = view.assemble_view({str(scored)!r}, 1, 0)\n"
        f"view.pick_view({str(scored)!r})\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_pick_prefers_most_distinct_kinds_then_detections(scored):
    # star 1 night 0: flare, transit, cloud (3 kinds, 6 detections in view)
    # star 3 night 1: flare, hot_pixel, tracking_drift (3 kinds, 3 detections)
    p = view.pick_view(scored)
    assert (p.star_id, p.night) == (1, 0)
    assert p.kinds == ("cloud", "flare", "transit")
    assert p.n_detections == 6


def test_pick_tie_on_kinds_breaks_on_detections(tmp_path, result):
    run = store.write_run(tmp_path, result, run_id="t")
    extra = [
        Detection("alpha", f"alpha-x{i}", 3, PERIOD + 100 * i, PERIOD + 100 * i + 50, 1.0)
        for i in range(9)
    ]
    store.write_detections(run, [*DETS, *extra])
    p = view.pick_view(run)
    assert (p.star_id, p.night) == (3, 1)
    assert p.kinds == ("flare", "hot_pixel", "tracking_drift")


def test_pick_tie_on_kinds_and_detections_breaks_on_lowest_star(tmp_path, result):
    run = store.write_run(tmp_path, result, run_id="t")  # no detections at all
    p = view.pick_view(run)
    assert (p.star_id, p.night) == (1, 0)  # star 1 (3 kinds) beats star 3 (3 kinds) by id


def test_pick_ties_on_star_break_on_lowest_night(tmp_path, result):
    truth = [
        _label("flare", 2, 100, 200),
        _label("flare", 2, PERIOD + 100, PERIOD + 200, night=1),
    ]
    run = store.write_run(tmp_path, replace(result, truth=truth), run_id="t")
    p = view.pick_view(run)
    assert (p.star_id, p.night) == (2, 0)


def test_pick_counts_field_wide_kinds_for_every_star(tmp_path, result):
    truth = [_label("cloud", None, 100, 900), _label("flare", 3, 100, 200)]
    run = store.write_run(tmp_path, replace(result, truth=truth), run_id="t")
    p = view.pick_view(run)
    assert (p.star_id, p.night) == (3, 0) and p.kinds == ("cloud", "flare")


def test_pick_ignores_unobservable_labels(tmp_path, result):
    # star 2's third kind is unobservable (footprint 0): invisible in the flux, so it must
    # not make star 2 the picture. Counted, star 2 would win with 3 kinds; ignored, stars 1
    # and 2 tie on 2 kinds and the lowest id wins.
    truth = [
        _label("flare", 1, 100, 200),
        _label("transit", 1, 300, 400, k=1),
        _label("flare", 2, 100, 200),
        _label("transit", 2, 300, 400, k=1),
        _label("hot_pixel", 2, 500, 600, k=2, footprint=0),
    ]
    run = store.write_run(tmp_path, replace(result, truth=truth), run_id="t")
    p = view.pick_view(run)
    assert (p.star_id, p.night) == (1, 0)
    assert p.kinds == ("flare", "transit")


def test_pick_ignores_unobservable_field_wide_labels(tmp_path, result):
    truth = [_label("flare", 3, 100, 200), _label("cloud", None, 100, 900, footprint=0)]
    run = store.write_run(tmp_path, replace(result, truth=truth), run_id="t")
    assert view.pick_view(run).kinds == ("flare",)


def test_pick_respects_detector_subset(scored):
    p = view.pick_view(scored, detectors=["field"])
    assert p.n_detections == 1  # only field-0 touches star 1 night 0


def test_pick_describe_mentions_every_part(scored):
    text = view.pick_view(scored).describe()
    assert "star 1" in text and "night 0" in text and "flare" in text and "6 detections" in text


def test_default_out_path(tmp_path):
    assert (
        view.default_out_path(tmp_path / "r", 5, 1) == tmp_path / "r" / "views" / "star5-night1.png"
    )
