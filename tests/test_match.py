from skywatch.scoring.match import Match, match_detections, span_iou
from skywatch.types import Detection, TruthLabel

C = 60


def det(did, star, a, b, score=1.0, detector="d"):
    return Detection(detector, did, star, a, b, score)


def lab(lid, star, a, b, kind="transit"):
    origin = "science" if kind == "transit" else "instrument"
    return TruthLabel(lid, kind, origin, star, a, b)


def test_span_iou():
    assert span_iou(0, 10, 0, 10) == 1.0
    assert span_iou(0, 10, 5, 15) == 5 / 15
    assert span_iou(0, 10, 10, 20) == 0.0
    assert span_iou(0, 10, 2, 4) == 0.2


def test_highest_iou_label_wins_and_one_to_one():
    labels = [lab("a", 1, 0, 100), lab("b", 1, 0, 40)]
    dets = [det("d1", 1, 0, 50, score=2.0), det("d2", 1, 0, 50, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, labels, C)}
    assert ms["d1"].label_id == "b" and ms["d1"].iou == 0.8 and ms["d1"].grade == "bounded"
    assert ms["d2"].label_id == "a" and ms["d2"].iou == 0.5
    # only one label: second detection unmatched
    ms = {m.det_id: m for m in match_detections(dets, [labels[0]], C)}
    assert ms["d1"].label_id == "a"
    assert ms["d2"] == Match("d", "d2", None, 0.0)


def test_star_mismatch_and_field_wide():
    wide = lab("w", None, 0, 100, kind="cloud")
    assert match_detections([det("x", 7, 0, 100)], [wide], C)[0].label_id == "w"
    one = lab("o", 1, 0, 100)
    assert match_detections([det("x", 2, 0, 100)], [one], C)[0].label_id is None


def test_disjoint_rejected():
    assert match_detections([det("x", 1, 0, 100)], [lab("o", 1, 100, 200)], C)[0].label_id is None


def test_single_cadence_tolerance():
    cr = lab("cr", 1, 600, 660, kind="cosmic_ray")
    # detection one cadence away, IoU 0
    assert match_detections([det("x", 1, 660, 720)], [cr], C)[0].label_id == "cr"
    assert match_detections([det("x", 1, 540, 600)], [cr], C)[0].label_id == "cr"
    # two cadences away: no
    assert match_detections([det("x", 1, 720, 780)], [cr], C)[0].label_id is None
    # tolerance not applied to multi-cadence labels
    multi = lab("m", 1, 600, 720)
    assert match_detections([det("x", 1, 720, 780)], [multi], C)[0].label_id is None


def test_detectors_independent():
    labels = [lab("a", 1, 0, 100)]
    dets = [det("p", 1, 0, 100, detector="A"), det("q", 1, 0, 100, detector="B")]
    ms = match_detections(dets, labels, C)
    assert sorted((m.detector, m.label_id) for m in ms) == [("A", "a"), ("B", "a")]


def test_deterministic_ties():
    labels = [lab("b", 1, 0, 100), lab("a", 1, 0, 100)]
    ms = match_detections([det("x", 1, 0, 100)], labels, C)
    assert ms[0].label_id == "a"


def test_none_star_detection_vs_labels():
    d = det("x", None, 0, 100)
    assert match_detections([d], [lab("o", 1, 0, 100)], C)[0].label_id is None
    wide = lab("w", None, 0, 100, kind="cloud")
    assert match_detections([d], [wide], C)[0].label_id == "w"


def test_bounded_vs_touched_grade():
    label = lab("o", 1, 0, 1000)
    ms = match_detections([det("b", 1, 0, 400), det("t", 1, 900, 1000, score=0.5)], [label], C)
    got = {m.det_id: m for m in ms}
    assert got["b"].grade == "bounded" and got["b"].label_id == "o"
    # label already taken by "b" and star-level: "t" is unmatched, not corroborating
    assert got["t"] == Match("d", "t", None, 0.0, None)
    (m,) = match_detections([det("t", 1, 900, 1000)], [label], C)
    assert m.grade == "touched" and m.label_id == "o" and m.iou == 0.1


def test_single_cadence_tolerance_is_bounded():
    cr = lab("cr", 1, 600, 660, kind="cosmic_ray")
    (m,) = match_detections([det("x", 1, 660, 720)], [cr], C)
    assert m.grade == "bounded"


def test_prefers_bounded_over_touched():
    touched = lab("a", 1, 40, 1000)  # overlaps det 0..50 by 10: touched, id sorts first
    bounded = lab("z", 1, 0, 40)  # inter 40 / union 50 = 0.8: bounded
    (m,) = match_detections([det("x", 1, 0, 50)], [touched, bounded], C)
    assert m.label_id == "z" and m.grade == "bounded"


def test_falls_back_to_touched_when_bounded_taken():
    big = lab("big", 1, 0, 1000)
    other = lab("o2", 1, 0, 40)
    dets = [det("d1", 1, 0, 40, score=2.0), det("d2", 1, 0, 40, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, [big, other], C)}
    assert ms["d1"].label_id == "o2" and ms["d1"].grade == "bounded"
    assert ms["d2"].label_id == "big" and ms["d2"].grade == "touched"


def test_field_wide_label_corroborated_by_later_detections():
    wide = lab("w", None, 0, 1000, kind="cloud")
    dets = [det(f"d{s}", s, 0, 100, score=3.0 - s) for s in (1, 2, 3)]
    ms = {m.det_id: m for m in match_detections(dets, [wide], C)}
    assert ms["d1"].grade == "touched" and ms["d1"].label_id == "w"
    for k in ("d2", "d3"):
        assert ms[k].grade == "corroborating" and ms[k].label_id == "w"


def test_unclaimed_label_preferred_over_corroborating():
    wide = lab("w", None, 0, 1000, kind="cloud")
    own = lab("o", 2, 0, 100)
    dets = [det("d1", 1, 0, 100, score=2.0), det("d2", 2, 0, 100, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, [wide, own], C)}
    assert ms["d2"].label_id == "o" and ms["d2"].grade == "bounded"


def test_star_level_labels_stay_one_to_one():
    label = lab("o", 1, 0, 1000)
    dets = [det("d1", 1, 0, 500, score=2.0), det("d2", 1, 0, 600, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, [label], C)}
    assert ms["d1"].grade == "bounded"
    assert ms["d2"].label_id is None and ms["d2"].grade is None


def test_single_cadence_bounded_beats_higher_iou_touched():
    """The bounded-first preference is only observable through the single-cadence rule."""
    a = TruthLabel("A", "cosmic_ray", "instrument", 1, 600, 660)
    b = TruthLabel("B", "flare", "science", 1, 0, 10000)
    (m,) = match_detections([Detection("d", "x", 1, 660, 1000, 1.0)], [a, b], 60)
    assert m.label_id == "A" and m.grade == "bounded"
