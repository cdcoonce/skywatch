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
    assert ms["d1"].label_id == "b" and ms["d1"].iou == 0.8
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


def test_low_iou_rejected():
    assert match_detections([det("x", 1, 0, 100)], [lab("o", 1, 90, 190)], C)[0].label_id is None


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
