from skywatch.scoring.match import Match, match_detections, span_iou
from skywatch.types import ORIGIN_OF_KIND, Detection, TruthLabel

C = 60


def det(did, star, a, b, score=1.0, detector="d"):
    return Detection(detector, did, star, a, b, score)


def lab(lid, star, a, b, kind="transit"):
    return TruthLabel(lid, kind, ORIGIN_OF_KIND[kind], star, a, b)


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
    # abutting spans are only matched by the edge pass; with it off, or with a real gap, no
    o = lab("o", 1, 100, 200)
    assert (
        match_detections([det("x", 1, 0, 100)], [o], C, edge_tolerance_cadences=0)[0].label_id
        is None
    )
    assert match_detections([det("x", 1, 0, 40)], [o], C)[0].label_id is None


def test_single_cadence_tolerance():
    cr = lab("cr", 1, 600, 660, kind="cosmic_ray")
    # detection one cadence away, IoU 0
    assert match_detections([det("x", 1, 660, 720)], [cr], C)[0].label_id == "cr"
    assert match_detections([det("x", 1, 540, 600)], [cr], C)[0].label_id == "cr"
    # two cadences away: no
    assert match_detections([det("x", 1, 720, 780)], [cr], C)[0].label_id is None
    # the single-cadence tolerance (bounded) is not applied to multi-cadence labels; the
    # edge pass (touched) covers them, and is off at edge_tolerance_cadences=0
    multi = lab("m", 1, 600, 720)
    assert (
        match_detections([det("x", 1, 720, 780)], [multi], C, edge_tolerance_cadences=0)[0].label_id
        is None
    )


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


# --- edge tolerance (issue #11) -------------------------------------------------
#
# Edge semantics under test: with edge_tolerance_cadences = k, a detection that does
# not overlap a label at all but sits within k cadences of one of its edges is
# eligible for that label as "touched". The gap is the distance between the two
# half-open spans: det.t_start - label.t_end after the label, label.t_start -
# det.t_end before it; eligible when 0 <= gap < k * cadence_s (0 = abutting, i.e.
# the first sample after / last sample before the label).


def test_exit_edge_detection_on_multi_cadence_label_is_touched():
    hot = lab("h", 1, 600, 1200, kind="hot_pixel")
    # starts at the label's t_end: the first sample after the plateau (the step down)
    (m,) = match_detections([det("x", 1, 1200, 1260)], [hot], C)
    assert (m.label_id, m.grade, m.iou) == ("h", "touched", 0.0)


def test_entry_edge_detection_is_touched():
    hot = lab("h", 1, 600, 1200, kind="hot_pixel")
    (m,) = match_detections([det("x", 1, 540, 600)], [hot], C)
    assert (m.label_id, m.grade, m.iou) == ("h", "touched", 0.0)


def test_detection_two_cadences_after_is_still_unmatched():
    hot = lab("h", 1, 600, 1200, kind="hot_pixel")
    (m,) = match_detections([det("x", 1, 1260, 1320)], [hot], C)
    assert m == Match("d", "x", None, 0.0, None)
    (m,) = match_detections([det("x", 1, 480, 540)], [hot], C)  # two cadences before
    assert m.label_id is None


def test_edge_tolerance_is_a_cadence_count():
    hot = lab("h", 1, 600, 1200, kind="hot_pixel")
    two_after = det("x", 1, 1260, 1320)
    assert match_detections([two_after], [hot], C, edge_tolerance_cadences=1)[0].label_id is None
    (m,) = match_detections([two_after], [hot], C, edge_tolerance_cadences=2)
    assert (m.label_id, m.grade) == ("h", "touched")
    three_after = det("y", 1, 1320, 1380)
    assert match_detections([three_after], [hot], C, edge_tolerance_cadences=2)[0].label_id is None


def test_edge_tolerance_must_be_non_negative():
    import pytest

    with pytest.raises(ValueError):
        match_detections([], [], C, edge_tolerance_cadences=-1)


def test_edge_match_requires_same_star():
    hot = lab("h", 1, 600, 1200, kind="hot_pixel")
    (m,) = match_detections([det("x", 2, 1200, 1260)], [hot], C)
    assert m.label_id is None


def test_edge_match_on_field_wide_label_follows_the_field_wide_rules():
    wide = lab("w", None, 600, 1200, kind="cloud")
    dets = [det("a", 1, 1200, 1260, score=2.0), det("b", 2, 1200, 1260, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, [wide], C)}
    assert (ms["a"].label_id, ms["a"].grade) == ("w", "touched")
    assert (ms["b"].label_id, ms["b"].grade) == ("w", "corroborating")


def test_overlap_beats_edge_for_one_detection():
    # det [700, 720) overlaps A (IoU 20/120, touched) and abuts B's entry edge.
    # B's id sorts first, so only the overlap-beats-edge rule can pick A.
    a = lab("z-overlap", 1, 600, 720)
    b = lab("a-edge", 1, 720, 840)
    (m,) = match_detections([det("x", 1, 700, 720)], [a, b], C)
    assert m.label_id == "z-overlap" and m.grade == "touched" and m.iou > 0


def test_edge_match_does_not_steal_a_label_from_an_overlapping_detection():
    # The edge detection scores higher, so a single greedy pass would give it the label
    # and strand the bounded detection. Edge matching only runs on what overlap left over.
    label = lab("h", 1, 600, 1200, kind="hot_pixel")
    edge = det("edge", 1, 1200, 1260, score=9.0)
    onset = det("onset", 1, 600, 1200, score=1.0)
    ms = {m.det_id: m for m in match_detections([edge, onset], [label], C)}
    assert (ms["onset"].label_id, ms["onset"].grade) == ("h", "bounded")
    assert (ms["edge"].label_id, ms["edge"].grade) == ("h", "corroborating")


def test_edge_detection_on_a_claimed_star_level_label_is_corroborating_not_fp():
    label = lab("h", 1, 600, 1200, kind="hot_pixel")
    onset = det("onset", 1, 600, 660, score=2.0)  # IoU 0.1: touched
    exit_ = det("exit", 1, 1200, 1260, score=1.0)
    ms = {m.det_id: m for m in match_detections([onset, exit_], [label], C)}
    assert (ms["onset"].label_id, ms["onset"].grade) == ("h", "touched")
    assert (ms["exit"].label_id, ms["exit"].grade) == ("h", "corroborating")
    assert ms["exit"].iou == 0.0


def test_overlapping_duplicate_on_claimed_star_level_label_is_still_unmatched():
    # Unchanged: only non-overlapping edge detections are forgiven, not duplicates in-span.
    label = lab("h", 1, 600, 1200, kind="hot_pixel")
    dets = [det("d1", 1, 600, 1200, score=2.0), det("d2", 1, 700, 800, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, [label], C)}
    assert ms["d2"] == Match("d", "d2", None, 0.0, None)


def test_edge_prefers_unclaimed_label_then_nearer_edge():
    # Two labels flank a one-cadence hole; det [660, 720) is an exit edge of A (gap 0)
    # and an entry edge of B (gap 0). Equal gaps: the lower label_id wins, then the
    # other label is left for the next detection.
    a = lab("a", 1, 360, 660)
    b = lab("b", 1, 720, 1020)
    dets = [det("x", 1, 660, 720, score=2.0), det("y", 1, 660, 720, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, [a, b], C)}
    assert (ms["x"].label_id, ms["x"].grade) == ("a", "touched")
    assert (ms["y"].label_id, ms["y"].grade) == ("b", "touched")


def test_nearer_edge_wins_over_farther_edge():
    near = lab("z", 1, 720, 1020)  # det ends at 720: gap 0
    far = lab("a", 1, 780, 1080)  # gap 60, within k=2
    (m,) = match_detections([det("x", 1, 660, 720)], [near, far], C, edge_tolerance_cadences=2)
    assert m.label_id == "z"


def test_single_cadence_tolerance_still_bounded_with_edge_tolerance():
    cr = lab("cr", 1, 600, 660, kind="cosmic_ray")
    for k in (0, 1, 2):
        (m,) = match_detections([det("x", 1, 660, 720)], [cr], C, edge_tolerance_cadences=k)
        assert (m.label_id, m.grade) == ("cr", "bounded")


def test_edge_tolerance_zero_reproduces_the_pre_edge_matcher_exactly():
    labels = [
        lab("hot", 1, 600, 1200, kind="hot_pixel"),
        lab("cr", 2, 600, 660, kind="cosmic_ray"),
        lab("flare", 3, 0, 1000, kind="flare"),
        lab("cloud", None, 3000, 3600, kind="cloud"),
    ]
    dets = [
        det("a1", 1, 600, 660, score=5.0),  # touched onset
        det("a2", 1, 1200, 1260, score=4.0),  # exit edge (fp at k=0)
        det("b1", 2, 660, 720, score=3.0),  # single-cadence tolerance: bounded
        det("c1", 3, 0, 400, score=2.0),  # bounded
        det("c2", 3, 1000, 1060, score=1.5),  # exit edge of a taken label
        det("c3", 3, 1500, 1560, score=1.0),  # far away
        det("d1", 4, 3000, 3100, score=0.9),  # cloud touched
        det("d2", 5, 3600, 3660, score=0.8),  # cloud exit edge
        det("d3", 6, 3100, 3200, score=0.7),  # cloud corroborating
    ]
    got = match_detections(dets, labels, C, edge_tolerance_cadences=0)
    assert got == [
        Match("d", "a1", "hot", 0.1, "touched"),
        Match("d", "a2", None, 0.0, None),
        Match("d", "b1", "cr", 0.0, "bounded"),
        Match("d", "c1", "flare", 0.4, "bounded"),
        Match("d", "c2", None, 0.0, None),
        Match("d", "c3", None, 0.0, None),
        Match("d", "d1", "cloud", 100 / 600, "touched"),
        Match("d", "d2", None, 0.0, None),
        Match("d", "d3", "cloud", 100 / 600, "corroborating"),
    ]
    # and the default (1) leaves every overlap/tolerance outcome alone, only rescuing edges
    default = match_detections(dets, labels, C)
    changed = {a.det_id for a, b in zip(default, got, strict=True) if a != b}
    assert changed == {"a2", "c2", "d2"}
    for a, b in zip(default, got, strict=True):
        if a.det_id not in changed:
            assert a == b


def test_edge_pass_skips_a_detection_that_overlaps_a_claimed_label():
    # d2 overlaps A (already claimed by d1) and abuts B. On main it was unmatched; the
    # edge pass is for detections that overlap NO label, so it must stay unmatched.
    a = lab("A", 1, 600, 1200, kind="hot_pixel")
    b = lab("B", 1, 1260, 1500, kind="hot_pixel")
    d1 = det("d1", 1, 600, 1200, score=2.0)
    d2 = det("d2", 1, 1100, 1260, score=1.0)
    ms = {m.det_id: m for m in match_detections([d1, d2], [a, b], C)}
    assert (ms["d1"].label_id, ms["d1"].grade) == ("A", "bounded")
    assert ms["d2"] == Match("d", "d2", None, 0.0, None)
    # and it agrees with the overlap-only matcher
    off = {m.det_id: m for m in match_detections([d1, d2], [a, b], C, edge_tolerance_cadences=0)}
    assert ms == off
