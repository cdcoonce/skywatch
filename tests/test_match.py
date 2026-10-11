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
    # only one label: the second detection is a fragment of the event d1 found
    ms = {m.det_id: m for m in match_detections(dets, [labels[0]], C)}
    assert ms["d1"].label_id == "a"
    assert ms["d2"] == Match("d", "d2", "a", 0.5, "corroborating")


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
    # label already taken by "b" and star-level: "t" overlaps it, so it corroborates it
    assert got["t"] == Match("d", "t", "o", 0.1, "corroborating")
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
    # one tp only: the second detection overlaps the claimed label, so it corroborates
    assert (ms["d2"].label_id, ms["d2"].grade) == ("o", "corroborating")


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


def test_overlapping_duplicate_on_claimed_star_level_label_is_corroborating():
    # A fragment of an event the detector already found points at something real: it is
    # evidence for the claimed label, not a false alarm, and not a second tp.
    label = lab("h", 1, 600, 1200, kind="hot_pixel")
    dets = [det("d1", 1, 600, 1200, score=2.0), det("d2", 1, 700, 800, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, [label], C)}
    assert (ms["d1"].label_id, ms["d1"].grade) == ("h", "bounded")
    assert ms["d2"] == Match("d", "d2", "h", span_iou(700, 800, 600, 1200), "corroborating")
    # also when the claim is only "touched", and when edge tolerance is off
    dets = [det("d1", 1, 600, 660, score=2.0), det("d2", 1, 900, 960, score=1.0)]
    for k in (0, 1):
        ms = {m.det_id: m for m in match_detections(dets, [label], C, edge_tolerance_cadences=k)}
        assert (ms["d1"].label_id, ms["d1"].grade) == ("h", "touched")
        assert (ms["d2"].label_id, ms["d2"].grade) == ("h", "corroborating")


def test_the_tp_goes_to_the_first_by_score_and_a_later_better_fit_is_the_fragment():
    # The one-to-one rule is unchanged: greedy by score decides the tp. A later detection
    # that fits better (higher IoU) is still only corroborating; it never steals the tp.
    label = lab("h", 1, 600, 1200, kind="hot_pixel")
    low_iou = det("hi_score", 1, 600, 700, score=2.0)
    high_iou = det("lo_score", 1, 600, 1200, score=1.0)
    ms = {m.det_id: m for m in match_detections([low_iou, high_iou], [label], C)}
    assert (ms["hi_score"].label_id, ms["hi_score"].grade) == ("h", "touched")
    assert (ms["lo_score"].label_id, ms["lo_score"].grade) == ("h", "corroborating")
    tps = [m for m in ms.values() if m.grade in ("bounded", "touched")]
    assert len(tps) == 1


def test_duplicate_overlapping_two_claimed_labels_corroborates_the_higher_iou_one():
    a = lab("a", 1, 0, 300, kind="flare")
    b = lab("b", 1, 300, 400, kind="flare")
    # claims: d1 takes a, d2 takes b
    d1 = det("d1", 1, 0, 300, score=3.0)
    d2 = det("d2", 1, 300, 400, score=2.0)
    # dup overlaps a for 60 s of a 360 s union span, b for 100 s: b has the higher IoU
    dup = det("dup", 1, 240, 400, score=1.0)
    ms = {m.det_id: m for m in match_detections([d1, d2, dup], [a, b], C)}
    assert ms["dup"].label_id == "b" and ms["dup"].grade == "corroborating"
    assert ms["dup"].iou == span_iou(240, 400, 300, 400)
    # truth order must not matter
    ms = {m.det_id: m for m in match_detections([d1, d2, dup], [b, a], C)}
    assert ms["dup"].label_id == "b"


def test_equal_iou_duplicate_corroborates_the_lowest_label_id():
    a = lab("a", 1, 0, 100, kind="flare")
    b = lab("b", 1, 100, 200, kind="flare")
    d1 = det("d1", 1, 0, 100, score=3.0)
    d2 = det("d2", 1, 100, 200, score=2.0)
    dup = det("dup", 1, 50, 150, score=1.0)  # 50 s into each: equal IoU
    for truth in ([a, b], [b, a]):
        ms = {m.det_id: m for m in match_detections([d1, d2, dup], truth, C)}
        assert ms["dup"].label_id == "a" and ms["dup"].grade == "corroborating"


def test_field_wide_corroboration_keeps_precedence_over_a_better_fitting_star_level_label():
    # Pre-existing behaviour must not move: a detection that corroborated a claimed
    # field-wide label keeps doing so even if it now also overlaps a claimed star-level
    # label with a higher IoU.
    cloud = lab("cloud", None, 0, 1000, kind="cloud")
    flare = lab("flare", 1, 400, 500, kind="flare")
    d1 = det("d1", 1, 0, 1000, score=3.0)  # claims the cloud
    d2 = det("d2", 1, 400, 500, score=2.0)  # claims the flare
    dup = det("dup", 1, 400, 500, score=1.0)  # IoU 1.0 with flare, 0.1 with cloud
    ms = {m.det_id: m for m in match_detections([d1, d2, dup], [cloud, flare], C)}
    assert (ms["d1"].label_id, ms["d2"].label_id) == ("cloud", "flare")
    assert (ms["dup"].label_id, ms["dup"].grade) == ("cloud", "corroborating")


def test_duplicate_overlapping_one_claimed_and_one_free_label_takes_the_free_one():
    claimed = lab("claimed", 1, 0, 300, kind="flare")
    free = lab("free", 1, 300, 600, kind="flare")
    d1 = det("d1", 1, 0, 300, score=2.0)
    dup = det("dup", 1, 200, 400, score=1.0)  # IoU 100/400 each way
    ms = {m.det_id: m for m in match_detections([d1, dup], [claimed, free], C)}
    assert (ms["dup"].label_id, ms["dup"].grade) == ("free", "touched")


def test_detection_overlapping_no_label_is_still_unmatched():
    label = lab("h", 1, 600, 1200, kind="hot_pixel")
    far = det("far", 1, 3000, 3100)
    other_star = det("other", 2, 600, 1200)
    ms = {m.det_id: m for m in match_detections([far, other_star], [label], C)}
    assert ms["far"] == Match("d", "far", None, 0.0, None)
    assert ms["other"] == Match("d", "other", None, 0.0, None)


def test_duplicate_over_an_unobservable_label_is_still_unmatched():
    # An unobservable label is treated as absent, so it is nothing to corroborate.
    label = TruthLabel(
        "u", "flare", ORIGIN_OF_KIND["flare"], 1, 600, 1200, params={"footprint_samples": 0}
    )
    ms = {m.det_id: m for m in match_detections([det("d1", 1, 600, 1200)], [label], C)}
    assert ms["d1"] == Match("d", "d1", None, 0.0, None)


def test_corroborating_duplicates_do_not_change_the_tp_set():
    label = lab("h", 1, 600, 1200, kind="hot_pixel")
    base = [det("d1", 1, 600, 1200, score=2.0)]
    more = [*base, det("d2", 1, 700, 800, score=1.0), det("d3", 1, 900, 1000, score=0.5)]

    def tp(ms):
        return {(m.det_id, m.label_id, m.grade) for m in ms if m.grade in ("bounded", "touched")}

    assert tp(match_detections(more, [label], C)) == tp(match_detections(base, [label], C))


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
    # d2 overlaps A (already claimed by d1) and abuts B. The edge pass is for detections
    # that overlap NO label, so it must not take B: d2 corroborates A (the overlap rule).
    a = lab("A", 1, 600, 1200, kind="hot_pixel")
    b = lab("B", 1, 1260, 1500, kind="hot_pixel")
    d1 = det("d1", 1, 600, 1200, score=2.0)
    d2 = det("d2", 1, 1100, 1260, score=1.0)
    ms = {m.det_id: m for m in match_detections([d1, d2], [a, b], C)}
    assert (ms["d1"].label_id, ms["d1"].grade) == ("A", "bounded")
    assert (ms["d2"].label_id, ms["d2"].grade) == ("A", "corroborating")
    # and it agrees with the overlap-only matcher (B stays unclaimed, d2 not edge-matched)
    off = {m.det_id: m for m in match_detections([d1, d2], [a, b], C, edge_tolerance_cadences=0)}
    assert ms == off


def test_edge_tie_among_unclaimed_labels_takes_lowest_id_not_truth_order():
    # Equal gap (0) to both; the lower label_id is listed second in truth.
    a = lab("a", 1, 360, 660)
    b = lab("b", 1, 720, 1020)
    (m,) = match_detections([det("x", 1, 660, 720)], [b, a], C)
    assert (m.label_id, m.grade) == ("a", "touched")


def test_edge_tie_among_claimed_labels_takes_lowest_id_not_truth_order():
    # Both labels are claimed by overlapping detections; x abuts both at gap 0.
    a = lab("a", 1, 360, 660)
    b = lab("b", 1, 720, 1020)
    dets = [
        det("on_a", 1, 360, 660, score=3.0),
        det("on_b", 1, 720, 1020, score=2.0),
        det("x", 1, 660, 720, score=1.0),
    ]
    ms = {m.det_id: m for m in match_detections(dets, [b, a], C)}
    assert ms["on_a"].label_id == "a" and ms["on_b"].label_id == "b"
    assert (ms["x"].label_id, ms["x"].grade) == ("a", "corroborating")


def test_unclaimed_label_beats_a_nearer_claimed_label():
    # k=2: claimed A at gap 0, unclaimed B one cadence further. Unclaimed wins, touched.
    a = lab("a", 1, 360, 660)
    b = lab("b", 1, 780, 1080)
    dets = [det("on_a", 1, 360, 660, score=2.0), det("x", 1, 660, 720, score=1.0)]
    ms = {m.det_id: m for m in match_detections(dets, [a, b], C, edge_tolerance_cadences=2)}
    assert ms["on_a"].label_id == "a"
    assert (ms["x"].label_id, ms["x"].grade) == ("b", "touched")


def test_iou_threshold_is_exactly_point_three():
    label = lab("o", 1, 0, 1000)
    just_under = match_detections([det("u", 1, 0, 299)], [label], C)[0]
    exactly = match_detections([det("e", 1, 0, 300)], [label], C)[0]
    just_over = match_detections([det("v", 1, 0, 301)], [label], C)[0]
    assert (just_under.grade, just_under.iou) == ("touched", 0.299)
    assert (exactly.grade, exactly.iou) == ("bounded", 0.3)
    assert just_over.grade == "bounded"


# --- zero-footprint labels: no detector could see them -------------------------------


def unseen(lid, star, a, b, kind="flare"):
    return TruthLabel(lid, kind, ORIGIN_OF_KIND[kind], star, a, b, {"footprint_samples": 0})


def seen(lid, star, a, b, kind="flare", n=3):
    return TruthLabel(lid, kind, ORIGIN_OF_KIND[kind], star, a, b, {"footprint_samples": n})


def test_zero_footprint_label_is_never_matched():
    (m,) = match_detections([det("x", 1, 0, 600)], [unseen("u", 1, 0, 600)], C)
    assert m.label_id is None and m.grade is None  # as if the label were absent: a false alarm


def test_detection_over_zero_footprint_label_equals_label_absent():
    dets = [det("x", 1, 0, 600), det("y", 2, 100, 160)]
    labels = [unseen("u", 1, 0, 600), seen("s", 2, 100, 160, kind="cosmic_ray", n=1)]
    assert match_detections(dets, labels, C) == match_detections(dets, labels[1:], C)


def test_zero_footprint_label_does_not_steal_a_detection_from_a_real_one():
    labels = [unseen("u", 1, 0, 600), seen("s", 1, 0, 400)]
    (m,) = match_detections([det("x", 1, 0, 600)], labels, C)
    assert m.label_id == "s"


def test_one_sample_label_is_still_matched():
    (m,) = match_detections([det("x", 1, 0, 60)], [seen("s", 1, 0, 60, "cosmic_ray", n=1)], C)
    assert m.label_id == "s" and m.grade == "bounded"


def test_edge_tolerance_does_not_reach_a_zero_footprint_label():
    (m,) = match_detections([det("x", 1, 600, 660)], [unseen("u", 1, 0, 600)], C)
    assert m.label_id is None
    (m,) = match_detections([det("x", 1, 600, 660)], [seen("s", 1, 0, 600)], C)
    assert m.label_id == "s"


def test_labels_without_the_key_still_match():
    (m,) = match_detections([det("x", 1, 0, 600)], [lab("old", 1, 0, 600, "flare")], C)
    assert m.label_id == "old"
