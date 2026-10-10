import pytest

from skywatch.scoring.match import Match
from skywatch.scoring.metrics import confusion, format_scoreboard, score
from skywatch.types import ORIGIN_OF_KIND, Detection, TruthLabel

T = lambda lid, kind, origin: TruthLabel(lid, kind, origin, 1, 0, 100)
truth = [
    T("t1", "transit", "science"),
    T("t2", "transit", "science"),  # missed
    T("c1", "cosmic_ray", "instrument"),
]
dets = [
    Detection("d", "d1", 1, 0, 100, 3.0, "transit"),  # TP transit
    Detection("d", "d2", 1, 0, 100, 2.0, "transit"),  # TP cosmic_ray labelled as transit
    Detection("d", "d3", 1, 0, 100, 1.0, "flare"),  # FP flare
    Detection("d", "d4", 1, 0, 100, 0.5, None),  # FP unknown
]
matches = [
    Match("d", "d1", "t1", 1.0, "bounded"),
    Match("d", "d2", "c1", 1.0, "bounded"),
    Match("d", "d3", None, 0.0),
    Match("d", "d4", None, 0.0),
]


def by_kind(scores):
    return {s.kind: s for s in scores}


def test_score_rows():
    scores = score(matches, dets, truth)
    # ordered by detector, then kind alphabetically, with "all" last
    kinds = [s.kind for s in scores]
    assert kinds == [*sorted(set(ORIGIN_OF_KIND) | {"unknown"}), "all"]
    k = by_kind(scores)
    absent = next(x for x in ORIGIN_OF_KIND if x not in {"transit", "cosmic_ray", "flare"})
    assert (k[absent].tp, k[absent].fp, k[absent].fn) == (0, 0, 0)
    assert (k["transit"].tp, k["transit"].fn, k["transit"].fp) == (1, 1, 0)
    assert k["transit"].precision == 0.0 or k["transit"].fp == 0
    assert k["transit"].recall == 0.5 and k["transit"].precision == 1.0
    assert k["transit"].f1 == pytest.approx(2 / 3)
    assert (k["cosmic_ray"].tp, k["cosmic_ray"].fn, k["cosmic_ray"].fp) == (1, 0, 0)
    assert k["cosmic_ray"].recall == 1.0
    assert (k["flare"].tp, k["flare"].fp, k["flare"].fn) == (0, 1, 0)
    assert k["flare"].precision == 0.0 and k["flare"].recall == 0.0 and k["flare"].f1 == 0.0
    assert (k["unknown"].fp, k["unknown"].tp) == (1, 0)
    a = k["all"]
    assert (a.tp, a.fp, a.fn) == (2, 2, 1)
    assert a.precision == 0.5 and a.recall == pytest.approx(2 / 3)
    assert a.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_confusion():
    c = confusion(matches, dets, truth)
    assert c == {
        ("d", "transit", "transit"): 1,
        ("d", "cosmic_ray", "transit"): 1,
        ("d", "transit", "missed"): 1,
    }


def test_confusion_unknown_guess():
    d = [Detection("d", "x", 1, 0, 100, 1.0)]
    m = [Match("d", "x", "t1", 1.0, "bounded")]
    assert confusion(m, d, truth)[("d", "transit", "unknown")] == 1


def test_scoreboard():
    text = format_scoreboard(score(matches, dets, truth))
    lines = text.splitlines()
    assert lines[0].split() == [
        "detector",
        "kind",
        "tpB",
        "tpT",
        "corr",
        "fp",
        "fn",
        "P",
        "R",
        "F1",
        "bounded",
    ]
    row = next(ln for ln in lines if ln.split()[:2] == ["d", "transit"])
    assert row.split() == [
        "d",
        "transit",
        "1",
        "0",
        "0",
        "0",
        "1",
        "1.000",
        "0.500",
        "0.667",
        "1.000",
    ]


def test_silent_detector_gets_fn_rows():
    rows = score([], [], truth, detectors=["quiet"])
    k = by_kind(rows)
    assert {s.detector for s in rows} == {"quiet"}
    assert (k["transit"].tp, k["transit"].fp, k["transit"].fn) == (0, 0, 2)
    assert (k["cosmic_ray"].tp, k["cosmic_ray"].fp, k["cosmic_ray"].fn) == (0, 0, 1)
    assert (k["all"].tp, k["all"].fp, k["all"].fn) == (0, 0, 3)
    c = confusion([], [], truth, detectors=["quiet"])
    assert c == {("quiet", "transit", "missed"): 2, ("quiet", "cosmic_ray", "missed"): 1}


def test_no_detectors_arg_keeps_derived_behaviour():
    assert score([], [], truth) == []
    assert confusion([], [], truth) == {}


def test_kinds_arg_restricts_rows():
    rows = score(matches, dets, truth, kinds=["transit"])
    assert {s.kind for s in rows} == {"transit", "cosmic_ray", "flare", "unknown", "all"}
    rows = score([], [], truth, detectors=["q"], kinds=["transit"])
    assert [s.kind for s in rows] == ["cosmic_ray", "transit", "all"]


def test_grades_corroborating_and_formulas_by_hand():
    wide = TruthLabel("w", "cloud", "instrument", None, 0, 1000)
    flares = [TruthLabel(f"f{i}", "flare", "science", i, 0, 1000) for i in range(1, 5)]
    tr = [wide, *flares]
    ds = [Detection("d", f"x{i}", i, 0, 10, 1.0, "flare") for i in range(1, 6)]
    ms = [
        Match("d", "x1", "f1", 0.9, "bounded"),
        Match("d", "x2", "f2", 0.01, "touched"),
        Match("d", "x3", "w", 0.01, "touched"),
        Match("d", "x4", "w", 0.01, "corroborating"),
        Match("d", "x5", None, 0.0, None),
    ]
    k = by_kind(score(ms, ds, tr))
    f = k["flare"]
    assert (f.tp_bounded, f.tp_touched, f.corroborating, f.tp, f.fp, f.fn) == (1, 1, 0, 2, 1, 2)
    assert f.precision == pytest.approx(2 / 3)
    assert f.recall == 0.5
    assert f.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert f.bounded_rate == 0.5
    c = k["cloud"]
    assert (c.tp_bounded, c.tp_touched, c.corroborating, c.fp, c.fn) == (0, 1, 1, 0, 0)
    assert c.recall == 1.0 and c.bounded_rate == 0.0
    a = k["all"]
    assert (a.tp_bounded, a.tp_touched, a.corroborating, a.fp, a.fn) == (1, 2, 1, 1, 2)
    assert a.bounded_rate == pytest.approx(1 / 3)
    conf = confusion(ms, ds, tr)
    assert conf == {
        ("d", "flare", "flare"): 2,
        ("d", "cloud", "flare"): 1,
        ("d", "flare", "missed"): 2,
    }


def test_three_stars_one_field_wide_label():
    wide = TruthLabel("w", "cloud", "instrument", None, 0, 1000)
    ds = [Detection("d", f"x{i}", i, 0, 10, 1.0) for i in (1, 2, 3)]
    ms = [
        Match("d", "x1", "w", 0.01, "touched"),
        Match("d", "x2", "w", 0.01, "corroborating"),
        Match("d", "x3", "w", 0.01, "corroborating"),
    ]
    c = by_kind(score(ms, ds, [wide]))["cloud"]
    assert (c.tp_touched, c.corroborating, c.fp, c.fn) == (1, 2, 0, 0)
    assert by_kind(score(ms, ds, [wide]))["all"].fp == 0
