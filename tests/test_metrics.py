import pytest

from skywatch.scoring.match import Match
from skywatch.scoring.metrics import confusion, format_scoreboard, score
from skywatch.types import Detection, TruthLabel

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
    Match("d", "d1", "t1", 1.0),
    Match("d", "d2", "c1", 1.0),
    Match("d", "d3", None, 0.0),
    Match("d", "d4", None, 0.0),
]


def by_kind(scores):
    return {s.kind: s for s in scores}


def test_score_rows():
    scores = score(matches, dets, truth)
    assert [s.kind for s in scores] == sorted(s.kind for s in scores)
    k = by_kind(scores)
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
    m = [Match("d", "x", "t1", 1.0)]
    assert confusion(m, d, truth)[("d", "transit", "unknown")] == 1


def test_scoreboard():
    text = format_scoreboard(score(matches, dets, truth))
    lines = text.splitlines()
    assert "detector" in lines[0] and "F1" in lines[0]
    row = next(ln for ln in lines if ln.split()[:2] == ["d", "transit"])
    assert row.split() == ["d", "transit", "1", "0", "1", "1.000", "0.500", "0.667"]
