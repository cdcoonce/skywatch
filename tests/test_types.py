import numpy as np
import pytest

from skywatch.types import (
    INSTRUMENT_KINDS,
    ORIGIN_OF_KIND,
    SCIENCE_KINDS,
    Detection,
    LightCurve,
    TruthLabel,
)


def _arrays(n, t=None):
    t = np.arange(n, dtype=np.int64) * 60 if t is None else np.asarray(t, dtype=np.int64)
    return t, np.ones(n), np.ones(n), np.zeros(n, dtype=np.uint8)


def test_lightcurve_ok():
    t, f, e, q = _arrays(5)
    lc = LightCurve(1, t, f, e, q)
    assert len(lc) == 5


def test_lightcurve_rejects_unequal_lengths():
    t, f, e, q = _arrays(5)
    with pytest.raises(ValueError):
        LightCurve(1, t, f[:4], e, q)


@pytest.mark.parametrize("t", [[0, 60, 60, 120], [0, 120, 60, 180]])
def test_lightcurve_rejects_non_increasing_t(t):
    _, f, e, q = _arrays(4)
    with pytest.raises(ValueError):
        LightCurve(1, np.asarray(t, dtype=np.int64), f, e, q)


def test_truth_rejects_mismatched_origin():
    with pytest.raises(ValueError):
        TruthLabel("a", "transit", "instrument", 1, 0, 10)
    with pytest.raises(ValueError):
        TruthLabel("a", "cloud", "science", None, 0, 10)


def test_truth_rejects_unknown_kind():
    with pytest.raises(ValueError):
        TruthLabel("a", "supernova", "science", 1, 0, 10)


@pytest.mark.parametrize("kind", SCIENCE_KINDS + INSTRUMENT_KINDS)
def test_truth_accepts_every_kind(kind):
    origin = "science" if kind in SCIENCE_KINDS else "instrument"
    assert ORIGIN_OF_KIND[kind] == origin
    label = TruthLabel("a", kind, origin, None, 0, 10, {"x": 1})
    assert label.origin == origin


def test_detection_round_trips():
    d = Detection("rolling_z", "d1", 3, 0, 60, 4.2, "flare")
    assert Detection(**d.__dict__) == d
    assert Detection("x", "d2", None, 0, 1, 1.0).kind_guess is None
