import itertools

import numpy as np

from skywatch.detectors.nights import split_nights

C = 60
NIGHT = 480  # samples per night in these tests
GAP_S = 57600 - C  # a 16 h gap between a night's last sample and the next night's first


def three_nights():
    return np.concatenate([k * 86400 + np.arange(NIGHT, dtype=np.int64) * C for k in range(3)])


def test_single_night_is_one_full_slice():
    t = np.arange(NIGHT, dtype=np.int64) * C
    assert split_nights(t, C) == [slice(0, NIGHT)]


def test_three_nights_tile_the_array():
    t = three_nights()
    s = split_nights(t, C)
    assert s == [slice(0, NIGHT), slice(NIGHT, 2 * NIGHT), slice(2 * NIGHT, 3 * NIGHT)]


def test_dropout_below_threshold_does_not_split():
    t = np.arange(NIGHT, dtype=np.int64) * C
    t[200:] += 50 * C  # a 50-cadence hole inside one night
    assert split_nights(t, C) == [slice(0, NIGHT)]


def test_gap_exactly_at_threshold_does_not_split():
    t = np.concatenate([np.arange(10) * C, 9 * C + 100 * C + np.arange(10) * C]).astype(np.int64)
    assert np.diff(t).max() == 100 * C
    assert split_nights(t, C) == [slice(0, 20)]
    t[10:] += C  # now one cadence over the threshold
    assert split_nights(t, C) == [slice(0, 10), slice(10, 20)]


def test_empty_array():
    assert split_nights(np.array([], dtype=np.int64), C) == []


def test_single_sample():
    assert split_nights(np.array([5], dtype=np.int64), C) == [slice(0, 1)]


def test_custom_gap_cadences_changes_the_split():
    t = np.arange(NIGHT, dtype=np.int64) * C
    t[200:] += 50 * C
    assert split_nights(t, C, gap_cadences=100) == [slice(0, NIGHT)]
    assert split_nights(t, C, gap_cadences=10) == [slice(0, 200), slice(200, NIGHT)]


def test_slices_reconstruct_t():
    t = three_nights()
    parts = split_nights(t, C)
    assert np.array_equal(np.concatenate([t[s] for s in parts]), t)
    assert all(a.stop == b.start for a, b in itertools.pairwise(parts))
