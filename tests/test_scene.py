import numpy as np
import pytest

from skywatch.types import Star
from skywatch.world.scene import Scene


def _stars(n: int) -> list[Star]:
    return [
        Star(
            star_id=i,
            flux0=1000.0 * (i + 1),
            noise_floor=0.01,
            variability="constant",
            ra=0.0,
            dec=0.0,
        )
        for i in range(n)
    ]


def test_quiet_scene_is_flat_and_fully_observed():
    sc = Scene.quiet(_stars(3), t_start=100, n_samples=5, cadence_s=60, night_index=0)
    assert sc.t.tolist() == [100, 160, 220, 280, 340]
    assert sc.flux.shape == (3, 5) and sc.mask.all()
    assert np.allclose(sc.flux[1], 2000.0)
    assert sc.row(2) == 2


def test_scene_rejects_shape_mismatch_and_bad_grid():
    stars = _stars(2)
    t = np.array([0, 60, 120], dtype=np.int64)
    with pytest.raises(ValueError):
        Scene(
            t=t,
            flux=np.ones((2, 2)),
            mask=np.ones((2, 3), bool),
            stars=stars,
            night_index=0,
            cadence_s=60,
        )
    with pytest.raises(ValueError):
        Scene(
            t=np.array([0, 60, 130]),
            flux=np.ones((2, 3)),
            mask=np.ones((2, 3), bool),
            stars=stars,
            night_index=0,
            cadence_s=60,
        )
    with pytest.raises(KeyError):
        Scene.quiet(stars, 0, 3, 60, 0).row(99)
