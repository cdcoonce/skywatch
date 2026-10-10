import numpy as np

from skywatch.world.noise import apply_noise


def test_shapes_and_err_formula():
    flux = np.array([[100.0, 400.0, 900.0], [1e4, 1e4, 1e4]])
    nf = np.array([0.01, 0.02])
    obs, err = apply_noise(flux, nf, np.random.default_rng(0))
    assert obs.shape == err.shape == flux.shape
    assert np.allclose(err, np.sqrt(flux + (nf[:, None] * flux) ** 2))


def test_empirical_std_matches_err():
    flux = np.full((1, 200_000), 5000.0)
    nf = np.array([0.01])
    obs, err = apply_noise(flux, nf, np.random.default_rng(1))
    assert abs((obs - flux).std() / err[0, 0] - 1) < 0.1


def test_determinism():
    flux = np.full((3, 50), 1000.0)
    nf = np.full(3, 0.01)
    a = apply_noise(flux, nf, np.random.default_rng(5))
    b = apply_noise(flux, nf, np.random.default_rng(5))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_negative_flux_clipped():
    obs, err = apply_noise(np.array([[-5.0, 4.0]]), np.array([0.01]), np.random.default_rng(0))
    assert np.isfinite(obs).all() and np.isfinite(err).all()
