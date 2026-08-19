"""Ground-truth recovery for the pRF fitter.

This is the load-bearing test of the project. Synthetic activations are generated from pRFs
with *known* parameters and handed to the fitter; the fitter must recover them. If it cannot
recover a pRF it produced itself, then no pRF it reports from a real network means anything,
and a fitter bug would be indistinguishable from a scientific finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.config import FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.stimuli import build_apertures

RESOLUTION = 64
GROUND_TRUTH = [
    (0.0, 0.0, 5.0),
    (12.0, 8.0, 4.0),
    (-10.0, 14.0, 7.0),
    (16.0, -12.0, 6.0),
    (-15.0, -9.0, 9.0),
]


@pytest.fixture(scope="module")
def apertures() -> np.ndarray:
    config = StimulusConfig(resolution=RESOLUTION, n_steps=20, directions=(0, 45, 90, 135))
    return build_apertures(config, "bar").as_float()


@pytest.fixture(scope="module")
def grid() -> Grid:
    return Grid(RESOLUTION)


@pytest.fixture(scope="module")
def fitter(grid: Grid, apertures: np.ndarray) -> PRFFitter:
    return PRFFitter(grid, apertures, FitConfig(grid_size=10, sigma_bounds=(1.0, 20.0)))


def synthesise(
    grid: Grid,
    apertures: np.ndarray,
    truth: tuple,
    beta: float = 3.0,
    baseline: float = 0.5,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    x0, y0, sigma = truth
    clean = beta * predict(GaussianReceptiveField(x0, y0, sigma).weights(grid), apertures) + baseline
    if noise <= 0.0:
        return clean
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, noise * float(np.std(clean)), size=clean.shape)


@pytest.mark.parametrize("truth", GROUND_TRUTH)
def test_recovers_noiseless_prf(fitter: PRFFitter, grid: Grid, apertures: np.ndarray, truth: tuple) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, truth))
    x0, y0, sigma = truth

    assert fit.converged
    assert fit.r2 > 0.99
    assert np.hypot(fit.x0 - x0, fit.y0 - y0) < 0.5
    assert abs(fit.sigma - sigma) / sigma < 0.05


@pytest.mark.parametrize("truth", GROUND_TRUTH)
def test_recovers_prf_under_moderate_noise(
    fitter: PRFFitter, grid: Grid, apertures: np.ndarray, truth: tuple
) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, truth, noise=0.2, seed=7))
    x0, y0, _ = truth

    assert fit.converged
    assert fit.r2 > 0.8
    assert np.hypot(fit.x0 - x0, fit.y0 - y0) < 1.5


def test_rejects_pure_noise(fitter: PRFFitter) -> None:
    rng = np.random.default_rng(11)
    responses = rng.normal(size=(len(fitter.apertures), 40))

    fits = fitter.fit_all(responses)
    rejected = sum(not fit.converged or fit.r2 < fitter.config.r2_threshold for fit in fits)

    assert rejected / len(fits) >= 0.95


def test_amplitude_and_baseline_recovered(fitter: PRFFitter, grid: Grid, apertures: np.ndarray) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (8.0, -6.0, 5.0), beta=4.0, baseline=1.5))

    assert fit.beta == pytest.approx(4.0, rel=0.05)
    assert fit.baseline == pytest.approx(1.5, abs=0.05)


def test_constant_response_is_not_fitted(fitter: PRFFitter) -> None:
    fit = fitter.fit_unit(np.full(len(fitter.apertures), 2.0))

    assert not fit.converged
    assert fit.r2 == 0.0
    assert np.isnan(fit.x0)


def test_non_finite_response_is_not_fitted(fitter: PRFFitter) -> None:
    response = np.zeros(len(fitter.apertures))
    response[3] = np.nan

    assert not fitter.fit_unit(response).converged


def test_wrong_length_response_raises(fitter: PRFFitter) -> None:
    with pytest.raises(ValueError):
        fitter.fit_unit(np.zeros(len(fitter.apertures) + 1))


def test_fitted_parameters_respect_bounds(fitter: PRFFitter, grid: Grid, apertures: np.ndarray) -> None:
    lower, upper = fitter.bounds

    for truth in GROUND_TRUTH:
        fit = fitter.fit_unit(synthesise(grid, apertures, truth))
        assert lower[0] <= fit.x0 <= upper[0]
        assert lower[1] <= fit.y0 <= upper[1]
        assert lower[2] <= fit.sigma <= upper[2]


def test_fit_is_deterministic(fitter: PRFFitter, grid: Grid, apertures: np.ndarray) -> None:
    response = synthesise(grid, apertures, (10.0, 10.0, 6.0), noise=0.1, seed=3)

    first, second = fitter.fit_unit(response), fitter.fit_unit(response)

    assert (first.x0, first.y0, first.sigma) == (second.x0, second.y0, second.sigma)


def test_fit_all_rejects_wrong_frame_count(fitter: PRFFitter) -> None:
    with pytest.raises(ValueError):
        fitter.fit_all(np.zeros((len(fitter.apertures) + 2, 3)))
