"""Edge cases where a pRF fit is most likely to fail silently.

Interior pRFs of moderate size are the easy case. The failure modes that matter are centres
pushed to the edge of the stimulated field and sigmas pressed against their search bounds,
because there the optimiser can return a confident-looking parameter that is really an
artefact of where the search stopped.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.config import FitConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import BOUND_TOLERANCE, PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField

from .conftest import synthesise


@pytest.mark.parametrize(
    "truth",
    [
        (28.0, 0.0, 4.0),
        (-28.0, 0.0, 4.0),
        (0.0, 28.0, 4.0),
        (0.0, -28.0, 4.0),
        (20.0, 20.0, 4.0),
    ],
)
def test_recovers_prf_near_field_boundary(fitter, grid, apertures, truth) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, truth))
    x0, y0, _ = truth

    assert fit.converged
    assert np.hypot(fit.x0 - x0, fit.y0 - y0) < 1.0


def test_centre_never_escapes_the_field(fitter, grid, apertures) -> None:
    lower, upper = fitter.bounds

    for truth in [(31.0, 0.0, 3.0), (0.0, -31.0, 3.0), (30.0, 30.0, 3.0)]:
        fit = fitter.fit_unit(synthesise(grid, apertures, truth))
        assert lower[0] <= fit.x0 <= upper[0]
        assert lower[1] <= fit.y0 <= upper[1]


def test_recovers_smallest_admissible_sigma(grid, apertures) -> None:
    config = FitConfig(grid_size=10, sigma_bounds=(1.0, 20.0))
    fitter = PRFFitter(grid, apertures, config)

    fit = fitter.fit_unit(synthesise(grid, apertures, (0.0, 0.0, 1.5)))

    assert fit.converged
    assert abs(fit.sigma - 1.5) < 0.5


def test_recovers_largest_admissible_sigma(grid, apertures) -> None:
    config = FitConfig(grid_size=10, sigma_bounds=(1.0, 25.0))
    fitter = PRFFitter(grid, apertures, config)

    fit = fitter.fit_unit(synthesise(grid, apertures, (0.0, 0.0, 18.0)))

    assert fit.converged
    assert abs(fit.sigma - 18.0) / 18.0 < 0.15


def test_sigma_outside_search_range_is_flagged_at_bound(grid, apertures) -> None:
    """A pRF wider than the search allows must be reported as pinned, not as a measurement."""
    config = FitConfig(grid_size=8, sigma_bounds=(1.0, 6.0))
    fitter = PRFFitter(grid, apertures, config)

    fit = fitter.fit_unit(synthesise(grid, apertures, (0.0, 0.0, 20.0)))

    assert fit.at_bound
    assert abs(fit.sigma - 6.0) <= BOUND_TOLERANCE


def test_interior_fit_is_not_flagged_at_bound(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (8.0, -6.0, 5.0)))

    assert not fit.at_bound


def test_fitter_rejects_too_few_frames(grid) -> None:
    with pytest.raises(ValueError, match="frames"):
        PRFFitter(grid, np.ones((4, 64, 64)), FitConfig())


def test_fitter_rejects_mismatched_grid(grid) -> None:
    with pytest.raises(ValueError, match="grid"):
        PRFFitter(grid, np.ones((40, 32, 32)), FitConfig())


def test_fitter_rejects_non_frame_stack(grid) -> None:
    with pytest.raises(ValueError, match="frames, height, width"):
        PRFFitter(grid, np.ones((64, 64)), FitConfig())


def test_zero_response_is_not_fitted(fitter) -> None:
    assert not fitter.fit_unit(np.zeros(fitter.n_frames)).converged


def test_infinite_response_is_not_fitted(fitter) -> None:
    response = np.zeros(fitter.n_frames)
    response[0] = np.inf

    assert not fitter.fit_unit(response).converged


def test_receptive_field_far_outside_field_is_finite(grid) -> None:
    weights = GaussianReceptiveField(500.0, 500.0, 3.0).weights(grid)

    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(0.0, abs=1e-12)
