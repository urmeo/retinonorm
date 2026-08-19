"""Parameter uncertainty.

A pRF position reported without an error bar invites the reader to treat 12.0 and 12.4 as
different when the data cannot distinguish them. These tests check that the reported interval
tracks the actual noise level and covers the parameter that generated the signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.prf.fit import UnitFit

from .conftest import synthesise

TRUTH = (12.0, 8.0, 5.0)


def test_standard_errors_are_finite_for_a_good_fit(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=0.2, seed=5))

    assert np.isfinite([fit.se_x0, fit.se_y0, fit.se_sigma]).all()
    assert fit.se_x0 > 0.0


def test_uncertainty_grows_with_noise(fitter, grid, apertures) -> None:
    widths = []
    for noise in (0.1, 0.3, 0.6):
        fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=noise, seed=5))
        low, high = fit.confidence_interval("x0")
        widths.append(high - low)

    assert widths == sorted(widths)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_interval_covers_the_generating_parameter(fitter, grid, apertures, seed) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=0.2, seed=seed))

    low, high = fit.confidence_interval("x0")

    assert low <= TRUTH[0] <= high


def test_degrees_of_freedom_account_for_projected_parameters(fitter, grid, apertures) -> None:
    """Amplitude and baseline are estimated too, even though the optimiser never sees them."""
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH))

    assert fit.dof == fitter.n_frames - 5


def test_wider_confidence_level_gives_wider_interval(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=8))

    narrow = fit.confidence_interval("x0", level=0.80)
    wide = fit.confidence_interval("x0", level=0.99)

    assert wide[0] < narrow[0] and narrow[1] < wide[1]


def test_failed_fit_reports_undefined_interval() -> None:
    low, high = UnitFit.failed().confidence_interval("x0")

    assert np.isnan(low) and np.isnan(high)


def test_invalid_confidence_level_raises(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH))

    with pytest.raises(ValueError, match="level"):
        fit.confidence_interval("x0", level=1.5)
