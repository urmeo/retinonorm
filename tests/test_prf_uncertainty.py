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

    assert wide[0] < narrow[0]
    assert narrow[1] < wide[1]


def test_failed_fit_reports_undefined_interval() -> None:
    low, high = UnitFit.failed().confidence_interval("x0")

    assert np.isnan(low)
    assert np.isnan(high)


def test_invalid_confidence_level_raises(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH))

    with pytest.raises(ValueError, match="level"):
        fit.confidence_interval("x0", level=1.5)


def test_duplicate_frames_do_not_shrink_the_error_bars(grid, apertures, fit_config) -> None:
    """Showing a frame twice is not a second observation.

    Without an haemodynamic response the default eight-direction bar stimulus presents every
    frame exactly twice, since a sweep and its 180 degree return are bit-identical. Counting
    the copies would have reported standard errors a factor of root two too small.
    """
    from cortexprobe.prf.fit import PRFFitter

    response = synthesise(grid, apertures, TRUTH, noise=0.2, seed=5)

    once = PRFFitter(grid, apertures, fit_config)
    twice = PRFFitter(grid, np.concatenate([apertures, apertures]), fit_config)

    single = once.fit_unit(response)
    repeated = twice.fit_unit(np.concatenate([response, response]))

    assert twice.n_frames == 2 * once.n_frames
    assert twice.n_independent_frames == once.n_independent_frames
    assert repeated.dof == single.dof
    assert repeated.se_x0 == pytest.approx(single.se_x0, rel=1e-6)
    assert repeated.se_sigma == pytest.approx(single.se_sigma, rel=1e-6)


def test_degrees_of_freedom_count_distinct_frames(grid, fit_config) -> None:
    from cortexprobe.config import StimulusConfig
    from cortexprobe.prf.fit import PRFFitter
    from cortexprobe.stimuli import build_apertures

    from .conftest import RESOLUTION

    eight = build_apertures(
        StimulusConfig(resolution=RESOLUTION, n_steps=20), "bar"
    ).as_float()
    fitter = PRFFitter(grid, eight, fit_config)

    assert fitter.n_frames == 160
    assert fitter.n_independent_frames == 80


@pytest.mark.slow
@pytest.mark.parametrize("truth", [(12.0, 8.0, 5.0), (28.0, 0.0, 4.0)])
@pytest.mark.parametrize("noise", [0.2, 0.5])
def test_interval_coverage_is_calibrated(fitter, grid, apertures, truth, noise) -> None:
    """Empirical coverage of the nominal 95 per cent interval, over many realisations.

    Five successes prove very little about a 95 per cent interval: you would see 5/5 about 77
    per cent of the time even at 95 per cent true coverage, and about 59 per cent of the time
    at 90 per cent. Two hundred realisations narrow that enough to be worth asserting.

    Both an interior position and one near the field edge are tested, since the linearisation
    behind the interval is weakest where the stimulus constrains the fit least. Measured
    coverage runs 0.915 to 0.935 against a nominal 0.95, so the interval is mildly optimistic
    -- which is a property worth knowing and stating, not one to paper over with a loose bound.
    """
    realisations = 200
    covered = 0
    for seed in range(realisations):
        response = synthesise(grid, apertures, truth, noise=noise, seed=1000 + seed)
        low, high = fitter.fit_unit(response).confidence_interval("x0")
        covered += bool(low <= truth[0] <= high)

    assert 0.90 <= covered / realisations <= 0.99
