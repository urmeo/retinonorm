"""Held-out validation and the leakage it is designed to expose.

The decisive pair is :func:`test_random_frame_split_leaks_on_autocorrelated_noise` and its
white-noise counterpart. Together they show why folds are split by sweep group, and pin the
mechanism: the leak needs both overlapping apertures and a temporally correlated response.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.config import FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import _r_squared
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator, LeaveOneGroupOut
from cortexprobe.stimuli import build_apertures

from .conftest import autocorrelated_noise, synthesise

TRUTH = (12.0, 8.0, 5.0)


@pytest.fixture(scope="module")
def validator(grid, apertures, sequence, fit_config) -> CrossValidator:
    return CrossValidator(grid, apertures, sequence.group, fit_config)


def test_folds_hold_out_whole_groups(sequence) -> None:
    splitter = LeaveOneGroupOut(sequence.group)

    for train, test in splitter.splits():
        assert set(sequence.group[train]).isdisjoint(sequence.group[test])
        assert len(train) + len(test) == len(sequence.group)


def test_splitter_requires_two_groups() -> None:
    with pytest.raises(ValueError, match="two sweep groups"):
        LeaveOneGroupOut(np.zeros(10, dtype=int))


def test_real_signal_generalises(validator, grid, apertures) -> None:
    result = validator.validate_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=9))

    assert result.cv_r2 > 0.8
    assert result.n_folds == 4


def test_noise_fails_to_generalise(validator) -> None:
    """In-sample R² can look mildly encouraging on noise; held-out R² should go negative."""
    response = np.random.default_rng(4).normal(size=len(validator.apertures))

    result = validator.validate_unit(response)

    assert result.cv_r2 < result.fit.r2
    assert result.cv_r2 < 0.1


def test_optimism_is_larger_for_noise_than_for_signal(validator, grid, apertures) -> None:
    signal = validator.validate_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=9))
    noise = validator.validate_unit(np.random.default_rng(4).normal(size=len(validator.apertures)))

    assert noise.optimism > signal.optimism


def test_random_frame_split_leaks_on_autocorrelated_noise(grid, apertures, sequence, fit_config) -> None:
    """A random frame split turns a failing model into an apparently successful one.

    The leak needs two ingredients: apertures that barely change between neighbouring frames,
    and a response that is correlated across those frames. Both hold for real activations. The
    signal here has no pRF structure at all, so an honest split must score it below zero.
    """
    response = autocorrelated_noise(len(apertures), width=3)

    grouped = CrossValidator(grid, apertures, sequence.group, fit_config)
    grouped_score = grouped.validate_unit(response).cv_r2

    shuffled = np.random.default_rng(0).permutation(len(apertures)) % 4
    leaky = CrossValidator(grid, apertures, shuffled, fit_config)
    leaky_score = leaky.validate_unit(response).cv_r2

    assert grouped_score < 0.0
    assert leaky_score > grouped_score


def test_white_noise_does_not_leak_across_a_random_split(grid, apertures, sequence, fit_config) -> None:
    """The counterpart: with no temporal correlation there is nothing for a split to leak.

    This pins the mechanism. If this test ever starts failing alongside the one above, the
    cause is aperture overlap rather than response autocorrelation.
    """
    response = np.random.default_rng(21).normal(size=len(apertures))

    grouped = CrossValidator(grid, apertures, sequence.group, fit_config)
    shuffled = np.random.default_rng(0).permutation(len(apertures)) % 4
    leaky = CrossValidator(grid, apertures, shuffled, fit_config)

    difference = leaky.validate_unit(response).cv_r2 - grouped.validate_unit(response).cv_r2

    assert abs(difference) < 0.2


def test_held_out_score_uses_train_amplitude(validator, grid, apertures, sequence, fit_config) -> None:
    """Re-solving amplitude on the test fold would leak; confirm it is not done."""
    response = synthesise(grid, apertures, TRUTH, noise=0.2, seed=3)
    train, test, fitter = validator._folds[0]

    train_fit = fitter.fit_unit(response[train])
    field = GaussianReceptiveField(train_fit.x0, train_fit.y0, train_fit.sigma)
    expected = _r_squared(
        response[test],
        train_fit.beta * predict(field.weights(grid), apertures[test]) + train_fit.baseline,
    )

    assert validator.validate_unit(response).fold_r2[0] == pytest.approx(expected)


def test_validate_all_rejects_wrong_frame_count(validator) -> None:
    with pytest.raises(ValueError, match="one row per stimulus frame"):
        validator.validate_all(np.zeros((len(validator.apertures) + 1, 2)))


def test_cross_validated_fit_reports_fold_dispersion(validator, grid, apertures) -> None:
    result = validator.validate_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=9))

    assert np.isfinite(result.cv_r2_sd)
    assert result.cv_r2_sd >= 0.0


# --- fold sizing --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_stack():
    """A deliberately tiny stimulus, so fold-sizing tests do not build eighty fitters."""
    config = StimulusConfig(resolution=32, n_steps=4, directions=(0, 90))
    grid = Grid(32)
    apertures = build_apertures(config, "bar").as_float()
    return grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, 5.0))


def test_every_frame_can_be_its_own_group(small_stack) -> None:
    """Nothing forbids single-frame groups: each still leaves plenty of training frames."""
    grid, apertures, config = small_stack
    groups = np.arange(len(apertures))

    validator = CrossValidator(grid, apertures, groups, config)

    assert len(validator._folds) == len(apertures)


def test_a_fold_with_too_few_training_frames_is_skipped(small_stack) -> None:
    """Holding out almost everything leaves too few frames to fit five parameters."""
    grid, apertures, config = small_stack
    groups = np.array([0] * (len(apertures) - 1) + [1])

    validator = CrossValidator(grid, apertures, groups, config)

    assert len(validator._folds) == 1
    _, test, _ = validator._folds[0]
    assert list(test) == [len(apertures) - 1]


def test_a_split_where_no_fold_is_large_enough_is_rejected(small_stack) -> None:
    grid, apertures, config = small_stack
    half = len(apertures) // 2
    groups = np.array([0] * half + [1] * (len(apertures) - half))

    with pytest.raises(ValueError, match="no fold retains enough frames"):
        CrossValidator(grid, apertures, groups, config)


def test_splitter_rejects_multidimensional_groups() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        LeaveOneGroupOut(np.zeros((4, 2), dtype=int))


def test_splitter_length_is_the_group_count(sequence) -> None:
    assert len(LeaveOneGroupOut(sequence.group)) == len(np.unique(sequence.group))


def test_an_unfittable_unit_scores_undefined_rather_than_zero(validator) -> None:
    """A fold that could not be fitted must not contribute a score of zero to the mean."""
    result = validator.validate_unit(np.full(len(validator.apertures), 3.0))

    assert np.isnan(result.cv_r2)
    assert np.isnan(result.cv_r2_sd)


def test_validate_all_rejects_a_non_matrix(validator) -> None:
    with pytest.raises(ValueError, match="frames, units"):
        validator.validate_all(np.zeros(len(validator.apertures)))


def test_validate_all_fits_every_column(validator, grid, apertures) -> None:
    activations = np.column_stack(
        [synthesise(grid, apertures, TRUTH, noise=0.2, seed=s) for s in (1, 2)]
    )

    results = validator.validate_all(activations)

    assert len(results) == 2
    assert all(r.cv_r2 > 0.8 for r in results)
