"""Held-out validation and the leakage it is designed to expose.

The decisive pair is :func:`test_random_frame_split_leaks_on_autocorrelated_noise` and its
white-noise counterpart. Together they show why folds are split by sweep group, and pin the
mechanism: the leak needs both overlapping apertures and a temporally correlated response.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.prf.fit import PRFFitter, _r_squared
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator, LeaveOneGroupOut

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


def test_validator_rejects_undersized_folds(grid, apertures, fit_config) -> None:
    tiny = PRFFitter(grid, apertures, fit_config)
    groups = np.arange(tiny.n_frames)  # every frame its own group -> folds still large enough

    validator = CrossValidator(grid, apertures, groups, fit_config)

    assert len(validator._folds) == tiny.n_frames
