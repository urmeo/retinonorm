"""The pRF model itself: the weighting, and the response it predicts.

This is the mathematics the rest of the instrument rests on. The unit-volume normalisation in
particular is a scientific choice, not a numerical convenience -- under unit-peak normalisation
the overlap between a pRF and an aperture would grow with sigma by construction, which is the
very quantity the size-versus-depth hypothesis intends to measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.geometry import Grid
from cortexprobe.prf.model import GaussianReceptiveField, design_matrix, predict


def test_a_receptive_field_needs_a_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        GaussianReceptiveField(0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="sigma must be positive"):
        GaussianReceptiveField(0.0, 0.0, -1.0)


@pytest.mark.parametrize(
    ("x0", "y0", "eccentricity", "angle"),
    [(3.0, 4.0, 5.0, 53.13010235), (0.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 1.0, 180.0),
     (0.0, -2.0, 2.0, 270.0)],
)
def test_position_is_reported_in_polar_coordinates_too(x0, y0, eccentricity, angle) -> None:
    field = GaussianReceptiveField(x0, y0, 3.0)

    assert field.eccentricity == pytest.approx(eccentricity)
    assert field.polar_angle == pytest.approx(angle)


def test_weights_carry_unit_volume() -> None:
    grid = Grid(64)

    for sigma in (1.0, 3.0, 5.0, 10.0):
        assert GaussianReceptiveField(0.0, 0.0, sigma).weights(grid).sum() == pytest.approx(
            1.0, abs=5e-3
        )


def test_a_wider_field_spreads_the_same_weight_more_thinly() -> None:
    """The property that keeps pRF size against depth a measurement rather than a tautology."""
    grid = Grid(64)

    narrow = GaussianReceptiveField(0.0, 0.0, 2.0).weights(grid)
    wide = GaussianReceptiveField(0.0, 0.0, 6.0).weights(grid)

    assert wide.max() < narrow.max()
    assert wide.sum() == pytest.approx(narrow.sum(), rel=1e-3)


def test_weights_peak_at_the_declared_centre() -> None:
    grid = Grid(64)

    weights = GaussianReceptiveField(10.0, -6.0, 3.0).weights(grid)
    row, column = np.unravel_index(int(np.argmax(weights)), weights.shape)

    assert grid.x[row, column] == pytest.approx(10.0, abs=1.0)
    assert grid.y[row, column] == pytest.approx(-6.0, abs=1.0)


def test_prediction_is_the_overlap_with_each_aperture() -> None:
    grid = Grid(32)
    weights = GaussianReceptiveField(0.0, 0.0, 3.0).weights(grid)
    apertures = np.stack([np.ones(grid.shape), np.zeros(grid.shape)])

    response = predict(weights, apertures)

    assert response[0] == pytest.approx(weights.sum())
    assert response[1] == pytest.approx(0.0)


def test_prediction_rejects_a_field_that_does_not_match_the_frames() -> None:
    grid = Grid(32)
    weights = GaussianReceptiveField(0.0, 0.0, 3.0).weights(grid)

    with pytest.raises(ValueError, match="does not match aperture frames"):
        predict(weights, np.ones((4, 16, 16)))


def test_prediction_scales_linearly_with_the_aperture() -> None:
    grid = Grid(32)
    weights = GaussianReceptiveField(2.0, 2.0, 4.0).weights(grid)
    apertures = np.random.default_rng(0).random((5, *grid.shape))

    assert np.allclose(predict(weights, 2.0 * apertures), 2.0 * predict(weights, apertures))


def test_design_matrix_stacks_one_column_per_candidate() -> None:
    grid = Grid(32)
    apertures = np.random.default_rng(1).random((7, *grid.shape))
    fields = [
        GaussianReceptiveField(0.0, 0.0, 3.0),
        GaussianReceptiveField(5.0, -5.0, 2.0),
        GaussianReceptiveField(-4.0, 4.0, 4.0),
    ]

    matrix = design_matrix(fields, grid, apertures)

    assert matrix.shape == (7, 3)
    for column, field in enumerate(fields):
        assert np.allclose(matrix[:, column], predict(field.weights(grid), apertures))


def test_a_field_far_outside_the_grid_contributes_nothing() -> None:
    grid = Grid(32)

    weights = GaussianReceptiveField(500.0, 500.0, 2.0).weights(grid)

    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(0.0, abs=1e-12)
