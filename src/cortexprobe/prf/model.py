"""The population receptive field model.

A pRF is a weighting over the visual field. Its predicted response to a stimulus frame is the
overlap between that weighting and the exposed aperture, following Dumoulin & Wandell (2008),
*NeuroImage* 39:647-660.

One deliberate departure from the fMRI procedure: no haemodynamic convolution. A convolutional
network has no haemodynamics, and adding an HRF would be a biological detail this model does not
possess. The predicted timecourse is therefore the raw overlap sequence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..arrays import FloatArray
from ..geometry import Grid


class ReceptiveField(ABC):
    """A normalised weighting over the visual field."""

    @abstractmethod
    def weights(self, grid: Grid) -> FloatArray:
        """Return an ``(H, W)`` array of field weights on ``grid``."""


@dataclass(frozen=True)
class GaussianReceptiveField(ReceptiveField):
    """An isotropic 2D Gaussian receptive field.

    Parameters are in the pixel coordinates defined by :class:`~cortexprobe.geometry.Grid`:
    the origin is the centre of gaze, x increases rightward, y increases upward.
    """

    x0: float
    y0: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def eccentricity(self) -> float:
        return float(np.hypot(self.x0, self.y0))

    @property
    def polar_angle(self) -> float:
        return float(np.degrees(np.arctan2(self.y0, self.x0)) % 360.0)

    def weights(self, grid: Grid) -> FloatArray:
        """Field weights on ``grid``, normalised to unit volume.

        Dividing by ``2 * pi * sigma**2`` makes a wide pRF spread the same total weight more
        thinly rather than accumulate more of it. Under unit-peak normalisation every pRF
        would share a maximum of 1.0, so its overlap with the aperture would grow with sigma
        by construction, and H1 -- pRF size increasing with depth -- would be guaranteed by
        the parameterisation instead of measured.

        The Gaussian is not truncated by this function, but the grid truncates it anyway: the
        unit-volume property holds only while ``sigma`` is small relative to the field. Ninety
        nine per cent of the volume lies within +/- 3 sigma, so it survives while sigma stays
        under roughly ``resolution / 6``; on a 64 px grid a sigma of 20 retains 0.723 of it.
        :class:`~cortexprobe.prf.fit.PRFFitter` refuses a sigma ceiling that breaches this,
        because the shortfall grows with sigma and would bias size-versus-depth comparisons.
        """
        dx = grid.x - self.x0
        dy = grid.y - self.y0
        variance = self.sigma**2
        weights: FloatArray = np.exp(-(dx**2 + dy**2) / (2.0 * variance)) / (
            2.0 * np.pi * variance
        )
        return weights


def predict(weights: FloatArray, apertures: FloatArray) -> FloatArray:
    """Predicted response timecourse for a receptive field under a stimulus sequence.

    The response at frame ``t`` is the summed overlap of the field with the exposed aperture,
    ``r(t) = sum_xy w(x, y) * aperture_t(x, y)``.
    """
    if weights.shape != apertures.shape[1:]:
        raise ValueError(
            f"receptive field {weights.shape} does not match aperture frames {apertures.shape[1:]}"
        )
    return apertures.reshape(len(apertures), -1) @ weights.ravel()


def design_matrix(fields: list[GaussianReceptiveField], grid: Grid, apertures: FloatArray) -> FloatArray:
    """Stack predicted timecourses for a set of candidate fields, one column each.

    Used by the coarse grid search, which scores many candidates against the same activations.
    """
    return np.column_stack([predict(field.weights(grid), apertures) for field in fields])
