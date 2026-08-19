"""Visual field coordinate system.

Stimuli and receptive fields must agree on where the centre of gaze is and which way is up,
or a fitted pRF position means nothing. Both sides take their coordinates from :class:`Grid`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .arrays import BoolArray, FloatArray


@dataclass(frozen=True)
class Grid:
    """A square visual field sampled on a pixel lattice.

    The origin sits at the geometric centre of the field. Coordinates are in pixels, x
    increasing rightward and y increasing upward, so a fitted ``y0`` reads the way a person
    would expect rather than the way image arrays are indexed.
    """

    resolution: int

    def __post_init__(self) -> None:
        if self.resolution < 2:
            raise ValueError("grid resolution must be at least 2")

    @property
    def radius(self) -> float:
        """Largest eccentricity that fits inside the square field."""
        return self.resolution / 2.0

    @cached_property
    def _axis(self) -> FloatArray:
        centre = (self.resolution - 1) / 2.0
        return np.arange(self.resolution, dtype=np.float64) - centre

    @cached_property
    def x(self) -> FloatArray:
        return np.tile(self._axis, (self.resolution, 1))

    @cached_property
    def y(self) -> FloatArray:
        return np.flipud(np.tile(self._axis[:, None], (1, self.resolution)))

    @cached_property
    def eccentricity(self) -> FloatArray:
        return np.hypot(self.x, self.y)

    @cached_property
    def polar_angle(self) -> FloatArray:
        """Angle in degrees, counter-clockwise from the positive x axis, in [0, 360)."""
        angle: FloatArray = np.degrees(np.arctan2(self.y, self.x)) % 360.0
        return angle

    @cached_property
    def field_mask(self) -> BoolArray:
        """Circular aperture inscribed in the square field."""
        return self.eccentricity <= self.radius

    @property
    def shape(self) -> tuple[int, int]:
        return (self.resolution, self.resolution)
