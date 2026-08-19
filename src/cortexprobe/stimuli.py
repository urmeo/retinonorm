"""Retinotopic mapping stimuli.

The same aperture sequence serves two roles: it is shown to the network, and it is the
regressor the pRF model is fitted against. Generating it once, here, keeps those two uses
from drifting apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import StimulusConfig
from .geometry import Grid

BoolArray = np.ndarray
FloatArray = np.ndarray


@dataclass(frozen=True)
class ApertureSequence:
    """A stack of binary apertures with the frame metadata needed to interpret it."""

    apertures: BoolArray
    grid: Grid
    kind: str
    frame_index: np.ndarray

    def __post_init__(self) -> None:
        if self.apertures.dtype != np.bool_:
            raise TypeError("apertures must be boolean")
        if self.apertures.shape[1:] != self.grid.shape:
            raise ValueError("aperture frames must match the grid")
        if len(self.frame_index) != len(self.apertures):
            raise ValueError("frame_index must label every frame")

    @property
    def n_frames(self) -> int:
        return int(self.apertures.shape[0])

    @property
    def coverage(self) -> FloatArray:
        """Fraction of the field exposed in each frame."""
        return self.apertures.mean(axis=(1, 2))

    def as_float(self) -> FloatArray:
        return self.apertures.astype(np.float64)


class ApertureGenerator(ABC):
    """Builds one family of mapping apertures on a shared grid."""

    kind: str

    def __init__(self, config: StimulusConfig) -> None:
        self.config = config
        self.grid = Grid(config.resolution)

    def build(self) -> ApertureSequence:
        frames, labels = self._frames()
        stack = np.stack(frames) & self.grid.field_mask
        return ApertureSequence(
            apertures=stack,
            grid=self.grid,
            kind=self.kind,
            frame_index=np.asarray(labels),
        )

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Frame count this generator will produce."""

    @abstractmethod
    def _frames(self) -> Tuple[List[BoolArray], List[int]]:
        """Return the unmasked frames and their sweep labels."""


class BarSweep(ApertureGenerator):
    """A bar traverses the field once per direction, perpendicular to its heading."""

    kind = "bar"

    @property
    def n_frames(self) -> int:
        return self.config.n_bar_frames

    def _frames(self) -> Tuple[List[BoolArray], List[int]]:
        grid = self.grid
        half_width = self.config.bar_width_px / 2.0
        travel = np.linspace(-grid.radius, grid.radius, self.config.n_steps)

        frames: List[BoolArray] = []
        labels: List[int] = []
        for direction in self.config.directions:
            theta = np.radians(direction)
            projection = grid.x * np.cos(theta) + grid.y * np.sin(theta)
            for step, offset in enumerate(travel):
                frames.append(np.abs(projection - offset) <= half_width)
                labels.append(direction * 1000 + step)
        return frames, labels


class RotatingWedge(ApertureGenerator):
    """A polar-angle wedge rotates through a full revolution."""

    kind = "wedge"

    @property
    def n_frames(self) -> int:
        return self.config.n_steps

    def _frames(self) -> Tuple[List[BoolArray], List[int]]:
        grid = self.grid
        span = self.config.wedge_span_deg
        starts = np.linspace(0.0, 360.0, self.config.n_steps, endpoint=False)

        frames: List[BoolArray] = []
        labels: List[int] = []
        for step, start in enumerate(starts):
            delta = (grid.polar_angle - start) % 360.0
            frames.append(delta <= span)
            labels.append(step)
        return frames, labels


class ExpandingRing(ApertureGenerator):
    """An annulus expands from the centre to the edge of the field."""

    kind = "ring"

    @property
    def n_frames(self) -> int:
        return self.config.n_steps

    def _frames(self) -> Tuple[List[BoolArray], List[int]]:
        grid = self.grid
        thickness = self.config.ring_thickness_px
        centres = np.linspace(0.0, grid.radius, self.config.n_steps)

        frames: List[BoolArray] = []
        labels: List[int] = []
        for step, centre in enumerate(centres):
            frames.append(np.abs(grid.eccentricity - centre) <= thickness / 2.0)
            labels.append(step)
        return frames, labels


class CarrierPattern:
    """Binary noise refreshed per frame.

    The carrier gives the network something to respond to inside the aperture. It is
    deliberately unstructured: any retinotopic signal recovered from the activations must come
    from the aperture, not from the texture filling it.
    """

    def __init__(self, config: StimulusConfig) -> None:
        self.config = config

    def build(self, n_frames: int) -> FloatArray:
        rng = np.random.default_rng(self.config.carrier_seed)
        shape = (n_frames, self.config.resolution, self.config.resolution)
        return rng.integers(0, 2, size=shape).astype(np.float64)


GENERATORS = {
    generator.kind: generator for generator in (BarSweep, RotatingWedge, ExpandingRing)
}


def build_apertures(config: StimulusConfig, kind: str = "bar") -> ApertureSequence:
    try:
        generator = GENERATORS[kind]
    except KeyError:
        raise ValueError(f"unknown stimulus kind {kind!r}; expected one of {sorted(GENERATORS)}")
    return generator(config).build()
