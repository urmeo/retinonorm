"""Retinotopic mapping stimuli.

The same aperture sequence serves two roles: it is shown to the network, and it is the
regressor the pRF model is fitted against. Generating it once, here, keeps those two uses
from drifting apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .config import ConfigError, StimulusConfig
from .geometry import Grid

BoolArray = np.ndarray
FloatArray = np.ndarray


def frame_similarity(stack: BoolArray) -> FloatArray:
    """Pairwise cosine similarity between aperture frames.

    Cosine rather than raw overlap, so a large frame and a small one are not judged similar
    merely because the large one covers the small one's pixels along with many others.
    """
    flat = stack.reshape(len(stack), -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1)
    norms[norms == 0.0] = 1.0
    unit = flat / norms[:, None]
    return unit @ unit.T


def _prune_leaking_frames(stack: BoolArray, groups: np.ndarray, threshold: float) -> np.ndarray:
    """Frame indices to keep so no cross-group pair overlaps above ``threshold``.

    Group labels come from index arithmetic -- a wedge start angle, a ring block -- which knows
    nothing about how much neighbouring apertures actually share. Frames adjacent to a group
    boundary can therefore overlap a training frame almost completely. Rather than trust the
    arithmetic, the offending frames are measured and dropped.

    Frames are removed one at a time, always the one in the most surviving violations, until
    none remain. Dropping frames costs a little data; leaving them makes every held-out score
    on this design an overestimate of unknown size.
    """
    similarity = frame_similarity(stack)
    violating = (similarity > threshold) & (groups[:, None] != groups[None, :])
    keep = np.ones(len(stack), dtype=bool)
    while True:
        live = violating & keep[:, None] & keep[None, :]
        counts = live.sum(axis=1)
        if not counts.any():
            return np.flatnonzero(keep)
        keep[int(np.argmax(counts))] = False


@dataclass(frozen=True)
class ApertureSequence:
    """A stack of binary apertures with the frame metadata needed to interpret it."""

    apertures: BoolArray
    grid: Grid
    kind: str
    frame_index: np.ndarray
    group: np.ndarray

    def __post_init__(self) -> None:
        if self.apertures.dtype != np.bool_:
            raise TypeError("apertures must be boolean")
        if self.apertures.shape[1:] != self.grid.shape:
            raise ValueError("aperture frames must match the grid")
        if len(self.frame_index) != len(self.apertures):
            raise ValueError("frame_index must label every frame")
        if len(self.group) != len(self.apertures):
            raise ValueError("group must label every frame")

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
        frames, labels, groups = self._frames()
        stack = np.stack(frames) & self.grid.field_mask
        group_array = np.asarray(groups)

        keep = _prune_leaking_frames(stack, group_array, self.config.max_fold_similarity)
        if len(np.unique(group_array)) >= 2 > len(np.unique(group_array[keep])):
            raise ConfigError(
                f"{self.kind} apertures define "
                f"{len(np.unique(group_array))} cross-validation groups but only "
                f"{len(np.unique(group_array[keep]))} survive pruning at "
                f"max_fold_similarity={self.config.max_fold_similarity}; raise the threshold or "
                "increase n_steps so neighbouring frames overlap less"
            )
        return ApertureSequence(
            apertures=stack[keep],
            grid=self.grid,
            kind=self.kind,
            frame_index=np.asarray(labels)[keep],
            group=group_array[keep],
        )

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Frames this design lays out, before leakage pruning removes any.

        :meth:`build` may return fewer: see :func:`_prune_leaking_frames`. Read
        :attr:`ApertureSequence.n_frames` for the count actually presented.
        """

    @abstractmethod
    def _frames(self) -> tuple[list[BoolArray], list[int], list[int]]:
        """Return the unmasked frames, their labels, and their cross-validation groups.

        Frames within a group are strongly overlapping and must never be split across a
        train/test boundary. Groups are the smallest unit a fold may contain.
        """


class BarSweep(ApertureGenerator):
    """A bar traverses the field once per direction, perpendicular to its heading.

    Frames are grouped by sweep *axis* (``direction % 180``), not by direction. The bar
    position depends on ``x cos(theta) + y sin(theta)``, which negates under a 180 degree
    turn, while the sweep offsets run symmetrically from ``-radius`` to ``+radius``. Frame
    ``(d, k)`` is therefore bit-identical to frame ``(d + 180, n_steps - 1 - k)``. Without an
    haemodynamic response there is no temporal asymmetry to break the tie, so a return sweep
    carries no information its outbound partner does not. Grouping by direction would place
    those identical frames on opposite sides of a cross-validation boundary and score
    memorisation as generalisation.
    """

    kind = "bar"

    @property
    def n_frames(self) -> int:
        return self.config.n_bar_frames

    def _frames(self) -> tuple[list[BoolArray], list[int], list[int]]:
        grid = self.grid
        half_width = self.config.bar_width_px / 2.0
        travel = np.linspace(-grid.radius, grid.radius, self.config.n_steps)

        frames: list[BoolArray] = []
        labels: list[int] = []
        groups: list[int] = []
        for direction in self.config.directions:
            theta = np.radians(direction)
            projection = grid.x * np.cos(theta) + grid.y * np.sin(theta)
            for step, offset in enumerate(travel):
                frames.append(np.abs(projection - offset) <= half_width)
                labels.append(direction * 1000 + step)
                groups.append(direction % 180)
        return frames, labels, groups


class RotatingWedge(ApertureGenerator):
    """A polar-angle wedge rotates through a full revolution."""

    kind = "wedge"

    @property
    def n_frames(self) -> int:
        return self.config.n_steps

    def _frames(self) -> tuple[list[BoolArray], list[int], list[int]]:
        grid = self.grid
        span = self.config.wedge_span_deg
        starts = np.linspace(0.0, 360.0, self.config.n_steps, endpoint=False)

        frames: list[BoolArray] = []
        labels: list[int] = []
        groups: list[int] = []
        for step, start in enumerate(starts):
            delta = (grid.polar_angle - start) % 360.0
            frames.append(delta <= span)
            labels.append(step)
            groups.append(int(start // 90.0))
        return frames, labels, groups


class ExpandingRing(ApertureGenerator):
    """An annulus expands from the centre to the edge of the field."""

    kind = "ring"

    @property
    def n_frames(self) -> int:
        return self.config.n_steps

    def _frames(self) -> tuple[list[BoolArray], list[int], list[int]]:
        grid = self.grid
        thickness = self.config.ring_thickness_px
        centres = np.linspace(0.0, grid.radius, self.config.n_steps)

        block = max(1, self.config.n_steps // 4)
        frames: list[BoolArray] = []
        labels: list[int] = []
        groups: list[int] = []
        for step, centre in enumerate(centres):
            frames.append(np.abs(grid.eccentricity - centre) <= thickness / 2.0)
            labels.append(step)
            groups.append(step // block)
        return frames, labels, groups


GENERATORS = {
    generator.kind: generator for generator in (BarSweep, RotatingWedge, ExpandingRing)
}


def build_apertures(config: StimulusConfig, kind: str = "bar") -> ApertureSequence:
    try:
        generator = GENERATORS[kind]
    except KeyError:
        raise ValueError(f"unknown stimulus kind {kind!r}; expected one of {sorted(GENERATORS)}")
    return generator(config).build()
