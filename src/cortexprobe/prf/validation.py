"""Held-out validation of fitted pRFs.

In-sample R² measures how well three free parameters can be bent to fit one timecourse. It
says nothing about whether the pRF generalises to stimulus positions the fit never saw.

Folds are split by *sweep group*, never by individual frame. Adjacent frames within a sweep
show overlapping apertures, so a random frame split would put near-duplicates on both sides of
the train/test boundary and report a generalisation score that is really memorisation.

A group is a sweep *axis*, not a sweep direction: see :class:`~cortexprobe.stimuli.BarSweep`
for why a 180 degree return sweep is the same stimulus run backwards.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ..arrays import FloatArray, IntArray
from ..config import FitConfig
from ..geometry import Grid
from .fit import N_PARAMETERS, PRFFitter, UnitFit, _r_squared
from .model import GaussianReceptiveField, predict


@dataclass(frozen=True)
class CrossValidatedFit:
    """A pRF fitted on all frames, scored on frames it never saw."""

    fit: UnitFit
    fold_r2: tuple[float, ...]

    @property
    def cv_r2(self) -> float:
        """Mean held-out R². Lower than in-sample R² for any honest fit."""
        scores = [score for score in self.fold_r2 if np.isfinite(score)]
        return float(np.mean(scores)) if scores else float("nan")

    @property
    def cv_r2_sd(self) -> float:
        scores = [score for score in self.fold_r2 if np.isfinite(score)]
        return float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan")

    @property
    def n_folds(self) -> int:
        return len(self.fold_r2)

    @property
    def optimism(self) -> float:
        """In-sample R² minus held-out R². Large values indicate overfitting."""
        return self.fit.r2 - self.cv_r2


class LeaveOneGroupOut:
    """Iterates folds that hold out one whole sweep group at a time."""

    def __init__(self, groups: Sequence[int]) -> None:
        self.groups = np.asarray(groups)
        if self.groups.ndim != 1:
            raise ValueError("groups must be one-dimensional")
        self.unique = np.unique(self.groups)
        if len(self.unique) < 2:
            raise ValueError("cross-validation needs at least two sweep groups")

    def __len__(self) -> int:
        return len(self.unique)

    def splits(self) -> Iterator[tuple[IntArray, IntArray]]:
        for held_out in self.unique:
            test = np.flatnonzero(self.groups == held_out)
            train = np.flatnonzero(self.groups != held_out)
            yield train, test


class CrossValidator:
    """Fits and cross-validates pRFs against a grouped stimulus sequence.

    One fitter is built per fold and reused across every unit. Building them per unit would
    rebuild identical candidate predictions for each of possibly thousands of units.
    """

    def __init__(
        self,
        grid: Grid,
        apertures: FloatArray,
        groups: Sequence[int],
        config: FitConfig,
    ) -> None:
        self.grid = grid
        self.apertures = np.asarray(apertures, dtype=np.float64)
        self.config = config
        self.splitter = LeaveOneGroupOut(groups)

        self._full = PRFFitter(grid, self.apertures, config)
        self._folds: list[tuple[IntArray, IntArray, PRFFitter]] = []
        for train, test in self.splitter.splits():
            if len(train) <= N_PARAMETERS:
                continue
            self._folds.append((train, test, PRFFitter(grid, self.apertures[train], config)))
        if not self._folds:
            raise ValueError("no fold retains enough frames to fit a pRF")

    def validate_unit(self, response: FloatArray) -> CrossValidatedFit:
        response = np.asarray(response, dtype=np.float64)
        full_fit = self._full.fit_unit(response)

        scores: list[float] = []
        for train, test, fitter in self._folds:
            train_fit = fitter.fit_unit(response[train])
            scores.append(self._score_held_out(train_fit, test, response))
        return CrossValidatedFit(fit=full_fit, fold_r2=tuple(scores))

    def _score_held_out(self, train_fit: UnitFit, test: IntArray, response: FloatArray) -> float:
        """Apply train-fold parameters, unchanged, to held-out frames.

        Amplitude and baseline come from the training fold too. Re-solving them on the test
        frames would leak the held-out data back into the prediction.
        """
        if not np.isfinite([train_fit.x0, train_fit.y0, train_fit.sigma]).all():
            return float("nan")
        field = GaussianReceptiveField(train_fit.x0, train_fit.y0, train_fit.sigma)
        prediction = predict(field.weights(self.grid), self.apertures[test])
        fitted = train_fit.beta * prediction + train_fit.baseline
        return _r_squared(response[test], fitted)

    def validate_all(self, activations: FloatArray) -> list[CrossValidatedFit]:
        activations = np.asarray(activations, dtype=np.float64)
        if activations.ndim != 2:
            raise ValueError("activations must have shape (frames, units)")
        if activations.shape[0] != len(self.apertures):
            raise ValueError("activations must have one row per stimulus frame")
        return [self.validate_unit(activations[:, unit]) for unit in range(activations.shape[1])]
