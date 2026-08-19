"""Two-stage pRF fitting.

A coarse grid picks the basin, then nonlinear refinement finds the minimum inside it. Grid
search alone resolves position no better than its own spacing; refinement alone settles into
whichever local minimum happens to be nearest the starting point.

Amplitude and baseline are never searched. At every trial position they are solved in closed
form by linear least squares, so the nonlinear optimiser only ever explores three parameters:
``x0``, ``y0``, ``sigma``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from ..config import FitConfig
from ..geometry import Grid
from .model import GaussianReceptiveField, predict

FloatArray = np.ndarray


@dataclass(frozen=True)
class UnitFit:
    """The fitted pRF for one unit."""

    x0: float
    y0: float
    sigma: float
    beta: float
    baseline: float
    r2: float
    converged: bool
    n_fev: int

    @classmethod
    def failed(cls) -> UnitFit:
        """A unit that could not be fitted. Never reported as a pRF."""
        nan = float("nan")
        return cls(nan, nan, nan, nan, nan, 0.0, False, 0)

    @property
    def eccentricity(self) -> float:
        return float(np.hypot(self.x0, self.y0))


def _solve_amplitude(prediction: FloatArray, response: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """Least-squares fit of ``response ~ beta * prediction + baseline``."""
    design = np.column_stack([prediction, np.ones_like(prediction)])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return coefficients, design @ coefficients


def _r_squared(response: FloatArray, fitted: FloatArray) -> float:
    total = float(np.sum((response - response.mean()) ** 2))
    if total <= 0.0:
        return 0.0
    residual = float(np.sum((response - fitted) ** 2))
    return 1.0 - residual / total


class PRFFitter:
    """Fits Gaussian pRFs to activation timecourses recorded under a known aperture sequence.

    The candidate predictions are built once per fitter because they depend only on the
    stimulus, not on the unit being fitted. Rebuilding them per unit would repeat identical
    work for every unit in a layer.
    """

    def __init__(self, grid: Grid, apertures: FloatArray, config: FitConfig) -> None:
        if apertures.ndim != 3:
            raise ValueError("apertures must have shape (frames, height, width)")
        if apertures.shape[1:] != grid.shape:
            raise ValueError("aperture frames must match the grid")
        self.grid = grid
        self.apertures = apertures.astype(np.float64, copy=False)
        self.config = config
        self.candidates = self._build_candidates()
        self._predictions = np.column_stack(
            [predict(candidate.weights(grid), self.apertures) for candidate in self.candidates]
        )

    @property
    def bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Search box: centres stay inside the field, sigma inside its configured range."""
        radius = self.grid.radius
        sigma_low, sigma_high = self.config.sigma_bounds
        return (-radius, -radius, sigma_low), (radius, radius, sigma_high)

    def _build_candidates(self) -> List[GaussianReceptiveField]:
        radius = self.grid.radius
        sigma_low, sigma_high = self.config.sigma_bounds
        n_sigma = max(3, self.config.grid_size // 2)

        centres = np.linspace(-radius, radius, self.config.grid_size)
        sigmas = np.geomspace(sigma_low, sigma_high, n_sigma)

        candidates = [
            GaussianReceptiveField(float(x), float(y), float(sigma))
            for x in centres
            for y in centres
            if np.hypot(x, y) <= radius
            for sigma in sigmas
        ]
        if not candidates:
            raise ValueError("candidate grid is empty; check grid_size and sigma_bounds")
        return candidates

    def _coarse_search(self, response: FloatArray) -> GaussianReceptiveField:
        best_index, best_r2 = 0, -np.inf
        for index in range(self._predictions.shape[1]):
            _, fitted = _solve_amplitude(self._predictions[:, index], response)
            score = _r_squared(response, fitted)
            if score > best_r2:
                best_index, best_r2 = index, score
        return self.candidates[best_index]

    def fit_unit(self, response: FloatArray) -> UnitFit:
        response = np.asarray(response, dtype=np.float64)
        if response.shape != (len(self.apertures),):
            raise ValueError("response must have one value per stimulus frame")
        if not np.isfinite(response).all():
            return UnitFit.failed()
        if np.ptp(response) == 0.0:
            return UnitFit.failed()

        start = self._coarse_search(response)

        def residual(params: Sequence[float]) -> FloatArray:
            x0, y0, sigma = params
            field = GaussianReceptiveField(float(x0), float(y0), float(sigma))
            _, fitted = _solve_amplitude(predict(field.weights(self.grid), self.apertures), response)
            return fitted - response

        try:
            solution = least_squares(
                residual,
                x0=[start.x0, start.y0, start.sigma],
                bounds=self.bounds,
                max_nfev=self.config.max_nfev,
            )
        except (ValueError, np.linalg.LinAlgError):
            return UnitFit.failed()

        x0, y0, sigma = (float(v) for v in solution.x)
        field = GaussianReceptiveField(x0, y0, sigma)
        coefficients, fitted = _solve_amplitude(
            predict(field.weights(self.grid), self.apertures), response
        )
        score = _r_squared(response, fitted)

        return UnitFit(
            x0=x0,
            y0=y0,
            sigma=sigma,
            beta=float(coefficients[0]),
            baseline=float(coefficients[1]),
            r2=score,
            converged=bool(solution.success) and score >= self.config.r2_threshold,
            n_fev=int(solution.nfev),
        )

    def fit_all(self, activations: FloatArray) -> List[UnitFit]:
        """Fit every unit in an ``(frames, units)`` activation matrix."""
        activations = np.asarray(activations, dtype=np.float64)
        if activations.ndim != 2:
            raise ValueError("activations must have shape (frames, units)")
        if activations.shape[0] != len(self.apertures):
            raise ValueError("activations must have one row per stimulus frame")
        return [self.fit_unit(activations[:, unit]) for unit in range(activations.shape[1])]
