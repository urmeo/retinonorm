"""Two-stage pRF fitting with parameter uncertainty.

A coarse grid picks the basin, then nonlinear refinement finds the minimum inside it. Grid
search alone resolves position no better than its own spacing; refinement alone settles into
whichever local minimum happens to be nearest the starting point.

Amplitude and baseline are never searched. At every trial position they are solved in closed
form by linear least squares, so the nonlinear optimiser only ever explores three parameters:
``x0``, ``y0``, ``sigma``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from ..arrays import FloatArray
from ..config import ConfigError, FitConfig
from ..geometry import Grid
from .model import GaussianReceptiveField, predict

# Three nonlinear parameters plus amplitude and baseline solved by projection.
N_PARAMETERS = 5

# A fitted value this close to its search bound is reported as pinned rather than estimated.
BOUND_TOLERANCE = 1e-3

# Fraction of its unit volume a Gaussian must retain inside the sampled field for its overlap
# with an aperture to be a measurement rather than an artefact of truncation. 99 per cent of a
# Gaussian lies within +/- 3 sigma, and the field radius is resolution / 2, so this is met while
# sigma stays under about resolution / 6.
MIN_ON_GRID_VOLUME = 0.99


@dataclass(frozen=True)
class UnitFit:
    """The fitted pRF for one unit, with the uncertainty of its parameters.

    ``converged`` and :attr:`accepted` answer different questions and must not be conflated.
    ``converged`` is the optimiser's own report: did the search terminate successfully?
    :attr:`accepted` is the scientific question: should this unit be reported as a pRF at all?
    A fit can converge cleanly onto an answer that explains almost no variance, and a rejected
    unit is not evidence that the numerics failed.
    """

    x0: float
    y0: float
    sigma: float
    beta: float
    baseline: float
    r2: float
    converged: bool
    n_fev: int
    se_x0: float = float("nan")
    se_y0: float = float("nan")
    se_sigma: float = float("nan")
    second_field_r2: float = float("nan")
    """Variance a second receptive field would explain on top of this one.

    A single Gaussian cannot represent a unit driven by two separated lobes. It settles on one
    of them, reports a sigma belonging to neither, and nothing else in this record would show
    it. Multi-peaked spatial tuning is common in the deeper layers this project intends to
    tap, so a misspecified fit there would be a plausible-looking wrong pRF.

    Measured on a four-direction bar sweep at 64 px, a genuine single-Gaussian unit stays
    under 0.03 from noiseless to 80 per cent noise, pure noise reaches 0.06, and a two-lobe
    unit scores 0.29 to 0.44. It is reported rather than enforced: acceptance does not depend
    on it, so downstream analysis chooses its own cut and states it.
    """
    x0_at_bound: bool = False
    y0_at_bound: bool = False
    sigma_at_bound: bool = False
    dof: int = 0
    r2_threshold: float = float("inf")
    """Variance the fit must explain to be accepted. Defaults to a threshold nothing clears,
    so a fit assembled without one is never mistaken for an accepted pRF."""

    @classmethod
    def failed(cls) -> UnitFit:
        """A unit that could not be fitted. Never reported as a pRF."""
        nan = float("nan")
        return cls(nan, nan, nan, nan, nan, 0.0, False, 0)

    @property
    def eccentricity(self) -> float:
        return float(np.hypot(self.x0, self.y0))

    @property
    def at_bound(self) -> bool:
        """Whether any fitted parameter is pinned against its search bound."""
        return self.x0_at_bound or self.y0_at_bound or self.sigma_at_bound

    @property
    def accepted(self) -> bool:
        """Whether this unit should be reported as a measured pRF.

        A pRF sitting at the edge of the visual field is a real measurement -- receptive
        fields do lie near the field boundary -- so ``x0`` or ``y0`` at a bound is not
        disqualifying. A sigma pinned against the search ceiling is different: it says the
        true size lies outside the range that was searched, so the reported value records
        where the search stopped rather than what the data support.

        A negative ``beta`` is disqualifying too. The amplitude is solved by unconstrained
        least squares, so a unit whose response *falls* when the aperture covers a location
        fits a perfect pRF there with the sign flipped. That is a suppressed unit, not a
        receptive field, and surround suppression and normalisation both produce them in a
        real network. Reported as an ordinary pRF it would contaminate every downstream
        size-versus-depth regression and lesion contrast.
        """
        return (
            self.converged
            and self.r2 >= self.r2_threshold
            and self.beta > 0.0
            and not self.sigma_at_bound
        )

    def confidence_interval(self, parameter: str, level: float = 0.95) -> tuple[float, float]:
        """Two-sided interval for one parameter from the linearised covariance.

        The interval assumes the residuals are approximately Gaussian and the model is locally
        linear at the solution. It is a summary of fit precision, not a guarantee of coverage
        under model misspecification.
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        estimate = getattr(self, parameter)
        error = getattr(self, f"se_{parameter}")
        if not np.isfinite(estimate) or not np.isfinite(error) or self.dof <= 0:
            return float("nan"), float("nan")
        critical = float(student_t.ppf(0.5 + level / 2.0, self.dof))
        return estimate - critical * error, estimate + critical * error


def _solve_amplitude(prediction: FloatArray, response: FloatArray) -> tuple[FloatArray, FloatArray]:
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


def _count_distinct_frames(apertures: FloatArray) -> int:
    """Frames carrying information the rest do not.

    A design can present the same aperture twice: without an haemodynamic response a bar
    sweep and its 180 degree return are bit-identical, so the default eight-direction stimulus
    shows every frame exactly twice. A duplicate contributes a second copy of its own residual
    and its own Jacobian rows. Those copies cancel out of the fitted parameters but not out of
    the degrees of freedom, so counting them would shrink every standard error as though a
    genuinely independent observation had been made.
    """
    flat = apertures.reshape(len(apertures), -1)
    return len({row.tobytes() for row in flat})


def _standard_errors(
    jacobian: FloatArray, cost: float, n_independent: int
) -> tuple[float, float, float]:
    """Linearised standard errors from the Jacobian at the solution.

    ``cov = residual_variance * (J^T J)^-1`` is the usual Gauss-Newton approximation. The
    pseudo-inverse is used because a pRF sitting outside the stimulated field can make
    ``J^T J`` singular, and that case should yield undefined errors rather than an exception.

    ``n_independent`` counts distinct frames, while ``cost`` and ``jacobian`` still cover every
    frame presented. With ``k`` copies of each frame both ``cost`` and ``J^T J`` scale by ``k``
    and the factors cancel, leaving the covariance a duplicate-free stimulus would have given.
    """
    dof = n_independent - N_PARAMETERS
    if dof <= 0:
        return (float("nan"),) * 3
    residual_variance = 2.0 * cost / dof
    covariance = residual_variance * np.linalg.pinv(jacobian.T @ jacobian)
    variances = np.diag(covariance)
    if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
        return (float("nan"),) * 3
    errors = np.sqrt(variances)
    return float(errors[0]), float(errors[1]), float(errors[2])


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
        if len(apertures) <= N_PARAMETERS:
            raise ValueError(
                f"need more than {N_PARAMETERS} frames to fit a pRF; got {len(apertures)}"
            )
        self.grid = grid
        self.apertures = apertures.astype(np.float64, copy=False)
        self.config = config
        self._check_sigma_ceiling()
        self.n_independent_frames = _count_distinct_frames(self.apertures)
        self.candidates = self._build_candidates()
        self._predictions = np.column_stack(
            [predict(candidate.weights(grid), self.apertures) for candidate in self.candidates]
        )

    def _check_sigma_ceiling(self) -> None:
        """Reject a sigma ceiling this grid cannot represent.

        :class:`~cortexprobe.config.FitConfig` already refuses a lower bound under one pixel,
        because a Gaussian narrower than the pixel pitch is under-sampled and silently loses
        its unit volume. The same failure occurs at the other end of the range, by truncation
        rather than under-sampling, and it is the more dangerous of the two: the missing
        volume grows with sigma, so the bias runs against large pRFs. That is precisely the
        axis along which pRF size is compared across depth, so an unguarded ceiling would
        push a headline result in a consistent direction for a reason that is pure geometry.

        Only the fitter can make this check. The bound comes from the fit configuration and
        the field size from the grid, and neither knows about the other on its own.
        """
        _, sigma_high = self.config.sigma_bounds
        weights = GaussianReceptiveField(0.0, 0.0, sigma_high).weights(self.grid)
        volume = float(weights[self.grid.field_mask].sum())
        if volume < MIN_ON_GRID_VOLUME:
            raise ConfigError(
                f"upper sigma bound {sigma_high:g} px keeps only {volume:.3f} of its unit "
                f"volume inside a {self.grid.resolution} px field, under the "
                f"{MIN_ON_GRID_VOLUME} tolerance. A Gaussian this wide relative to the field "
                "is truncated, so its overlap with an aperture understates the true value by "
                "an amount that grows with sigma. Lower the bound to about "
                f"{self.grid.resolution // 6} px, or raise the grid resolution to about "
                f"{int(sigma_high * 6)} px."
            )

    @property
    def n_frames(self) -> int:
        return int(len(self.apertures))

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Search box: centres stay inside the field, sigma inside its configured range."""
        radius = self.grid.radius
        sigma_low, sigma_high = self.config.sigma_bounds
        return (-radius, -radius, sigma_low), (radius, radius, sigma_high)

    def _build_candidates(self) -> list[GaussianReceptiveField]:
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

    def _bounds_hit(self, x0: float, y0: float, sigma: float) -> tuple[bool, bool, bool]:
        """Which of the three searched parameters came to rest against a search bound."""
        lower, upper = self.bounds
        pinned = [
            abs(value - low) <= BOUND_TOLERANCE or abs(value - high) <= BOUND_TOLERANCE
            for value, low, high in zip((x0, y0, sigma), lower, upper)
        ]
        return pinned[0], pinned[1], pinned[2]

    def _second_field_r2(self, response: FloatArray, fitted: FloatArray) -> float:
        """Largest R2 a second candidate pRF would add to the fitted one.

        This is the R2 gain from a two-Gaussian alternative, but in closed form. Every
        candidate prediction is already built, so each is orthogonalised against the fitted
        prediction and the intercept, and its incremental R2 read off directly. No second
        nonlinear search runs, which keeps per-unit cost flat.
        """
        total = float(np.sum((response - response.mean()) ** 2))
        if total <= 0.0:
            return 0.0
        residual = response - fitted
        design = np.column_stack([fitted, np.ones_like(fitted)])
        coefficients, *_ = np.linalg.lstsq(design, self._predictions, rcond=None)
        perpendicular = self._predictions - design @ coefficients
        norms = np.sum(perpendicular**2, axis=0)
        usable = norms > 1e-12
        if not usable.any():
            return 0.0
        gains = (residual @ perpendicular[:, usable]) ** 2 / (norms[usable] * total)
        return float(np.max(gains))

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
        if response.shape != (self.n_frames,):
            raise ValueError("response must have one value per stimulus frame")
        if not np.isfinite(response).all() or np.ptp(response) == 0.0:
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

        x0, y0, sigma = (float(value) for value in solution.x)
        field = GaussianReceptiveField(x0, y0, sigma)
        coefficients, fitted = _solve_amplitude(
            predict(field.weights(self.grid), self.apertures), response
        )
        score = _r_squared(response, fitted)
        se_x0, se_y0, se_sigma = _standard_errors(
            solution.jac, float(solution.cost), self.n_independent_frames
        )
        x0_at_bound, y0_at_bound, sigma_at_bound = self._bounds_hit(x0, y0, sigma)

        return UnitFit(
            x0=x0,
            y0=y0,
            sigma=sigma,
            beta=float(coefficients[0]),
            baseline=float(coefficients[1]),
            r2=score,
            converged=bool(solution.success),
            n_fev=int(solution.nfev),
            se_x0=se_x0,
            se_y0=se_y0,
            se_sigma=se_sigma,
            second_field_r2=self._second_field_r2(response, fitted),
            x0_at_bound=x0_at_bound,
            y0_at_bound=y0_at_bound,
            sigma_at_bound=sigma_at_bound,
            dof=self.n_independent_frames - N_PARAMETERS,
            r2_threshold=self.config.r2_threshold,
        )

    def fit_all(self, activations: FloatArray) -> list[UnitFit]:
        """Fit every unit in a ``(frames, units)`` activation matrix."""
        activations = np.asarray(activations, dtype=np.float64)
        if activations.ndim != 2:
            raise ValueError("activations must have shape (frames, units)")
        if activations.shape[0] != self.n_frames:
            raise ValueError("activations must have one row per stimulus frame")
        return [self.fit_unit(activations[:, unit]) for unit in range(activations.shape[1])]
