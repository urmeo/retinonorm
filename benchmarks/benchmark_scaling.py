"""Runtime scaling of the pRF fitter.

Answers two questions the design depends on. First, does per-unit cost stay flat as the
number of units grows, or does something quadratic hide in the loop? Second, how much does
leave-one-group-out cross-validation multiply that cost?

Run directly; it prints a table and writes nothing.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from cortexprobe.config import FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures

RESOLUTION = 64
UNIT_COUNTS = (1, 5, 10, 25, 50, 100)


def make_units(grid: Grid, apertures: np.ndarray, n_units: int, seed: int = 0) -> np.ndarray:
    """Activations from randomly placed pRFs, one column per unit."""
    rng = np.random.default_rng(seed)
    radius = grid.radius * 0.7
    columns = []
    for _ in range(n_units):
        x0, y0 = rng.uniform(-radius, radius, size=2)
        sigma = rng.uniform(3.0, 10.0)
        clean = predict(GaussianReceptiveField(x0, y0, sigma).weights(grid), apertures)
        columns.append(clean + rng.normal(0.0, 0.2 * clean.std(), size=clean.shape))
    return np.column_stack(columns)


def time_call(function, *args) -> tuple[float, object]:
    start = time.perf_counter()
    result = function(*args)
    return time.perf_counter() - start, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--max-units", type=int, default=max(UNIT_COUNTS))
    arguments = parser.parse_args()

    grid = Grid(arguments.resolution)
    config = StimulusConfig(
        resolution=arguments.resolution, n_steps=20, directions=(0, 45, 90, 135)
    )
    sequence = build_apertures(config, "bar")
    apertures = sequence.as_float()
    fit_config = FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0))

    setup, fitter = time_call(PRFFitter, grid, apertures, fit_config)
    print(
        f"grid {arguments.resolution}px | {fitter.n_frames} frames | "
        f"{len(fitter.candidates)} candidates | setup {setup * 1000:.0f} ms\n"
    )

    counts: list[int] = [n for n in UNIT_COUNTS if n <= arguments.max_units]
    print(
        f"{'units':>6} {'fit (s)':>9} {'per unit (ms)':>14} "
        f"{'cv (s)':>9} {'cv per unit (ms)':>17} {'cv factor':>10}"
    )
    print("-" * 70)

    validator = CrossValidator(grid, apertures, sequence.group, fit_config)
    for n_units in counts:
        activations = make_units(grid, apertures, n_units)
        fit_seconds, _ = time_call(fitter.fit_all, activations)
        cv_seconds, _ = time_call(validator.validate_all, activations)
        print(
            f"{n_units:>6} {fit_seconds:>9.3f} {fit_seconds / n_units * 1000:>14.1f} "
            f"{cv_seconds:>9.3f} {cv_seconds / n_units * 1000:>17.1f} "
            f"{cv_seconds / fit_seconds:>10.1f}x"
        )

    largest = counts[-1]
    activations = make_units(grid, apertures, largest)
    per_unit = time_call(fitter.fit_all, activations)[0] / largest
    print(f"\nprojected: 10k units ~ {per_unit * 10_000 / 60:.1f} min single-threaded")


if __name__ == "__main__":
    main()
