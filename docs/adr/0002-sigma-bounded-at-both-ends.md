# 0002. Sigma is bounded at both ends by the grid

**Status:** Accepted

## Context

[0001](0001-unit-volume-gaussian.md) relies on every receptive field carrying the same total
weight. The grid breaks that at both extremes, by two different mechanisms.

Below the pixel pitch the Gaussian is under-sampled, and the sum depends entirely on where the
centre falls relative to the lattice. `Grid` places pixel centres at half-integers, so the field
origin is itself half a pixel off-lattice in both axes. At `sigma = 0.2` on a 64 px field:

| pRF centre | on-grid sum |
|---|---|
| the field origin — half a pixel off-lattice in both axes | 0.031 |
| a quarter pixel from the origin | 0.160 |
| half a pixel from the origin, on-lattice in x | 0.350 |
| exactly on a pixel centre | **3.979** |

None is 1.0, and the error runs in *both* directions: a pRF sitting on a pixel collects nearly
four times unit volume, one sitting between pixels a thirtieth of it. An earlier draft of this
record quoted only the undershoot, and mislabelled it as the on-pixel case.

Above roughly `resolution / 6.1` it is truncated by the edge of the field. The radial mass of a
**two-dimensional** Gaussian within 3σ is `1 − exp(−4.5) = 0.9889` — the familiar 99 % figure is
the one-dimensional one — so the 0.99 tolerance needs slightly tighter than `3σ ≤ resolution / 2`.
Measured on the grid, the largest admissible sigma is `resolution / 6.095` at 64, 128 and 256 px
alike, and a sigma of exactly `resolution / 6` is rejected. Measured on a 64 px field:
`sigma = 10` retains 0.994, `sigma = 20` retains 0.723, `sigma = 40` retains 0.275.

Only the floor was guarded. The ceiling is the more dangerous of the two: the shortfall grows
monotonically with sigma, so it biases the fit against large pRFs — the exact axis H1 measures.
The shipped default `sigma_bounds` of `(1.0, 40.0)` retained 0.723 on the default 128 px field.

## Decision

Both ends are guarded, in the place that has the information to do it.

`FitConfig` rejects a lower bound under 1 px, with an error that explains why. It can do this
alone, because the pixel pitch is 1 by definition of the coordinate system.

`PRFFitter` rejects an upper bound whose Gaussian retains less than `MIN_ON_GRID_VOLUME` (0.99)
of its volume inside the field mask. Only the fitter can: the bound comes from the fit
configuration and the field size from the grid, and neither knows about the other.

The default `sigma_bounds` becomes `(1.0, 20.0)`, which the default 128 px stimulus supports.

## Consequences

Configurations that previously ran now fail fast with an explicit message. That includes test
fixtures that fitted sigma up to 18 on a 64 px field; they moved to a ceiling of 10. Those tests
passed before because the same truncated model both generated and fitted the data, so the error
cancelled and the suite could not see it.

Measuring pRFs larger than `resolution / 6.1` requires raising the resolution, which is the
honest cost: the field has to be big enough to contain what is being measured. The fitter's error
message offers both routes and sizes them with the same 6.1 factor; an earlier version suggested
`sigma × 6`, and following that advice reproduced the very error it was explaining.
