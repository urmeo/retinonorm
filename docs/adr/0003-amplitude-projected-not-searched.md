# 0003. Amplitude is projected, not searched, and its sign is checked

**Status:** Accepted

## Context

Fitting `response ~ beta * overlap(x0, y0, sigma) + baseline` has five free parameters. Two of
them, `beta` and `baseline`, enter linearly, so at any trial position they have a closed-form
least-squares solution. Searching them nonlinearly alongside the three geometric parameters adds
two dimensions and a family of local minima where a poor position is compensated by a large
amplitude.

Unconstrained least squares admits negative `beta`. A unit whose response *falls* where the
aperture covers a location then fits a flawless pRF at that location with the sign reversed:
measured at `beta = -3.000`, `R² = 1.0000`, centre recovered exactly. Surround suppression and
divisive normalisation both produce such units in a real network, and nothing downstream
inspected the sign.

## Decision

`beta` and `baseline` are solved by projection at every trial position; the optimiser explores
only `x0`, `y0`, `sigma`. A fit with `beta <= 0` is not accepted as a pRF.

## Consequences

The search is three-dimensional, which is why per-unit cost stays flat and the coarse grid can
be small. Degrees of freedom must still count the projected parameters — five, not three — or
the reported uncertainty would be too narrow.

Suppressed units are excluded rather than counted as pRFs at the location that suppresses them.
Their fits are still returned, with parameters and `beta` intact, so an analysis that wants to
study suppression can find them; they are simply not `accepted`. See
[0004](0004-convergence-is-not-acceptance.md).
