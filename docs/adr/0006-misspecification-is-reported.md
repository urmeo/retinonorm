# 0006. Misspecification is reported, not enforced

**Status:** Accepted

## Context

A single Gaussian cannot represent a unit driven by two separated lobes. Measured on lobes at
`x = ±14` with sigma 3, the fit settles on one lobe, reports `sigma = 2.27` — belonging to
neither — and scores `R² = 0.5366`, comfortably above the acceptance threshold. Nothing in
`UnitFit` indicated the model was wrong for that unit.

Multi-peaked spatial tuning is common in the deeper layers this project intends to tap, so this
would produce real, confidently wrong pRFs the moment a network is attached.

## Decision

Each fit reports `second_field_r2`: the largest additional R² a second candidate receptive field
would explain on top of the fitted one. Candidate predictions are already built for the coarse
search, so each is orthogonalised against the fitted prediction and the intercept and its
incremental R² read off directly. This is the R² gain from a two-Gaussian alternative without
running a second nonlinear search, so per-unit cost stays flat.

It is reported, not enforced. Acceptance does not depend on it.

## Consequences

Measured separation on a four-direction bar sweep at 64 px, over noise from 0 to 80 %, and over
two lobe geometries (`x = ±14, σ = 3` and `(±16, ±10), σ = 3`):

| Unit | `second_field_r2` |
|---|---|
| single Gaussian | 0.000 – 0.023 |
| pure noise | 0.056 |
| two lobes | 0.206 – 0.473 |

The separation is widest at low noise and narrows as noise grows: the weakest two-lobe case
(80 % noise, closer lobes) scores 0.206, only about four times the pure-noise level, while the
strongest scores 0.473. A cut near 0.1 separates them in this configuration. That number is not
claimed to transfer to a different stimulus, resolution, or lobe geometry.

A naive alternative was rejected: correlating the residual against candidate predictions without
weighting by remaining variance. That scores 0.925 on a *perfect* noiseless fit, because a
residual at machine precision still correlates with something and correlation is scale-free.

Because the diagnostic is reported rather than acted on, downstream analysis must choose a cut
and state it. That is deliberate: the right threshold depends on what the analysis is for, and
burying it here would hide a scientific choice inside the instrument.
