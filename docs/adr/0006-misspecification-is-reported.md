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

Measured separation on a four-direction bar sweep at 64 px: genuine single-Gaussian units stay
under 0.03 from noiseless to 80 % noise, pure noise reaches 0.06, two-lobe units score 0.29 to
0.44 — an order of magnitude apart.

A naive alternative was rejected: correlating the residual against candidate predictions without
weighting by remaining variance. That scores 0.925 on a *perfect* noiseless fit, because a
residual at machine precision still correlates with something and correlation is scale-free.

Because the diagnostic is reported rather than acted on, downstream analysis must choose a cut
and state it. That is deliberate: the right threshold depends on what the analysis is for, and
burying it here would hide a scientific choice inside the instrument.
