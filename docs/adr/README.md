# Architecture decision records

One record per decision that would be expensive to reverse or easy to reverse by accident.
Each states the context that forced the choice, the choice, and what it costs.

These are decisions about the *instrument*. None of them is a scientific finding, and none
presumes an outcome for the hypotheses in [`../PLAN.md`](../PLAN.md).

| # | Decision | Status |
|---|---|---|
| [0001](0001-unit-volume-gaussian.md) | Receptive fields are normalised to unit volume | Accepted |
| [0002](0002-sigma-bounded-at-both-ends.md) | Sigma is bounded at both ends by the grid | Accepted |
| [0003](0003-amplitude-projected-not-searched.md) | Amplitude is projected, not searched, and its sign is checked | Accepted |
| [0004](0004-convergence-is-not-acceptance.md) | Convergence and acceptance are separate | Accepted |
| [0005](0005-folds-grouped-by-sweep-axis.md) | Folds are grouped by sweep axis and measured overlap | Accepted |
| [0006](0006-misspecification-is-reported.md) | Misspecification is reported, not enforced | Accepted |
