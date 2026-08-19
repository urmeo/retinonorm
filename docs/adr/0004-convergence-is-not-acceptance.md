# 0004. Convergence and acceptance are separate

**Status:** Accepted

## Context

`UnitFit.converged` was `solution.success and r2 >= threshold`. That conflated two unrelated
questions. A fit that converged cleanly onto a unit with no spatial tuning was reported as *not
converged*, and there was no way to distinguish an optimiser failure from a successful
measurement of an uninteresting unit. The "40 of 40 rejected" claim about pure noise rested on
that overloaded flag, and so did not say what it appeared to say.

The same conflation appeared in `at_bound`, which fired on `x0`, `y0` or `sigma` together. A pRF
at the edge of the visual field and a sigma pinned against the search ceiling are different
situations with opposite implications.

## Decision

`converged` is the optimiser's report and nothing else. `accepted` is a separate property: the
fit converged, cleared the R² threshold, has positive amplitude, and did not pin sigma against
its ceiling. The threshold travels on `UnitFit` so acceptance is self-contained, defaulting to a
value nothing clears so a hand-assembled fit is never mistaken for a measured pRF.

`at_bound` splits into `x0_at_bound`, `y0_at_bound` and `sigma_at_bound`, and remains available
as their disjunction. Only `sigma_at_bound` blocks acceptance.

## Consequences

Reports can distinguish "the numerics failed" from "this unit has no pRF", which are different
findings and would prompt different investigations.

A centre pinned at the field edge stays accepted, because receptive fields genuinely sit near
the boundary and excluding them would bias the eccentricity distribution inward — which is the
axis H2 measures. A pinned sigma is excluded, because that value records where the search
stopped rather than what the data support.
