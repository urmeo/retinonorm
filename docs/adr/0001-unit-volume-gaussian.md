# 0001. Receptive fields are normalised to unit volume

**Status:** Accepted

## Context

A Gaussian receptive field can be normalised so its peak is 1.0, or so its total weight is 1.0.
The predicted response of a pRF is its overlap with the exposed aperture, summed over the field.

Under unit-peak normalisation, a wider pRF has more total weight, so its overlap with any
aperture is larger by construction. H1 — that pRF size increases with depth — is measured by
comparing fitted sigma across layers. If overlap grows with sigma for reasons of
parameterisation, a deeper layer with any systematic difference in response magnitude would
appear to have larger pRFs regardless of its actual spatial tuning.

## Decision

Weights are divided by `2 * pi * sigma**2`, so every receptive field carries the same total
weight and a wider one spreads it more thinly.

## Consequences

The overlap of a pRF with a fixed aperture *decreases* with sigma rather than increasing, so a
size-versus-depth result cannot be manufactured by the normalisation. Amplitude is recovered
separately by projection (see [0003](0003-amplitude-projected-not-searched.md)), so nothing is
lost by fixing the scale of the weighting.

Unit volume is a property of the continuous Gaussian. On a sampled, bounded grid it holds only
across a range of sigma, which is why [0002](0002-sigma-bounded-at-both-ends.md) exists.
