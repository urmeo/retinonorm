# 0005. Folds are grouped by sweep axis and measured overlap

**Status:** Accepted

## Context

Held-out validation is meaningless if a test frame also appears in training. Two ways that
happened, both invisible to the test suite.

A bar sweep's position depends on `x cos θ + y sin θ`, which negates under a 180° turn, while
the travel offsets run symmetrically from `-radius` to `+radius`. Frame `(d, k)` is therefore
bit-identical to frame `(d + 180, n_steps - 1 - k)`. Grouping by direction placed those copies
in different folds. Under the shipped eight-direction default (128 px, 32 steps), every one of
the 32 held-out frames in every group appeared verbatim in the training set — 256 of 256 overall
— and mean held-out-to-training cosine similarity was 1.000. The suite could not see it:
`conftest.py` pins the directions `(0, 45, 90, 135)`, which happens to contain no pair 180°
apart. That is a property of those particular angles, not of having four of them: `(0, 90, 180,
270)` is a four-direction set with two such pairs, and the configuration accepts it.

Wedge and ring groups came from index arithmetic — `start // 90`, `step // block` — which never
consulted how much neighbouring apertures actually share. Ring was worst: a held-out frame reached
cosine 0.871 against a training frame under the default, and 0.806 under the test fixture.

## Decision

Bar frames are grouped by sweep *axis*, `direction % 180`. Without an haemodynamic response
there is no temporal asymmetry between a sweep and its return, so the return carries no
information its partner does not; the two belong in the same fold.

For every design, frames whose cosine similarity to a frame in another group exceeds
`StimulusConfig.max_fold_similarity` are dropped when the sequence is built, greedily, removing
the frame in the most surviving violations first. The threshold lives in the configuration, so
it is part of the run digest.

Considered and rejected: grouping by connected components of the above-threshold similarity
graph. Under the default, every consecutive ring pair has similarity of at least 0.811 — above the
0.75 threshold — so the chain links all 32 frames into one component and leaves no folds at all.
(On the smaller fixture the collapse is partial rather than total, its weakest lag-1 pair being
0.736; the default is the case that matters.)

## Consequences

The default configuration now yields four honest folds with worst-case similarity 0.319, close
to the test fixture's 0.312. Under the default, ring loses three of its thirty-two frames; bar and
wedge already sat under the threshold and lose none. (On the fixture, ring loses three of twenty.)

The duplicate frames remain in the eight-direction default. They cost fitting time and carry no
information, and degrees of freedom must count distinct frames or every standard error is too
narrow by a factor of root two. See `_count_distinct_frames`.

The invariants are asserted against the **default** configuration as well as the test fixture,
in `tests/test_stimuli.py`. A test that only ever builds the fixture cannot catch a defect that
lives in the default.
