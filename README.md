# CortexProbe

**Population receptive fields and lesion effects in convolutional models of the visual hierarchy.**

Status: instrument validated, no network probed yet. Every number below comes from a run in
this repository. No result is claimed that has not been measured.

---

## What this is

A measurement instrument for **population receptive fields (pRFs) in convolutional networks**.

In human fMRI, a voxel's pRF is estimated by sweeping a bar aperture across the visual field and
regressing the voxel's timecourse against candidate 2D Gaussians (Dumoulin & Wandell, 2008,
*NeuroImage* 39:647-660). That procedure applies unchanged to a convolutional unit: present the
same stimulus, record the unit's activation timecourse, fit the same model. The output is a
*measured* pRF for an artificial unit, in the same units as the human quantity.

Once units have pRFs, two things become possible that a plain encoding model cannot do:

- **Lesion a network and measure the consequence.** Silence units in an early layer, refit
  downstream pRFs, and quantify the distortion — an artificial scotoma.
- **Treat architectural variation as individual difference.** Independently seeded instances
  give different pRF maps; the spread is measurable and testable against fit noise.

### Why it is built this way

Two neighbouring projects stall on the same wall: they require authorised restricted data (HCP
7T, NSD) before a single empirical number can exist, so neither reports one. CortexProbe has no
such dependency. Stimuli are generated in-repo; the "brain" is a frozen network. Every number
here is producible, reproducible, and defensible on a laptop.

### Scope boundary

A frozen ImageNet-trained CNN is **not** a biologically realistic model of visual cortex. It has
no recurrence, no separate excitatory and inhibitory populations, and no built-in cortical
magnification. This project does not claim otherwise. It builds the measurement apparatus that
biologically realistic architectures would need in order to be evaluated. This boundary is not
softened anywhere in this repository.

---

## Working rules

| Rule | Enforcement |
|---|---|
| No fabricated numbers | Nothing appears in any document that was not produced by a recorded run |
| Prove the instrument before using it | The fitter must recover pRFs it generated itself, or no downstream result counts |
| Held-out or it does not count | Leave-one-sweep-group-out CV; train-fold amplitude applied unchanged to test frames |
| Every estimate carries a dispersion | A point estimate without an error bar is treated as a defect |
| Configuration is the only source of truth | Frozen dataclasses with a SHA-256 run digest; no loose keyword arguments |
| State the boundary | The scope limit above is repeated wherever results are reported |
| Granular commits | One concern per commit, short human messages |

---

## Pipeline

```
  config ──▶ stimuli ──▶ model ──▶ activations ──▶ prf.fit ──┬──▶ lesion
                                                             ├──▶ individuals
                                                             └──▶ evaluation ──▶ report
```

| Stage | Module | State |
|---|---|---|
| Run configuration, digest, IO | `config.py` | done |
| Visual field coordinates | `geometry.py` | done |
| Bar / wedge / ring apertures, CV groups | `stimuli.py` | done |
| Gaussian pRF and predicted response | `prf/model.py` | done |
| Two-stage fit with uncertainty | `prf/fit.py` | done |
| Grouped cross-validation | `prf/validation.py` | done |
| Typed result container, Parquet IO | `prf/result.py` | not started |
| Frozen network loading, layer taps | `models.py` | not started (needs PyTorch) |
| Hooked activation extraction | `activations.py` | not started (needs PyTorch) |
| Lesion operators | `lesion.py` | not started |
| Seeded model instances | `individuals.py` | not started |
| Hypothesis statistics | `evaluation.py` | not started |
| Command line interface | `cli.py` | not started |

---

## Design decisions

**Unit-volume Gaussians.** The receptive field is normalised by `1 / (2 * pi * sigma**2)`, so a
wide pRF spreads the same total weight more thinly rather than accumulating more of it. Under
unit-peak normalisation every pRF would share a maximum of 1.0, its overlap with the aperture
would grow with sigma by construction, and the hypothesis that pRF size increases with depth
would be guaranteed by the parameterisation instead of measured.

**Amplitude is never searched.** At each trial position, amplitude and baseline are solved in
closed form by linear least squares, so the nonlinear optimiser explores only `x0`, `y0`,
`sigma`. This keeps the search three-dimensional and removes a family of local minima.

**Folds split by sweep group, never by frame.** Adjacent frames within a sweep show
near-identical apertures. A random frame split places near-duplicates on both sides of the
train/test boundary and reports memorisation as generalisation. Measured below.

**Sigma floor at one pixel.** Below the pixel pitch a Gaussian is under-sampled and silently
loses unit volume: at `sigma = 0.2` the field sums to 0.031 rather than 1.0. The configuration
rejects any lower bound under 1.0 px with an error explaining why.

---

## Results

**These validate the instrument, not any scientific claim.** No network has been probed. The
four hypotheses this project exists to test — pRF size increasing with depth, size increasing
with eccentricity, lesion-induced downstream distortion, and measurable spread across seeded
instances — are all **untested**. They are experiments, and a negative result will be reported
as a negative result.

### Recovery of known pRFs

Activations are synthesised from known `(x0, y0, sigma)`, fitted, and compared against the
parameters that generated them.

| Condition | Position error | Sigma error | Minimum R² |
|---|---|---|---|
| Noiseless | **0.000 px** | **0.0 %** | 1.0000 |
| 20 % noise | 0.287 px | 5.2 % | 0.9696 |
| 50 % noise | 0.772 px | 12.8 % | 0.8326 |
| Pure noise | — | — | **40 / 40 rejected** |

On pure noise the fitter never once claimed a pRF, and its best spurious R² was 0.195 — below
the 0.2 acceptance threshold. This is the property that stops a fitter bug from being mistaken
for a finding.

### Parameter uncertainty

Standard errors come from the Jacobian at the solution; intervals use Student's *t* with
`dof = n_frames - 5`, since the projected amplitude and baseline are estimated parameters too.

| Noise | Fitted x0 | SE | 95 % CI width |
|---|---|---|---|
| 0.0 | 12.000 | 0.0000 | 0.000 |
| 0.1 | 11.962 | 0.0961 | 0.383 |
| 0.3 | 11.909 | 0.2806 | 1.118 |
| 0.6 | 11.881 | 0.5354 | 2.133 |

Across five noise realisations the interval contained the generating parameter in every case.

### Cross-validation and leakage

A first version of the leakage test asserted that a random frame split would inflate scores on
white noise. Measurement showed it does not. The leak requires **two** ingredients: overlapping
apertures *and* a response correlated across neighbouring frames. Real activation timecourses
have both, because a bar moving one step barely changes the input.

| Response | Grouped CV | Random-split CV | Leak |
|---|---|---|---|
| White noise | −0.204 | −0.209 | −0.004 |
| Autocorrelated, lag-1 r = 0.83 | −0.672 | **+0.116** | **+0.788** |

The second row is the failure mode in one line: a random split turns a model that is clearly
failing into one that appears to work. Both cases are tests, so the mechanism stays pinned; if
the white-noise case ever starts failing too, the cause is aperture overlap rather than response
autocorrelation.

### Runtime

| Units | Fit (s) | Per unit (ms) | CV (s) | CV factor |
|---|---|---|---|---|
| 1 | 0.010 | 9.6 | 0.043 | 4.5× |
| 10 | 0.081 | 8.1 | 0.409 | 5.0× |
| 100 | 0.821 | 8.2 | 4.090 | 5.0× |

Per-unit cost is flat, so nothing quadratic is hiding in the loop. Cross-validation costs a
constant 5.0× — four folds plus the full fit. Extrapolating, 10 000 units take roughly 1.4
minutes single-threaded, so parallelism is not yet worth its complexity.

Measured on an Apple M2, 8 GB RAM, at 64 px resolution with 80 frames and 300 candidates.
Reproduce with `python benchmarks/benchmark_scaling.py`.

---

## Install and verify

Python 3.9 or newer. The implemented core needs only NumPy, SciPy, and pandas; PyTorch becomes
necessary at `models.py`, which is not yet written.

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest tests -q
python3 benchmarks/benchmark_scaling.py
```

71 tests currently pass. They use synthetic fixtures only and download nothing.

---

## Testing strategy

The central risk is a fitter that returns confident nonsense, so the suite is built around
proving it does not.

| Suite | What it establishes |
|---|---|
| `test_prf_recovery.py` | Known parameters are recovered; noise is rejected |
| `test_prf_edge_cases.py` | Boundary centres, sigma at both bounds, degenerate inputs |
| `test_prf_uncertainty.py` | Errors scale with noise; intervals cover the truth |
| `test_cross_validation.py` | Folds hold out whole groups; the leakage mechanism is pinned |
| `test_prf_designs.py` | Recovery across bar, wedge, ring and across pRF sizes |

---

## References

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual
cortex. *NeuroImage*, 39(2), 647-660.

## Licence

MIT. See [`LICENSE`](LICENSE).
