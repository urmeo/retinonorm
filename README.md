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

<!-- BEGIN GENERATED: recovery -->
| Condition | Position error, worst (px) | mean (px) | Sigma error, worst (%) | mean (%) | Minimum R² |
|---|---|---|---|---|---|
| Noiseless | 0.000 | 0.000 | 0.0 | 0.0 | 1.0000 |
| 20% noise | 0.287 | 0.197 | 5.2 | 2.3 | 0.9696 |
| 50% noise | 0.772 | 0.511 | 12.8 | 5.6 | 0.8326 |
| Pure noise | — | — | — | — | **40 / 40 rejected** |

Worst and mean are taken over n = 5 ground-truth pRFs. On pure noise the fitter accepted none of 40 units, and its best spurious R² was 0.1947, below the 0.2 acceptance threshold.

<sub>Recorded on macOS-26.6.1 arm64, Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0 | config digest `4d6a856a499e` | generated 2026-08-19</sub>
<!-- END GENERATED: recovery -->

### Parameter uncertainty

Standard errors come from the Jacobian at the solution; intervals use Student's *t* with
`dof = distinct_frames - 5`, since the projected amplitude and baseline are estimated parameters
too, and a frame shown twice is not a second observation.

<!-- BEGIN GENERATED: uncertainty -->
| Noise | Fitted x0 | SE | 95 % CI width |
|---|---|---|---|
| 0.0 | 12.000 | 0.0000 | 0.000 |
| 0.1 | 11.962 | 0.0961 | 0.383 |
| 0.3 | 11.909 | 0.2806 | 1.118 |
| 0.6 | 11.881 | 0.5354 | 2.133 |

The noiseless row is a numerical floor, not a measurement: the residual is at machine precision, so the linearised interval collapses. It is shown to make the scaling of the rows below it legible.

Empirical coverage of the nominal 95% interval:

| Position | Noise | n | Coverage | 95 % binomial CI |
|---|---|---|---|---|
| interior | 0.2 | 200 | 0.935 | [0.901, 0.969] |
| interior | 0.5 | 200 | 0.930 | [0.895, 0.965] |
| near edge | 0.2 | 200 | 0.930 | [0.895, 0.965] |
| near edge | 0.5 | 200 | 0.915 | [0.876, 0.954] |

Coverage runs slightly under nominal, and lowest near the field edge at high noise, where the linearisation behind the interval is weakest.

<sub>Recorded on macOS-26.6.1 arm64, Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0 | config digest `4d6a856a499e` | generated 2026-08-19</sub>
<!-- END GENERATED: uncertainty -->

### Cross-validation and leakage

A first version of the leakage test asserted that a random frame split would inflate scores on
white noise. Measurement showed it does not. The leak requires **two** ingredients: overlapping
apertures *and* a response correlated across neighbouring frames. Real activation timecourses
have both, because a bar moving one step barely changes the input.

<!-- BEGIN GENERATED: cross_validation -->
| Response | n seeds | Grouped CV | Random-split CV | Leak | Leak positive |
|---|---|---|---|---|---|
| White | 20 | -0.248 ± 0.135 | -0.339 ± 0.192 | -0.091 ± 0.231 | 8 / 20 |
| Autocorrelated | 20 | -0.624 ± 0.385 | -0.164 ± 0.189 | +0.459 ± 0.402 | 17 / 20 |

Mean ± SD over seeds 21–40. The carrier is a width-3 boxcar, whose lag-1 autocorrelation measured 0.651 ± 0.086 against a theoretical 0.667.

<sub>Recorded on macOS-26.6.1 arm64, Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0 | config digest `4d6a856a499e` | generated 2026-08-19</sub>
<!-- END GENERATED: cross_validation -->

A random frame split raises the apparent score on temporally correlated noise in most
realisations tested, while on white noise the effect is indistinguishable from zero given the
realisation-to-realisation spread. Both cases are tests, so the mechanism stays pinned: if the
white-noise case ever starts leaking too, the cause is aperture overlap rather than response
autocorrelation.

### Runtime

<!-- BEGIN GENERATED: runtime -->
| Units | Fit (s) | Per unit (ms) | CV (s) | CV factor |
|---|---|---|---|---|
| 1 | 0.008 | 8.2 | 0.036 | 4.4× |
| 10 | 0.073 | 7.3 | 0.379 | 5.2× |
| 100 | 0.765 | 7.7 | 3.958 | 5.2× |

Per-unit cost is flat, so nothing quadratic hides in the loop. Cross-validation costs a roughly constant factor: 4 folds plus the full fit. On this machine 10 000 units extrapolate to about 1.3 minutes single-threaded. Timings are hardware-specific; the environment is recorded below.

<sub>Recorded on macOS-26.6.1 arm64, Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0 | config digest `4d6a856a499e` | generated 2026-08-19</sub>
<!-- END GENERATED: runtime -->

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
