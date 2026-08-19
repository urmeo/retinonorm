# CortexProbe — Build Specification

Executable specification. Every module below states its contract, its invariants, and the
acceptance criteria that must pass before the corresponding commit lands.

Companion to [`PLAN.md`](PLAN.md). Where the two disagree, this file governs implementation
detail and `PLAN.md` governs intent.

---

## 0. Toolchain

```
Python      >=3.9; CI runs 3.9 and 3.12, project-local .venv
torch       >=2.2,<3        MPS backend on Apple silicon
torchvision >=0.17,<1       frozen pretrained weights
numpy       >=1.24,<3
scipy       >=1.11,<2       least_squares (Levenberg–Marquardt / TRF)
matplotlib  >=3.8,<4        figures only, never in library code paths
```

Dev: `ruff`, `mypy` (strict), `pytest`, `pytest-cov`, `hypothesis`.

Hard constraints from measured hardware (8 GB RAM, no CUDA):
- Peak RSS budget **4 GB**. Any stage that would exceed it must stream.
- No stage may hold `(T, C, H, W)` activations for a full layer in RAM. Write to `np.memmap`.

---

## 1. `config.py`

Frozen dataclasses. The only source of run truth; nothing reads loose kwargs.

```python
@dataclass(frozen=True, slots=True)
class StimulusConfig:
    resolution: int  # square field, pixels
    n_steps: int  # frames per sweep direction
    bar_width_frac: float  # bar width as fraction of field
    directions: tuple[int, ...]  # degrees
    wedge_span_deg: float
    ring_thickness_frac: float
    max_fold_similarity: float  # largest tolerated held-out/training frame overlap


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str  # registry key
    layers: tuple[str, ...]  # tap points
    weights_seed: int | None  # None = pretrained


@dataclass(frozen=True, slots=True)
class FitConfig:
    grid_size: int  # coarse search resolution
    sigma_bounds: tuple[float, float]
    max_nfev: int
    r2_threshold: float  # below this, unit is not reported as fitted


@dataclass(frozen=True, slots=True)
class RunConfig:
    stimulus: StimulusConfig
    model: ModelConfig
    fit: FitConfig
    seed: int

    def digest(self) -> str: ...  # sha256 over canonical JSON
```

**Invariant:** `RunConfig.digest()` is stable across processes and platforms. Two runs with
equal digests must produce bit-identical stimuli.

**Acceptance:** round-trip `RunConfig → JSON → RunConfig` is identity; digest stable across
two interpreter sessions. Both are covered by `tests/test_config.py`, along with every
`__post_init__` guard.

**Corrected during implementation.** `carrier_seed` is not in `StimulusConfig`. The carrier has
no consumer until `models.py` provides a network to drive, and a configuration knob that
controls nothing is the same unbacked claim this project forbids in prose. It arrives with the
carrier.

---

## 2. `stimuli.py`

Generates the aperture sequence used for both prediction and model input.

```python
def bar_apertures(cfg: StimulusConfig) -> np.ndarray:       # (T, H, W) bool
def wedge_apertures(cfg: StimulusConfig) -> np.ndarray:
def ring_apertures(cfg: StimulusConfig) -> np.ndarray:
def carrier(cfg: StimulusConfig) -> np.ndarray:             # (T, H, W) float32 in [0,1]  -- with models.py
def render(apertures, carrier) -> np.ndarray:               # (T, 3, H, W) float32, ImageNet-normalised
```

Bar sweeps traverse the field edge-to-edge for each direction in `cfg.directions`. The carrier is
a seeded binary noise pattern refreshed per frame — it drives the network without carrying
retinotopic information itself. It lands with `models.py`, when there is a network input for it
to drive.

**Invariants**
- `apertures.dtype == bool`; no frame is entirely empty.
- Every lit pixel lies inside the circular field mask.
- Each generator's declared `n_frames` is the count *before* leakage pruning; `build` may return
  fewer, and `ApertureSequence.n_frames` is what is actually presented.
- No frame assigned to one cross-validation group is bit-identical to a frame in another, and
  none exceeds `cfg.max_fold_similarity` cosine overlap with one. Asserted for every design and
  for the **default** configuration, in `tests/test_stimuli.py`.
- `render` output is finite and matches torchvision ImageNet normalisation.
- Same `carrier_seed` ⇒ bit-identical carrier.

**Corrected during implementation.** An earlier draft of this spec required bar aperture area to
be constant across frames. Measurement showed it is not, and should not be: the circular field
mask clips the bar as it approaches the edge, so area rises and falls across a sweep
(86 → 512 → 86 px at resolution 64). This matches human retinotopy, where the stimulus is
circularly masked for the same reason. The invariant was wrong; the behaviour is correct.
Coverage per frame is exposed as `ApertureSequence.coverage` so the fit can account for it.

**Acceptance:** determinism test on repeated generation; field-containment test; declared-vs-actual
frame count test for every generator.

---

## 3. `models.py`

```python
LAYER_REGISTRY: dict[str, tuple[str, ...]]     # model name -> valid tap points

def load_model(cfg: ModelConfig) -> torch.nn.Module   # eval(), requires_grad_(False)
def resolve_layers(model, names) -> dict[str, torch.nn.Module]
def weight_digest(model) -> str                        # sha256 over state_dict
```

**Invariants**
- Returned model is in `eval()` mode with all gradients disabled. Asserted, not assumed.
- Requesting a layer name absent from the registry raises `UnknownLayerError`, never a silent
  fallback.
- `weight_digest` changes iff weights change.

**Acceptance:** loading twice yields equal `weight_digest`; unknown layer name raises.

---

## 4. `activations.py`

```python
def extract(
    model, layers: dict[str, Module], stimuli: np.ndarray,
    out_dir: Path, batch_size: int = 8,
) -> dict[str, ActivationHandle]
```

Registers forward hooks, runs the stimulus stack in batches, writes each layer's activations to a
`np.memmap` of shape `(T, U)` where `U = channels × pooled_positions`.

Spatial handling (decision D-2/D-3 in `PLAN.md`): retain the spatial map but downsample to at most
`8 × 8` positions per channel by average pooling, then flatten. A "unit" is one
`(channel, pooled_position)` pair. This keeps `U` tractable while preserving the retinotopic
structure the pRF fit depends on — pooling the spatial map away entirely would destroy exactly
the signal being measured.

**Invariants**
- Hooks are removed on exit, including on exception (`try/finally`).
- Peak RSS stays under budget: batches are written and released, never accumulated.
- `handle.shape == (T, U)`; no NaN or Inf.

**Acceptance:** integration test on a 2-layer toy CNN checks shape, finiteness, hook cleanup, and
that a second identical call reproduces the array bit-for-bit.

---

## 5. `prf/model.py`

```python
def gaussian_rf(x0, y0, sigma, grid: Grid) -> np.ndarray          # (H, W), unit volume
def predict(rf: np.ndarray, apertures: np.ndarray) -> np.ndarray  # (T,) overlap timecourse
```

The predicted response at time *t* is the summed overlap of the receptive field with the aperture:
`r(t) = Σ_xy rf(x,y) · aperture_t(x,y)`, following Dumoulin & Wandell (2008). No haemodynamic
convolution — a CNN has no haemodynamics, and adding one would be a fabricated biological detail.
This departure from the fMRI procedure is deliberate and is documented in the README.

**Invariants**
- `gaussian_rf` integrates to 1.0 within 1e-6 for sigma comfortably inside the field.
- `predict` is linear in `rf`.
- Translating `(x0, y0)` translates the response of a translated aperture identically.

**Acceptance:** normalisation test; linearity test; translation-equivariance test.

---

## 6. `prf/fit.py`

Two stages. Coarse grid search over `(x0, y0, sigma)` selects a basin; `scipy.optimize.least_squares`
refines within it. Grid search alone is too coarse; refinement alone lands in local minima.

```python
def fit_unit(timecourse, apertures, grid, cfg: FitConfig) -> UnitFit
def fit_all(activations, apertures, cfg, n_jobs) -> PRFResult
```

`UnitFit` carries `x0, y0, sigma, beta, baseline, r2, converged, n_fev`.

**Invariants**
- A unit whose timecourse is constant returns `converged=False`, `r2=0.0` — never a fitted pRF.
- `r2` is computed against the *same* timecourse used for fitting; no separate scaling.
- Fitted `sigma` is inside `cfg.sigma_bounds`, or `converged=False`.
- Deterministic: same inputs ⇒ same output, to floating tolerance.

**Acceptance — this is the project's load-bearing test.**
Generate synthetic activations from *known* `(x0, y0, sigma)` pRFs across a spread of positions and
sizes, add controlled noise at several SNRs, run `fit_all`, assert:

| Condition | Requirement |
|---|---|
| noiseless | position error < 0.5 px, sigma error < 5%, `r2 > 0.99` |
| moderate noise | position error < 1.5 px, `r2 > 0.8` |
| pure noise input | `converged=False` or `r2 < r2_threshold` for ≥95% of units |

If this test does not pass, no downstream result is reported. Non-negotiable.

---

## 7. `prf/result.py`

```python
@dataclass(frozen=True)
class PRFResult:
    units: pd.DataFrame          # one row per unit
    layer: str
    run_digest: str
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> "PRFResult"
    def fitted(self, threshold: float) -> pd.DataFrame
```

Persisted as Parquet plus a JSON sidecar holding `run_digest`, config, library versions, and
timestamp.

**Invariant:** `load(save(x)) == x`. Loading a result whose sidecar digest does not match the
requesting config raises `DigestMismatchError` rather than returning mismatched data.

---

## 8. `lesion.py`

```python
def spatial_lesion(activations, positions, layer) -> np.ndarray   # zero-ablation
def channel_lesion(activations, channels, layer) -> np.ndarray
def lesion_delta(intact: PRFResult, lesioned: PRFResult) -> pd.DataFrame
```

`lesion_delta` joins on unit id and reports per-unit shift in `x0, y0, sigma, r2`. Units unfitted
in either condition are excluded and the exclusion count is reported — never silently dropped.

**Acceptance:** lesioning zero units yields an all-zero delta; lesioning every unit yields no
fitted downstream units.

---

## 9. `individuals.py`

```python
def make_individuals(base: ModelConfig, seeds: Sequence[int]) -> list[ModelConfig]
def between_individual_variance(results: list[PRFResult]) -> pd.DataFrame
```

An "individual" is the frozen backbone plus a seeded random projection applied at the tap point —
this varies the readout without retraining, which is the only affordable source of variation on
this hardware (D-5).

**Critical control:** between-individual variance must be compared against within-individual fit
uncertainty. If seed-driven spread does not exceed fitting noise, H4 is **not** supported and must
be reported as such. `between_individual_variance` returns both quantities so the comparison
cannot be skipped.

---

## 10. `evaluation.py`

Pure functions, one per hypothesis. Each returns an estimate, a dispersion, an *n*, and a plain
statement of what would falsify it.

```python
def size_vs_depth(results)        -> H1Result   # regression of sigma on layer index
def size_vs_eccentricity(result)  -> H2Result   # regression within layer
def lesion_effect(delta)          -> H3Result   # effect size vs distance from lesion
def individual_spread(results)    -> H4Result   # between- vs within-variance ratio
```

**Invariant:** every result object carries `n` and a confidence interval. A point estimate without
a dispersion is a bug.

---

## 11. `cli.py`

```
cortexprobe stimuli   --config C --out D
cortexprobe extract   --config C --out D
cortexprobe fit       --config C --out D
cortexprobe lesion    --config C --spec S --out D
cortexprobe individuals --config C --seeds 0,1,2,3,4 --out D
cortexprobe evaluate  --run D
cortexprobe report    --run D --out R
```

Every subcommand: writes a run manifest, is idempotent, refuses to overwrite an existing completed
stage without `--force`, and exits non-zero with a typed error code on failure.

---

## 12. Definition of done

A milestone is complete only when all hold:

1. `ruff format --check` and `ruff check` pass.
2. `mypy --strict` reports zero issues on `src/`.
3. `pytest` passes; branch coverage ≥ 85% on `src/cortexprobe/`.
4. The ground-truth recovery test (§6) passes.
5. No number appears in any `.md` that was not produced by a run recorded in `results/`.
   Enforced by `scripts/generate_validation_report.py --check` in CI, which re-measures every
   quoted figure and confirms the committed tables were rendered from it.
6. Commit message is a short human phrase, one concern per commit.

---

## 13. Sequenced build order

Strictly bottom-up; each step is testable before the next begins.

```
 1. scaffold: pyproject, ruff, mypy, pytest, CI, MIT license, .gitignore
 2. config.py            + tests   → commits "add config", "add config tests"
 3. stimuli.py           + tests   → commits "bar apertures", "wedge and ring",
                                     "add stimulus leakage invariant tests"
 4. prf/model.py         + tests   → commit "gaussian prf model"
 5. prf/fit.py           + RECOVERY TEST → commits "grid search init", "nonlinear refine", "recovery test"
 6. prf/result.py        + tests   → commit "prf result io"
 7. models.py            + tests   → commit "model registry"
 8. activations.py       + tests   → commit "activation extraction"
 9. cli.py (stimuli/extract/fit)   → commit "add cli"
10. FIRST REAL RUN — intact AlexNet → commit "run intact"
11. lesion.py, individuals.py, evaluation.py → commits per module
12. viz.py, report                → commit "add report"
13. FINAL RUNS + RESULTS.md       → commits "run lesion", "run individuals", "updated results"
```

Steps 4–6 precede any model work deliberately: the fitter must be proven correct against known
ground truth before it is ever pointed at a real network, or a fitter bug becomes indistinguishable
from a scientific finding.
