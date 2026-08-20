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
matplotlib  >=3.7,<4        `viz` extra; figures only, never in library code paths
pandas      >=2.0,<3        `io` extra, with pyarrow>=15; arrives with prf/result.py
```

Dev: `ruff`, `mypy` (strict), `pytest`, `pytest-cov`, `hypothesis`.

Hard constraints from measured hardware (8 GB RAM, no CUDA):
- Peak RSS budget **4 GB**. Any stage that would exceed it must stream.
- No stage may hold `(T, C, H, W)` activations for a full layer in RAM. Write to `np.memmap`.

---

## 1. `config.py`

Frozen dataclasses. The only source of run truth; nothing reads loose kwargs.

```python
@dataclass(frozen=True)
class StimulusConfig(ConfigBase):
    resolution: int  # square field, pixels
    n_steps: int  # frames per sweep direction
    bar_width_frac: float  # bar width as fraction of field
    directions: tuple[int, ...]  # degrees
    wedge_span_deg: float
    ring_thickness_frac: float
    max_fold_similarity: float  # largest tolerated held-out/training frame overlap


@dataclass(frozen=True)
class ModelConfig(ConfigBase):
    name: str  # registry key
    layers: tuple[str, ...]  # tap points
    weights_seed: Optional[int]  # None = pretrained; not `int | None`, see below
    pool_to: int  # spatial positions per channel after pooling; >= 2


@dataclass(frozen=True)
class FitConfig(ConfigBase):
    grid_size: int  # coarse search resolution
    sigma_bounds: tuple[float, float]
    max_nfev: int
    r2_threshold: float  # below this, unit is not reported as fitted


@dataclass(frozen=True)
class RunConfig(ConfigBase):
    stimulus: StimulusConfig
    model: ModelConfig
    fit: FitConfig
    seed: int

    def digest(self) -> str: ...  # sha256 over canonical JSON
```

**Invariant:** `RunConfig.digest()` is stable across processes and platforms. Two runs with
equal digests must produce bit-identical stimuli.

**Acceptance:** round-trip `RunConfig → JSON → RunConfig` is identity, and every
`__post_init__` guard fires — both covered by `tests/test_config.py`. Digest stability *across
processes and platforms* is not a unit test: it is enforced by the `reproducibility` CI job,
which re-derives `config_digest` on Linux/3.9 and Linux/3.12 and compares it against the value
recorded on macOS in `results/validation.json`.

`slots=True` and `int | None` both appear obvious here and both are wrong for this project.
`slots=` requires Python 3.10; the floor is 3.9. `int | None` is evaluated at runtime by
`get_type_hints` inside `from_dict`, which raises `TypeError` on 3.9 — hence `Optional[int]`, as
recorded in the `per-file-ignores` note in `pyproject.toml`.

**Corrected during implementation.** `carrier_seed` is not in `StimulusConfig`. The carrier has
no consumer until `models.py` provides a network to drive, and a configuration knob that
controls nothing is the same unbacked claim this project forbids in prose. It arrives with the
carrier.

---

## 2. `stimuli.py`

Generates the aperture sequence used for both prediction and model input.

```python
class ApertureGenerator(ABC):                               # BarSweep / RotatingWedge / ExpandingRing
    def build(self) -> ApertureSequence: ...
    @property
    def n_frames(self) -> int: ...                          # before leakage pruning

@dataclass(frozen=True)
class ApertureSequence:                                     # apertures (T,H,W) bool, grid, kind,
    ...                                                     # frame_index, group; .coverage, .as_float()

def build_apertures(config: StimulusConfig, kind: str = "bar") -> ApertureSequence
def frame_similarity(stack: BoolArray) -> FloatArray        # (T, T) pairwise cosine

# with models.py, when there is a network input to drive:
def carrier(cfg: StimulusConfig) -> np.ndarray:             # (T, H, W) float32 in [0,1]
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
(86 px at the edge to 542 px at the centre, resolution 64). This matches human retinotopy, where the stimulus is
circularly masked for the same reason. The invariant was wrong; the behaviour is correct.
Coverage per frame is exposed as `ApertureSequence.coverage` so the fit can account for it.

**Acceptance:** `tests/test_stimuli.py` — determinism on repeated generation; field containment; declared-vs-actual
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
@dataclass(frozen=True)
class GaussianReceptiveField(ReceptiveField):                     # x0, y0, sigma
    def weights(self, grid: Grid) -> FloatArray                   # (H, W), unit volume
    @property
    def eccentricity(self) -> float
    @property
    def polar_angle(self) -> float

def predict(weights: FloatArray, apertures: FloatArray) -> FloatArray   # (T,) overlap timecourse
def design_matrix(fields, grid: Grid, apertures: FloatArray) -> FloatArray  # (T, n_fields)
```

The predicted response at time *t* is the summed overlap of the receptive field with the aperture:
`r(t) = Σ_xy rf(x,y) · aperture_t(x,y)`, following Dumoulin & Wandell (2008). No haemodynamic
convolution — a CNN has no haemodynamics, and adding one would be a fabricated biological detail.
This departure from the fMRI procedure is deliberate and is documented in the README.

**Invariants**
- `GaussianReceptiveField.weights` integrates to 1.0 for sigma comfortably inside the field;
  outside that range it does not, which is why both bounds are guarded (see §6 and ADR-0002).
- `predict` is linear in the receptive field.
- Translating `(x0, y0)` translates the response of a translated aperture identically.

**Acceptance:** `tests/test_prf_model.py` — unit-volume test; linearity in the receptive field;
translation equivariance (pRF and aperture shifted together predict identically).

---

## 6. `prf/fit.py`

Two stages. Coarse grid search over `(x0, y0, sigma)` selects a basin; `scipy.optimize.least_squares`
refines within it. Grid search alone is too coarse; refinement alone lands in local minima.

```python
class PRFFitter:
    def __init__(self, grid: Grid, apertures: FloatArray, config: FitConfig) -> None
    def fit_unit(self, response: FloatArray) -> UnitFit
    def fit_all(self, activations: FloatArray) -> list[UnitFit]      # (frames, units)
```

Nothing parallelises today; per-unit cost is flat, so there is no `n_jobs`.

`UnitFit` carries the fitted geometry `x0, y0, sigma`, the projected `beta, baseline`, the score
`r2`, the optimiser's `converged` and `n_fev`, the linearised `se_x0, se_y0, se_sigma` with `dof`
and the `r2_threshold` used, the misspecification diagnostic `second_field_r2`, and the three
bound flags `x0_at_bound, y0_at_bound, sigma_at_bound` (with `at_bound` as their disjunction).

**`converged` is not `accepted`.** `converged` is the optimiser's own report. `accepted` is the
scientific claim, and is the flag downstream analysis must gate on:
`converged and r2 >= r2_threshold and beta > 0 and not sigma_at_bound`. The optimiser converges
happily on pure noise, so gating on `converged` alone would report every noise unit as a pRF. A
negative `beta` is a suppressed unit, not a receptive field; a sigma pinned at the ceiling records
where the search stopped. See ADR-0003 and ADR-0004.

**Invariants**
- A unit whose timecourse is constant returns `accepted=False`, `r2=0.0` — never a fitted pRF.
- `r2` is computed against the *same* timecourse used for fitting; no separate scaling.
- Fitted `sigma` is inside `cfg.sigma_bounds`; at the ceiling it sets `sigma_at_bound` and is not
  accepted.
- Degrees of freedom count *distinct* frames, not presented frames: a frame shown twice is not a
  second observation.
- `PRFFitter` refuses a `sigma_bounds` ceiling the grid cannot represent, and refuses non-finite
  apertures, at construction.
- Deterministic: same inputs ⇒ same output, to floating tolerance.

**Acceptance — this is the project's load-bearing test.**
Generate synthetic activations from *known* `(x0, y0, sigma)` pRFs across a spread of positions and
sizes, add controlled noise at several SNRs, run `fit_all`, assert:

| Condition | Requirement |
|---|---|
| noiseless | position error < 0.5 px, sigma error < 5%, `r2 > 0.99` |
| moderate noise | position error < 1.5 px, `r2 > 0.8` |
| pure noise input | `accepted=False` for ≥95% of units |

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
   Enforced for `README.md` by `scripts/generate_validation_report.py --check` in CI, which
   re-measures every figure inside the generated regions and confirms `README.md` and
   `results/VALIDATION.md` were rendered from them. Numbers quoted in `docs/` — this file, the
   ADRs — are not machine-checked; they are verified by hand against the code and cite what to
   run.
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
