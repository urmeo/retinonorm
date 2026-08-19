# CortexProbe — Plan

**Population receptive fields and lesion effects in convolutional models of the visual hierarchy.**

Status: planning. No code written. No results claimed.

---

## 1. Analysis

### 1.1 The gap this fills

IndiBrain Project 5 is defined as the intersection of three things: *biologically realistic
artificial neural networks*, *pRF and connective-field modelling*, and *individual differences /
lesion simulation* (vacancy, "What You Will Do").

Existing portfolio coverage:

| Capability | CalmSense | RetinoNorm | Gap |
|---|---|---|---|
| Deep networks (training, PyTorch) | yes (1D-CNN) | no | — |
| Individual differences | yes (subject-level) | yes (normative deviation) | — |
| Leakage-safe evaluation | yes | yes | — |
| Visual cortex / retinotopy | no | yes (7T maps) | — |
| **pRF fitting, implemented** | no | no | **open** |
| **Lesion / perturbation simulation** | no | no | **open** |
| **DNN probed *as* a visual system** | no | no | **open** |

The last three rows are the core method of Project 5 and are absent from both existing projects.
CortexProbe targets exactly those three.

### 1.2 Why this design is feasible where the others were not

RetinoNorm and the NSD project both stall at the same wall: they need authorised restricted data
(HCP 7T, NSD) before a single empirical number can be produced. That is why neither reports a
result.

CortexProbe has no such dependency. The stimuli are synthetic and generated in-repo; the "brain"
is a frozen convolutional network. Every number in this project can therefore be produced,
re-produced, and defended. **This project will report real measured results.**

### 1.3 Scientific framing

In human fMRI, a population receptive field is fitted by presenting a moving aperture over a
carrier pattern and regressing each voxel's timecourse against the predicted response of a
candidate 2D Gaussian receptive field (Dumoulin & Wandell, 2008, *NeuroImage* 39:647–660).

The same procedure applies without modification to a convolutional unit: present the identical
stimulus sequence, record the unit's activation timecourse, fit the same Gaussian model. This
gives a *measured* pRF for an artificial unit, directly comparable to the human quantity.

Claims to be tested (none assumed true; each is an experiment):

- **H1 — hierarchy.** Fitted pRF size increases with depth, mirroring V1 → V2 → V3 → hV4.
- **H2 — eccentricity.** Within a layer, pRF size increases with eccentricity.
- **H3 — lesion.** Silencing units in an early layer produces a measurable, spatially localised
  distortion of downstream pRFs (an artificial scotoma).
- **H4 — individual differences.** Independently seeded model instances yield systematically
  different pRF maps; the spread is quantifiable and is *not* fitting noise.

H4 is the bridge to IndiBrain: it treats architectural/initialisation variation as the model
analogue of between-subject variation.

### 1.4 Honest scope boundary

A frozen ImageNet-trained CNN is **not** a biologically realistic model of visual cortex. It has
no recurrence, no separate excitatory/inhibitory populations, no cortical magnification built in.
This project does not claim otherwise. It builds the *measurement apparatus* — a pRF and lesion
probe that runs on any vision model — which is the prerequisite for the biologically realistic
architectures Project 5 proposes to develop. This boundary is stated in the README and must not
be softened.

---

## 2. Environment (measured, not assumed)

| Property | Value | Consequence for design |
|---|---|---|
| Machine | Apple M2, 8 cores | MPS backend; no CUDA |
| RAM | 8 GB | Rules out large models and big batches |
| Free disk | 25 GB | torch + torchvision (~2.5 GB) fits |
| System Python | 3.9.6 (`/usr/bin/python3`) | Too old for target tooling |
| Homebrew | present (`/opt/homebrew/bin/brew`) | Install Python 3.12 |
| torch / torchvision | **absent** | Must install |
| numpy 2.0.2, scipy 1.13.1, sklearn 1.6.1, matplotlib 3.9.4 | present | Available under 3.9 only; reinstall under 3.12 |

Design consequences:

- Target **Python 3.12** via Homebrew, in a project-local `.venv`.
- Model must be small. Baseline is **AlexNet** (~61M params, 5 conv layers, maps cleanly onto a
  shallow hierarchy) with **ResNet-18** as a second architecture. Both frozen — no training, so
  no GPU-hours and full determinism.
- Activation extraction must stream and downsample; never hold a full
  `n_stimuli × n_units × H × W` tensor in memory. Hard budget: peak RSS < 4 GB.

---

## 3. Pipeline

```
                 ┌──────────────┐
  config ───────▶│  stimuli     │  bar / wedge / ring apertures over noise carrier
                 └──────┬───────┘  → (T, H, W) binary aperture stack + metadata
                        │
                 ┌──────▼───────┐
                 │  model       │  frozen CNN, named layer taps via forward hooks
                 └──────┬───────┘  → (T, U) activation matrix, streamed to disk
                        │
                 ┌──────▼───────┐
                 │  prf.fit     │  coarse grid search → nonlinear refine (Levenberg–Marquardt)
                 └──────┬───────┘  → per-unit (x0, y0, sigma, beta, R²)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
 ┌──────▼─────┐  ┌──────▼─────┐  ┌──────▼──────┐
 │  lesion    │  │ individuals│  │  evaluation │
 │  refit     │  │  seeds     │  │  H1–H4      │
 └──────┬─────┘  └──────┬─────┘  └──────┬──────┘
        └───────────────┼───────────────┘
                 ┌──────▼───────┐
                 │  report      │  figures + metrics.json, values only from real runs
                 └──────────────┘
```

### Stage contracts

| Stage | Input | Output | Determinism |
|---|---|---|---|
| `stimuli` | `StimulusConfig` | `(T,H,W) uint8`, `stimuli.json` | seeded; hash recorded |
| `model` | model id, layer names | `(T,U) float32` memmap | frozen weights; hash recorded |
| `prf.fit` | activations, apertures | `PRFResult` table | seeded init; converged flag per unit |
| `lesion` | lesion spec | refitted `PRFResult` | same seed as intact run |
| `individuals` | seed list | stacked `PRFResult` | one seed per instance |
| `evaluation` | results | `metrics.json` | pure function |

**Fabrication guard:** `report` renders only from a `metrics.json` produced by a completed run,
and refuses to render if the run manifest hash does not match. No number is ever typed by hand.

---

## 4. Repository architecture

```
cortexprobe/
├── src/cortexprobe/
│   ├── config.py         frozen dataclasses; single source of run truth
│   ├── stimuli.py        aperture generators (bar, wedge, ring), carrier
│   ├── models.py         registry, frozen loading, layer taps
│   ├── activations.py    hooked extraction, streaming, spatial pooling
│   ├── prf/
│   │   ├── model.py      Gaussian pRF → predicted timecourse
│   │   ├── fit.py        grid search + nonlinear refine
│   │   └── result.py     PRFResult container, IO, validation
│   ├── lesion.py         channel / spatial / random lesion operators
│   ├── individuals.py    seeded instance generation
│   ├── evaluation.py     H1–H4 statistics
│   ├── viz.py            retinotopy maps, size-vs-depth, lesion deltas
│   └── cli.py            subcommands
├── tests/                unit + property + integration (synthetic ground truth)
├── configs/              versioned run configs
├── docs/                 PLAN.md, BUILD_SPEC.md, ADRs, RESULTS.md
├── results/              gitignored; run outputs
├── .github/workflows/    ci.yml (lint, type, test, coverage)
├── pyproject.toml
├── LICENSE               MIT
├── CITATION.cff
└── README.md
```

**Key design decision:** `prf/` is a package, not a module, because the pRF model, the fitting
procedure, and the result container have genuinely different reasons to change — the model is
mathematics, the fitter is numerics, the container is IO. Keeping them in one file would couple
three independent change axes.

---

## 5. Correctness strategy

The central risk is a pRF fitter that returns confident nonsense. Mitigation is a
**ground-truth recovery test**: synthesise activations from a *known* Gaussian pRF, run the
fitter, and assert recovery of `(x0, y0, sigma)` within tolerance. If the fitter cannot recover a
pRF it generated itself, no downstream result is trustworthy.

| Layer | What it proves |
|---|---|
| Unit tests | each function behaves on edge cases |
| Property tests | aperture area conserved; Gaussian normalised; fit invariant to activation scale |
| **Ground-truth recovery** | **fitter is correct** — parameters recovered from synthetic pRFs |
| Integration | CLI end-to-end on a tiny config completes and writes valid artifacts |
| CI | lint (ruff), types (mypy strict), tests, coverage floor 85% |

---

## 6. Milestones and commits

Granular, short messages, one concern each.

| # | Milestone | Commits |
|---|---|---|
| 0 | Scaffold | `init repo`, `add license`, `add pyproject`, `add ci` |
| 1 | Stimuli | `add stimulus config`, `bar apertures`, `wedge and ring`, `stimulus tests` |
| 2 | Models | `model registry`, `layer taps`, `activation extraction`, `model tests` |
| 3 | pRF core | `gaussian prf model`, `grid search init`, `nonlinear refine`, `recovery test` |
| 4 | Results IO | `prf result container`, `result io`, `result tests` |
| 5 | Experiments | `lesion operators`, `seeded individuals`, `evaluation metrics` |
| 6 | Reporting | `retinotopy plots`, `size depth plot`, `report renderer` |
| 7 | Runs | `run intact`, `run lesion`, `run individuals`, `updated results` |
| 8 | Docs | `updated readme`, `add adrs`, `add citation` |

Rule: no commit both adds a feature and changes unrelated formatting.

---

## 7. Open decisions

| ID | Decision | Assumption taken | Revisit when |
|---|---|---|---|
| D-1 | Backbone | AlexNet primary, ResNet-18 second | if depth resolution insufficient |
| D-2 | Activation pooling | channel-mean over spatial map per unit-position | if pRFs come out degenerate |
| D-3 | Unit definition | one "unit" = one (channel, spatial position) | if unit count explodes memory |
| D-4 | Lesion type | zero-ablation of a contiguous spatial patch | may add channel-wise |
| D-5 | Individuals | seeded random readout + dropout mask over frozen features | if variance too small to measure |

Each becomes an ADR in `docs/adr/` once settled with evidence.

---

## 8. What this plan does not promise

- No claim that any hypothesis H1–H4 will hold. They are tests, and a negative result is reported
  as a negative result.
- No biological realism claim (§1.4).
- No performance number appears in any document until a real run produces it.
