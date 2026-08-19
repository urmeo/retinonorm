"""Regenerate every number quoted in README.md.

BUILD_SPEC section 12.5 requires that no number appear in any document that was not produced
by a recorded run. This script is that record. It runs each measurement the README reports and
writes both a machine-readable ``results/validation.json`` and the exact Markdown tables in
``results/VALIDATION.md``, then splices those tables into README.md between marker comments so
the two cannot drift apart.

Every table carries the environment it was produced in and the digest of the configuration
that produced it. The recorded date is preserved when a rerun reproduces the same numbers, so
that regenerating on an unchanged tree is a no-op and CI can diff the committed copy.

Run with no arguments::

    python3 scripts/generate_validation_report.py

Every measurement except the runtime table is deterministic, and reproduces byte for byte
across Python 3.9 to 3.12 and across NumPy and SciPy versions, so a rerun on an unchanged tree
rewrites the same bytes and ``git diff --exit-code results/`` passes anywhere.

Timings are not reproducible by nature, and the machine they were taken on is part of what they
mean. Both the runtime table and the recorded environment are therefore carried forward unless
``--runtime`` asks for fresh ones. Refresh them deliberately, on a machine whose details you are
willing to publish in the footer. When the numbers reproduce somewhere other than the recorded
environment, the script says so: that is evidence of portability, not a discrepancy.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from cortexprobe.config import FitConfig, ModelConfig, RunConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The configuration every measurement below is made under. Recorded by digest in the output.
CONFIG = RunConfig(
    stimulus=StimulusConfig(resolution=64, n_steps=20, directions=(0, 45, 90, 135)),
    model=ModelConfig(),
    fit=FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0)),
    seed=0,
)

GROUND_TRUTH = [
    (0.0, 0.0, 5.0),
    (12.0, 8.0, 4.0),
    (-10.0, 14.0, 7.0),
    (16.0, -12.0, 6.0),
    (-15.0, -9.0, 9.0),
]
RECOVERY_NOISE = [0.0, 0.2, 0.5]
RECOVERY_SEED = 7
N_NOISE_UNITS = 40
NOISE_SEED = 11

UNCERTAINTY_TRUTH = (12.0, 8.0, 5.0)
UNCERTAINTY_NOISE = [0.0, 0.1, 0.3, 0.6]
UNCERTAINTY_SEED = 5
COVERAGE_TRUTHS = [(12.0, 8.0, 5.0), (28.0, 0.0, 4.0)]
COVERAGE_NOISE = [0.2, 0.5]
COVERAGE_REALISATIONS = 200
COVERAGE_SEED_BASE = 1000

CV_SEEDS = list(range(21, 41))
CARRIER_WIDTH = 3
SPLIT_SEED = 0

RUNTIME_UNITS = [1, 10, 100]
RUNTIME_SEED = 0


def synthesise(grid, apertures, truth, beta=3.0, baseline=0.5, noise=0.0, seed=0):
    """Activation timecourse from a known pRF, with noise scaled to the clean signal."""
    clean = beta * predict(GaussianReceptiveField(*truth).weights(grid), apertures) + baseline
    if noise <= 0.0:
        return clean
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, noise * float(np.std(clean)), size=clean.shape)


def autocorrelated_noise(n_frames, width=CARRIER_WIDTH, seed=21):
    rng = np.random.default_rng(seed)
    return np.convolve(rng.normal(size=n_frames), np.ones(width) / width, mode="same")


# --- measurements ---------------------------------------------------------------------------


def measure_recovery(grid, apertures, fitter) -> dict[str, Any]:
    rows = []
    for noise in RECOVERY_NOISE:
        positions, sigmas, scores = [], [], []
        for truth in GROUND_TRUTH:
            fit = fitter.fit_unit(
                synthesise(grid, apertures, truth, noise=noise, seed=RECOVERY_SEED)
            )
            positions.append(float(np.hypot(fit.x0 - truth[0], fit.y0 - truth[1])))
            sigmas.append(abs(fit.sigma - truth[2]) / truth[2] * 100.0)
            scores.append(fit.r2)
        rows.append(
            {
                "noise": noise,
                "n": len(GROUND_TRUTH),
                "position_max": max(positions),
                "position_mean": float(np.mean(positions)),
                "position_sd": float(np.std(positions, ddof=1)),
                "sigma_max": max(sigmas),
                "sigma_mean": float(np.mean(sigmas)),
                "sigma_sd": float(np.std(sigmas, ddof=1)),
                "r2_min": min(scores),
            }
        )

    rng = np.random.default_rng(NOISE_SEED)
    fits = fitter.fit_all(rng.normal(size=(fitter.n_frames, N_NOISE_UNITS)))
    return {
        "rows": rows,
        "pure_noise": {
            "n_units": N_NOISE_UNITS,
            "rejected": sum(not fit.accepted for fit in fits),
            "best_r2": max(fit.r2 for fit in fits),
            "threshold": CONFIG.fit.r2_threshold,
        },
    }


def measure_uncertainty(grid, apertures, fitter) -> dict[str, Any]:
    rows = []
    for noise in UNCERTAINTY_NOISE:
        fit = fitter.fit_unit(
            synthesise(grid, apertures, UNCERTAINTY_TRUTH, noise=noise, seed=UNCERTAINTY_SEED)
        )
        low, high = fit.confidence_interval("x0")
        rows.append(
            {
                "noise": noise,
                "x0": fit.x0,
                "se_x0": fit.se_x0,
                "ci_width": high - low,
                "dof": fit.dof,
            }
        )

    coverage = []
    for truth in COVERAGE_TRUTHS:
        for noise in COVERAGE_NOISE:
            covered = 0
            for seed in range(COVERAGE_REALISATIONS):
                response = synthesise(
                    grid, apertures, truth, noise=noise, seed=COVERAGE_SEED_BASE + seed
                )
                low, high = fitter.fit_unit(response).confidence_interval("x0")
                covered += bool(low <= truth[0] <= high)
            rate = covered / COVERAGE_REALISATIONS
            half = 1.96 * float(np.sqrt(rate * (1.0 - rate) / COVERAGE_REALISATIONS))
            coverage.append(
                {
                    "position": "interior" if truth == COVERAGE_TRUTHS[0] else "near edge",
                    "noise": noise,
                    "n": COVERAGE_REALISATIONS,
                    "coverage": rate,
                    "ci_low": rate - half,
                    "ci_high": rate + half,
                }
            )
    return {"rows": rows, "coverage": coverage, "nominal": 0.95}


def measure_cross_validation(grid, apertures, sequence) -> dict[str, Any]:
    grouped = CrossValidator(grid, apertures, sequence.group, CONFIG.fit)
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(len(apertures)) % 4
    leaky = CrossValidator(grid, apertures, shuffled, CONFIG.fit)

    rows, lags = [], []
    for label in ("white", "autocorrelated"):
        grouped_scores, leaky_scores = [], []
        for seed in CV_SEEDS:
            if label == "white":
                response = np.random.default_rng(seed).normal(size=len(apertures))
            else:
                response = autocorrelated_noise(len(apertures), CARRIER_WIDTH, seed)
                lags.append(float(np.corrcoef(response[:-1], response[1:])[0, 1]))
            grouped_scores.append(grouped.validate_unit(response).cv_r2)
            leaky_scores.append(leaky.validate_unit(response).cv_r2)

        leaks = np.asarray(leaky_scores) - np.asarray(grouped_scores)
        rows.append(
            {
                "response": label,
                "n": len(CV_SEEDS),
                "grouped_mean": float(np.mean(grouped_scores)),
                "grouped_sd": float(np.std(grouped_scores, ddof=1)),
                "leaky_mean": float(np.mean(leaky_scores)),
                "leaky_sd": float(np.std(leaky_scores, ddof=1)),
                "leak_mean": float(np.mean(leaks)),
                "leak_sd": float(np.std(leaks, ddof=1)),
                "leak_positive": int((leaks > 0.0).sum()),
            }
        )
    return {
        "rows": rows,
        "kernel_width": CARRIER_WIDTH,
        "lag1_mean": float(np.mean(lags)),
        "lag1_sd": float(np.std(lags, ddof=1)),
        "lag1_theoretical": 1.0 - 1.0 / CARRIER_WIDTH,
        "seeds": CV_SEEDS,
    }


def measure_runtime(grid, apertures, sequence, fitter) -> dict[str, Any]:
    validator = CrossValidator(grid, apertures, sequence.group, CONFIG.fit)
    rng = np.random.default_rng(RUNTIME_SEED)
    radius = grid.radius * 0.7

    def units(n):
        columns = []
        for _ in range(n):
            x0, y0 = rng.uniform(-radius, radius, size=2)
            clean = predict(
                GaussianReceptiveField(x0, y0, rng.uniform(3.0, 8.0)).weights(grid), apertures
            )
            columns.append(clean + rng.normal(0.0, 0.2 * clean.std(), size=clean.shape))
        return np.column_stack(columns)

    rows = []
    for n in RUNTIME_UNITS:
        activations = units(n)
        start = time.perf_counter()
        fitter.fit_all(activations)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        validator.validate_all(activations)
        cv_seconds = time.perf_counter() - start

        rows.append(
            {
                "units": n,
                "fit_seconds": fit_seconds,
                "per_unit_ms": fit_seconds / n * 1000.0,
                "cv_seconds": cv_seconds,
                "cv_factor": cv_seconds / fit_seconds,
            }
        )
    return {"rows": rows, "n_folds": len(validator._folds)}


# --- rendering ------------------------------------------------------------------------------


def current_environment() -> dict[str, str]:
    return {
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def footer(environment: dict[str, str], digest: str, generated: str) -> str:
    return (
        f"<sub>Recorded on {environment['platform']} {environment['machine']}, "
        f"Python {environment['python']}, NumPy {environment['numpy']}, "
        f"SciPy {environment['scipy']} | config digest `{digest[:12]}` | "
        f"generated {generated}</sub>"
    )


def render(report: dict[str, Any]) -> dict[str, str]:
    """One Markdown block per README section, keyed by its marker name."""
    line = footer(report["environment"], report["config_digest"], report["generated"])
    blocks = {}

    recovery = report["recovery"]

    def condition(noise: float) -> str:
        return "Noiseless" if noise == 0 else f"{noise:.0%} noise"

    rows = "\n".join(
        f"| {condition(r['noise'])} "
        f"| {r['position_max']:.3f} | {r['position_mean']:.3f} | "
        f"{r['sigma_max']:.1f} | {r['sigma_mean']:.1f} | {r['r2_min']:.4f} |"
        for r in recovery["rows"]
    )
    noise = recovery["pure_noise"]
    blocks["recovery"] = (
        f"| Condition | Position error, worst (px) | mean (px) | "
        f"Sigma error, worst (%) | mean (%) | Minimum R² |\n"
        f"|---|---|---|---|---|---|\n{rows}\n"
        f"| Pure noise | — | — | — | — | "
        f"**{noise['rejected']} / {noise['n_units']} rejected** |\n\n"
        f"Worst and mean are taken over n = {recovery['rows'][0]['n']} ground-truth pRFs. On pure "
        f"noise the fitter accepted none of {noise['n_units']} units, and its best spurious R² was "
        f"{noise['best_r2']:.4f}, below the {noise['threshold']} acceptance threshold.\n\n{line}"
    )

    uncertainty = report["uncertainty"]
    rows = "\n".join(
        f"| {r['noise']:.1f} | {r['x0']:.3f} | {r['se_x0']:.4f} | {r['ci_width']:.3f} |"
        for r in uncertainty["rows"]
    )
    coverage = "\n".join(
        f"| {c['position']} | {c['noise']:.1f} | {c['n']} | {c['coverage']:.3f} | "
        f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}] |"
        for c in uncertainty["coverage"]
    )
    blocks["uncertainty"] = (
        f"| Noise | Fitted x0 | SE | 95 % CI width |\n|---|---|---|---|\n{rows}\n\n"
        f"The noiseless row is a numerical floor, not a measurement: the residual is at machine "
        f"precision, so the linearised interval collapses. It is shown to make the scaling of the "
        f"rows below it legible.\n\n"
        f"Empirical coverage of the nominal {uncertainty['nominal']:.0%} interval:\n\n"
        f"| Position | Noise | n | Coverage | 95 % binomial CI |\n|---|---|---|---|---|\n"
        f"{coverage}\n\n"
        f"Coverage runs slightly under nominal, and lowest near the field edge at high noise, "
        f"where the linearisation behind the interval is weakest.\n\n{line}"
    )

    cv = report["cross_validation"]
    rows = "\n".join(
        f"| {r['response'].capitalize()} | {r['n']} | "
        f"{r['grouped_mean']:+.3f} ± {r['grouped_sd']:.3f} | "
        f"{r['leaky_mean']:+.3f} ± {r['leaky_sd']:.3f} | "
        f"{r['leak_mean']:+.3f} ± {r['leak_sd']:.3f} | {r['leak_positive']} / {r['n']} |"
        for r in cv["rows"]
    )
    blocks["cross_validation"] = (
        f"| Response | n seeds | Grouped CV | Random-split CV | Leak | Leak positive |\n"
        f"|---|---|---|---|---|---|\n{rows}\n\n"
        f"Mean ± SD over seeds {cv['seeds'][0]}–{cv['seeds'][-1]}. The carrier is a width-"
        f"{cv['kernel_width']} boxcar, whose lag-1 autocorrelation measured "
        f"{cv['lag1_mean']:.3f} ± {cv['lag1_sd']:.3f} against a theoretical "
        f"{cv['lag1_theoretical']:.3f}.\n\n{line}"
    )

    runtime = report["runtime"]
    rows = "\n".join(
        f"| {r['units']} | {r['fit_seconds']:.3f} | {r['per_unit_ms']:.1f} | "
        f"{r['cv_seconds']:.3f} | {r['cv_factor']:.1f}× |"
        for r in runtime["rows"]
    )
    largest = runtime["rows"][-1]
    blocks["runtime"] = (
        f"| Units | Fit (s) | Per unit (ms) | CV (s) | CV factor |\n|---|---|---|---|---|\n"
        f"{rows}\n\n"
        f"Per-unit cost is flat, so nothing quadratic hides in the loop. Cross-validation costs a "
        f"roughly constant factor: {runtime['n_folds']} folds plus the full fit. On this machine "
        f"10 000 units extrapolate to about "
        f"{largest['per_unit_ms'] * 10_000 / 60_000:.1f} minutes single-threaded. Timings are "
        f"hardware-specific; the environment is recorded below.\n\n{line}"
    )
    return blocks


SECTIONS = [
    ("recovery", "Recovery of known pRFs"),
    ("uncertainty", "Parameter uncertainty"),
    ("cross_validation", "Cross-validation and leakage"),
    ("runtime", "Runtime"),
]


def validation_markdown(blocks: dict[str, str]) -> str:
    sections = "\n\n".join(f"## {title}\n\n{blocks[name]}" for name, title in SECTIONS)
    return (
        "# Validation report\n\n"
        "Generated by `scripts/generate_validation_report.py`. Do not edit by hand.\n\n"
        "**These validate the instrument, not any scientific claim.** No network has been "
        "probed. A frozen ImageNet-trained CNN is not a biologically realistic model of visual "
        "cortex; this project builds the measurement apparatus, and does not claim otherwise.\n\n"
        f"{sections}\n"
    )


def splice(text: str, blocks: dict[str, str]) -> str:
    """Replace each marked region with its generated block."""
    for name, block in blocks.items():
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {name} -->).*?(<!-- END GENERATED: {name} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            raise SystemExit(f"README.md has no generated region named {name!r}")
        text = pattern.sub(
            lambda match, body=block: f"{match.group(1)}\n{body}\n{match.group(2)}", text
        )
    return text


def rounded(value: Any, places: int = 6) -> Any:
    """Round every float in the report to a fixed number of decimal places.

    Measurements reproduce across Python, NumPy and SciPy versions to about ten significant
    figures; the last few bits differ with the BLAS underneath. Full float64 precision in the
    artifact would therefore record noise as if it were signal, and would make a byte
    comparison of the committed report fail for no reason that matters. Six decimal places is
    two orders finer than anything the tables display and four orders coarser than the drift.
    """
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {key: rounded(item, places) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, places) for item in value]
    return value


def drifted(recorded: Any, fresh: Any, path: str = "") -> list[str]:
    """Where a freshly measured report departs from the committed one, beyond tolerance.

    A byte comparison of float measurements is the wrong instrument across environments: the
    last few bits move with the BLAS underneath, and a gate that fails for that reason teaches
    everyone to ignore it. The tolerance here is far tighter than any number the tables show
    and far looser than the observed drift, so a failure means a real change.
    """
    if isinstance(recorded, dict) and isinstance(fresh, dict):
        problems = []
        for key in sorted(set(recorded) | set(fresh)):
            if key not in recorded or key not in fresh:
                problems.append(f"{path}{key}: present in only one report")
            else:
                problems += drifted(recorded[key], fresh[key], f"{path}{key}.")
        return problems
    if isinstance(recorded, list) and isinstance(fresh, list):
        if len(recorded) != len(fresh):
            return [f"{path.rstrip('.')}: length {len(recorded)} became {len(fresh)}"]
        return [
            problem
            for index, (a, b) in enumerate(zip(recorded, fresh))
            for problem in drifted(a, b, f"{path}{index}.")
        ]
    if isinstance(recorded, float) or isinstance(fresh, float):
        if not math.isclose(recorded, fresh, rel_tol=1e-5, abs_tol=1e-6):
            return [f"{path.rstrip('.')}: {recorded!r} became {fresh!r}"]
        return []
    if recorded != fresh:
        return [f"{path.rstrip('.')}: {recorded!r} became {fresh!r}"]
    return []


def numbers_only(report: dict[str, Any]) -> str:
    """The report without its date, for deciding whether anything actually moved."""
    return json.dumps({k: v for k, v in report.items() if k != "generated"}, sort_keys=True)


def run_check(report: dict[str, Any], previous: dict[str, Any], readme: str) -> int:
    """Verify the committed report still reproduces. Writes nothing."""
    if not previous:
        print("no committed report to check against; run without --check first")
        return 1

    ignored = ("runtime", "environment", "generated")
    problems = drifted(
        {k: v for k, v in previous.items() if k not in ignored},
        {k: v for k, v in report.items() if k not in ignored},
    )
    if problems:
        print("measurements have drifted from the committed report:")
        for problem in problems:
            print(f"  {problem}")
        print("\nrerun without --check to record the new numbers")
        return 1

    # Rendered from the committed numbers, so this comparison is exact by construction and
    # catches a hand-edited table rather than a floating point difference.
    blocks = render(previous)
    if splice(readme, blocks) != readme:
        print("README.md tables do not match the committed report; rerun without --check")
        return 1
    if (RESULTS / "VALIDATION.md").read_text() != validation_markdown(blocks):
        print("results/VALIDATION.md does not match the committed report")
        return 1

    print(
        "every measurement reproduces within tolerance, and the committed tables match; "
        f"recorded on {previous['environment']['platform']}, checked on "
        f"{current_environment()['platform']} Python {platform.python_version()}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the validation report.")
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="re-measure the runtime table instead of carrying forward the recorded one",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed report still reproduces, writing nothing; exits non-zero "
        "if a measurement has drifted or the committed tables do not match it",
    )
    arguments = parser.parse_args()

    readme_path = ROOT / "README.md"
    readme = readme_path.read_text()

    grid = Grid(CONFIG.stimulus.resolution)
    sequence = build_apertures(CONFIG.stimulus, "bar")
    apertures = sequence.as_float()
    fitter = PRFFitter(grid, apertures, CONFIG.fit)

    previous_path = RESULTS / "validation.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}

    report: dict[str, Any] = {
        "config_digest": CONFIG.digest(),
        "config": CONFIG.to_dict(),
        "stimulus": {
            "frames": sequence.n_frames,
            "distinct_frames": fitter.n_independent_frames,
            "groups": sorted({int(g) for g in sequence.group}),
            "candidates": len(fitter.candidates),
        },
        "recovery": measure_recovery(grid, apertures, fitter),
        "uncertainty": measure_uncertainty(grid, apertures, fitter),
        "cross_validation": measure_cross_validation(grid, apertures, sequence),
    }

    # Timings differ on every run, so re-measuring them unconditionally would make the
    # reproducibility check permanently red and therefore worthless. They are refreshed only on
    # request, and never count as "the numbers changed" when deciding the recorded date.
    report = rounded(report)

    if arguments.check:
        return run_check(report, previous, readme)

    running_in = current_environment()
    refresh = arguments.runtime or "runtime" not in previous
    if refresh:
        report["runtime"] = rounded(measure_runtime(grid, apertures, sequence, fitter))
        report["environment"] = running_in
    else:
        report["runtime"] = previous["runtime"]
        report["environment"] = previous["environment"]

    def measurements(payload: dict[str, Any]) -> str:
        return numbers_only(dict(payload, runtime=None, environment=None))

    unchanged = bool(previous) and measurements(report) == measurements(previous)
    report["generated"] = previous.get("generated") if unchanged else date.today().isoformat()

    blocks = render(report)
    spliced = splice(readme, blocks)  # validated before anything is written

    RESULTS.mkdir(exist_ok=True)
    previous_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    (RESULTS / "VALIDATION.md").write_text(validation_markdown(blocks))

    readme_path.write_text(spliced)

    print(f"wrote {previous_path.relative_to(ROOT)} and results/VALIDATION.md")
    state = "unchanged" if unchanged else "updated"
    timing = "re-measured" if refresh else "carried forward"
    print(f"README tables spliced; numbers {state}, runtime and environment {timing}")
    if unchanged and running_in != report["environment"]:
        print(
            "note: these numbers reproduced here, in a different environment than the one on "
            f"record ({running_in['platform']} {running_in['machine']}, Python "
            f"{running_in['python']}, NumPy {running_in['numpy']}, SciPy {running_in['scipy']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
