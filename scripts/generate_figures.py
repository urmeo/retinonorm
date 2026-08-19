"""Regenerate the figures in README.md.

Every figure is drawn from the same measurements and the same configuration as
``generate_validation_report.py``, so a figure cannot disagree with the table beside it.
Needs the ``viz`` extra::

    python3 -m pip install -e '.[viz]'
    python3 scripts/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_validation_report import (
    CONFIG,
    CV_SEEDS,
    GROUND_TRUTH,
    RECOVERY_NOISE,
    RECOVERY_SEED,
    SPLIT_SEED,
    autocorrelated_noise,
    synthesise,
)

from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures, frame_similarity

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
INK = "#1b1f23"
ACCENT = "#2f6f9f"
WARN = "#b4442e"
GOOD = "#3f7d58"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#d0d7de",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def worst_cross_group_similarity(apertures, groups) -> float:
    similarity = frame_similarity(apertures)
    worst = 0.0
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        if len(train):
            worst = max(worst, float(similarity[np.ix_(test, train)].max(axis=1).max()))
    return worst


def figure_stimulus() -> None:
    """One row per design, frames coloured by cross-validation group."""
    fig, axes = plt.subplots(3, 6, figsize=(8.4, 4.4))
    for row, kind in enumerate(("bar", "wedge", "ring")):
        sequence = build_apertures(CONFIG.stimulus, kind)
        picks = np.linspace(0, sequence.n_frames - 1, 6).astype(int)
        for column, index in enumerate(picks):
            axis = axes[row, column]
            axis.imshow(sequence.apertures[index], cmap="gray_r", interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"group {sequence.group[index]}", fontsize=7, pad=2)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            f"{kind}\n{sequence.n_frames} frames",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=26,
        )
    fig.suptitle("Mapping stimuli, frames labelled by cross-validation group", fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "stimuli.png", dpi=150)
    plt.close(fig)


def figure_fold_leakage() -> None:
    """Worst held-out/training similarity, grouping by direction versus by axis."""
    grid_config = CONFIG.stimulus
    default = grid_config.__class__(resolution=64, n_steps=20)  # shipped 8 directions

    labels, before, after = [], [], []

    sequence = build_apertures(default, "bar")
    by_direction = np.array([d for d in default.directions for _ in range(default.n_steps)])
    labels.append("bar\n(8 directions)")
    before.append(worst_cross_group_similarity(sequence.apertures, by_direction))
    after.append(worst_cross_group_similarity(sequence.apertures, sequence.group))

    for kind in ("wedge", "ring"):
        unpruned = build_apertures(
            grid_config.__class__(resolution=64, n_steps=20, max_fold_similarity=0.999), kind
        )
        pruned = build_apertures(default, kind)
        labels.append(kind)
        before.append(worst_cross_group_similarity(unpruned.apertures, unpruned.group))
        after.append(worst_cross_group_similarity(pruned.apertures, pruned.group))

    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(6.2, 3.2))
    axis.bar(x - 0.19, before, 0.36, label="before", color=WARN)
    axis.bar(x + 0.19, after, 0.36, label="after", color=GOOD)
    axis.axhline(
        default.max_fold_similarity,
        color=INK,
        ls="--",
        lw=1,
        label=f"threshold {default.max_fold_similarity}",
    )
    for index, (b, a) in enumerate(zip(before, after)):
        axis.text(index - 0.19, b + 0.02, f"{b:.3f}", ha="center", fontsize=8, color=WARN)
        axis.text(index + 0.19, a + 0.02, f"{a:.3f}", ha="center", fontsize=8, color=GOOD)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("worst held-out ↔ training\ncosine similarity")
    axis.set_title("A held-out frame must not resemble a training frame")
    axis.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES / "fold-leakage.png", dpi=150)
    plt.close(fig)


def figure_recovery() -> None:
    """Position and size error against noise, one line per ground-truth pRF."""
    grid = Grid(CONFIG.stimulus.resolution)
    apertures = build_apertures(CONFIG.stimulus, "bar").as_float()
    fitter = PRFFitter(grid, apertures, CONFIG.fit)

    position = np.zeros((len(GROUND_TRUTH), len(RECOVERY_NOISE)))
    size = np.zeros_like(position)
    for row, truth in enumerate(GROUND_TRUTH):
        for column, noise in enumerate(RECOVERY_NOISE):
            fit = fitter.fit_unit(
                synthesise(grid, apertures, truth, noise=noise, seed=RECOVERY_SEED)
            )
            position[row, column] = float(np.hypot(fit.x0 - truth[0], fit.y0 - truth[1]))
            size[row, column] = abs(fit.sigma - truth[2]) / truth[2] * 100.0

    noise_pct = [n * 100 for n in RECOVERY_NOISE]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for axis, data, label, places in (
        (axes[0], position, "position error (px)", 3),
        (axes[1], size, "sigma error (%)", 1),
    ):
        for row in range(len(GROUND_TRUTH)):
            axis.plot(
                noise_pct,
                data[row],
                marker="o",
                ms=3.5,
                lw=1,
                color="#c3cad1",
                zorder=1,
                label="individual pRF" if row == 0 else None,
            )
        axis.plot(
            noise_pct,
            data.max(axis=0),
            marker="o",
            ms=6,
            lw=2.2,
            color=WARN,
            zorder=3,
            label="worst of 5",
        )
        axis.plot(
            noise_pct,
            data.mean(axis=0),
            marker="o",
            ms=6,
            lw=2.2,
            color=ACCENT,
            zorder=3,
            label="mean of 5",
        )
        for column, value in enumerate(data.max(axis=0)):
            # Noiseless recovery is exact to machine precision; show it as zero, not as 3e-09.
            shown = 0.0 if value < 1e-6 else value
            axis.annotate(
                f"{shown:.{places}f}",
                (noise_pct[column], value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                color=WARN,
                fontweight="bold",
            )
        axis.set_xlabel("noise (% of signal standard deviation)")
        axis.set_ylabel(label)
        axis.set_xticks(noise_pct)
        axis.set_xticklabels([f"{int(n)}" for n in noise_pct])
        axis.margins(y=0.28)
        axis.set_ylim(bottom=-data.max() * 0.06)
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Accuracy degrades gracefully with noise", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "recovery.png", dpi=150)
    plt.close(fig)


def figure_leakage() -> None:
    """Held-out score under an honest split and a random frame split, over 20 seeds."""
    grid = Grid(CONFIG.stimulus.resolution)
    sequence = build_apertures(CONFIG.stimulus, "bar")
    apertures = sequence.as_float()
    grouped = CrossValidator(grid, apertures, sequence.group, CONFIG.fit)
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(len(apertures)) % 4
    leaky = CrossValidator(grid, apertures, shuffled, CONFIG.fit)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for axis, label in zip(axes, ("white", "autocorrelated")):
        honest, split = [], []
        for seed in CV_SEEDS:
            if label == "white":
                response = np.random.default_rng(seed).normal(size=len(apertures))
            else:
                response = autocorrelated_noise(len(apertures), 3, seed)
            honest.append(grouped.validate_unit(response).cv_r2)
            split.append(leaky.validate_unit(response).cv_r2)

        for h, s in zip(honest, split):
            axis.plot([0, 1], [h, s], color="#8c959f", lw=0.7, zorder=1)
        axis.scatter(np.zeros(len(honest)), honest, s=22, color=GOOD, zorder=2, label="grouped")
        axis.scatter(np.ones(len(split)), split, s=22, color=WARN, zorder=2, label="random split")
        axis.axhline(0, color=INK, lw=1)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["grouped\n(axis)", "random\nframe split"])
        axis.set_xlim(-0.35, 1.35)
        leak = np.mean(np.asarray(split) - np.asarray(honest))
        axis.set_title(
            f"{label} noise\nmean leak {leak:+.3f}, "
            f"{int((np.asarray(split) > np.asarray(honest)).sum())}/{len(CV_SEEDS)} positive"
        )
    axes[0].set_ylabel("held-out R²")
    fig.suptitle(
        "A random frame split inflates the score on temporally correlated noise", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "leakage.png", dpi=150)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for draw in (figure_stimulus, figure_fold_leakage, figure_recovery, figure_leakage):
        draw()
        print(f"drew {draw.__name__.removeprefix('figure_')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
