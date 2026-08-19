"""Configuration validation, serialisation, and the run digest.

Configuration is the only source of truth in this project, so two things have to hold. Every
guard must actually fire -- including the sigma floor the README elevates to a named design
decision -- and a configuration must survive a round trip through JSON unchanged, or the digest
that identifies a run would not identify it.
"""

from __future__ import annotations

import json

import pytest

from cortexprobe.config import (
    ConfigError,
    FitConfig,
    ModelConfig,
    RunConfig,
    StimulusConfig,
)

# --- stimulus guards ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"resolution": 4}, "at least 8 pixels"),
        ({"resolution": 65}, "must be even"),
        ({"n_steps": 1}, "n_steps must be at least 2"),
        ({"bar_width_frac": 0.0}, "bar_width_frac"),
        ({"bar_width_frac": 1.0}, "bar_width_frac"),
        ({"directions": ()}, "at least one sweep direction"),
        ({"directions": (360,)}, "degrees in .0, 360."),
        ({"directions": (-1,)}, "degrees in .0, 360."),
        ({"wedge_span_deg": 0.0}, "wedge_span_deg"),
        ({"wedge_span_deg": 361.0}, "wedge_span_deg"),
        ({"ring_thickness_frac": 0.0}, "ring_thickness_frac"),
        ({"ring_thickness_frac": 1.0}, "ring_thickness_frac"),
        ({"max_fold_similarity": 0.0}, "max_fold_similarity"),
        ({"max_fold_similarity": 1.0}, "max_fold_similarity"),
    ],
)
def test_stimulus_config_rejects_invalid_values(overrides, message) -> None:
    with pytest.raises(ConfigError, match=message):
        StimulusConfig(**overrides)


def test_stimulus_derived_quantities() -> None:
    config = StimulusConfig(resolution=64, n_steps=10, directions=(0, 90))

    assert config.n_bar_frames == 20
    assert config.bar_width_px == pytest.approx(8.0)
    assert config.ring_thickness_px == pytest.approx(8.0)


# --- model guards ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "model name"),
        ({"layers": ()}, "at least one tap layer"),
        ({"layers": ("a", "a")}, "unique"),
        ({"pool_to": 1}, "pool_to"),
    ],
)
def test_model_config_rejects_invalid_values(overrides, message) -> None:
    with pytest.raises(ConfigError, match=message):
        ModelConfig(**overrides)


# --- fit guards -----------------------------------------------------------------------------


def test_sigma_floor_under_a_pixel_is_rejected_with_a_reason() -> None:
    """The repository's marquee guard: below the pixel pitch a Gaussian loses unit volume."""
    with pytest.raises(ConfigError, match="at least 1 pixel"):
        FitConfig(sigma_bounds=(0.2, 10.0))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grid_size": 2}, "grid_size"),
        ({"sigma_bounds": (0.999, 10.0)}, "at least 1 pixel"),
        ({"sigma_bounds": (5.0, 5.0)}, "increasing"),
        ({"sigma_bounds": (10.0, 5.0)}, "increasing"),
        ({"max_nfev": 0}, "max_nfev"),
        ({"r2_threshold": -0.1}, "r2_threshold"),
        ({"r2_threshold": 1.1}, "r2_threshold"),
    ],
)
def test_fit_config_rejects_invalid_values(overrides, message) -> None:
    with pytest.raises(ConfigError, match=message):
        FitConfig(**overrides)


def test_default_sigma_ceiling_suits_the_default_stimulus() -> None:
    """The shipped defaults must be usable together, not merely valid apart."""
    from cortexprobe.geometry import Grid
    from cortexprobe.prf.fit import MIN_ON_GRID_VOLUME
    from cortexprobe.prf.model import GaussianReceptiveField

    config = RunConfig()
    grid = Grid(config.stimulus.resolution)
    _, sigma_high = config.fit.sigma_bounds

    weights = GaussianReceptiveField(0.0, 0.0, sigma_high).weights(grid)

    assert weights[grid.field_mask].sum() >= MIN_ON_GRID_VOLUME


# --- serialisation --------------------------------------------------------------------------


def test_tuples_survive_a_json_round_trip() -> None:
    """JSON has no tuple. Without restoration a round trip would change field types."""
    config = StimulusConfig(directions=(0, 45, 90))

    restored = StimulusConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert isinstance(restored.directions, tuple)


def test_nested_sections_survive_a_json_round_trip() -> None:
    config = RunConfig(
        stimulus=StimulusConfig(resolution=64, directions=(0, 90)),
        model=ModelConfig(layers=("features.1",), weights_seed=7),
        fit=FitConfig(grid_size=6),
        seed=3,
    )

    restored = RunConfig.from_dict(json.loads(config.to_json()))

    assert restored == config
    assert isinstance(restored.stimulus, StimulusConfig)
    assert isinstance(restored.fit, FitConfig)


def test_optional_field_round_trips_as_none() -> None:
    config = ModelConfig(weights_seed=None)

    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


@pytest.mark.parametrize("section", [StimulusConfig, ModelConfig, FitConfig, RunConfig])
def test_unknown_keys_are_rejected(section) -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        section.from_dict({"nonsense": 1})


def test_canonical_json_is_stable_under_key_order() -> None:
    config = RunConfig()
    shuffled = dict(reversed(list(json.loads(config.to_json()).items())))

    assert RunConfig.from_dict(shuffled).to_json() == config.to_json()


# --- digest ---------------------------------------------------------------------------------


def test_digest_is_stable_across_equal_configurations() -> None:
    assert RunConfig().digest() == RunConfig().digest()
    assert len(RunConfig().digest()) == 64


def test_digest_responds_to_the_seed() -> None:
    assert RunConfig(seed=0).digest() != RunConfig(seed=1).digest()


def test_digest_responds_to_a_nested_change() -> None:
    """A digest that ignored nested sections would identify two different runs alike."""
    baseline = RunConfig()
    changed = RunConfig(fit=FitConfig(grid_size=baseline.fit.grid_size + 1))

    assert baseline.digest() != changed.digest()


def test_digest_responds_to_the_leakage_threshold() -> None:
    baseline = RunConfig()
    changed = RunConfig(stimulus=StimulusConfig(max_fold_similarity=0.5))

    assert baseline.digest() != changed.digest()


def test_digest_survives_a_round_trip(tmp_path) -> None:
    config = RunConfig(seed=11, stimulus=StimulusConfig(resolution=64))
    path = tmp_path / "run.json"

    config.save(path)

    assert RunConfig.load(path) == config
    assert RunConfig.load(path).digest() == config.digest()


def test_saved_configuration_is_readable_json(tmp_path) -> None:
    path = tmp_path / "run.json"
    RunConfig().save(path)

    payload = json.loads(path.read_text())

    assert set(payload) == {"stimulus", "model", "fit", "seed"}
    assert path.read_text().endswith("\n")
