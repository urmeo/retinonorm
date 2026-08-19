"""Run configuration.

Every parameter that can change a result lives in one of these dataclasses. Nothing in the
pipeline reads loose keyword arguments, so a run is fully described by its :class:`RunConfig`
and reproducible from its digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T", bound="ConfigBase")


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


def _restore(annotation: Any, value: Any) -> Any:
    """Rebuild the declared container type from its JSON representation.

    JSON has no tuple, so every tuple field arrives as a list. Without this, a round-trip
    would silently change field types and break both hashing and frozen-dataclass equality.
    """
    if get_origin(annotation) is tuple:
        (element_type, *_) = get_args(annotation) or (Any,)
        return tuple(_restore(element_type, item) for item in value)
    return value


@dataclass(frozen=True)
class ConfigBase:
    """Shared serialisation for every configuration object."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], payload: Dict[str, Any]) -> T:
        hints = get_type_hints(cls)
        known = {field.name for field in fields(cls)}
        unexpected = set(payload) - known
        if unexpected:
            raise ConfigError(f"{cls.__name__} received unknown keys: {sorted(unexpected)}")
        restored = {name: _restore(hints[name], value) for name, value in payload.items()}
        return cls(**restored)


@dataclass(frozen=True)
class StimulusConfig(ConfigBase):
    """Retinotopic mapping stimulus.

    A bar aperture sweeps the visual field once per entry in ``directions``, exposing a noise
    carrier that drives the network without itself carrying retinotopic structure.
    """

    resolution: int = 128
    n_steps: int = 32
    bar_width_frac: float = 0.125
    directions: Tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
    carrier_seed: int = 0

    def __post_init__(self) -> None:
        if self.resolution < 8:
            raise ConfigError("resolution must be at least 8 pixels")
        if self.resolution % 2:
            raise ConfigError("resolution must be even so the field has a true centre")
        if self.n_steps < 2:
            raise ConfigError("n_steps must be at least 2")
        if not 0.0 < self.bar_width_frac < 1.0:
            raise ConfigError("bar_width_frac must lie in (0, 1)")
        if not self.directions:
            raise ConfigError("at least one sweep direction is required")
        if any(not 0 <= d < 360 for d in self.directions):
            raise ConfigError("directions must be degrees in [0, 360)")

    @property
    def n_frames(self) -> int:
        return self.n_steps * len(self.directions)

    @property
    def bar_width_px(self) -> float:
        return self.bar_width_frac * self.resolution


@dataclass(frozen=True)
class ModelConfig(ConfigBase):
    """Which network to probe, and where to tap it."""

    name: str = "alexnet"
    layers: Tuple[str, ...] = ("features.2", "features.5", "features.12")
    weights_seed: Optional[int] = None
    pool_to: int = 8

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("model name must not be empty")
        if not self.layers:
            raise ConfigError("at least one tap layer is required")
        if len(set(self.layers)) != len(self.layers):
            raise ConfigError("tap layers must be unique")
        if self.pool_to < 2:
            raise ConfigError("pool_to must be at least 2 to retain retinotopic structure")


@dataclass(frozen=True)
class FitConfig(ConfigBase):
    """Search settings for the two-stage pRF fit.

    A coarse grid selects the basin; nonlinear refinement finds the minimum inside it. Grid
    search alone is too coarse to trust, and refinement alone settles into local minima.
    """

    grid_size: int = 12
    sigma_bounds: Tuple[float, float] = (0.5, 40.0)
    max_nfev: int = 200
    r2_threshold: float = 0.2

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ConfigError("grid_size must be at least 3")
        low, high = self.sigma_bounds
        if low <= 0:
            raise ConfigError("lower sigma bound must be positive")
        if high <= low:
            raise ConfigError("sigma_bounds must be increasing")
        if self.max_nfev < 1:
            raise ConfigError("max_nfev must be positive")
        if not 0.0 <= self.r2_threshold <= 1.0:
            raise ConfigError("r2_threshold must lie in [0, 1]")


@dataclass(frozen=True)
class RunConfig(ConfigBase):
    """A complete, reproducible experiment."""

    stimulus: StimulusConfig = StimulusConfig()
    model: ModelConfig = ModelConfig()
    fit: FitConfig = FitConfig()
    seed: int = 0

    _SECTIONS = {"stimulus": StimulusConfig, "model": ModelConfig, "fit": FitConfig}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> RunConfig:
        sections: Dict[str, Any] = dict(payload)
        for name, section_type in cls._SECTIONS.items():
            if name in sections:
                sections[name] = section_type.from_dict(sections[name])
        return super().from_dict(sections)  # type: ignore[return-value]

    def to_json(self) -> str:
        """Canonical JSON: sorted keys and fixed separators, so the digest is stable."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_dict(json.loads(path.read_text()))
