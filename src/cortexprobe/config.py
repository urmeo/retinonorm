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
from typing import Any, ClassVar, Optional, TypeVar, get_args, get_origin, get_type_hints

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], payload: dict[str, Any]) -> T:
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

    A bar aperture sweeps the visual field once per entry in ``directions``. The noise carrier
    that will fill the aperture arrives with ``models.py``, when there is a network input for
    it to drive; it is not configured here until then.

    ``max_fold_similarity`` is the largest cosine overlap tolerated between a held-out frame
    and any training frame. Frames that exceed it across a fold boundary are dropped when the
    sequence is built, so a cross-validation score cannot be inflated by near-duplicate frames
    straddling the split.
    """

    resolution: int = 128
    n_steps: int = 32
    bar_width_frac: float = 0.125
    directions: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
    wedge_span_deg: float = 45.0
    ring_thickness_frac: float = 0.125
    max_fold_similarity: float = 0.75

    def __post_init__(self) -> None:
        if self.resolution < 8:
            raise ConfigError("resolution must be at least 8 pixels")
        if self.resolution % 2:
            raise ConfigError("resolution must be even to keep the field symmetric about the origin")
        if self.n_steps < 2:
            raise ConfigError("n_steps must be at least 2")
        if not 0.0 < self.bar_width_frac < 1.0:
            raise ConfigError("bar_width_frac must lie in (0, 1)")
        if not self.directions:
            raise ConfigError("at least one sweep direction is required")
        if any(not 0 <= d < 360 for d in self.directions):
            raise ConfigError("directions must be degrees in [0, 360)")
        if not 0.0 < self.wedge_span_deg <= 360.0:
            raise ConfigError("wedge_span_deg must lie in (0, 360]")
        if not 0.0 < self.ring_thickness_frac < 1.0:
            raise ConfigError("ring_thickness_frac must lie in (0, 1)")
        if not 0.0 < self.max_fold_similarity < 1.0:
            raise ConfigError("max_fold_similarity must lie in (0, 1)")

    @property
    def n_bar_frames(self) -> int:
        """Frame count for a bar run only. Wedge and ring runs are ``n_steps`` frames."""
        return self.n_steps * len(self.directions)

    @property
    def bar_width_px(self) -> float:
        return self.bar_width_frac * self.resolution

    @property
    def ring_thickness_px(self) -> float:
        return self.ring_thickness_frac * self.resolution


@dataclass(frozen=True)
class ModelConfig(ConfigBase):
    """Which network to probe, and where to tap it."""

    name: str = "alexnet"
    layers: tuple[str, ...] = ("features.2", "features.5", "features.12")
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

    Both ends of ``sigma_bounds`` are constrained, for the same reason from opposite
    directions. The floor is checked here: below the pixel pitch a Gaussian is under-sampled.
    The ceiling depends on the field size and so is checked by
    :class:`~cortexprobe.prf.fit.PRFFitter`, which knows the grid: much beyond
    ``resolution / 6`` a Gaussian is truncated by the field edge. The default ceiling of 20 px
    suits the default 128 px stimulus.
    """

    grid_size: int = 12
    sigma_bounds: tuple[float, float] = (1.0, 20.0)
    max_nfev: int = 200
    r2_threshold: float = 0.2

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ConfigError("grid_size must be at least 3")
        low, high = self.sigma_bounds
        if low < 1.0:
            raise ConfigError(
                "lower sigma bound must be at least 1 pixel; below the pixel pitch a Gaussian "
                "is under-sampled and loses its unit-volume normalisation"
            )
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

    _SECTIONS: ClassVar[dict[str, type[ConfigBase]]] = {
        "stimulus": StimulusConfig,
        "model": ModelConfig,
        "fit": FitConfig,
    }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunConfig:
        sections: dict[str, Any] = dict(payload)
        for name, section_type in cls._SECTIONS.items():
            if name in sections:
                sections[name] = section_type.from_dict(sections[name])
        return super().from_dict(sections)

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
