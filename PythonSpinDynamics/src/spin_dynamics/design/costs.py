"""Physical acquisition-cost models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def _valid_seconds(value: float) -> float:
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("physical acquisition cost must be finite and positive")
    return seconds


@dataclass(frozen=True)
class ConstantCost:
    """The same positive physical duration for every action."""

    seconds_per_action: float

    def __post_init__(self) -> None:
        _valid_seconds(self.seconds_per_action)

    def seconds(self, design: Any) -> float:
        del design
        return float(self.seconds_per_action)


@dataclass(frozen=True)
class CallableCost:
    """Physical duration supplied by ``function(design)``."""

    function: Callable[[Any], float]

    def seconds(self, design: Any) -> float:
        return _valid_seconds(self.function(design))

