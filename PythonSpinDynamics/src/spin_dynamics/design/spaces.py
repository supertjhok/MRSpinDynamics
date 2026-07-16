"""Finite candidate design spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _same_design(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        comparison = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(comparison, np.ndarray):
        return bool(np.all(comparison))
    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class CandidateDesignSpace:
    """Ordered finite actions available to an adaptive session."""

    actions: tuple[Any, ...]

    def __init__(self, actions: Iterable[Any]) -> None:
        values = tuple(actions)
        if not values:
            raise ValueError("actions must not be empty")
        object.__setattr__(self, "actions", values)

    def index(self, design: Any) -> int:
        """Return the stable action index used by session checkpoints."""

        for index, candidate in enumerate(self.actions):
            if _same_design(candidate, design):
                return index
        raise ValueError("design is not part of this candidate space")

