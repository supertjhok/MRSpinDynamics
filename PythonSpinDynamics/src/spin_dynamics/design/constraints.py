"""Composable feasibility constraints for candidate actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class ConstraintResult:
    """Feasibility and an optional human-readable reason."""

    feasible: bool
    message: str = ""


class DesignConstraint(Protocol):
    def evaluate(self, design: Any) -> ConstraintResult: ...


@dataclass(frozen=True)
class CallableConstraint:
    """Constraint backed by a Boolean predicate."""

    predicate: Callable[[Any], bool]
    message: str

    def evaluate(self, design: Any) -> ConstraintResult:
        feasible = bool(self.predicate(design))
        return ConstraintResult(feasible, "" if feasible else self.message)


def evaluate_constraints(
    design: Any, constraints: Sequence[DesignConstraint]
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate every constraint and collect rejection messages."""

    messages: list[str] = []
    feasible = True
    for constraint in constraints:
        result = constraint.evaluate(design)
        if not result.feasible:
            feasible = False
            messages.append(result.message or type(constraint).__name__)
    return feasible, tuple(messages)

