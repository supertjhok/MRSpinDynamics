"""Registry mapping (sequence spec type, probe) to validated workflows.

Entries delegate to existing ``spin_dynamics.workflows`` functions; the
facade never re-implements dynamics. ``honors`` lists the dotted spec fields
the wrapped workflow consumes so ``plan()`` can warn about spec fields that
would be silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from spin_dynamics.experiment.specs import PROBE_NAMES, Experiment


@dataclass(frozen=True)
class WorkflowEntry:
    """One registered workflow behind the facade."""

    name: str
    sequence_type: type
    probe: str
    func: Callable[..., Any]
    build_kwargs: Callable[[Experiment], dict[str, Any]]
    honors: frozenset[str]
    execution_kwargs: frozenset[str] = field(default_factory=frozenset)
    max_time: Callable[[Experiment], float] | None = None
    """Normalized total evolution time estimator, for the rephasing pre-check.

    ``None`` for workflows without a finite acquisition time (e.g. asymptotic
    CPMG), which the rephasing rule then skips.
    """


_REGISTRY: dict[tuple[type, str], WorkflowEntry] = {}


def register_workflow(entry: WorkflowEntry) -> WorkflowEntry:
    if entry.probe not in PROBE_NAMES:
        raise ValueError(f"unknown probe {entry.probe!r}; expected one of {PROBE_NAMES}")
    key = (entry.sequence_type, entry.probe)
    if key in _REGISTRY:
        raise ValueError(
            f"a workflow for ({entry.sequence_type.__name__}, {entry.probe!r}) "
            "is already registered"
        )
    _REGISTRY[key] = entry
    return entry


def resolve(sequence_type: type, probe: str) -> WorkflowEntry | None:
    return _REGISTRY.get((sequence_type, probe))


def available_workflows() -> tuple[WorkflowEntry, ...]:
    return tuple(_REGISTRY.values())


def probes_for(sequence_type: type) -> tuple[str, ...]:
    return tuple(
        entry.probe for (seq, _), entry in _REGISTRY.items() if seq is sequence_type
    )
