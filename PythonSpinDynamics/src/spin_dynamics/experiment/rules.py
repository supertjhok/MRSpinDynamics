"""Declarative compatibility rules run by ``Experiment.plan()``.

Each rule is a callable ``(Experiment, WorkflowEntry) -> Iterable[RuleFinding]``
that inspects a resolved experiment and reports zero or more findings with an
``ok``/``warn``/``error`` severity. This generalizes the guards that were
previously scattered across the workflows (rephasing grid adequacy, noise-spec
validity) into a single front-loaded pass, so users see problems before
``run()`` executes anything.

Rules run only once a workflow has been resolved; spec-shape errors (unknown
probe, bad ``numpts``) are handled earlier in :mod:`spin_dynamics.experiment.plan`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from spin_dynamics.core.isochromats import analyze_rephasing
from spin_dynamics.experiment.registry import WorkflowEntry
from spin_dynamics.experiment.specs import Acquisition, Experiment
from spin_dynamics.noise import as_noise_spec

SEVERITIES = ("ok", "warn", "error")


@dataclass(frozen=True)
class RuleFinding:
    """One outcome from a single rule."""

    rule: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")


Rule = Callable[[Experiment, WorkflowEntry], Iterable[RuleFinding]]


def _offset_grid(acquisition: Acquisition) -> np.ndarray:
    return np.linspace(
        -float(acquisition.maxoffs),
        float(acquisition.maxoffs),
        int(acquisition.numpts),
    )


def rephasing_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]:
    """Front-load the isochromat-grid rephasing check.

    Uses the workflow's own ``max_time`` estimator so the plan-time verdict
    matches the run-time ``check_rephasing`` call exactly. Respects
    ``acquisition.rephase_action`` (``ignore`` skips, ``raise`` escalates to an
    error, ``warn`` reports a warning). ``auto_refine_grid=True`` suppresses the
    finding because the workflow will grow the grid itself.
    """

    estimator = entry.max_time
    if estimator is None:
        return []
    acquisition = experiment.acquisition
    action = acquisition.rephase_action
    if action == "ignore":
        return []

    max_time = estimator(experiment)
    analysis = analyze_rephasing(
        _offset_grid(acquisition), max_time, acquisition.rephase_safety_factor
    )
    if analysis.ok:
        return []

    recommended = analysis.recommended_numpts
    if acquisition.auto_refine_grid and recommended is not None:
        return [
            RuleFinding(
                rule="rephasing",
                severity="ok",
                message=(
                    f"grid will auto-refine from numpts={acquisition.numpts} to "
                    f"at least numpts={recommended} to avoid rephasing"
                ),
                details={"recommended_numpts": recommended, "max_time": max_time},
            )
        ]

    message = (
        f"isochromat grid may rephase before the sequence finishes "
        f"(spacing rephases at t={analysis.rephase_time:.4g}, sequence runs to "
        f"t={max_time:.4g}); use at least numpts={recommended} over "
        f"maxoffs={acquisition.maxoffs:g}, enable acquisition.auto_refine_grid, "
        f"or set acquisition.rephase_action='ignore' to silence"
    )
    severity = "error" if action == "raise" else "warn"
    return [
        RuleFinding(
            rule="rephasing",
            severity=severity,
            message=message,
            details={
                "recommended_numpts": recommended,
                "max_time": max_time,
                "rephase_time": analysis.rephase_time,
            },
        )
    ]


def noise_spec_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]:
    """Validate the acquisition noise spec at plan time.

    Catches malformed noise inputs and the unsupported time-domain/non-white
    combination before ``run()`` reaches the workflow.
    """

    noise = experiment.acquisition.noise
    if noise is None:
        return []
    try:
        spec = as_noise_spec(noise)
    except (TypeError, ValueError) as exc:
        return [
            RuleFinding(
                rule="noise_spec",
                severity="error",
                message=f"invalid acquisition.noise: {exc}",
            )
        ]
    if spec is not None and spec.domain == "time" and spec.model != "white":
        return [
            RuleFinding(
                rule="noise_spec",
                severity="error",
                message="time-domain noise currently supports only the white model",
            )
        ]
    return []


DEFAULT_RULES: tuple[Rule, ...] = (rephasing_rule, noise_spec_rule)


def run_rules(
    experiment: Experiment,
    entry: WorkflowEntry,
    rules: Iterable[Rule] = DEFAULT_RULES,
) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    for rule in rules:
        findings.extend(rule(experiment, entry))
    return findings
