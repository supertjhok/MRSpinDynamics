"""Experiment planning: registry resolution plus compatibility checking.

``plan()`` is the front-loaded validation stage of the facade: it resolves
which workflow will run, surfaces errors that would make ``run()`` fail, and
warns about spec fields the resolved workflow does not honor. Later
milestones extend this with a declarative rule engine and runtime/memory
estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

from spin_dynamics.experiment.registry import WorkflowEntry, probes_for, resolve
from spin_dynamics.experiment.rules import RuleFinding, run_rules
from spin_dynamics.experiment.specs import (
    PROBE_NAMES,
    SEQUENCE_TYPES,
    CPMGIRTrain,
    CPMGTrain,
    Experiment,
    non_default_fields,
)


@dataclass(frozen=True)
class ExperimentPlan:
    """Resolved execution plan for an :class:`Experiment`."""

    experiment: Experiment
    workflow: str | None
    probe: str
    sequence: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    findings: tuple[RuleFinding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def report(self) -> str:
        lines = [f"sequence: {self.sequence}", f"probe: {self.probe}"]
        lines.append(f"workflow: {self.workflow or '(unresolved)'}")
        if self.errors:
            lines.append("errors:")
            lines.extend(f"  - {msg}" for msg in self.errors)
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {msg}" for msg in self.warnings)
        if not self.errors and not self.warnings:
            lines.append("checks: ok")
        return "\n".join(lines)


def _spec_sanity_errors(experiment: Experiment) -> list[str]:
    errors: list[str] = []
    acquisition = experiment.acquisition
    if acquisition.numpts <= 0:
        errors.append("acquisition.numpts must be positive")
    if acquisition.maxoffs <= 0:
        errors.append("acquisition.maxoffs must be positive")
    if acquisition.rephase_action not in ("ignore", "warn", "raise"):
        errors.append("acquisition.rephase_action must be 'ignore', 'warn', or 'raise'")
    sample = experiment.sample
    for name in ("t1_seconds", "t2_seconds"):
        value = getattr(sample, name)
        if value is not None and value <= 0:
            errors.append(f"sample.{name} must be positive when set")
    hardware = experiment.hardware
    if hardware.q_value is not None and hardware.q_value <= 0:
        errors.append("hardware.q_value must be positive when set")
    sequence = experiment.sequence
    if isinstance(sequence, (CPMGTrain, CPMGIRTrain)) and sequence.num_echoes <= 0:
        errors.append("sequence.num_echoes must be positive")
    if isinstance(sequence, CPMGIRTrain):
        if sequence.echo_spacing_seconds <= 0:
            errors.append("sequence.echo_spacing_seconds must be positive")
        if sequence.tauvect is not None and any(tau < 0 for tau in sequence.tauvect):
            errors.append("sequence.tauvect entries must be non-negative")
    return errors


def plan_experiment(experiment: Experiment) -> ExperimentPlan:
    errors: list[str] = []
    warnings: list[str] = []

    sequence_name = type(experiment.sequence).__name__
    probe = experiment.hardware.probe
    entry: WorkflowEntry | None = None

    if not isinstance(experiment.sequence, SEQUENCE_TYPES):
        known = ", ".join(cls.__name__ for cls in SEQUENCE_TYPES)
        errors.append(
            f"unknown sequence spec {sequence_name}; known sequences: {known}"
        )
    elif probe not in PROBE_NAMES:
        errors.append(f"unknown probe {probe!r}; expected one of {PROBE_NAMES}")
    else:
        entry = resolve(type(experiment.sequence), probe)
        if entry is None:
            supported = probes_for(type(experiment.sequence)) or ("none",)
            errors.append(
                f"no workflow registered for ({sequence_name}, {probe!r}); "
                f"probes supported for {sequence_name}: {', '.join(supported)}"
            )

    errors.extend(_spec_sanity_errors(experiment))

    findings: tuple[RuleFinding, ...] = ()
    if entry is not None:
        for dotted in sorted(non_default_fields(experiment)):
            if dotted == "hardware.probe" or dotted in entry.honors:
                continue
            if dotted == "sample.label":
                continue
            warnings.append(
                f"{dotted} is not honored by {entry.name} and will be ignored"
            )

        # Rules assume a resolved workflow and sane spec shape; skip them when
        # basic validation already failed so estimators never see bad inputs.
        if not errors:
            findings = tuple(run_rules(experiment, entry))
            for finding in findings:
                if finding.severity == "error":
                    errors.append(f"[{finding.rule}] {finding.message}")
                elif finding.severity == "warn":
                    warnings.append(f"[{finding.rule}] {finding.message}")

    return ExperimentPlan(
        experiment=experiment,
        workflow=entry.name if entry is not None else None,
        probe=probe,
        sequence=sequence_name,
        errors=tuple(errors),
        warnings=tuple(warnings),
        findings=findings,
    )
