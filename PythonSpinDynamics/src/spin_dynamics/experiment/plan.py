"""Experiment planning: registry resolution plus compatibility checking.

``plan()`` is the front-loaded validation stage of the facade: it resolves
which workflow will run, surfaces errors that would make ``run()`` fail, and
warns about spec fields the resolved workflow does not honor. Later
milestones extend this with a declarative rule engine and runtime/memory
estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

from spin_dynamics.experiment.estimate import RuntimeEstimate, estimate_runtime
from spin_dynamics.experiment.hardware import UniformB0
from spin_dynamics.experiment.registry import WorkflowEntry, probes_for, resolve
from spin_dynamics.experiment.rules import RuleFinding, run_rules
from spin_dynamics.experiment.specs import (
    PROBE_NAMES,
    SEQUENCE_TYPES,
    CPMGImaging,
    CPMGIRTrain,
    CPMGTrain,
    ESRFID,
    ESRHahnEcho,
    Experiment,
    NQRSLSE,
    NQRSORC,
    PGSE,
    PGSEWalkers,
    TransportDomain2D,
    UniformFlow2D,
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
    estimate: RuntimeEstimate | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def report(self) -> str:
        lines = [f"sequence: {self.sequence}", f"probe: {self.probe}"]
        lines.append(f"workflow: {self.workflow or '(unresolved)'}")
        if self.estimate is not None:
            lines.append(f"estimate: {self.estimate.summary()}")
            lines.extend(f"  note: {note}" for note in self.estimate.notes)
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
    if sample.diffusion_coefficient is not None and sample.diffusion_coefficient < 0:
        errors.append("sample.diffusion_coefficient must be non-negative when set")
    hardware = experiment.hardware
    if hardware.q_value is not None and hardware.q_value <= 0:
        errors.append("hardware.q_value must be positive when set")
    sequence = experiment.sequence
    if (
        isinstance(sequence, (CPMGTrain, CPMGIRTrain, CPMGImaging))
        and sequence.num_echoes <= 0
    ):
        errors.append("sequence.num_echoes must be positive")
    if isinstance(sequence, CPMGIRTrain):
        if sequence.echo_spacing_seconds <= 0:
            errors.append("sequence.echo_spacing_seconds must be positive")
        if sequence.tauvect is not None and any(tau < 0 for tau in sequence.tauvect):
            errors.append("sequence.tauvect entries must be non-negative")
    if isinstance(sequence, PGSE):
        if sequence.num_echoes <= 0:
            errors.append("sequence.num_echoes must be positive")
        if sequence.gradient_duration < 0:
            errors.append("sequence.gradient_duration must be non-negative")
        if sequence.diffusion_time < sequence.gradient_duration:
            errors.append(
                "sequence.diffusion_time must be at least gradient_duration"
            )
        if (
            sequence.first_echo_time_seconds is not None
            and sequence.first_echo_time_seconds <= 0
        ):
            errors.append("sequence.first_echo_time_seconds must be positive when set")
        if (
            sequence.num_echoes > 1
            and sequence.echo_spacing_seconds is not None
            and sequence.echo_spacing_seconds <= 0
        ):
            errors.append("sequence.echo_spacing_seconds must be positive when set")
        if sequence.gamma <= 0:
            errors.append("sequence.gamma must be positive")
    if isinstance(sequence, PGSEWalkers):
        if not isinstance(sample.transport_domain, TransportDomain2D):
            errors.append(
                "sequence PGSEWalkers requires sample.transport_domain "
                "(a TransportDomain2D)"
            )
        if sample.flow is not None and not isinstance(sample.flow, UniformFlow2D):
            errors.append("sample.flow must be a UniformFlow2D when set")
        if sequence.num_echoes <= 0:
            errors.append("sequence.num_echoes must be positive")
        if sequence.gradient_duration <= 0:
            errors.append("sequence.gradient_duration must be positive")
        if sequence.diffusion_time <= (
            sequence.gradient_duration + sequence.refocusing_duration
        ):
            errors.append(
                "sequence.diffusion_time must exceed gradient_duration + "
                "refocusing_duration"
            )
        if sequence.gamma <= 0:
            errors.append("sequence.gamma must be positive")
        if sequence.gradient_axis not in ("x", "z"):
            errors.append("sequence.gradient_axis must be 'x' or 'z'")
        if sequence.walkers_per_cell <= 0:
            errors.append("sequence.walkers_per_cell must be positive")
        if sequence.seed is not None and sequence.seed < 0:
            errors.append("sequence.seed must be non-negative when set")
        if sequence.excitation_duration <= 0 or sequence.refocusing_duration <= 0:
            errors.append("sequence RF pulse durations must be positive")
        if sequence.echo_spacing_seconds is not None and sequence.echo_spacing_seconds <= 0:
            errors.append("sequence.echo_spacing_seconds must be positive when set")
        if sequence.boundary not in ("reflect", "periodic", "clip"):
            errors.append("sequence.boundary must be 'reflect', 'periodic', or 'clip'")
        if sequence.substeps_per_interval <= 0:
            errors.append("sequence.substeps_per_interval must be positive")
    if isinstance(sequence, (NQRSLSE, NQRSORC)) and sample.site is None:
        errors.append(
            f"sequence {type(sequence).__name__} requires sample.site "
            "(a spin_dynamics.nqr.QuadrupolarSite)"
        )
    if isinstance(sequence, (ESRFID, ESRHahnEcho)):
        if sample.esr_system is None:
            errors.append(
                f"sequence {type(sequence).__name__} requires sample.esr_system "
                "(a spin_dynamics.esr.ESRSpinSystem)"
            )
        b0 = experiment.hardware.b0
        if not isinstance(b0, UniformB0) or b0.field_tesla is None:
            errors.append(
                f"sequence {type(sequence).__name__} requires hardware.b0 = "
                "UniformB0(field_tesla=...) to fix the electron Larmor frequency"
            )
    if isinstance(sequence, CPMGImaging):
        if sample.phantom is None:
            errors.append("sequence CPMGImaging requires sample.phantom")
        if sequence.ny <= 0:
            errors.append("sequence.ny must be positive")
        if sequence.maxoffs <= 0:
            errors.append("sequence.maxoffs must be positive")
        if any(v <= 0 for v in sequence.fov):
            errors.append("sequence.fov values must be positive")
        if sample.phantom is not None:
            if sample.phantom.t1_map is not None and sample.t1_seconds is not None:
                errors.append(
                    "provide either sample.t1_seconds or phantom.t1_map, not both"
                )
            if sample.phantom.t2_map is not None and sample.t2_seconds is not None:
                errors.append(
                    "provide either sample.t2_seconds or phantom.t2_map, not both"
                )
    return errors


def plan_experiment(experiment: Experiment, *, estimate: bool = True) -> ExperimentPlan:
    """Resolve, validate, and (optionally) cost a declarative experiment.

    ``estimate=False`` skips the runtime/memory prediction; the first estimate
    per process triggers a sub-second calibration dry run (see
    :mod:`spin_dynamics.experiment.estimate`).
    """

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

    runtime_estimate = None
    if estimate and entry is not None and entry.cost is not None and not errors:
        runtime_estimate = estimate_runtime(entry.cost(experiment))

    workflow_name = entry.name if entry is not None else None
    if entry is not None and entry.resolve_func is not None and not errors:
        try:
            workflow_name = entry.resolve_func(experiment).__name__
        except ValueError:
            pass  # the matching rule already reports the problem

    return ExperimentPlan(
        experiment=experiment,
        workflow=workflow_name,
        probe=probe,
        sequence=sequence_name,
        errors=tuple(errors),
        warnings=tuple(warnings),
        findings=findings,
        estimate=runtime_estimate,
    )
