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


def hardware_wiring_rule(
    experiment: Experiment, entry: WorkflowEntry
) -> list[RuleFinding]:
    """Solve requested coil fields at plan time and surface wiring problems.

    The solve is cached, so ``run()`` reuses the plan-time result; a spec that
    cannot be wired (bad geometry types, a coil axis parallel to B0 leaving no
    transverse B1) becomes a plan error instead of a mid-run exception.
    """

    from spin_dynamics.experiment import wiring
    from spin_dynamics.experiment.hardware import RxArray
    from spin_dynamics.experiment.specs import CPMGImaging

    if not isinstance(experiment.sequence, CPMGImaging):
        # Unused coil specs on other sequences are reported by the generic
        # ignored-field warnings via each entry's `honors` set.
        return []
    is_receiver_array = isinstance(experiment.hardware.rx_coil, RxArray)
    if is_receiver_array and experiment.hardware.probe != "ideal":
        return [
            RuleFinding(
                rule="hardware_wiring",
                severity="error",
                message=(
                    "RxArray CPMG imaging currently requires probe='ideal'; "
                    "per-channel tuned/matched transfer functions are deferred "
                    "to the multiport coupling phase"
                ),
            )
        ]
    if is_receiver_array and experiment.acquisition.noise is not None:
        return [
            RuleFinding(
                rule="hardware_wiring",
                severity="error",
                message=(
                    "RxArray imaging uses acquisition.receiver_noise_std or "
                    "acquisition.receiver_noise_covariance instead of the "
                    "single-channel acquisition.noise specification"
                ),
            )
        ]
    has_array_noise = (
        experiment.acquisition.receiver_noise_std > 0.0
        or experiment.acquisition.receiver_noise_covariance is not None
        or experiment.acquisition.receiver_noise_seed is not None
    )
    if not is_receiver_array and has_array_noise:
        return [
            RuleFinding(
                rule="hardware_wiring",
                severity="error",
                message=(
                    "acquisition receiver-array noise fields require an "
                    "RxArray imaging receiver"
                ),
            )
        ]
    if not wiring.uses_hardware_fields(experiment.hardware):
        return []
    try:
        wiring.solve_for_experiment(experiment)
        diagnostics = wiring.solve_diagnostics(
            experiment.sample.phantom, experiment.hardware
        )
    except ValueError as exc:
        return [
            RuleFinding(rule="hardware_wiring", severity="error", message=str(exc))
        ]

    findings = [
        RuleFinding(
            rule="hardware_wiring",
            severity="ok",
            message="coil fields solved onto the phantom grid (cached)",
            details=dict(diagnostics),
        )
    ]
    for key, which in (
        ("tx_transverse_fraction", "transmit"),
        ("rx_transverse_fraction", "receive"),
    ):
        fraction = diagnostics.get(key)
        if fraction is not None and fraction < 0.5:
            findings.append(
                RuleFinding(
                    rule="hardware_wiring",
                    severity="warn",
                    message=(
                        f"only {fraction:.0%} of the {which} coil's B1 over the "
                        "sample is transverse to B0; the normalized maps hide "
                        "this inefficiency (check coil orientation)"
                    ),
                    details={key: fraction},
                )
            )
    return findings


def nqr_model_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]:
    """Run ``select_nqr_model`` at plan time and check the engine dispatch.

    Reports the recommended model with its reasons; warns when the user
    forces a model against the recommendation (or uses SORC, which has no
    full-model implementation, on a site that needs one); errors when the
    dispatch would hit an engine constraint (reduced selective-pulse
    workflows support spin-1 only).
    """

    from spin_dynamics.experiment import nqr_adapter
    from spin_dynamics.experiment.specs import NQRSLSE, NQRSORC

    sequence = experiment.sequence
    if not isinstance(sequence, (NQRSLSE, NQRSORC)):
        return []
    try:
        selection = nqr_adapter.model_selection(experiment)
    except ValueError as exc:
        return [RuleFinding(rule="nqr_model", severity="error", message=str(exc))]

    findings = [
        RuleFinding(
            rule="nqr_model",
            severity="ok",
            message=(
                f"model selection recommends the {selection.recommended_model} "
                f"engine for transition {selection.target_label!r} "
                f"(isolation ratio {selection.isolation_ratio:.2g})"
            ),
            details={
                "recommended_model": selection.recommended_model,
                "target_label": selection.target_label,
                "isolation_ratio": selection.isolation_ratio,
                "reasons": list(selection.reasons),
            },
        )
    ]

    site = experiment.sample.site
    if isinstance(sequence, NQRSORC):
        resolved = "reduced"  # SORC has no full-model implementation
        if selection.recommended_model == "full":
            findings.append(
                RuleFinding(
                    rule="nqr_model",
                    severity="warn",
                    message=(
                        "model selection recommends the full engine but SORC "
                        "only has a reduced implementation: "
                        + "; ".join(selection.reasons)
                    ),
                )
            )
    else:
        resolved = nqr_adapter.resolved_slse_model(experiment)
        if sequence.model != "auto" and sequence.model != selection.recommended_model:
            findings.append(
                RuleFinding(
                    rule="nqr_model",
                    severity="warn",
                    message=(
                        f"model={sequence.model!r} overrides the "
                        f"{selection.recommended_model!r} recommendation: "
                        + "; ".join(selection.reasons)
                    ),
                )
            )

    if resolved == "reduced" and site is not None and float(site.spin) != 1.0:
        findings.append(
            RuleFinding(
                rule="nqr_model",
                severity="error",
                message=(
                    "the reduced selective-pulse engine supports spin-1 only; "
                    f"this site has spin {site.spin}. Use model='full' "
                    "(SLSE) or the full-dynamics workflows directly"
                ),
            )
        )
    return findings


def spectroscopy_inputs_rule(
    experiment: Experiment, entry: WorkflowEntry
) -> list[RuleFinding]:
    """Resolve spectroscopy sample objects and transition labels at plan time."""

    from spin_dynamics.experiment import (
        esr_adapter,
        esr_multidim_adapter,
        nqr_adapter,
    )
    from spin_dynamics.experiment.specs import (
        ESRCWSweep,
        ESRDaviesENDOR,
        ESRFID,
        ESRHYSCORE,
        ESRHahnEcho,
        ESRMimsENDOR,
        ESRThreePulseESEEM,
        ESRTwoPulseESEEM,
        NQRFID,
        NQRPopulationTransfer,
    )

    sequence = experiment.sequence
    try:
        if isinstance(sequence, NQRFID):
            nqr_adapter.require_site(experiment)
        elif isinstance(sequence, NQRPopulationTransfer):
            site = nqr_adapter.require_site(experiment)
            nqr_adapter.target_transition(site, sequence.perturbation_transition)
            nqr_adapter.target_transition(site, sequence.detection_transition)
        elif isinstance(sequence, (ESRFID, ESRHahnEcho, ESRCWSweep)):
            esr_adapter.require_system(experiment)
        elif isinstance(
            sequence,
            (
                ESRTwoPulseESEEM,
                ESRThreePulseESEEM,
                ESRHYSCORE,
                ESRDaviesENDOR,
                ESRMimsENDOR,
            ),
        ):
            coupling = esr_multidim_adapter.require_coupling(experiment)
            if isinstance(sequence, (ESRTwoPulseESEEM, ESRThreePulseESEEM)):
                offset = (
                    sequence.electron_offset_hz
                    if isinstance(sequence, ESRTwoPulseESEEM)
                    else 0.0
                )
                esr_multidim_adapter.resolved_eseem_model(
                    sequence.model, coupling, electron_offset_hz=offset
                )
        else:
            return []
    except ValueError as exc:
        return [
            RuleFinding(
                rule="spectroscopy_inputs", severity="error", message=str(exc)
            )
        ]
    return []


def transport_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]:
    """Report uniform-flow scale and flag closed reflecting transport."""

    from spin_dynamics.experiment.specs import PGSEWalkers

    if not isinstance(experiment.sequence, PGSEWalkers):
        return []
    flow = experiment.sample.flow
    domain = experiment.sample.transport_domain
    if flow is None or domain is None:
        return []
    velocity = flow.as_array()
    speed = float(np.linalg.norm(velocity))
    widths = np.array(
        [domain.x_axis[-1] - domain.x_axis[0], domain.z_axis[-1] - domain.z_axis[0]]
    )
    crossing_times = np.divide(
        widths,
        np.abs(velocity),
        out=np.full(2, np.inf),
        where=np.abs(velocity) > 0.0,
    )
    details = {
        "velocity_m_per_s": velocity.tolist(),
        "speed_m_per_s": speed,
        "axis_crossing_times_seconds": crossing_times.tolist(),
        "boundary": experiment.sequence.boundary,
    }
    if speed > 0.0 and experiment.sequence.boundary == "reflect":
        return [
            RuleFinding(
                rule="transport",
                severity="warn",
                message=(
                    "nonzero uniform flow with reflecting boundaries models a "
                    "closed bouncing ensemble, not through-flow; use periodic "
                    "boundaries for translational bulk flow"
                ),
                details=details,
            )
        ]
    return [
        RuleFinding(
            rule="transport",
            severity="ok",
            message=f"uniform-flow transport speed is {speed:.3g} m/s",
            details=details,
        )
    ]


def sequence_ir_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]:
    """Compile general IR at plan time and reject unsupported backend policy."""

    from spin_dynamics.experiment import sequence_adapter
    from spin_dynamics.experiment.specs import SequenceIRExecution

    if not isinstance(experiment.sequence, SequenceIRExecution):
        return []
    try:
        compiled, _steps = sequence_adapter.prepare_for_experiment(experiment)
    except (TypeError, ValueError, NotImplementedError) as exc:
        return [
            RuleFinding(rule="sequence_ir", severity="error", message=str(exc))
        ]
    return [
        RuleFinding(
            rule="sequence_ir",
            severity="ok",
            message=(
                f"compiled {compiled.durations_seconds.size} intervals and "
                f"{compiled.adc.times_seconds.size} ADC samples for a "
                f"{len(experiment.sample.sequence_domain.axes)}-D motion backend"
            ),
            details={
                "intervals": int(compiled.durations_seconds.size),
                "adc_samples": int(compiled.adc.times_seconds.size),
                "particles": int(
                    np.count_nonzero(experiment.sample.sequence_domain.density > 0.0)
                    * experiment.sequence.walkers_per_cell
                ),
                "source_format": compiled.source_format,
            },
        )
    ]


DEFAULT_RULES: tuple[Rule, ...] = (
    rephasing_rule,
    noise_spec_rule,
    hardware_wiring_rule,
    nqr_model_rule,
    spectroscopy_inputs_rule,
    transport_rule,
    sequence_ir_rule,
)


def run_rules(
    experiment: Experiment,
    entry: WorkflowEntry,
    rules: Iterable[Rule] = DEFAULT_RULES,
) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    for rule in rules:
        findings.extend(rule(experiment, entry))
    return findings
