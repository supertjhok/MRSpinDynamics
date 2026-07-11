"""NQR engine adapter: spec -> model selection -> reduced/full dispatch.

The facade's ``NQRSLSE``/``NQRSORC`` sequence specs use the reduced engine's
nutation convention — ``nutation_hz`` is the *effective* two-level Rabi rate
of the addressed transition at full RF coupling (flip angle
``2*pi*nutation_hz*duration``), matching ``SelectivePulse``. The bare field
nutation ``gamma*B1/(2*pi)`` that ``select_nqr_model`` and the full
density-matrix engine expect is derived here as
``bare = effective / (2 * transition.strength)``; owning this conversion in
one place avoids the factor-of-2/matrix-element mistakes the two conventions
invite.

For SLSE, ``model="auto"`` dispatches to ``simulate_slse`` (reduced,
stroboscopic per-cycle model) or ``simulate_full_slse`` (full ``2I+1``
excitation + refocusing train) following ``select_nqr_model``; the two are
different fidelity models of the same experiment, not parameter-identical
re-encodings. SORC has a reduced implementation only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import numpy as np

from spin_dynamics.nqr import (
    NQRTransition,
    QuadrupolarSite,
    SelectivePulse,
    SLSESequence,
    SORCSequence,
    diagonalize_site,
    select_nqr_model,
    simulate_full_fid,
    simulate_full_slse,
    simulate_population_transfer,
    simulate_slse,
    simulate_sorc,
)
from spin_dynamics.nqr.model_selection import NQRModelSelection

REFOCUS_PHASE_SHIFT = np.pi / 2.0


def require_site(experiment: Any) -> QuadrupolarSite:
    site = experiment.sample.site
    if site is None:
        raise ValueError("NQR sequences require sample.site")
    if not isinstance(site, QuadrupolarSite):
        raise ValueError("sample.site must be a spin_dynamics.nqr.QuadrupolarSite")
    return site


B1_DIRECTION_PAS = (1.0, 0.0, 0.0)  # both engines' default drive polarization


@lru_cache(maxsize=64)
def target_transition(site: QuadrupolarSite, label: str) -> NQRTransition:
    """Resolve a transition label.

    ``"auto"`` picks the line most strongly coupled to the default x-polarized
    B1 (not the largest bare strength — a strong line can still be RF-dark for
    the drive polarization, e.g. the spin-1 ``z`` transition under B1 || x).
    """

    eigensystem = diagonalize_site(site)
    if label == "auto":
        b1 = np.asarray(B1_DIRECTION_PAS, dtype=np.float64)
        return max(
            eigensystem.transitions,
            key=lambda t: float(abs(np.vdot(b1, t.dipole_vector))),
        )
    return eigensystem.transition(label)


def bare_nutation_hz(
    site: QuadrupolarSite, label: str, effective_nutation_hz: float
) -> float:
    """Convert the effective two-level Rabi rate to bare gamma*B1/(2*pi)."""

    transition = target_transition(site, label)
    if transition.strength <= 0:
        raise ValueError(
            f"transition {transition.label!r} is RF-dark; it cannot be driven"
        )
    return float(effective_nutation_hz) / (2.0 * transition.strength)


@lru_cache(maxsize=64)
def _cached_selection(
    site: QuadrupolarSite,
    label: str,
    effective_nutation_hz: float,
    pulse_duration_seconds: float,
) -> NQRModelSelection:
    transition = target_transition(site, label)
    return select_nqr_model(
        site,
        transition.label,
        nutation_hz=bare_nutation_hz(site, label, effective_nutation_hz),
        pulse_duration_seconds=pulse_duration_seconds,
    )


def model_selection(experiment: Any) -> NQRModelSelection:
    """Run (cached) reduced-vs-full model selection for an NQR experiment."""

    site = require_site(experiment)
    sequence = experiment.sequence
    return _cached_selection(
        site,
        sequence.transition,
        sequence.nutation_hz,
        sequence.pulse_duration_seconds,
    )


def resolved_slse_model(experiment: Any) -> str:
    model = experiment.sequence.model
    if model != "auto":
        return model
    return model_selection(experiment).recommended_model


def resolve_slse_func(experiment: Any) -> Callable[..., Any]:
    return (
        simulate_slse
        if resolved_slse_model(experiment) == "reduced"
        else simulate_full_slse
    )


def _t2e(sequence: Any) -> float:
    return float("inf") if sequence.t2e_seconds is None else sequence.t2e_seconds


def slse_kwargs(experiment: Any) -> dict[str, Any]:
    site = require_site(experiment)
    sequence = experiment.sequence
    transition = target_transition(site, sequence.transition)
    if resolved_slse_model(experiment) == "reduced":
        return {
            "site": site,
            "sequence": SLSESequence(
                detection=SelectivePulse(
                    transition_label=transition.label,
                    duration_seconds=sequence.pulse_duration_seconds,
                    nutation_hz=sequence.nutation_hz,
                    phase=sequence.phase,
                    rf_frequency_hz=sequence.rf_frequency_hz,
                ),
                echo_spacing_seconds=sequence.echo_spacing_seconds,
                num_echoes=sequence.num_echoes,
            ),
            "orientations": sequence.orientations,
            "b0_tesla": sequence.b0_tesla,
            "t2e_seconds": _t2e(sequence),
        }
    carrier = (
        sequence.rf_frequency_hz
        if sequence.rf_frequency_hz is not None
        else transition.frequency_hz
    )
    return {
        "site": site,
        "nutation_hz": bare_nutation_hz(site, sequence.transition, sequence.nutation_hz),
        "excitation_duration_seconds": sequence.pulse_duration_seconds,
        "refocus_duration_seconds": sequence.pulse_duration_seconds,
        "echo_spacing_seconds": sequence.echo_spacing_seconds,
        "num_echoes": sequence.num_echoes,
        "rf_frequency_hz": carrier,
        "excitation_phase": sequence.phase,
        "refocus_phase": sequence.phase + REFOCUS_PHASE_SHIFT,
        "orientations": sequence.orientations,
        "b0_tesla": sequence.b0_tesla,
        "t2e_seconds": _t2e(sequence),
    }


def sorc_kwargs(experiment: Any) -> dict[str, Any]:
    site = require_site(experiment)
    sequence = experiment.sequence
    transition = target_transition(site, sequence.transition)
    return {
        "site": site,
        "sequence": SORCSequence(
            detection=SelectivePulse(
                transition_label=transition.label,
                duration_seconds=sequence.pulse_duration_seconds,
                nutation_hz=sequence.nutation_hz,
                phase=sequence.phase,
                rf_frequency_hz=sequence.rf_frequency_hz,
            ),
            half_spacing_seconds=sequence.half_spacing_seconds,
            num_pulses=sequence.num_pulses,
        ),
        "orientations": sequence.orientations,
        "b0_tesla": sequence.b0_tesla,
        "t2e_seconds": _t2e(sequence),
    }


def fid_kwargs(experiment: Any) -> dict[str, Any]:
    site = require_site(experiment)
    sequence = experiment.sequence
    kwargs: dict[str, Any] = {
        "site": site,
        "nutation_hz": sequence.nutation_hz,
        "pulse_duration_seconds": sequence.pulse_duration_seconds,
        "times_seconds": np.linspace(
            0.0, sequence.acquisition_seconds, sequence.num_points
        ),
        "phase": sequence.phase,
    }
    if sequence.rf_frequency_hz is not None:
        kwargs["rf_frequency_hz"] = sequence.rf_frequency_hz
    if sequence.b0_tesla != 0.0:
        kwargs["b0_vector_tesla_pas"] = (0.0, 0.0, sequence.b0_tesla)
    return kwargs


def population_transfer_kwargs(experiment: Any) -> dict[str, Any]:
    site = require_site(experiment)
    sequence = experiment.sequence
    perturbation = target_transition(site, sequence.perturbation_transition)
    detection = target_transition(site, sequence.detection_transition)
    return {
        "site": site,
        "perturbation": SelectivePulse(
            transition_label=perturbation.label,
            duration_seconds=sequence.perturbation_duration_seconds,
            nutation_hz=sequence.perturbation_nutation_hz,
            phase=sequence.perturbation_phase,
            rf_frequency_hz=sequence.perturbation_frequency_hz,
        ),
        "detection_sequence": SLSESequence(
            detection=SelectivePulse(
                transition_label=detection.label,
                duration_seconds=sequence.detection_duration_seconds,
                nutation_hz=sequence.detection_nutation_hz,
                phase=sequence.detection_phase,
                rf_frequency_hz=sequence.detection_frequency_hz,
            ),
            echo_spacing_seconds=sequence.echo_spacing_seconds,
            num_echoes=sequence.num_echoes,
        ),
        "orientations": sequence.orientations,
        "b0_tesla": sequence.b0_tesla,
        "t2e_seconds": _t2e(sequence),
    }


__all__ = [
    "bare_nutation_hz",
    "fid_kwargs",
    "model_selection",
    "population_transfer_kwargs",
    "require_site",
    "resolve_slse_func",
    "resolved_slse_model",
    "slse_kwargs",
    "sorc_kwargs",
    "simulate_slse",
    "simulate_sorc",
    "simulate_full_slse",
    "simulate_full_fid",
    "simulate_population_transfer",
    "target_transition",
]
