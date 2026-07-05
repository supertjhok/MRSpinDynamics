"""ESR engine adapter: pulsed FID and Hahn-echo dispatch.

Wraps the pulsed core of ``spin_dynamics.esr`` (``simulate_fid``,
``simulate_hahn_echo``) behind the facade specs. The electron spin system
comes from ``Sample.esr_system`` and the static field from ``Hardware.b0``
(a :class:`~spin_dynamics.experiment.hardware.UniformB0` with ``field_tesla``
set — ESR needs the absolute field to fix the Larmor frequency, unlike the
rotating-frame NMR/NQR workflows).

``nutation_hz`` passes through in the engine's own convention (the electron
Rabi rate; spin-1/2 has a single transition so there is no matrix-element
bridge like the NQR adapter's). The specialized ESR experiment families
(DEER, ESEEM, HYSCORE, ENDOR) are analysis-style modules with their own
parameter surfaces and are not wrapped here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spin_dynamics.esr import ESRSpinSystem, simulate_fid, simulate_hahn_echo
from spin_dynamics.experiment.hardware import UniformB0


def require_system(experiment: Any) -> ESRSpinSystem:
    system = experiment.sample.esr_system
    if system is None:
        raise ValueError("ESR sequences require sample.esr_system")
    if not isinstance(system, ESRSpinSystem):
        raise ValueError("sample.esr_system must be a spin_dynamics.esr.ESRSpinSystem")
    return system


def require_b0_vector(experiment: Any) -> tuple[float, float, float]:
    b0 = experiment.hardware.b0
    if b0 is None or not isinstance(b0, UniformB0) or b0.field_tesla is None:
        raise ValueError(
            "ESR sequences require hardware.b0 = UniformB0(field_tesla=...) "
            "to fix the electron Larmor frequency"
        )
    return b0.vector_tesla()


def fid_kwargs(experiment: Any) -> dict[str, Any]:
    system = require_system(experiment)
    sequence = experiment.sequence
    kwargs: dict[str, Any] = {
        "system": system,
        "b0_vector_tesla_g": require_b0_vector(experiment),
        "nutation_hz": sequence.nutation_hz,
        "pulse_duration_seconds": sequence.pulse_duration_seconds,
        "times_seconds": np.linspace(
            0.0, sequence.acquisition_seconds, sequence.num_points
        ),
        "phase": sequence.phase,
    }
    if sequence.rf_frequency_hz is not None:
        kwargs["rf_frequency_hz"] = sequence.rf_frequency_hz
    if sequence.t2_seconds is not None:
        kwargs["t2_seconds"] = sequence.t2_seconds
    return kwargs


def hahn_kwargs(experiment: Any) -> dict[str, Any]:
    system = require_system(experiment)
    sequence = experiment.sequence
    refocus = (
        sequence.refocus_duration_seconds
        if sequence.refocus_duration_seconds is not None
        else 2.0 * sequence.excitation_duration_seconds
    )
    window = (
        sequence.acquisition_seconds
        if sequence.acquisition_seconds is not None
        else 2.0 * sequence.tau_seconds
    )
    kwargs: dict[str, Any] = {
        "system": system,
        "b0_vector_tesla_g": require_b0_vector(experiment),
        "nutation_hz": sequence.nutation_hz,
        "excitation_duration_seconds": sequence.excitation_duration_seconds,
        "refocus_duration_seconds": refocus,
        "tau_seconds": sequence.tau_seconds,
        "times_seconds": np.linspace(0.0, window, sequence.num_points),
        "excitation_phase": sequence.excitation_phase,
    }
    if sequence.refocus_phase is not None:
        kwargs["refocus_phase"] = sequence.refocus_phase
    if sequence.rf_frequency_hz is not None:
        kwargs["rf_frequency_hz"] = sequence.rf_frequency_hz
    if sequence.t2_seconds is not None:
        kwargs["t2_seconds"] = sequence.t2_seconds
    return kwargs


__all__ = [
    "fid_kwargs",
    "hahn_kwargs",
    "require_b0_vector",
    "require_system",
    "simulate_fid",
    "simulate_hahn_echo",
]
