"""High-level NQR simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.orientations import (
    OrientationSample,
    normalize_orientations,
    powder_average_grid,
)
from spin_dynamics.nqr.pulses import SelectivePulse, apply_selective_pulse
from spin_dynamics.nqr.sequences import SLSESequence
from spin_dynamics.nqr.systems import NQREigensystem, NQRTransition, QuadrupolarSite


@dataclass(frozen=True)
class SLSEResult:
    """Simulated SLSE echo train."""

    echo_times: np.ndarray
    echo_amplitudes: np.ndarray
    local_echo_amplitudes: np.ndarray
    orientation_weights: np.ndarray
    transition: NQRTransition
    eigensystem: NQREigensystem


@dataclass(frozen=True)
class PopulationTransferResult:
    """Perturbation plus SLSE detection result."""

    signal: SLSEResult
    reference: SLSEResult
    normalized_difference: np.ndarray
    perturbation: SelectivePulse


def equilibrium_density(levels_hz: np.ndarray) -> np.ndarray:
    """Return a trace-zero high-temperature density matrix in the energy basis."""

    levels = np.asarray(levels_hz, dtype=np.float64).reshape(-1)
    populations = -(levels - np.mean(levels))
    scale = float(np.max(np.abs(populations))) if populations.size else 0.0
    if scale > 0:
        populations = populations / scale
    return np.diag(populations.astype(np.complex128))


def transition_signal(
    density: np.ndarray,
    transition: NQRTransition,
    *,
    b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float],
) -> complex:
    """Return the complex single-coil signal for a transition coherence."""

    direction = np.asarray(b1_direction_pas, dtype=np.float64).reshape(3)
    direction = direction / np.linalg.norm(direction)
    receive = np.vdot(direction, transition.dipole_vector)
    coherence = density[transition.upper, transition.lower]
    return complex(receive * coherence)


def _orientation_b0_vector(
    orientation: OrientationSample,
    b0_tesla: float,
) -> np.ndarray | None:
    if b0_tesla == 0:
        return None
    direction = orientation.b0_direction_pas
    if direction is None:
        direction = orientation.b1_direction_pas
    return float(b0_tesla) * direction


def _as_orientations(
    orientations: str | tuple[OrientationSample, ...] | list[OrientationSample],
) -> tuple[OrientationSample, ...]:
    if isinstance(orientations, str):
        if orientations == "powder":
            return powder_average_grid()
        if orientations == "single":
            return normalize_orientations(
                [OrientationSample(b1_direction_pas=(1.0, 0.0, 0.0))]
            )
        raise ValueError("orientations string must be 'powder' or 'single'")
    return normalize_orientations(tuple(orientations))


def simulate_slse(
    site: QuadrupolarSite,
    sequence: SLSESequence,
    *,
    orientations: str | tuple[OrientationSample, ...] | list[OrientationSample] = "powder",
    b0_tesla: float = 0.0,
    t2e_seconds: float = np.inf,
    initial_density: np.ndarray | None = None,
) -> SLSEResult:
    """Simulate a selective-pulse SLSE echo train."""

    samples = _as_orientations(orientations)
    t2e_seconds = float(t2e_seconds)
    if t2e_seconds <= 0:
        raise ValueError("t2e_seconds must be positive")
    local: list[np.ndarray] = []
    first_eigensystem: NQREigensystem | None = None
    first_transition: NQRTransition | None = None

    echo_times = (
        np.arange(sequence.num_echoes, dtype=np.float64) + 1.0
    ) * sequence.echo_spacing_seconds
    decay = np.exp(-echo_times / t2e_seconds) if np.isfinite(t2e_seconds) else 1.0

    for sample in samples:
        eigensystem = diagonalize_site(site, _orientation_b0_vector(sample, b0_tesla))
        transition = eigensystem.transition(sequence.detection.transition_label)
        density = (
            equilibrium_density(eigensystem.levels_hz)
            if initial_density is None
            else np.asarray(initial_density, dtype=np.complex128).copy()
        )
        echoes = np.zeros(sequence.num_echoes, dtype=np.complex128)
        for echo_idx in range(sequence.num_echoes):
            density = apply_selective_pulse(
                density,
                transition,
                sequence.detection,
                b1_direction_pas=sample.b1_direction_pas,
            )
            echoes[echo_idx] = transition_signal(
                density,
                transition,
                b1_direction_pas=sample.b1_direction_pas,
            )
        echoes = echoes * decay
        local.append(echoes)
        if first_eigensystem is None:
            first_eigensystem = eigensystem
            first_transition = transition

    weights = np.array([sample.weight for sample in samples], dtype=np.float64)
    local_echoes = np.asarray(local, dtype=np.complex128)
    averaged = weights @ local_echoes
    if first_eigensystem is None or first_transition is None:
        raise AssertionError("orientation validation should prevent empty samples")
    return SLSEResult(
        echo_times=echo_times,
        echo_amplitudes=averaged,
        local_echo_amplitudes=local_echoes,
        orientation_weights=weights,
        transition=first_transition,
        eigensystem=first_eigensystem,
    )


def simulate_population_transfer(
    site: QuadrupolarSite,
    perturbation: SelectivePulse,
    detection_sequence: SLSESequence,
    *,
    orientations: str | tuple[OrientationSample, ...] | list[OrientationSample] = "powder",
    b0_tesla: float = 0.0,
    t2e_seconds: float = np.inf,
) -> PopulationTransferResult:
    """Simulate a perturbation pulse followed by SLSE detection."""

    samples = _as_orientations(orientations)
    perturbed_local_density: list[np.ndarray] = []
    for sample in samples:
        eigensystem = diagonalize_site(site, _orientation_b0_vector(sample, b0_tesla))
        density = equilibrium_density(eigensystem.levels_hz)
        density = apply_selective_pulse(
            density,
            eigensystem.transition(perturbation.transition_label),
            perturbation,
            b1_direction_pas=sample.b1_direction_pas,
        )
        perturbed_local_density.append(density)

    signal_local: list[np.ndarray] = []
    reference_local: list[np.ndarray] = []
    echo_times: np.ndarray | None = None
    first_eigensystem: NQREigensystem | None = None
    first_transition: NQRTransition | None = None
    for sample, density0 in zip(samples, perturbed_local_density):
        signal = simulate_slse(
            site,
            detection_sequence,
            orientations=[sample],
            b0_tesla=b0_tesla,
            t2e_seconds=t2e_seconds,
            initial_density=density0,
        )
        reference = simulate_slse(
            site,
            detection_sequence,
            orientations=[sample],
            b0_tesla=b0_tesla,
            t2e_seconds=t2e_seconds,
        )
        signal_local.append(signal.echo_amplitudes)
        reference_local.append(reference.echo_amplitudes)
        echo_times = signal.echo_times
        if first_eigensystem is None:
            first_eigensystem = signal.eigensystem
            first_transition = signal.transition

    weights = np.array([sample.weight for sample in samples], dtype=np.float64)
    signal_local_arr = np.asarray(signal_local, dtype=np.complex128)
    reference_local_arr = np.asarray(reference_local, dtype=np.complex128)
    signal_avg = weights @ signal_local_arr
    reference_avg = weights @ reference_local_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = signal_avg / reference_avg - 1.0
    if echo_times is None or first_eigensystem is None or first_transition is None:
        raise AssertionError("orientation validation should prevent empty samples")

    signal_result = SLSEResult(
        echo_times=echo_times,
        echo_amplitudes=signal_avg,
        local_echo_amplitudes=signal_local_arr,
        orientation_weights=weights,
        transition=first_transition,
        eigensystem=first_eigensystem,
    )
    reference_result = SLSEResult(
        echo_times=echo_times,
        echo_amplitudes=reference_avg,
        local_echo_amplitudes=reference_local_arr,
        orientation_weights=weights,
        transition=first_transition,
        eigensystem=first_eigensystem,
    )
    return PopulationTransferResult(
        signal=signal_result,
        reference=reference_result,
        normalized_difference=normalized,
        perturbation=perturbation,
    )
