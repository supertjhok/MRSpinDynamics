"""High-level NQR simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from spin_dynamics.coupling.evolution import propagator
from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.orientations import (
    OrientationSample,
    normalize_orientations,
    powder_average_grid,
)
from spin_dynamics.nqr.pulses import (
    SelectivePulse,
    apply_selective_pulse,
    selective_pulse_hamiltonian,
)
from spin_dynamics.nqr.relaxation import (
    NQRRelaxationModel,
    cycle_superoperator,
    effective_decay_time,
)
from spin_dynamics.nqr.sequences import SLSESequence
from spin_dynamics.nqr.systems import NQREigensystem, NQRTransition, QuadrupolarSite


OrientationInput = str | tuple[OrientationSample, ...] | list[OrientationSample]


def _require_spin_one_selective_pulse_site(site: QuadrupolarSite) -> None:
    """Reject sites outside the current embedded two-level pulse model."""

    if not np.isclose(site.spin, 1.0):
        raise NotImplementedError(
            "selective-pulse NQR workflows currently support spin=1 only; "
            "spin=3/2 requires a degenerate-doublet manifold RF model"
        )


@dataclass(frozen=True)
class SLSEResult:
    """Simulated SLSE echo train."""

    echo_times: np.ndarray
    echo_amplitudes: np.ndarray
    local_echo_amplitudes: np.ndarray
    orientation_weights: np.ndarray
    transition: NQRTransition
    eigensystem: NQREigensystem
    local_effective_t2eff_seconds: np.ndarray | None = None
    local_cycle_eigenvalues: np.ndarray | None = None


@dataclass(frozen=True)
class PopulationTransferResult:
    """Perturbation plus SLSE detection result."""

    signal: SLSEResult
    reference: SLSEResult
    normalized_difference: np.ndarray
    perturbation: SelectivePulse


@dataclass(frozen=True)
class SLSESweepResult:
    """SLSE response as one pulse-sequence parameter is swept."""

    sweep_values: np.ndarray
    selected_echo_amplitudes: np.ndarray
    effective_t2eff_seconds: np.ndarray
    results: tuple[SLSEResult, ...]
    sweep_name: str
    transition_label: str


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
    orientations: OrientationInput,
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


def _pulse_detuning_hz(pulse: SelectivePulse, transition: NQRTransition) -> float:
    rf_frequency_hz = (
        transition.frequency_hz
        if pulse.rf_frequency_hz is None
        else pulse.rf_frequency_hz
    )
    return float(rf_frequency_hz - transition.frequency_hz)


def _relaxing_slse_steps(
    dimension: int,
    transition: NQRTransition,
    sequence: SLSESequence,
    b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float],
) -> tuple[tuple[np.ndarray, float], ...]:
    pulse = sequence.detection
    detuning_hz = _pulse_detuning_hz(pulse, transition)
    pulse_hamiltonian = selective_pulse_hamiltonian(
        dimension,
        transition,
        nutation_hz=pulse.nutation_hz,
        phase=pulse.phase,
        b1_direction_pas=b1_direction_pas,
        detuning_hz=detuning_hz,
    )
    free_hamiltonian = selective_pulse_hamiltonian(
        dimension,
        transition,
        nutation_hz=0.0,
        detuning_hz=detuning_hz,
    )
    free_duration = max(
        sequence.echo_spacing_seconds - pulse.duration_seconds,
        0.0,
    )
    steps = [(pulse_hamiltonian, pulse.duration_seconds)]
    if free_duration > 0:
        steps.append((free_hamiltonian, free_duration))
    return tuple(steps)


def _effective_t2eff_average(result: SLSEResult) -> float:
    values = result.local_effective_t2eff_seconds
    if values is None:
        return np.inf
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.inf
    weights = np.asarray(result.orientation_weights, dtype=np.float64)[finite]
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.mean(values[finite]))
    return float(np.sum(weights * values[finite]) / total)


def _selected_echo(echoes: np.ndarray, echo_index: int) -> complex:
    echoes = np.asarray(echoes, dtype=np.complex128)
    index = int(echo_index)
    if index < 0:
        index += echoes.size
    if index < 0 or index >= echoes.size:
        raise IndexError("echo_index is out of range")
    return complex(echoes[index])


def simulate_slse(
    site: QuadrupolarSite,
    sequence: SLSESequence,
    *,
    orientations: OrientationInput = "powder",
    b0_tesla: float = 0.0,
    t2e_seconds: float = np.inf,
    initial_density: np.ndarray | None = None,
    relaxation: NQRRelaxationModel | None = None,
) -> SLSEResult:
    """Simulate a selective-pulse SLSE echo train."""

    _require_spin_one_selective_pulse_site(site)
    samples = _as_orientations(orientations)
    t2e_seconds = float(t2e_seconds)
    if t2e_seconds <= 0:
        raise ValueError("t2e_seconds must be positive")
    if relaxation is not None and np.isfinite(t2e_seconds):
        warnings.warn(
            "both a finite t2e_seconds envelope and a Liouville-space "
            "relaxation model were given; their T2 damping composes "
            "multiplicatively. Pass t2e_seconds=inf (the default) when using a "
            "relaxation model so coherence decay is not double-counted.",
            RuntimeWarning,
            stacklevel=2,
        )
    local: list[np.ndarray] = []
    local_t2eff: list[float] = []
    local_eigenvalues: list[np.ndarray] = []
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
        if relaxation is None:
            detuning_hz = _pulse_detuning_hz(sequence.detection, transition)
            hamiltonian = selective_pulse_hamiltonian(
                site.dimension,
                transition,
                nutation_hz=sequence.detection.nutation_hz,
                phase=sequence.detection.phase,
                b1_direction_pas=sample.b1_direction_pas,
                detuning_hz=detuning_hz,
            )
            unitary = propagator(
                hamiltonian,
                sequence.detection.duration_seconds,
            )
            unitary_dagger = unitary.conj().T
            for echo_idx in range(sequence.num_echoes):
                density = unitary @ density @ unitary_dagger
                echoes[echo_idx] = transition_signal(
                    density,
                    transition,
                    b1_direction_pas=sample.b1_direction_pas,
                )
        else:
            steps = _relaxing_slse_steps(
                site.dimension,
                transition,
                sequence,
                sample.b1_direction_pas,
            )
            cycle = cycle_superoperator(steps, relaxation=relaxation)
            eigenvalues = np.linalg.eigvals(cycle)
            local_eigenvalues.append(eigenvalues)
            local_t2eff.append(
                effective_decay_time(eigenvalues, sequence.echo_spacing_seconds)
            )
            vector = density.reshape(-1, order="F")
            for echo_idx in range(sequence.num_echoes):
                vector = cycle @ vector
                density = vector.reshape(density.shape, order="F")
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
        local_effective_t2eff_seconds=(
            np.asarray(local_t2eff, dtype=np.float64) if local_t2eff else None
        ),
        local_cycle_eigenvalues=(
            np.asarray(local_eigenvalues, dtype=np.complex128)
            if local_eigenvalues
            else None
        ),
    )


def simulate_slse_offset_sweep(
    site: QuadrupolarSite,
    transition_label: str,
    offsets_hz: np.ndarray | list[float] | tuple[float, ...],
    *,
    pulse_duration_seconds: float,
    nutation_hz: float,
    echo_spacing_seconds: float,
    num_echoes: int = 16,
    phase: float = 0.0,
    orientations: OrientationInput = "powder",
    b0_tesla: float = 0.0,
    t2e_seconds: float = np.inf,
    relaxation: NQRRelaxationModel | None = None,
    echo_index: int = -1,
) -> SLSESweepResult:
    """Sweep irradiation offset and return SLSE amplitude and decay estimates."""

    offsets = np.asarray(offsets_hz, dtype=np.float64).reshape(-1)
    if offsets.size == 0:
        raise ValueError("offsets_hz must not be empty")
    reference = diagonalize_site(site).transition(transition_label).frequency_hz
    results: list[SLSEResult] = []
    amplitudes = np.empty(offsets.size, dtype=np.complex128)
    t2eff = np.empty(offsets.size, dtype=np.float64)

    for idx, offset in enumerate(offsets):
        sequence = SLSESequence(
            detection=SelectivePulse(
                transition_label,
                duration_seconds=pulse_duration_seconds,
                nutation_hz=nutation_hz,
                phase=phase,
                rf_frequency_hz=reference + float(offset),
            ),
            echo_spacing_seconds=echo_spacing_seconds,
            num_echoes=num_echoes,
        )
        result = simulate_slse(
            site,
            sequence,
            orientations=orientations,
            b0_tesla=b0_tesla,
            t2e_seconds=t2e_seconds,
            relaxation=relaxation,
        )
        results.append(result)
        amplitudes[idx] = _selected_echo(result.echo_amplitudes, echo_index)
        t2eff[idx] = _effective_t2eff_average(result)

    return SLSESweepResult(
        sweep_values=offsets,
        selected_echo_amplitudes=amplitudes,
        effective_t2eff_seconds=t2eff,
        results=tuple(results),
        sweep_name="offset_hz",
        transition_label=str(transition_label),
    )


def simulate_slse_spacing_sweep(
    site: QuadrupolarSite,
    transition_label: str,
    echo_spacing_seconds: np.ndarray | list[float] | tuple[float, ...],
    *,
    pulse_duration_seconds: float,
    nutation_hz: float,
    num_echoes: int = 16,
    phase: float = 0.0,
    rf_offset_hz: float = 0.0,
    orientations: OrientationInput = "powder",
    b0_tesla: float = 0.0,
    t2e_seconds: float = np.inf,
    relaxation: NQRRelaxationModel | None = None,
    echo_index: int = -1,
) -> SLSESweepResult:
    """Sweep SLSE pulse period and return amplitude plus effective decay."""

    spacings = np.asarray(echo_spacing_seconds, dtype=np.float64).reshape(-1)
    if spacings.size == 0:
        raise ValueError("echo_spacing_seconds must not be empty")
    if np.any(spacings < pulse_duration_seconds):
        raise ValueError("echo spacings must be at least the pulse duration")
    reference = diagonalize_site(site).transition(transition_label).frequency_hz
    results: list[SLSEResult] = []
    amplitudes = np.empty(spacings.size, dtype=np.complex128)
    t2eff = np.empty(spacings.size, dtype=np.float64)

    for idx, spacing in enumerate(spacings):
        sequence = SLSESequence(
            detection=SelectivePulse(
                transition_label,
                duration_seconds=pulse_duration_seconds,
                nutation_hz=nutation_hz,
                phase=phase,
                rf_frequency_hz=reference + float(rf_offset_hz),
            ),
            echo_spacing_seconds=float(spacing),
            num_echoes=num_echoes,
        )
        result = simulate_slse(
            site,
            sequence,
            orientations=orientations,
            b0_tesla=b0_tesla,
            t2e_seconds=t2e_seconds,
            relaxation=relaxation,
        )
        results.append(result)
        amplitudes[idx] = _selected_echo(result.echo_amplitudes, echo_index)
        t2eff[idx] = _effective_t2eff_average(result)

    return SLSESweepResult(
        sweep_values=spacings,
        selected_echo_amplitudes=amplitudes,
        effective_t2eff_seconds=t2eff,
        results=tuple(results),
        sweep_name="echo_spacing_seconds",
        transition_label=str(transition_label),
    )


def simulate_population_transfer(
    site: QuadrupolarSite,
    perturbation: SelectivePulse,
    detection_sequence: SLSESequence,
    *,
    orientations: OrientationInput = "powder",
    b0_tesla: float = 0.0,
    t2e_seconds: float = np.inf,
) -> PopulationTransferResult:
    """Simulate a perturbation pulse followed by SLSE detection."""

    _require_spin_one_selective_pulse_site(site)
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
