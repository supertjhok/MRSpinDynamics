"""Quantitative comparison of NQR crossover RWA and laboratory-frame pulses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density
from spin_dynamics.nqr.crossover import boltzmann_populations
from spin_dynamics.nqr.full_dynamics import pulse_hamiltonian, rotating_indices
from spin_dynamics.nqr.hamiltonians import TAU, diagonalize_site
from spin_dynamics.nqr.lab_frame import sample_linear_rf_pulse, simulate_lab_frame_rf
from spin_dynamics.nqr.systems import QuadrupolarSite


@dataclass(frozen=True)
class RWAComparisonResult:
    """One exact laboratory-frame versus single-band RWA pulse comparison."""

    lab_density_rotating: np.ndarray
    rwa_density_rotating: np.ndarray
    initial_density_eigenbasis: np.ndarray
    maximum_element_error: float
    frobenius_error: float
    relative_response_error: float
    rf_frequency_hz: float
    nutation_hz: float
    pulse_duration_seconds: float
    zeeman_to_quadrupole_ratio: float
    rf_strength_ratio: float
    samples_per_carrier_cycle: int
    site: QuadrupolarSite


@dataclass(frozen=True)
class RWAValidityMap:
    """RWA error over static-interaction and RF-strength ratios."""

    interaction_ratios: np.ndarray
    rf_strength_ratios: np.ndarray
    maximum_element_error: np.ndarray
    relative_response_error: np.ndarray
    carrier_frequencies_hz: np.ndarray
    minimum_target_isolation_hz: np.ndarray
    b0_direction_pas: np.ndarray
    b1_direction_pas: np.ndarray
    duration_in_carrier_cycles: float
    samples_per_carrier_cycle: int
    site: QuadrupolarSite


def _unit_real(vector: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(out))
    if not np.all(np.isfinite(out)) or norm <= 0.0:
        raise ValueError(f"{name} must be finite and non-zero")
    return out / norm


def _strongest_transition_frequency(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: np.ndarray,
    b1_direction_pas: np.ndarray,
) -> tuple[float, float]:
    eigensystem = diagonalize_site(site, b0_vector_tesla_pas)
    if not eigensystem.transitions:
        raise ValueError("site has no RF-active transitions")
    strengths = np.array(
        [
            abs(np.vdot(b1_direction_pas, transition.dipole_vector))
            for transition in eigensystem.transitions
        ]
    )
    reference_frequency = max(
        site.quadrupole_frequency_hz,
        abs(site.gamma_hz_per_t) * float(np.linalg.norm(b0_vector_tesla_pas)),
    )
    # In weak fields the strongest matrix element can be an intra-manifold
    # Zeeman splitting near zero frequency.  The validity map is intended to
    # follow the conventional NQR/NMR band, so reject lines far below both the
    # quadrupole and Larmor scales before selecting its strongest member.
    in_target_band = np.array(
        [
            transition.frequency_hz >= 0.5 * reference_frequency
            for transition in eigensystem.transitions
        ]
    )
    candidates = np.flatnonzero(in_target_band)
    if candidates.size == 0:
        candidates = np.arange(strengths.size)
    target_index = int(candidates[np.argmax(strengths[candidates])])
    target = eigensystem.transitions[target_index]
    distinct_tolerance = 1e-10 * reference_frequency
    competitors = [
        abs(item.frequency_hz - target.frequency_hz)
        for index, item in enumerate(eigensystem.transitions)
        if (
            index != target_index
            and strengths[index] > 1e-12
            and abs(item.frequency_hz - target.frequency_hz) > distinct_tolerance
        )
    ]
    isolation = min(competitors) if competitors else np.inf
    return float(target.frequency_hz), float(isolation)


def compare_rwa_to_lab_frame(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray,
    *,
    nutation_hz: float,
    rf_frequency_hz: float,
    pulse_duration_seconds: float,
    phase_radians: float = 0.0,
    b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0),
    temperature_kelvin: float = 300.0,
    samples_per_carrier_cycle: int = 80,
) -> RWAComparisonResult:
    """Compare identical linear RF pulses in the exact lab frame and RWA."""

    b0 = np.asarray(b0_vector_tesla_pas, dtype=np.float64).reshape(3)
    b1 = _unit_real(b1_direction_pas, "b1_direction_pas")
    nutation = float(nutation_hz)
    carrier = float(rf_frequency_hz)
    duration = float(pulse_duration_seconds)
    samples = int(samples_per_carrier_cycle)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_pas must be finite")
    if nutation < 0.0 or not np.isfinite(nutation):
        raise ValueError("nutation_hz must be non-negative and finite")
    if carrier <= 0.0 or not np.isfinite(carrier):
        raise ValueError("rf_frequency_hz must be positive and finite")
    if duration <= 0.0 or not np.isfinite(duration):
        raise ValueError("pulse_duration_seconds must be positive and finite")
    if samples < 8:
        raise ValueError("samples_per_carrier_cycle must be at least eight")
    signed_gamma = float(site.gamma_hz_per_t)
    gamma = abs(signed_gamma)
    if gamma <= 0.0:
        raise ValueError("site.gamma_hz_per_t must be non-zero")

    eigensystem = diagonalize_site(site, b0)
    populations = boltzmann_populations(eigensystem.levels_hz, temperature_kelvin)
    initial_eigen = np.diag(populations.astype(np.complex128))
    rwa_hamiltonian = pulse_hamiltonian(
        eigensystem,
        nutation_hz=nutation,
        rf_frequency_hz=carrier,
        phase=phase_radians,
        b1_direction_pas=b1,
    )
    rwa_final = evolve_density(initial_eigen, rwa_hamiltonian, duration)

    peak_b1_tesla = 2.0 * nutation / signed_gamma
    segment_duration, rf_fields = sample_linear_rf_pulse(
        duration,
        1.0 / (samples * carrier),
        peak_b1_tesla,
        carrier,
        phase_radians=phase_radians,
        direction_pas=b1,
    )
    lab = simulate_lab_frame_rf(
        site,
        b0,
        segment_duration,
        rf_fields,
        temperature_kelvin=temperature_kelvin,
    )
    vectors = eigensystem.eigenvectors
    lab_eigen = vectors.conj().T @ lab.density_matrices[-1] @ vectors
    winding = rotating_indices(eigensystem.levels_hz, carrier)
    rotation = np.diag(np.exp(-1.0j * TAU * carrier * winding * duration))
    lab_rotating = rotation.conj().T @ lab_eigen @ rotation

    difference = rwa_final - lab_rotating
    response = rwa_final - initial_eigen
    response_norm = float(np.linalg.norm(response))
    relative = float(np.linalg.norm(difference)) / max(response_norm, 1e-30)
    zeeman_frequency = gamma * float(np.linalg.norm(b0))
    return RWAComparisonResult(
        lab_density_rotating=lab_rotating,
        rwa_density_rotating=rwa_final,
        initial_density_eigenbasis=initial_eigen,
        maximum_element_error=float(np.max(np.abs(difference))),
        frobenius_error=float(np.linalg.norm(difference)),
        relative_response_error=relative,
        rf_frequency_hz=carrier,
        nutation_hz=nutation,
        pulse_duration_seconds=duration,
        zeeman_to_quadrupole_ratio=(
            zeeman_frequency / site.quadrupole_frequency_hz
        ),
        rf_strength_ratio=2.0 * nutation / site.quadrupole_frequency_hz,
        samples_per_carrier_cycle=samples,
        site=site,
    )


def scan_rwa_validity(
    site: QuadrupolarSite,
    interaction_ratios: Sequence[float] | np.ndarray,
    rf_strength_ratios: Sequence[float] | np.ndarray,
    *,
    b0_direction_pas: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
    b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0),
    detuning_hz: float = 0.0,
    phase_radians: float = 0.0,
    duration_in_carrier_cycles: float = 5.0,
    temperature_kelvin: float = 300.0,
    samples_per_carrier_cycle: int = 60,
) -> RWAValidityMap:
    """Map RWA error while automatically following the strongest RF line."""

    static_ratios = np.asarray(interaction_ratios, dtype=np.float64).reshape(-1)
    drive_ratios = np.asarray(rf_strength_ratios, dtype=np.float64).reshape(-1)
    if (
        static_ratios.size == 0
        or drive_ratios.size == 0
        or not np.all(np.isfinite(static_ratios))
        or not np.all(np.isfinite(drive_ratios))
        or np.any(static_ratios < 0.0)
        or np.any(drive_ratios <= 0.0)
    ):
        raise ValueError(
            "interaction ratios must be finite and non-negative; "
            "RF-strength ratios must be finite and positive"
        )
    b0_direction = _unit_real(b0_direction_pas, "b0_direction_pas")
    b1_direction = _unit_real(b1_direction_pas, "b1_direction_pas")
    cycles = float(duration_in_carrier_cycles)
    if cycles <= 0.0 or not np.isfinite(cycles):
        raise ValueError("duration_in_carrier_cycles must be positive and finite")
    gamma = abs(site.gamma_hz_per_t)
    if gamma <= 0.0:
        raise ValueError("site.gamma_hz_per_t must be non-zero")

    shape = (static_ratios.size, drive_ratios.size)
    maximum_error = np.empty(shape, dtype=np.float64)
    relative_error = np.empty(shape, dtype=np.float64)
    carriers = np.empty(static_ratios.size, dtype=np.float64)
    isolation = np.empty(static_ratios.size, dtype=np.float64)
    for static_index, static_ratio in enumerate(static_ratios):
        field = static_ratio * site.quadrupole_frequency_hz / gamma
        b0 = field * b0_direction
        target, target_isolation = _strongest_transition_frequency(site, b0, b1_direction)
        carrier = target + float(detuning_hz)
        if carrier <= 0.0:
            raise ValueError("detuning produces a non-positive RF carrier")
        carriers[static_index] = carrier
        isolation[static_index] = target_isolation
        duration = cycles / carrier
        for drive_index, drive_ratio in enumerate(drive_ratios):
            result = compare_rwa_to_lab_frame(
                site,
                b0,
                nutation_hz=0.5 * drive_ratio * site.quadrupole_frequency_hz,
                rf_frequency_hz=carrier,
                pulse_duration_seconds=duration,
                phase_radians=phase_radians,
                b1_direction_pas=b1_direction,
                temperature_kelvin=temperature_kelvin,
                samples_per_carrier_cycle=samples_per_carrier_cycle,
            )
            maximum_error[static_index, drive_index] = result.maximum_element_error
            relative_error[static_index, drive_index] = result.relative_response_error

    return RWAValidityMap(
        interaction_ratios=static_ratios,
        rf_strength_ratios=drive_ratios,
        maximum_element_error=maximum_error,
        relative_response_error=relative_error,
        carrier_frequencies_hz=carriers,
        minimum_target_isolation_hz=isolation,
        b0_direction_pas=b0_direction,
        b1_direction_pas=b1_direction,
        duration_in_carrier_cycles=cycles,
        samples_per_carrier_cycle=int(samples_per_carrier_cycle),
        site=site,
    )
