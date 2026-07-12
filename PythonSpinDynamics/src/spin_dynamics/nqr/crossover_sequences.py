"""Pulsed crossover sequences with exact RF and static-field relaxation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.nqr.field_relaxation import field_dependent_equilibrium
from spin_dynamics.nqr.floquet import simulate_floquet_rf
from spin_dynamics.nqr.hamiltonians import diagonalize_site, nqr_hamiltonian
from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.systems import QuadrupolarSite
from spin_dynamics.relaxation import (
    RelaxationSuperoperator,
    liouville_hamiltonian,
    matrix_exponential,
)


@dataclass(frozen=True)
class CrossoverSLSEResult:
    """Single-crystal SLSE echo train across an arbitrary static-field regime."""

    echo_times_seconds: np.ndarray
    echo_amplitudes: np.ndarray
    density_matrices_pas: np.ndarray
    rf_frequency_hz: float
    nutation_hz: float
    b0_vector_tesla_pas: np.ndarray
    zeeman_to_quadrupole_ratio: float
    equilibrium_density_pas: np.ndarray
    equilibrium_signal: complex
    excitation_pulse_unitarity_error: float
    refocus_pulse_unitarity_error: float
    relaxation_model: RelaxationSuperoperator
    site: QuadrupolarSite


def _unit_real(vector: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(out))
    if not np.all(np.isfinite(out)) or norm <= 0.0:
        raise ValueError(f"{name} must be finite and non-zero")
    return out / norm


def _target_carrier_hz(
    site: QuadrupolarSite,
    b0: np.ndarray,
    b1: np.ndarray,
) -> float:
    eigensystem = diagonalize_site(site, b0)
    if not eigensystem.transitions:
        raise ValueError("site has no RF-active transitions")
    reference = max(
        site.quadrupole_frequency_hz,
        abs(site.gamma_hz_per_t) * float(np.linalg.norm(b0)),
    )
    candidates = [
        transition
        for transition in eigensystem.transitions
        if transition.frequency_hz >= 0.5 * reference
    ]
    if not candidates:
        candidates = list(eigensystem.transitions)
    target = max(
        candidates,
        key=lambda transition: abs(np.vdot(b1, transition.dipole_vector)),
    )
    return float(target.frequency_hz)


def _default_quadrature(b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    static_axis = b0 / np.linalg.norm(b0) if np.linalg.norm(b0) > 0.0 else np.array(
        [0.0, 0.0, 1.0]
    )
    quadrature = np.cross(static_axis, b1)
    if np.linalg.norm(quadrature) < 1.0e-12:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, b1)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        quadrature = fallback - np.dot(fallback, b1) * b1
    return quadrature / np.linalg.norm(quadrature)


def simulate_crossover_slse(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray,
    *,
    nutation_hz: float,
    excitation_duration_seconds: float,
    refocus_duration_seconds: float,
    echo_spacing_seconds: float,
    num_echoes: int,
    relaxation: RelaxationSuperoperator,
    rf_frequency_hz: float | None = None,
    excitation_phase_radians: float = 0.0,
    refocus_phase_radians: float = np.pi / 2.0,
    b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0),
    receive_quadrature_direction_pas: Sequence[float] | np.ndarray | None = None,
    floquet_sidebands: int = 4,
    initial_density: np.ndarray | None = None,
) -> CrossoverSLSEResult:
    """Simulate an SLSE train with Floquet pulses and static-field dissipation.

    Relaxation is included during free evolution. Pulses use the exact
    finite-sideband monochromatic propagator and are treated as hard compared
    with the relaxation timescale. The carrier follows the strongest member of
    the conventional NQR/NMR band unless supplied explicitly.
    """

    b0 = np.asarray(b0_vector_tesla_pas, dtype=np.float64).reshape(3)
    b1 = _unit_real(b1_direction_pas, "b1_direction_pas")
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_pas must be finite")
    nutation = float(nutation_hz)
    excitation_duration = float(excitation_duration_seconds)
    refocus_duration = float(refocus_duration_seconds)
    echo_spacing = float(echo_spacing_seconds)
    echo_count = int(num_echoes)
    if nutation < 0.0 or not np.isfinite(nutation):
        raise ValueError("nutation_hz must be non-negative and finite")
    if excitation_duration <= 0.0 or refocus_duration <= 0.0:
        raise ValueError("pulse durations must be positive")
    if echo_count <= 0:
        raise ValueError("num_echoes must be positive")
    if echo_spacing < refocus_duration:
        raise ValueError("echo_spacing_seconds must be at least refocus duration")
    free_half = 0.5 * (echo_spacing - refocus_duration)
    carrier = (
        _target_carrier_hz(site, b0, b1)
        if rf_frequency_hz is None
        else float(rf_frequency_hz)
    )
    if carrier <= 0.0 or not np.isfinite(carrier):
        raise ValueError("rf_frequency_hz must be positive and finite")

    temperature = float(getattr(relaxation, "temperature_kelvin", 300.0))
    equilibrium = field_dependent_equilibrium(
        site,
        b0,
        temperature_kelvin=temperature,
    )
    if initial_density is None:
        density = equilibrium.density_matrix_pas.copy()
    else:
        density = np.asarray(initial_density, dtype=np.complex128).copy()
        if density.shape != (site.dimension, site.dimension):
            raise ValueError("initial_density has the wrong shape")
        if not np.allclose(density, density.conj().T):
            raise ValueError("initial_density must be Hermitian")

    excite_result = simulate_floquet_rf(
        site,
        b0,
        nutation_hz=nutation,
        rf_frequency_hz=carrier,
        pulse_duration_seconds=excitation_duration,
        phase_radians=excitation_phase_radians,
        b1_direction_pas=b1,
        sidebands=floquet_sidebands,
        initial_density=density,
        temperature_kelvin=temperature,
    )
    excite = excite_result.physical_propagator
    refocus_result = simulate_floquet_rf(
        site,
        b0,
        nutation_hz=nutation,
        rf_frequency_hz=carrier,
        pulse_duration_seconds=refocus_duration,
        phase_radians=refocus_phase_radians,
        b1_direction_pas=b1,
        sidebands=floquet_sidebands,
        initial_density=density,
        temperature_kelvin=temperature,
    )
    refocus = refocus_result.physical_propagator
    hamiltonian = nqr_hamiltonian(site, b0)
    free_generator = liouville_hamiltonian(hamiltonian) + relaxation.superoperator(
        hamiltonian
    )
    free = matrix_exponential(free_generator, free_half)

    quadrature = (
        _default_quadrature(b0, b1)
        if receive_quadrature_direction_pas is None
        else _unit_real(
            receive_quadrature_direction_pas,
            "receive_quadrature_direction_pas",
        )
    )
    ops = spin_matrices(site.spin)
    in_phase = b1[0] * ops.ix + b1[1] * ops.iy + b1[2] * ops.iz
    out_of_phase = (
        quadrature[0] * ops.ix
        + quadrature[1] * ops.iy
        + quadrature[2] * ops.iz
    )
    detector = in_phase + 1.0j * out_of_phase
    baseline = complex(np.trace(equilibrium.density_matrix_pas @ detector))

    density = excite @ density @ excite.conj().T
    echoes = np.empty(echo_count, dtype=np.complex128)
    densities = np.empty(
        (echo_count, site.dimension, site.dimension),
        dtype=np.complex128,
    )
    for index in range(echo_count):
        vector = free @ density.reshape(-1, order="F")
        density = vector.reshape((site.dimension, site.dimension), order="F")
        density = refocus @ density @ refocus.conj().T
        vector = free @ density.reshape(-1, order="F")
        density = vector.reshape((site.dimension, site.dimension), order="F")
        densities[index] = density
        echoes[index] = np.trace(density @ detector) - baseline

    ratio = (
        abs(site.gamma_hz_per_t)
        * float(np.linalg.norm(b0))
        / site.quadrupole_frequency_hz
    )
    return CrossoverSLSEResult(
        echo_times_seconds=(np.arange(echo_count) + 1.0) * echo_spacing,
        echo_amplitudes=echoes,
        density_matrices_pas=densities,
        rf_frequency_hz=carrier,
        nutation_hz=nutation,
        b0_vector_tesla_pas=b0,
        zeeman_to_quadrupole_ratio=ratio,
        equilibrium_density_pas=equilibrium.density_matrix_pas,
        equilibrium_signal=baseline,
        excitation_pulse_unitarity_error=excite_result.unitarity_error,
        refocus_pulse_unitarity_error=refocus_result.unitarity_error,
        relaxation_model=relaxation,
        site=site,
    )
