"""Hydrogenative parahydrogen-induced polarization (PHIP) workflows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density
from spin_dynamics.coupling.hamiltonians import (
    isotropic_j_hamiltonian,
    secular_j_hamiltonian,
    zeeman_hamiltonian,
)
from spin_dynamics.coupling.operators import spin_operator
from spin_dynamics.coupling.systems import CoupledSpinSystem
from spin_dynamics.hyperpolarization.singlet import singlet_order_operator

PHIPProtocol = Literal["pasadena", "altadena"]
CouplingRegime = Literal["isotropic", "secular"]


@dataclass(frozen=True)
class PHIPFieldSegment:
    """One constant-field segment used during PHIP product transport.

    ``field_scale`` multiplies the product's detection-field offsets.  Scalar
    couplings remain in hertz.  Low-field segments normally use the isotropic
    coupling Hamiltonian; high-field PASADENA segments use the secular model.
    """

    field_scale: float
    duration_seconds: float
    coupling_regime: CouplingRegime = "isotropic"

    def __post_init__(self) -> None:
        if not np.isfinite(self.field_scale) or self.field_scale < 0.0:
            raise ValueError("field_scale must be non-negative and finite")
        if not np.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be non-negative and finite")
        if self.coupling_regime not in ("isotropic", "secular"):
            raise ValueError("coupling_regime must be 'isotropic' or 'secular'")


@dataclass(frozen=True)
class HydrogenativePHIPState:
    """Mapped product state following pairwise parahydrogen addition."""

    para_fraction: float
    pairwise_addition_fraction: float
    product_pair: tuple[int, int]
    density_matrix: np.ndarray
    deviation_density: np.ndarray

    @property
    def effective_para_excess(self) -> float:
        """Return para excess scaled by the pairwise-addition fraction."""

        return self.pairwise_addition_fraction * (self.para_fraction - 0.25)


@dataclass(frozen=True)
class PHIPAcquisitionResult:
    """State and complex FID from a hydrogenative PHIP protocol."""

    protocol: PHIPProtocol
    times_seconds: np.ndarray
    signal: np.ndarray
    product_density: np.ndarray
    density_after_transport: np.ndarray
    density_after_pulse: np.ndarray
    pulse_flip_angle_radians: float
    pulse_phase_radians: float


def hydrogenative_phip_state(
    product_nspin: int,
    *,
    para_fraction: float,
    pairwise_addition_fraction: float = 1.0,
    product_pair: tuple[int, int] = (0, 1),
) -> HydrogenativePHIPState:
    """Map parahydrogen singlet order into specified product spin sites.

    Unpaired or non-pairwise reaction products contribute an unpolarized
    background.  Consequently the usable product order is linear in both para
    excess above ``1/4`` and the externally supplied pairwise-addition
    fraction.  Catalyst kinetics and chemical yield are outside this map.
    """

    count = int(product_nspin)
    if count < 2:
        raise ValueError("product_nspin must be at least 2")
    fraction = float(para_fraction)
    paired = float(pairwise_addition_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("para_fraction must lie between 0 and 1")
    if not np.isfinite(paired) or not 0.0 <= paired <= 1.0:
        raise ValueError("pairwise_addition_fraction must lie between 0 and 1")

    order = singlet_order_operator(count, product_pair)
    deviation = paired * (fraction - 0.25) * order
    dimension = 2**count
    density = np.eye(dimension, dtype=np.complex128) / dimension + deviation
    density.setflags(write=False)
    deviation.setflags(write=False)
    return HydrogenativePHIPState(
        para_fraction=fraction,
        pairwise_addition_fraction=paired,
        product_pair=(int(product_pair[0]), int(product_pair[1])),
        density_matrix=density,
        deviation_density=deviation,
    )


def evolve_phip_field_trajectory(
    density: np.ndarray,
    system: CoupledSpinSystem,
    segments: Sequence[PHIPFieldSegment],
) -> np.ndarray:
    """Evolve a PHIP product through a piecewise-constant field trajectory."""

    out = np.asarray(density, dtype=np.complex128)
    if out.shape != (system.dimension, system.dimension):
        raise ValueError("density shape must match the product spin system")
    zeeman = zeeman_hamiltonian(system)
    isotropic = isotropic_j_hamiltonian(system)
    secular = secular_j_hamiltonian(system)
    for segment in segments:
        coupling = isotropic if segment.coupling_regime == "isotropic" else secular
        out = evolve_density(
            out,
            segment.field_scale * zeeman + coupling,
            segment.duration_seconds,
        )
    return out


def secularize_high_field_product(density: np.ndarray) -> np.ndarray:
    """Remove nonsecular product-basis coherences after high-field formation.

    PASADENA's familiar longitudinal two-spin order is the high-field secular
    part of the newly mapped singlet state.  Retaining the entire coherent
    singlet would make a nonselective hard pulse leave it NMR-silent.
    """

    matrix = np.asarray(density, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density must be a square matrix")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
        raise ValueError("density must be Hermitian")
    return np.diag(np.diag(matrix)).astype(np.complex128)


def apply_hard_pulse(
    density: np.ndarray,
    system: CoupledSpinSystem,
    flip_angle_radians: float,
    *,
    phase_radians: float = 0.0,
    indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Apply an ideal hard pulse to selected product spins."""

    angle = float(flip_angle_radians)
    phase = float(phase_radians)
    if not np.isfinite(angle) or not np.isfinite(phase):
        raise ValueError("pulse angle and phase must be finite")
    selected = tuple(range(system.nspin) if indices is None else (int(i) for i in indices))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("indices must select distinct product spins")
    if any(index < 0 or index >= system.nspin for index in selected):
        raise ValueError("pulse index lies outside the product spin system")
    axis = sum(
        (
            np.cos(phase) * spin_operator(system.nspin, index, "x")
            + np.sin(phase) * spin_operator(system.nspin, index, "y")
            for index in selected
        ),
        start=np.zeros((system.dimension, system.dimension), dtype=np.complex128),
    )
    return evolve_density(density, axis, angle)


def acquire_phip_fid(
    density: np.ndarray,
    system: CoupledSpinSystem,
    times_seconds: Iterable[float] | np.ndarray,
    *,
    detect_indices: Iterable[int] | None = None,
    t2_seconds: float | None = None,
) -> np.ndarray:
    """Acquire a high-field weak-coupling PHIP FID from selected spins."""

    times = np.array(times_seconds, dtype=np.float64, copy=True).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times_seconds must be finite and non-negative")
    selected = tuple(range(system.nspin) if detect_indices is None else (int(i) for i in detect_indices))
    if not selected or any(index < 0 or index >= system.nspin for index in selected):
        raise ValueError("detect_indices must select spins in the product system")
    if t2_seconds is not None and (
        not np.isfinite(t2_seconds) or float(t2_seconds) <= 0.0
    ):
        raise ValueError("t2_seconds must be positive and finite")
    detect = sum(
        (
            spin_operator(system.nspin, index, "x")
            + 1j * spin_operator(system.nspin, index, "y")
            for index in selected
        ),
        start=np.zeros((system.dimension, system.dimension), dtype=np.complex128),
    )
    hamiltonian = zeeman_hamiltonian(system) + secular_j_hamiltonian(system)
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    eigen_density = eigenvectors.conj().T @ np.asarray(density) @ eigenvectors
    eigen_detect = eigenvectors.conj().T @ detect @ eigenvectors
    coefficients = eigen_density * eigen_detect.T
    frequency_differences = energies[:, np.newaxis] - energies[np.newaxis, :]
    signal = np.sum(
        coefficients[np.newaxis, :, :]
        * np.exp(-1j * times[:, np.newaxis, np.newaxis] * frequency_differences),
        axis=(1, 2),
    )
    if t2_seconds is not None:
        signal *= np.exp(-times / float(t2_seconds))
    return signal


def simulate_hydrogenative_phip(
    system: CoupledSpinSystem,
    times_seconds: Iterable[float] | np.ndarray,
    *,
    protocol: PHIPProtocol,
    para_fraction: float,
    pairwise_addition_fraction: float = 1.0,
    product_pair: tuple[int, int] = (0, 1),
    reaction_time_seconds: float = 0.0,
    field_trajectory: Sequence[PHIPFieldSegment] | None = None,
    pulse_flip_angle_radians: float = np.pi / 4.0,
    pulse_phase_radians: float = 0.0,
    pulse_indices: Iterable[int] | None = None,
    detect_indices: Iterable[int] | None = None,
    t2_seconds: float | None = None,
) -> PHIPAcquisitionResult:
    """Simulate hydrogenative PASADENA or trajectory-defined ALTADENA.

    PASADENA product formation evolves at the detection field under the
    high-field secular Hamiltonian for ``reaction_time_seconds``.  ALTADENA
    requires an explicit low-to-high-field trajectory, preventing a silent
    assumption about transport speed or adiabaticity.
    """

    if protocol not in ("pasadena", "altadena"):
        raise ValueError("protocol must be 'pasadena' or 'altadena'")
    reaction_time = float(reaction_time_seconds)
    if not np.isfinite(reaction_time) or reaction_time < 0.0:
        raise ValueError("reaction_time_seconds must be non-negative and finite")
    mapped = hydrogenative_phip_state(
        system.nspin,
        para_fraction=para_fraction,
        pairwise_addition_fraction=pairwise_addition_fraction,
        product_pair=product_pair,
    )
    transported = mapped.deviation_density
    if protocol == "pasadena":
        if field_trajectory is not None:
            raise ValueError("field_trajectory is only used for ALTADENA")
        transported = secularize_high_field_product(transported)
        transported = evolve_phip_field_trajectory(
            transported,
            system,
            [PHIPFieldSegment(1.0, reaction_time, "secular")],
        )
    else:
        if field_trajectory is None or len(field_trajectory) == 0:
            raise ValueError("ALTADENA requires a non-empty field_trajectory")
        if reaction_time:
            transported = evolve_phip_field_trajectory(
                transported,
                system,
                [PHIPFieldSegment(0.0, reaction_time, "isotropic")],
            )
        transported = evolve_phip_field_trajectory(transported, system, field_trajectory)

    pulsed = apply_hard_pulse(
        transported,
        system,
        pulse_flip_angle_radians,
        phase_radians=pulse_phase_radians,
        indices=pulse_indices,
    )
    times = np.array(times_seconds, dtype=np.float64, copy=True).reshape(-1)
    signal = acquire_phip_fid(
        pulsed,
        system,
        times,
        detect_indices=detect_indices,
        t2_seconds=t2_seconds,
    )
    for array in (times, signal, transported, pulsed):
        array.setflags(write=False)
    return PHIPAcquisitionResult(
        protocol=protocol,
        times_seconds=times,
        signal=signal,
        product_density=mapped.density_matrix,
        density_after_transport=transported,
        density_after_pulse=pulsed,
        pulse_flip_angle_radians=float(pulse_flip_angle_radians),
        pulse_phase_radians=float(pulse_phase_radians),
    )


__all__ = [
    "CouplingRegime",
    "HydrogenativePHIPState",
    "PHIPAcquisitionResult",
    "PHIPFieldSegment",
    "PHIPProtocol",
    "acquire_phip_fid",
    "apply_hard_pulse",
    "evolve_phip_field_trajectory",
    "hydrogenative_phip_state",
    "secularize_high_field_product",
    "simulate_hydrogenative_phip",
]
