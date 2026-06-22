"""Dense electron-nuclear hyperfine helpers for ESR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.esr.hamiltonians import (
    TAU,
    effective_g_value,
    resonance_field_tesla,
)
from spin_dynamics.esr.lineshapes import spectrum_from_lines
from spin_dynamics.esr.systems import BOHR_MAGNETON_HZ_PER_T, ESRSpinSystem, as_g_tensor
from spin_dynamics.nqr.operators import spin_dimension, spin_matrices, validate_spin


@dataclass(frozen=True)
class NuclearSite:
    """One nucleus coupled to the ESR electron spin."""

    label: str
    isotope: str = "1H"
    spin: float = 0.5
    gamma_hz_per_t: float = 42.57747892e6

    def __post_init__(self) -> None:
        spin = validate_spin(self.spin)
        gamma_hz_per_t = float(self.gamma_hz_per_t)
        if not np.isfinite(gamma_hz_per_t):
            raise ValueError("gamma_hz_per_t must be finite")
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "isotope", str(self.isotope))
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "gamma_hz_per_t", gamma_hz_per_t)

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension for this nucleus."""

        return spin_dimension(self.spin)


@dataclass(frozen=True)
class ElectronNuclearSystem:
    """One electron spin coupled isotropically to one or more nuclei."""

    nuclei: tuple[NuclearSite, ...]
    hyperfine_hz: np.ndarray | Sequence[float]
    g_tensor: float | np.ndarray | list[float] | tuple[float, ...] = 2.00231930436256
    electron_label: str = "e"

    def __post_init__(self) -> None:
        nuclei = tuple(self.nuclei)
        if not nuclei:
            raise ValueError("at least one nuclear site is required")
        hyperfine = np.asarray(self.hyperfine_hz, dtype=np.float64).reshape(-1)
        if hyperfine.size != len(nuclei):
            raise ValueError("hyperfine_hz must match the number of nuclei")
        if not np.all(np.isfinite(hyperfine)):
            raise ValueError("hyperfine_hz must be finite")
        object.__setattr__(self, "nuclei", nuclei)
        object.__setattr__(self, "hyperfine_hz", hyperfine)
        object.__setattr__(self, "g_tensor", as_g_tensor(self.g_tensor))
        object.__setattr__(self, "electron_label", str(self.electron_label))

    @property
    def n_nuclei(self) -> int:
        """Number of coupled nuclei."""

        return len(self.nuclei)

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Hilbert-space dimensions in electron-first product order."""

        return (2,) + tuple(nucleus.dimension for nucleus in self.nuclei)

    @property
    def dimension(self) -> int:
        """Total Hilbert-space dimension."""

        out = 1
        for dimension in self.dimensions:
            out *= dimension
        return out


@dataclass(frozen=True)
class HyperfineTransition:
    """One ESR-active transition in an electron-nuclear spin system."""

    label: str
    lower: int
    upper: int
    frequency_hz: float
    dipole_vector: np.ndarray
    strength: float

    def __post_init__(self) -> None:
        dipole_vector = np.asarray(self.dipole_vector, dtype=np.complex128).reshape(3)
        strength = float(self.strength)
        if self.lower < 0 or self.upper < 0 or self.lower == self.upper:
            raise ValueError("transition levels must be distinct non-negative indices")
        if self.frequency_hz < 0 or not np.isfinite(self.frequency_hz):
            raise ValueError("frequency_hz must be non-negative and finite")
        if strength < 0 or not np.isfinite(strength):
            raise ValueError("strength must be non-negative and finite")
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "lower", int(self.lower))
        object.__setattr__(self, "upper", int(self.upper))
        object.__setattr__(self, "frequency_hz", float(self.frequency_hz))
        object.__setattr__(self, "dipole_vector", dipole_vector)
        object.__setattr__(self, "strength", strength)


@dataclass(frozen=True)
class HyperfineEigensystem:
    """Energy levels, eigenvectors, and ESR transitions for a hyperfine system."""

    system: ElectronNuclearSystem
    b0_vector_tesla_g: np.ndarray
    levels_hz: np.ndarray
    eigenvectors: np.ndarray
    transitions: tuple[HyperfineTransition, ...]


@dataclass(frozen=True)
class HyperfineFieldPoint:
    """One field-sweep contribution from one hyperfine transition."""

    field_tesla: float
    frequency_hz: float
    detuning_hz: float
    intensity: float
    lower: int
    upper: int
    label: str


@dataclass(frozen=True)
class HyperfineFieldSweepResult:
    """Field-swept ESR spectrum for an electron-nuclear hyperfine system."""

    fields_tesla: np.ndarray
    spectrum: np.ndarray
    transition_points: tuple[HyperfineFieldPoint, ...]
    microwave_frequency_hz: float
    broadening_hz: float
    system: ElectronNuclearSystem
    lineshape: str = "gaussian"
    detection_mode: str = "absorption"


def electron_nuclear_system(
    hyperfine_hz: Sequence[float] | np.ndarray,
    *,
    nuclei: Sequence[NuclearSite] | None = None,
    g_tensor: float | np.ndarray | list[float] | tuple[float, ...] = 2.00231930436256,
) -> ElectronNuclearSystem:
    """Build an electron-nuclear spin system from isotropic hyperfine constants."""

    hyperfine = np.asarray(hyperfine_hz, dtype=np.float64).reshape(-1)
    if nuclei is None:
        nuclei = tuple(
            NuclearSite(label=f"I{idx + 1}") for idx in range(hyperfine.size)
        )
    return ElectronNuclearSystem(
        nuclei=tuple(nuclei),
        hyperfine_hz=hyperfine,
        g_tensor=g_tensor,
    )


def electron_operator(system: ElectronNuclearSystem, axis: str) -> np.ndarray:
    """Return an electron spin operator embedded in the full Hilbert space."""

    return _embedded_operator(system, 0, axis)


def nuclear_operator(
    system: ElectronNuclearSystem,
    nucleus_index: int,
    axis: str,
) -> np.ndarray:
    """Return a nuclear spin operator embedded in the full Hilbert space."""

    index = int(nucleus_index)
    if index < 0 or index >= system.n_nuclei:
        raise ValueError("nucleus_index must select an existing nucleus")
    return _embedded_operator(system, index + 1, axis)


def zeeman_hamiltonian(
    system: ElectronNuclearSystem,
    b0_vector_tesla_g: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return electron plus nuclear Zeeman Hamiltonian in radians per second."""

    b0 = np.asarray(b0_vector_tesla_g, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_g must be finite")
    effective_field = system.g_tensor.T @ b0
    hamiltonian = TAU * BOHR_MAGNETON_HZ_PER_T * (
        effective_field[0] * electron_operator(system, "x")
        + effective_field[1] * electron_operator(system, "y")
        + effective_field[2] * electron_operator(system, "z")
    )
    for idx, nucleus in enumerate(system.nuclei):
        hamiltonian = hamiltonian - TAU * nucleus.gamma_hz_per_t * (
            b0[0] * nuclear_operator(system, idx, "x")
            + b0[1] * nuclear_operator(system, idx, "y")
            + b0[2] * nuclear_operator(system, idx, "z")
        )
    return hamiltonian


def isotropic_hyperfine_hamiltonian(system: ElectronNuclearSystem) -> np.ndarray:
    """Return isotropic ``S . A . I`` hyperfine Hamiltonian in radians per second."""

    hamiltonian = np.zeros((system.dimension, system.dimension), dtype=np.complex128)
    for idx, coupling_hz in enumerate(system.hyperfine_hz):
        if coupling_hz == 0:
            continue
        pair = (
            electron_operator(system, "x") @ nuclear_operator(system, idx, "x")
            + electron_operator(system, "y") @ nuclear_operator(system, idx, "y")
            + electron_operator(system, "z") @ nuclear_operator(system, idx, "z")
        )
        hamiltonian = hamiltonian + TAU * coupling_hz * pair
    return hamiltonian


def hyperfine_hamiltonian(
    system: ElectronNuclearSystem,
    b0_vector_tesla_g: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return Zeeman plus isotropic hyperfine Hamiltonian."""

    return zeeman_hamiltonian(system, b0_vector_tesla_g) + (
        isotropic_hyperfine_hamiltonian(system)
    )


def diagonalize_hyperfine_system(
    system: ElectronNuclearSystem,
    b0_vector_tesla_g: np.ndarray | Sequence[float],
    *,
    strength_tolerance: float = 1e-12,
    frequency_tolerance_hz: float = 1e-9,
) -> HyperfineEigensystem:
    """Diagonalize a hyperfine Hamiltonian and return ESR-active transitions."""

    b0 = np.asarray(b0_vector_tesla_g, dtype=np.float64).reshape(3)
    hamiltonian = hyperfine_hamiltonian(system, b0)
    values, vectors = np.linalg.eigh(hamiltonian)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    levels_hz = values / TAU

    operator_components = tuple(
        electron_operator(system, axis) for axis in ("x", "y", "z")
    )
    transitions: list[HyperfineTransition] = []
    for lower in range(system.dimension):
        for upper in range(lower + 1, system.dimension):
            frequency_hz = float(levels_hz[upper] - levels_hz[lower])
            if frequency_hz <= frequency_tolerance_hz:
                continue
            dipole = np.array(
                [
                    vectors[:, lower].conj().T @ op @ vectors[:, upper]
                    for op in operator_components
                ],
                dtype=np.complex128,
            )
            strength = float(np.linalg.norm(dipole))
            if strength <= strength_tolerance:
                continue
            transitions.append(
                HyperfineTransition(
                    label=f"{system.electron_label}{len(transitions) + 1}",
                    lower=lower,
                    upper=upper,
                    frequency_hz=frequency_hz,
                    dipole_vector=dipole,
                    strength=strength,
                )
            )

    transitions.sort(key=lambda item: item.frequency_hz)
    return HyperfineEigensystem(
        system=system,
        b0_vector_tesla_g=b0,
        levels_hz=levels_hz,
        eigenvectors=vectors,
        transitions=tuple(transitions),
    )


def simulate_hyperfine_field_sweep(
    system: ElectronNuclearSystem,
    microwave_frequency_hz: float,
    *,
    b0_direction_g: np.ndarray | Sequence[float] = (0.0, 0.0, 1.0),
    b1_direction_g: np.ndarray | Sequence[float] = (1.0, 0.0, 0.0),
    broadening_hz: float = 1.0e6,
    points: int = 1024,
    span_tesla: float | None = None,
    fields_tesla: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lineshape: str = "gaussian",
    detection_mode: str = "absorption",
    intensity_tolerance: float = 1e-14,
) -> HyperfineFieldSweepResult:
    """Return a field-swept ESR spectrum including isotropic hyperfine coupling."""

    frequency = float(microwave_frequency_hz)
    if frequency <= 0 or not np.isfinite(frequency):
        raise ValueError("microwave_frequency_hz must be positive and finite")
    broadening = float(broadening_hz)
    if broadening <= 0 or not np.isfinite(broadening):
        raise ValueError("broadening_hz must be positive and finite")
    b0_direction = _unit(b0_direction_g)
    b1_direction = _unit(b1_direction_g)
    if fields_tesla is None:
        fields = _auto_field_axis(
            system,
            frequency,
            b0_direction,
            broadening_hz=broadening,
            points=points,
            span_tesla=span_tesla,
        )
    else:
        fields = np.asarray(fields_tesla, dtype=np.float64).reshape(-1)
        if fields.size < 2:
            raise ValueError("fields_tesla must contain at least two points")
        if not np.all(np.isfinite(fields)):
            raise ValueError("fields_tesla must be finite")

    spectrum = np.zeros(fields.size, dtype=np.float64)
    transition_points: list[HyperfineFieldPoint] = []
    for idx, field in enumerate(fields):
        eigensystem = diagonalize_hyperfine_system(system, field * b0_direction)
        centers = []
        intensities = []
        for transition in eigensystem.transitions:
            rf_amplitude = np.vdot(b1_direction, transition.dipole_vector)
            intensity = float(abs(rf_amplitude) ** 2)
            if intensity <= intensity_tolerance:
                continue
            centers.append(transition.frequency_hz)
            intensities.append(intensity)
            transition_points.append(
                HyperfineFieldPoint(
                    field_tesla=float(field),
                    frequency_hz=transition.frequency_hz,
                    detuning_hz=transition.frequency_hz - frequency,
                    intensity=intensity,
                    lower=transition.lower,
                    upper=transition.upper,
                    label=transition.label,
                )
            )
        if centers:
            spectrum[idx] = spectrum_from_lines(
                np.array([frequency], dtype=np.float64),
                centers,
                intensities,
                width=broadening,
                lineshape=lineshape,
                detection_mode=detection_mode,
            )[0]

    return HyperfineFieldSweepResult(
        fields_tesla=fields,
        spectrum=spectrum,
        transition_points=tuple(transition_points),
        microwave_frequency_hz=frequency,
        broadening_hz=broadening,
        system=system,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )


def _embedded_operator(
    system: ElectronNuclearSystem,
    site_index: int,
    axis: str,
) -> np.ndarray:
    factors = []
    for idx, spin in enumerate(
        (0.5,) + tuple(nucleus.spin for nucleus in system.nuclei)
    ):
        ops = spin_matrices(spin)
        if idx == site_index:
            factors.append({"x": ops.ix, "y": ops.iy, "z": ops.iz}[axis.lower()])
        else:
            factors.append(ops.identity)
    return _kron_all(factors)


def _kron_all(factors: Sequence[np.ndarray]) -> np.ndarray:
    out: np.ndarray | None = None
    for factor in factors:
        out = factor if out is None else np.kron(out, factor)
    if out is None:
        raise ValueError("at least one factor is required")
    return out


def _unit(direction) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("direction must be a finite non-zero vector")
    return vec / norm


def _auto_field_axis(
    system: ElectronNuclearSystem,
    microwave_frequency_hz: float,
    b0_direction_g: np.ndarray,
    *,
    broadening_hz: float,
    points: int,
    span_tesla: float | None,
) -> np.ndarray:
    points = int(points)
    if points < 2:
        raise ValueError("points must be at least two")
    base = ESRSpinSystem(g_tensor=system.g_tensor)
    center = resonance_field_tesla(base, microwave_frequency_hz, b0_direction_g)
    g_eff = effective_g_value(base, b0_direction_g)
    hz_per_t = BOHR_MAGNETON_HZ_PER_T * g_eff
    if span_tesla is None:
        hyperfine_span = float(np.sum(np.abs(system.hyperfine_hz))) / hz_per_t
        broadening_span = 5.0 * broadening_hz / hz_per_t
        half_span = max(0.6 * hyperfine_span + broadening_span, broadening_span)
    else:
        half_span = 0.5 * float(span_tesla)
        if half_span <= 0 or not np.isfinite(half_span):
            raise ValueError("span_tesla must be positive and finite")
    return np.linspace(center - half_span, center + half_span, points)
