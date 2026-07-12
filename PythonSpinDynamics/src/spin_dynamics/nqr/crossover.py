"""Exact static spectra across the complete NQR-to-NMR crossover.

The weak-field helpers in :mod:`spin_dynamics.nqr.zeeman` intentionally follow
selected zero-field NQR lines. This module instead treats every magnetic-dipole
transition of ``H_Q + H_Z`` and remains valid when the Zeeman and quadrupolar
interactions are comparable or when the Zeeman interaction dominates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from spin_dynamics.nqr.hamiltonians import diagonalize_sites_over_b0
from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.orientations import (
    OrientationSample,
    b0_b1_powder_average_grid,
    normalize_orientations,
)
from spin_dynamics.nqr.systems import NQREigensystem, QuadrupolarSite


_PLANCK_J_S = 6.62607015e-34
_BOLTZMANN_J_PER_K = 1.380649e-23


def _unit_real(vector: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(out))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return out / norm


def _unit_complex(
    vector: Sequence[complex] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    out = np.asarray(vector, dtype=np.complex128).reshape(3)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(out))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return out / norm


@dataclass(frozen=True)
class CrossoverOrientation:
    """Static-field, transmit, and receive directions for one crystallite.

    Directions are expressed in the EFG principal-axis system. Transmit and
    receive directions may be complex, allowing linear, circular, or elliptical
    polarization. By default the receive coil is reciprocal with the transmit
    coil.
    """

    b0_direction_pas: np.ndarray | Sequence[float] = (0.0, 0.0, 1.0)
    transmit_direction_pas: np.ndarray | Sequence[complex] = (1.0, 0.0, 0.0)
    receive_direction_pas: np.ndarray | Sequence[complex] | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        b0 = _unit_real(self.b0_direction_pas, name="b0_direction_pas")
        transmit = _unit_complex(
            self.transmit_direction_pas,
            name="transmit_direction_pas",
        )
        receive_input = self.receive_direction_pas
        receive = (
            transmit.copy()
            if receive_input is None
            else _unit_complex(receive_input, name="receive_direction_pas")
        )
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be non-negative and finite")
        object.__setattr__(self, "b0_direction_pas", b0)
        object.__setattr__(self, "transmit_direction_pas", transmit)
        object.__setattr__(self, "receive_direction_pas", receive)
        object.__setattr__(self, "weight", weight)


CrossoverOrientationInput = (
    str | Sequence[CrossoverOrientation] | Sequence[OrientationSample]
)


@dataclass(frozen=True)
class CrossoverTransition:
    """One observable transition between energy manifolds.

    ``lower_levels`` and ``upper_levels`` may each contain more than one level
    at an exact degeneracy. Summing the response over complete manifolds makes
    it invariant to the eigensolver's basis choice inside a degenerate subspace.
    """

    orientation_index: int
    orientation_weight: float
    b0_vector_tesla_pas: np.ndarray
    zeeman_to_quadrupole_ratio: float
    lower_levels: tuple[int, ...]
    upper_levels: tuple[int, ...]
    frequency_hz: float
    population_difference: float
    excitation_strength: float
    detection_strength: float
    local_amplitude: complex
    amplitude: complex
    intensity: float


@dataclass(frozen=True)
class CrossoverSpectrumResult:
    """Exact equilibrium spectrum across the NQR-to-NMR crossover."""

    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    transitions: tuple[CrossoverTransition, ...]
    b0_tesla: float
    zeeman_frequency_hz: float
    zeeman_to_quadrupole_ratio: float
    quadrupole_reference_frequency_hz: float
    temperature_kelvin: float
    broadening_hz: float
    lineshape: str
    normalized: bool
    site: QuadrupolarSite


@dataclass(frozen=True)
class CrossoverFieldSweepResult:
    """Overlap-tracked eigenstates and transition responses over a B0 sweep."""

    b0_tesla: np.ndarray
    zeeman_to_quadrupole_ratio: np.ndarray
    levels_hz: np.ndarray
    eigenvectors: np.ndarray
    minimum_state_overlap: np.ndarray
    magnetic_quantum_expectation: np.ndarray
    state_pairs: tuple[tuple[int, int], ...]
    transition_frequencies_hz: np.ndarray
    transition_amplitudes: np.ndarray
    transition_intensities: np.ndarray
    orientation: CrossoverOrientation
    temperature_kelvin: float
    site: QuadrupolarSite


def boltzmann_populations(
    levels_hz: np.ndarray | Sequence[float],
    temperature_kelvin: float,
) -> np.ndarray:
    """Return normalized Boltzmann populations for energies supplied in Hz."""

    levels = np.asarray(levels_hz, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(levels)):
        raise ValueError("levels_hz must be finite")
    temperature = float(temperature_kelvin)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_kelvin must be positive and finite")
    if levels.size == 0:
        return np.array([], dtype=np.float64)
    shifted = levels - float(np.min(levels))
    exponent = -_PLANCK_J_S * shifted / (_BOLTZMANN_J_PER_K * temperature)
    weights = np.exp(exponent)
    return weights / np.sum(weights)


def _energy_manifolds(
    levels_hz: np.ndarray,
    *,
    degeneracy_tolerance_hz: float,
) -> tuple[tuple[int, ...], ...]:
    manifolds: list[list[int]] = []
    for index, level in enumerate(levels_hz):
        if not manifolds:
            manifolds.append([index])
            continue
        reference = float(np.mean(levels_hz[manifolds[-1]]))
        if abs(float(level) - reference) <= degeneracy_tolerance_hz:
            manifolds[-1].append(index)
        else:
            manifolds.append([index])
    return tuple(tuple(item) for item in manifolds)


def crossover_transitions_from_eigensystem(
    eigensystem: NQREigensystem,
    orientation: CrossoverOrientation,
    *,
    orientation_index: int = 0,
    temperature_kelvin: float = 300.0,
    degeneracy_tolerance_hz: float | None = None,
    coupling_tolerance: float = 1e-12,
) -> tuple[CrossoverTransition, ...]:
    """Return degeneracy-safe transition responses for one orientation."""

    levels = np.asarray(eigensystem.levels_hz, dtype=np.float64)
    vectors = np.asarray(eigensystem.eigenvectors, dtype=np.complex128)
    level_scale = max(float(np.ptp(levels)), float(np.max(np.abs(levels))), 1.0)
    if degeneracy_tolerance_hz is None:
        degeneracy_tolerance = max(1e-9, 1e-12 * level_scale)
    else:
        degeneracy_tolerance = float(degeneracy_tolerance_hz)
        if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0.0:
            raise ValueError(
                "degeneracy_tolerance_hz must be non-negative and finite"
            )
    coupling_tolerance = float(coupling_tolerance)
    if not np.isfinite(coupling_tolerance) or coupling_tolerance < 0.0:
        raise ValueError("coupling_tolerance must be non-negative and finite")

    populations = boltzmann_populations(levels, temperature_kelvin)
    manifolds = _energy_manifolds(
        levels,
        degeneracy_tolerance_hz=degeneracy_tolerance,
    )
    ops = spin_matrices(eigensystem.site.spin)
    operator_stack = (ops.ix, ops.iy, ops.iz)
    transmit = orientation.transmit_direction_pas
    receive = orientation.receive_direction_pas

    transitions: list[CrossoverTransition] = []
    for lower_index, lower_manifold in enumerate(manifolds):
        for upper_manifold in manifolds[lower_index + 1 :]:
            local_amplitude = 0.0j
            population_difference = 0.0
            excitation_strength = 0.0
            detection_strength = 0.0
            coupling_norm = 0.0
            for lower in lower_manifold:
                for upper in upper_manifold:
                    dipole = np.array(
                        [
                            vectors[:, lower].conj().T @ op @ vectors[:, upper]
                            for op in operator_stack
                        ],
                        dtype=np.complex128,
                    )
                    tx = complex(np.vdot(transmit, dipole))
                    rx = complex(np.vdot(receive, dipole))
                    coupling_norm += abs(tx) ** 2 + abs(rx) ** 2
                    delta_population = float(
                        populations[lower] - populations[upper]
                    )
                    population_difference += delta_population
                    excitation_strength += delta_population * abs(tx) ** 2
                    detection_strength += delta_population * abs(rx) ** 2
                    local_amplitude += delta_population * tx * np.conj(rx)
            if coupling_norm <= coupling_tolerance:
                continue
            lower_energy = float(np.mean(levels[list(lower_manifold)]))
            upper_energy = float(np.mean(levels[list(upper_manifold)]))
            amplitude = orientation.weight * local_amplitude
            transitions.append(
                CrossoverTransition(
                    orientation_index=int(orientation_index),
                    orientation_weight=float(orientation.weight),
                    b0_vector_tesla_pas=np.zeros(3, dtype=np.float64),
                    zeeman_to_quadrupole_ratio=0.0,
                    lower_levels=lower_manifold,
                    upper_levels=upper_manifold,
                    frequency_hz=upper_energy - lower_energy,
                    population_difference=population_difference,
                    excitation_strength=excitation_strength,
                    detection_strength=detection_strength,
                    local_amplitude=local_amplitude,
                    amplitude=amplitude,
                    intensity=float(abs(amplitude)),
                )
            )
    transitions.sort(key=lambda item: item.frequency_hz)
    return tuple(transitions)


def _normalize_crossover_orientations(
    orientations: CrossoverOrientationInput,
) -> tuple[CrossoverOrientation, ...]:
    if isinstance(orientations, str):
        if orientations == "single":
            return (CrossoverOrientation(),)
        if orientations == "powder":
            samples = b0_b1_powder_average_grid()
        else:
            raise ValueError("orientations string must be 'single' or 'powder'")
    else:
        supplied = tuple(orientations)
        if not supplied:
            raise ValueError("at least one orientation is required")
        if all(isinstance(item, CrossoverOrientation) for item in supplied):
            total = sum(item.weight for item in supplied)
            if total <= 0.0:
                raise ValueError("orientation weights must have positive sum")
            return tuple(replace(item, weight=item.weight / total) for item in supplied)
        if not all(isinstance(item, OrientationSample) for item in supplied):
            raise TypeError(
                "orientations must contain only CrossoverOrientation or "
                "OrientationSample values"
            )
        samples = normalize_orientations(supplied)

    return tuple(
        CrossoverOrientation(
            b0_direction_pas=(
                sample.b0_direction_pas
                if sample.b0_direction_pas is not None
                else sample.b1_direction_pas
            ),
            transmit_direction_pas=sample.b1_direction_pas,
            receive_direction_pas=sample.b1_direction_pas,
            weight=sample.weight,
        )
        for sample in samples
    )


def _line_profile(
    frequencies_hz: np.ndarray,
    center_hz: float,
    broadening_hz: float,
    lineshape: str,
) -> np.ndarray:
    offset = frequencies_hz - center_hz
    if lineshape == "gaussian":
        return np.exp(-0.5 * (offset / broadening_hz) ** 2)
    if lineshape == "lorentzian":
        return broadening_hz**2 / (offset**2 + broadening_hz**2)
    raise ValueError("lineshape must be 'gaussian' or 'lorentzian'")


def _maximum_overlap_assignment(overlap: np.ndarray) -> tuple[int, ...]:
    """Return the maximum-weight one-to-one column assignment for each row.

    Dynamic programming costs ``O(d 2**d)`` for a ``d x d`` overlap matrix,
    which is inexpensive for the small single-spin Hilbert spaces used here and
    avoids adding SciPy as a core dependency.
    """

    weights = np.asarray(overlap, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("overlap must be a square matrix")
    if not np.all(np.isfinite(weights)):
        raise ValueError("overlap must be finite")
    dimension = weights.shape[0]
    candidates: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(dimension):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (score, assignment) in candidates.items():
            for column in range(dimension):
                bit = 1 << column
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_score = score + float(weights[row, column])
                previous = updated.get(new_mask)
                if previous is None or new_score > previous[0]:
                    updated[new_mask] = (new_score, assignment + (column,))
        candidates = updated
    return candidates[(1 << dimension) - 1][1]


def track_crossover_field_sweep(
    site: QuadrupolarSite,
    b0_tesla: np.ndarray | Sequence[float],
    *,
    orientation: CrossoverOrientation | None = None,
    temperature_kelvin: float = 300.0,
    backend: str = "numpy",
) -> CrossoverFieldSweepResult:
    """Track eigenstate character and all state-pair responses over increasing B0.

    At each new field, the raw energy-ordered eigenvectors are assigned to the
    preceding tracked states by maximizing total squared overlap. Eigenvector
    phases are then aligned to make adjacent overlaps real and positive. Exact
    degeneracies still represent manifolds rather than uniquely labeled states;
    begin at a small nonzero field when individual branch identity is required.
    """

    fields = np.asarray(b0_tesla, dtype=np.float64).reshape(-1)
    if fields.size == 0 or not np.all(np.isfinite(fields)):
        raise ValueError("b0_tesla must contain finite values")
    if np.any(fields < 0.0) or np.any(np.diff(fields) <= 0.0):
        raise ValueError("b0_tesla must be non-negative and strictly increasing")
    temperature = float(temperature_kelvin)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_kelvin must be positive and finite")
    sample = CrossoverOrientation() if orientation is None else orientation

    b0_vectors = fields[:, np.newaxis] * sample.b0_direction_pas[np.newaxis, :]
    eigensystems = diagonalize_sites_over_b0(site, b0_vectors, backend=backend)
    dimension = site.dimension
    tracked_levels = np.empty((fields.size, dimension), dtype=np.float64)
    tracked_vectors = np.empty(
        (fields.size, dimension, dimension),
        dtype=np.complex128,
    )
    minimum_overlap = np.ones(fields.size, dtype=np.float64)

    for field_index, eigensystem in enumerate(eigensystems):
        levels = np.asarray(eigensystem.levels_hz, dtype=np.float64)
        vectors = np.asarray(eigensystem.eigenvectors, dtype=np.complex128).copy()
        if field_index > 0:
            previous = tracked_vectors[field_index - 1]
            overlap = np.abs(previous.conj().T @ vectors) ** 2
            assignment = _maximum_overlap_assignment(overlap)
            levels = levels[list(assignment)]
            vectors = vectors[:, list(assignment)]
            assigned_overlap = np.array(
                [overlap[row, column] for row, column in enumerate(assignment)]
            )
            minimum_overlap[field_index] = float(np.min(assigned_overlap))
            for state in range(dimension):
                adjacent = np.vdot(previous[:, state], vectors[:, state])
                if abs(adjacent) > 0.0:
                    vectors[:, state] *= np.exp(-1.0j * np.angle(adjacent))
        tracked_levels[field_index] = levels
        tracked_vectors[field_index] = vectors

    ops = spin_matrices(site.spin)
    operator_stack = (ops.ix, ops.iy, ops.iz)
    b0_operator = sum(
        component * operator
        for component, operator in zip(sample.b0_direction_pas, operator_stack)
    )
    magnetic_quantum = np.empty((fields.size, dimension), dtype=np.float64)
    state_pairs = tuple(
        (lower, upper)
        for lower in range(dimension)
        for upper in range(lower + 1, dimension)
    )
    frequencies = np.empty((fields.size, len(state_pairs)), dtype=np.float64)
    amplitudes = np.empty((fields.size, len(state_pairs)), dtype=np.complex128)

    for field_index in range(fields.size):
        levels = tracked_levels[field_index]
        vectors = tracked_vectors[field_index]
        populations = boltzmann_populations(levels, temperature)
        for state in range(dimension):
            vector = vectors[:, state]
            magnetic_quantum[field_index, state] = float(
                np.real(vector.conj().T @ b0_operator @ vector)
            )
        for pair_index, (first, second) in enumerate(state_pairs):
            if levels[first] <= levels[second]:
                lower, upper = first, second
            else:
                lower, upper = second, first
            dipole = np.array(
                [
                    vectors[:, lower].conj().T @ operator @ vectors[:, upper]
                    for operator in operator_stack
                ],
                dtype=np.complex128,
            )
            tx = complex(np.vdot(sample.transmit_direction_pas, dipole))
            rx = complex(np.vdot(sample.receive_direction_pas, dipole))
            population_difference = float(populations[lower] - populations[upper])
            frequencies[field_index, pair_index] = abs(
                levels[second] - levels[first]
            )
            amplitudes[field_index, pair_index] = (
                population_difference * tx * np.conj(rx)
            )

    ratios = abs(site.gamma_hz_per_t) * fields / site.quadrupole_frequency_hz
    return CrossoverFieldSweepResult(
        b0_tesla=fields,
        zeeman_to_quadrupole_ratio=ratios,
        levels_hz=tracked_levels,
        eigenvectors=tracked_vectors,
        minimum_state_overlap=minimum_overlap,
        magnetic_quantum_expectation=magnetic_quantum,
        state_pairs=state_pairs,
        transition_frequencies_hz=frequencies,
        transition_amplitudes=amplitudes,
        transition_intensities=np.abs(amplitudes),
        orientation=sample,
        temperature_kelvin=temperature,
        site=site,
    )


def simulate_crossover_spectrum(
    site: QuadrupolarSite,
    b0_tesla: float,
    *,
    orientations: CrossoverOrientationInput = "single",
    temperature_kelvin: float = 300.0,
    broadening_hz: float = 100.0,
    points: int = 2048,
    frequency_range_hz: tuple[float, float] | None = None,
    lineshape: str = "gaussian",
    normalize: bool = False,
    degeneracy_tolerance_hz: float | None = None,
    coupling_tolerance: float = 1e-12,
    backend: str = "numpy",
) -> CrossoverSpectrumResult:
    """Simulate every observable transition of ``H_Q + H_Z`` exactly.

    ``broadening_hz`` is the Gaussian standard deviation or Lorentzian HWHM.
    The returned complex spectrum retains the relative phase between distinct
    transmit and receive coils. Reciprocal coils produce a real, non-negative
    absorption spectrum.
    """

    b0 = float(b0_tesla)
    if not np.isfinite(b0) or b0 < 0.0:
        raise ValueError("b0_tesla must be finite and non-negative")
    temperature = float(temperature_kelvin)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_kelvin must be positive and finite")
    broadening = float(broadening_hz)
    if not np.isfinite(broadening) or broadening <= 0.0:
        raise ValueError("broadening_hz must be positive and finite")
    points = int(points)
    if points < 2:
        raise ValueError("points must be at least two")
    if lineshape not in {"gaussian", "lorentzian"}:
        raise ValueError("lineshape must be 'gaussian' or 'lorentzian'")

    samples = _normalize_crossover_orientations(orientations)
    b0_vectors = b0 * np.array(
        [sample.b0_direction_pas for sample in samples],
        dtype=np.float64,
    )
    eigensystems = diagonalize_sites_over_b0(site, b0_vectors, backend=backend)
    zeeman_frequency = abs(site.gamma_hz_per_t) * b0
    reference = float(site.quadrupole_frequency_hz)
    ratio = zeeman_frequency / reference

    transitions: list[CrossoverTransition] = []
    for orientation_index, (sample, eigensystem) in enumerate(
        zip(samples, eigensystems)
    ):
        local = crossover_transitions_from_eigensystem(
            eigensystem,
            sample,
            orientation_index=orientation_index,
            temperature_kelvin=temperature,
            degeneracy_tolerance_hz=degeneracy_tolerance_hz,
            coupling_tolerance=coupling_tolerance,
        )
        transitions.extend(
            replace(
                transition,
                b0_vector_tesla_pas=b0_vectors[orientation_index],
                zeeman_to_quadrupole_ratio=ratio,
            )
            for transition in local
        )

    if not transitions:
        raise ValueError("no observable transitions for the requested geometry")
    transitions.sort(key=lambda item: item.frequency_hz)

    if frequency_range_hz is None:
        minimum = min(item.frequency_hz for item in transitions) - 5.0 * broadening
        maximum = max(item.frequency_hz for item in transitions) + 5.0 * broadening
        minimum = max(0.0, minimum)
        if maximum <= minimum:
            maximum = minimum + 10.0 * broadening
    else:
        minimum, maximum = map(float, frequency_range_hz)
        if (
            not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or maximum <= minimum
        ):
            raise ValueError(
                "frequency_range_hz must contain finite increasing bounds"
            )

    frequencies = np.linspace(minimum, maximum, points)
    spectrum = np.zeros(points, dtype=np.complex128)
    for transition in transitions:
        spectrum += transition.amplitude * _line_profile(
            frequencies,
            transition.frequency_hz,
            broadening,
            lineshape,
        )
    normalized = bool(normalize)
    if normalized:
        scale = float(np.max(np.abs(spectrum)))
        if scale > 0.0:
            spectrum = spectrum / scale

    return CrossoverSpectrumResult(
        frequencies_hz=frequencies,
        spectrum=spectrum,
        transitions=tuple(transitions),
        b0_tesla=b0,
        zeeman_frequency_hz=zeeman_frequency,
        zeeman_to_quadrupole_ratio=ratio,
        quadrupole_reference_frequency_hz=reference,
        temperature_kelvin=temperature,
        broadening_hz=broadening,
        lineshape=lineshape,
        normalized=normalized,
        site=site,
    )
