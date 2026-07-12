"""Pulsed crossover sequences with exact RF and static-field relaxation."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import os

import numpy as np

from spin_dynamics.coupling.evolution import propagator
from spin_dynamics.nqr.field_relaxation import field_dependent_equilibrium
from spin_dynamics.nqr.floquet import simulate_floquet_rf
from spin_dynamics.nqr.full_dynamics import (
    detection_operator,
    pulse_hamiltonian,
    rotating_indices,
)
from spin_dynamics.nqr.hamiltonians import TAU, diagonalize_site, nqr_hamiltonian
from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.orientations import (
    OrientationSample,
    b0_b1_powder_average_grid,
    normalize_orientations,
    powder_average_grid,
)
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
    acquisition_offsets_seconds: np.ndarray | None
    echo_waveforms: np.ndarray | None
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


@dataclass(frozen=True)
class PowderCrossoverSLSEResult:
    """Orientation-averaged exact-pulse SLSE echo train."""

    echo_times_seconds: np.ndarray
    echo_amplitudes: np.ndarray
    local_echo_amplitudes: np.ndarray
    acquisition_offsets_seconds: np.ndarray | None
    unfiltered_echo_waveforms: np.ndarray | None
    echo_waveforms: np.ndarray | None
    matched_echo_amplitudes: np.ndarray | None
    prefix_matched_echo_amplitudes: np.ndarray | None
    orientation_weights: np.ndarray
    rf_frequency_hz: float
    b0_tesla: float
    zeeman_to_quadrupole_ratio: float
    local_results: tuple[CrossoverSLSEResult, ...]
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


def _powder_carrier_hz(site, field: float, samples, nutation_hz: float) -> float:
    frequencies: list[float] = []
    weights: list[float] = []
    for sample in samples:
        b0 = (
            np.zeros(3, dtype=np.float64)
            if sample.b0_direction_pas is None
            else field * sample.b0_direction_pas
        )
        eigensystem = diagonalize_site(site, b0)
        reference = max(
            site.quadrupole_frequency_hz,
            abs(site.gamma_hz_per_t) * field,
        )
        candidates = [
            transition
            for transition in eigensystem.transitions
            if transition.frequency_hz >= 0.5 * reference
        ]
        if not candidates:
            candidates = list(eigensystem.transitions)
        for transition in candidates:
            coupling = abs(
                np.vdot(sample.b1_direction_pas, transition.dipole_vector)
            )
            frequencies.append(float(transition.frequency_hz))
            weights.append(float(sample.weight) * coupling**2)
    if sum(weights) <= 0.0:
        raise ValueError("powder grid has no RF-active target transitions")
    frequencies_array = np.asarray(frequencies)
    weights_array = np.asarray(weights)
    bandwidth = max(2.0 * abs(float(nutation_hz)), 0.005 * site.quadrupole_frequency_hz)
    lower = float(np.min(frequencies_array)) - 0.5 * bandwidth
    upper = float(np.max(frequencies_array)) + 0.5 * bandwidth
    bin_count = max(1, int(np.ceil((upper - lower) / (0.25 * bandwidth))))
    histogram, edges = np.histogram(
        frequencies_array,
        bins=bin_count,
        range=(lower, upper),
        weights=weights_array,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = max(1, int(np.ceil(4.0 * bandwidth / (edges[1] - edges[0]))))
    offsets = np.arange(-radius, radius + 1) * (edges[1] - edges[0])
    smoothing = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    smoothed_full = np.convolve(histogram, smoothing, mode="full")
    start = (smoothing.size - 1) // 2
    smoothed = smoothed_full[start : start + histogram.size]
    peak_frequency = float(centers[int(np.argmax(smoothed))])
    local_weights = weights_array * np.exp(
        -0.5 * ((frequencies_array - peak_frequency) / bandwidth) ** 2
    )
    return float(np.average(frequencies_array, weights=local_weights))


def powder_carrier_frequency_hz(
    site: QuadrupolarSite,
    b0_tesla: float,
    orientations: Sequence[OrientationSample],
    *,
    nutation_hz: float,
) -> float:
    """Choose one intensity-weighted carrier from a powder orientation set."""

    field = float(b0_tesla)
    if field < 0.0 or not np.isfinite(field):
        raise ValueError("b0_tesla must be non-negative and finite")
    samples = normalize_orientations(tuple(orientations))
    return _powder_carrier_hz(site, field, samples, nutation_hz)


def select_powder_frequency_slice(
    site: QuadrupolarSite,
    b0_tesla: float,
    orientations: Sequence[OrientationSample],
    *,
    carrier_frequency_hz: float,
    half_width_hz: float,
) -> tuple[tuple[OrientationSample, ...], float]:
    """Select crystallites with an RF-active transition in a receiver slice.

    The returned orientations are renormalized within the selected spectral
    slice. The second return value is the fraction of the full powder weight
    retained before renormalization.
    """

    field = float(b0_tesla)
    carrier = float(carrier_frequency_hz)
    half_width = float(half_width_hz)
    if field <= 0.0 or not np.isfinite(field):
        raise ValueError("b0_tesla must be positive and finite")
    if carrier <= 0.0 or not np.isfinite(carrier):
        raise ValueError("carrier_frequency_hz must be positive and finite")
    if half_width <= 0.0 or not np.isfinite(half_width):
        raise ValueError("half_width_hz must be positive and finite")
    samples = normalize_orientations(tuple(orientations))
    selected: list[OrientationSample] = []
    retained_weight = 0.0
    for sample in samples:
        if sample.b0_direction_pas is None:
            raise ValueError("frequency-slice orientations require B0 directions")
        eigensystem = diagonalize_site(site, field * sample.b0_direction_pas)
        active = any(
            abs(transition.frequency_hz - carrier) <= half_width
            and abs(np.vdot(sample.b1_direction_pas, transition.dipole_vector))
            > 1.0e-10
            for transition in eigensystem.transitions
        )
        if active:
            selected.append(sample)
            retained_weight += sample.weight
    if not selected:
        raise ValueError("no crystallites lie in the requested frequency slice")
    return normalize_orientations(selected), float(retained_weight)


def matched_filter_echo_waveforms(
    coherent_waveforms: np.ndarray,
    acquisition_offsets_seconds: Sequence[float] | np.ndarray,
    *,
    receiver_bandwidth_hz: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter coherent echo waveforms and project them onto the first echo."""

    waveforms = np.asarray(coherent_waveforms, dtype=np.complex128)
    offsets = np.asarray(acquisition_offsets_seconds, dtype=np.float64).reshape(-1)
    if waveforms.ndim != 2 or waveforms.shape[1] != offsets.size:
        raise ValueError("coherent_waveforms must have shape (echoes, offsets)")
    if offsets.size < 3 or np.any(np.diff(offsets) <= 0.0):
        raise ValueError("acquisition offsets must be strictly increasing")
    bandwidth = float(receiver_bandwidth_hz)
    if bandwidth <= 0.0 or np.isnan(bandwidth):
        raise ValueError("receiver_bandwidth_hz must be positive")
    filtered = waveforms.copy()
    if np.isfinite(bandwidth):
        dt = float(offsets[1] - offsets[0])
        if bandwidth > 1.0 / dt:
            raise ValueError(
                "receiver bandwidth exceeds the acquisition sampling rate"
            )
        frequencies = np.fft.fftfreq(offsets.size, d=dt)
        response = np.exp(
            -0.5 * np.log(2.0) * (2.0 * frequencies / bandwidth) ** 2
        )
        filtered = np.fft.ifft(
            np.fft.fft(filtered, axis=1) * response[np.newaxis, :], axis=1
        )
    template = filtered[0]
    template_energy = float(np.vdot(template, template).real)
    if template_energy <= 0.0:
        raise ValueError("the first echo has zero matched-filter energy")
    amplitudes = filtered @ template.conj() / template_energy
    return filtered, amplitudes


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
    detection_mode: str = "baseband",
    floquet_sidebands: int = 4,
    initial_density: np.ndarray | None = None,
    acquisition_offsets_seconds: Sequence[float] | np.ndarray | None = None,
    pulse_model: str = "floquet",
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
    model = str(pulse_model).lower()
    if model not in {"floquet", "rwa"}:
        raise ValueError("pulse_model must be 'floquet' or 'rwa'")
    eigensystem = diagonalize_site(site, b0)
    vectors = eigensystem.eigenvectors
    winding = rotating_indices(eigensystem.levels_hz, carrier)

    def rotating_pulse_to_pas(rotating_pulse, start_time, duration):
        start_rotation = np.diag(
            np.exp(-1.0j * TAU * carrier * winding * start_time)
        )
        end_rotation = np.diag(
            np.exp(-1.0j * TAU * carrier * winding * (start_time + duration))
        )
        return (
            vectors
            @ end_rotation
            @ rotating_pulse
            @ start_rotation.conj().T
            @ vectors.conj().T
        )

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

    rwa_refocus = None
    if model == "floquet":
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
        excitation_unitarity_error = excite_result.unitarity_error
    else:
        rwa_excite = propagator(
            pulse_hamiltonian(
                eigensystem,
                nutation_hz=nutation,
                rf_frequency_hz=carrier,
                phase=excitation_phase_radians,
                b1_direction_pas=b1,
            ),
            excitation_duration,
        )
        rwa_refocus = propagator(
            pulse_hamiltonian(
                eigensystem,
                nutation_hz=nutation,
                rf_frequency_hz=carrier,
                phase=refocus_phase_radians,
                b1_direction_pas=b1,
            ),
            refocus_duration,
        )
        excite = rotating_pulse_to_pas(rwa_excite, 0.0, excitation_duration)
        excitation_unitarity_error = float(
            np.linalg.norm(excite.conj().T @ excite - np.eye(site.dimension))
        )
    hamiltonian = nqr_hamiltonian(site, b0)
    free_generator = liouville_hamiltonian(hamiltonian) + relaxation.superoperator(
        hamiltonian
    )
    free = matrix_exponential(free_generator, free_half)
    acquisition_offsets = None
    acquisition_propagators = None
    if acquisition_offsets_seconds is not None:
        acquisition_offsets = np.asarray(
            acquisition_offsets_seconds, dtype=np.float64
        ).reshape(-1)
        if acquisition_offsets.size < 2:
            raise ValueError("acquisition_offsets_seconds must contain at least two points")
        if not np.all(np.isfinite(acquisition_offsets)) or np.any(
            np.diff(acquisition_offsets) <= 0.0
        ):
            raise ValueError("acquisition offsets must be finite and strictly increasing")
        if acquisition_offsets[0] < -free_half or acquisition_offsets[-1] > free_half:
            raise ValueError("acquisition offsets must lie between adjacent RF pulses")
        generator_values, generator_vectors = np.linalg.eig(free_generator)
        generator_inverse = np.linalg.inv(generator_vectors)
        acquisition_propagators = tuple(
            (generator_vectors * np.exp(generator_values * (free_half + float(offset))))
            @ generator_inverse
            for offset in acquisition_offsets
        )

    if detection_mode not in {"baseband", "spin_quadrature"}:
        raise ValueError("detection_mode must be 'baseband' or 'spin_quadrature'")
    if detection_mode == "baseband":
        detector_eigen = detection_operator(eigensystem, carrier, b1)

        def detected_signal(state, time_seconds):
            state_eigen = vectors.conj().T @ state @ vectors
            rotation = np.diag(
                np.exp(-1.0j * TAU * carrier * winding * time_seconds)
            )
            state_rotating = rotation.conj().T @ state_eigen @ rotation
            return complex(np.trace(state_rotating @ detector_eigen))

    else:
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

        def detected_signal(state, time_seconds):
            del time_seconds
            return complex(np.trace(state @ detector))

    baseline = detected_signal(equilibrium.density_matrix_pas, 0.0)

    density = excite @ density @ excite.conj().T
    echoes = np.empty(echo_count, dtype=np.complex128)
    refocus_unitarity_errors = np.empty(echo_count, dtype=np.float64)
    densities = np.empty(
        (echo_count, site.dimension, site.dimension),
        dtype=np.complex128,
    )
    waveforms = (
        None
        if acquisition_offsets is None
        else np.empty((echo_count, acquisition_offsets.size), dtype=np.complex128)
    )
    for index in range(echo_count):
        vector = free @ density.reshape(-1, order="F")
        density = vector.reshape((site.dimension, site.dimension), order="F")
        pulse_start_time = excitation_duration + free_half + index * echo_spacing
        if model == "floquet":
            refocus_result = simulate_floquet_rf(
                site,
                b0,
                nutation_hz=nutation,
                rf_frequency_hz=carrier,
                pulse_duration_seconds=refocus_duration,
                phase_radians=(
                    refocus_phase_radians + TAU * carrier * pulse_start_time
                ),
                b1_direction_pas=b1,
                sidebands=floquet_sidebands,
                initial_density=density,
                temperature_kelvin=temperature,
            )
            refocus = refocus_result.physical_propagator
            refocus_unitarity_errors[index] = refocus_result.unitarity_error
        else:
            if rwa_refocus is None:
                raise AssertionError("RWA refocusing propagator was not initialized")
            refocus = rotating_pulse_to_pas(
                rwa_refocus, pulse_start_time, refocus_duration
            )
            refocus_unitarity_errors[index] = float(
                np.linalg.norm(refocus.conj().T @ refocus - np.eye(site.dimension))
            )
        density = refocus @ density @ refocus.conj().T
        if waveforms is not None and acquisition_propagators is not None:
            post_pulse_vector = density.reshape(-1, order="F")
            for sample_index, (offset, sample_propagator) in enumerate(
                zip(acquisition_offsets, acquisition_propagators, strict=True)
            ):
                sample_density = (sample_propagator @ post_pulse_vector).reshape(
                    (site.dimension, site.dimension), order="F"
                )
                sample_time = (
                    pulse_start_time
                    + refocus_duration
                    + free_half
                    + float(offset)
                )
                waveforms[index, sample_index] = (
                    detected_signal(sample_density, sample_time) - baseline
                )
        vector = free @ density.reshape(-1, order="F")
        density = vector.reshape((site.dimension, site.dimension), order="F")
        densities[index] = density
        echo_time = excitation_duration + (index + 1.0) * echo_spacing
        echoes[index] = detected_signal(density, echo_time) - baseline

    ratio = (
        abs(site.gamma_hz_per_t)
        * float(np.linalg.norm(b0))
        / site.quadrupole_frequency_hz
    )
    return CrossoverSLSEResult(
        echo_times_seconds=(np.arange(echo_count) + 1.0) * echo_spacing,
        echo_amplitudes=echoes,
        density_matrices_pas=densities,
        acquisition_offsets_seconds=acquisition_offsets,
        echo_waveforms=waveforms,
        rf_frequency_hz=carrier,
        nutation_hz=nutation,
        b0_vector_tesla_pas=b0,
        zeeman_to_quadrupole_ratio=ratio,
        equilibrium_density_pas=equilibrium.density_matrix_pas,
        equilibrium_signal=baseline,
        excitation_pulse_unitarity_error=excitation_unitarity_error,
        refocus_pulse_unitarity_error=float(np.max(refocus_unitarity_errors)),
        relaxation_model=relaxation,
        site=site,
    )


def _simulate_crossover_orientation_chunk(task):
    site, field, samples, relaxation, simulation_kwargs, compact = task
    results = tuple(
        simulate_crossover_slse(
            site,
            (
                np.zeros(3, dtype=np.float64)
                if sample.b0_direction_pas is None
                else field * sample.b0_direction_pas
            ),
            relaxation=relaxation,
            b1_direction_pas=sample.b1_direction_pas,
            **simulation_kwargs,
        )
        for sample in samples
    )
    if not compact:
        return results
    echoes = np.asarray([result.echo_amplitudes for result in results])
    waveforms = (
        None
        if results[0].echo_waveforms is None
        else np.asarray([result.echo_waveforms for result in results])
    )
    return echoes, waveforms


def simulate_crossover_slse_powder(
    site: QuadrupolarSite,
    b0_tesla: float,
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
    n_theta: int = 3,
    n_phi: int = 6,
    n_chi: int = 4,
    b1_b0_angle_radians: float = np.pi / 2.0,
    floquet_sidebands: int = 4,
    acquisition_duration_seconds: float | None = None,
    acquisition_points: int = 129,
    receiver_bandwidth_hz: float = np.inf,
    orientations: Sequence[OrientationSample] | None = None,
    pulse_model: str = "floquet",
    num_workers: int | None = 1,
    parallel_backend: str = "thread",
    retain_local_results: bool = True,
) -> PowderCrossoverSLSEResult:
    """Powder-average an exact-pulse SLSE train using one global carrier.

    At zero field the physical RF direction is integrated on the sphere, exactly
    matching :func:`powder_average_grid`; the redundant static-field frame is
    not sampled. For nonzero fields the correlated laboratory
    ``B0``/linear-``B1`` geometry is sampled over SO(3). If no carrier is
    supplied, one intensity-weighted center frequency is chosen for the complete
    powder and then held fixed for every crystallite.
    """

    field = float(b0_tesla)
    if field < 0.0 or not np.isfinite(field):
        raise ValueError("b0_tesla must be non-negative and finite")
    # At zero field only the RF direction exists physically. Sampling an
    # arbitrary B0 frame and its redundant chi angle gives a needlessly noisy
    # quadrature on coarse grids and does not reduce to the established NQR
    # powder average. Once B0 is non-zero, the correlated lab B0/B1 frame does
    # require the full SO(3) grid.
    if orientations is not None:
        samples = normalize_orientations(tuple(orientations))
        if field > 0.0 and any(
            sample.b0_direction_pas is None for sample in samples
        ):
            raise ValueError("nonzero-field powder orientations require B0 directions")
    else:
        samples = (
            powder_average_grid(n_theta, n_phi)
            if field == 0.0
            else b0_b1_powder_average_grid(
                n_theta,
                n_phi,
                n_chi,
                b1_b0_angle=b1_b0_angle_radians,
            )
        )
    carrier = (
        _powder_carrier_hz(site, field, samples, nutation_hz)
        if rf_frequency_hz is None
        else float(rf_frequency_hz)
    )
    acquisition_offsets = None
    if acquisition_duration_seconds is not None:
        acquisition_duration = float(acquisition_duration_seconds)
        points = int(acquisition_points)
        available = float(echo_spacing_seconds) - float(refocus_duration_seconds)
        if acquisition_duration <= 0.0 or acquisition_duration > available:
            raise ValueError(
                "acquisition_duration_seconds must be positive and fit between pulses"
            )
        if points < 3 or points % 2 == 0:
            raise ValueError("acquisition_points must be an odd integer of at least three")
        acquisition_offsets = np.linspace(
            -0.5 * acquisition_duration,
            0.5 * acquisition_duration,
            points,
        )
    workers = None if num_workers is None else int(num_workers)
    if workers is not None and workers <= 0:
        raise ValueError("num_workers must be positive or None")
    backend = str(parallel_backend).lower()
    if backend not in {"thread", "process"}:
        raise ValueError("parallel_backend must be 'thread' or 'process'")
    simulation_kwargs = dict(
        nutation_hz=nutation_hz,
        excitation_duration_seconds=excitation_duration_seconds,
        refocus_duration_seconds=refocus_duration_seconds,
        echo_spacing_seconds=echo_spacing_seconds,
        num_echoes=num_echoes,
        rf_frequency_hz=carrier,
        excitation_phase_radians=excitation_phase_radians,
        refocus_phase_radians=refocus_phase_radians,
        detection_mode="baseband",
        floquet_sidebands=floquet_sidebands,
        acquisition_offsets_seconds=acquisition_offsets,
        pulse_model=pulse_model,
    )
    effective_workers = workers or max(1, os.cpu_count() or 1)
    chunk_size = max(1, int(np.ceil(len(samples) / (4 * effective_workers))))
    sample_chunks = tuple(
        samples[index : index + chunk_size]
        for index in range(0, len(samples), chunk_size)
    )
    tasks = tuple(
        (site, field, chunk, relaxation, simulation_kwargs, not retain_local_results)
        for chunk in sample_chunks
    )
    if workers == 1:
        chunk_results = tuple(map(_simulate_crossover_orientation_chunk, tasks))
    else:
        executor_type = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        with executor_type(max_workers=workers) as executor:
            chunk_results = tuple(
                executor.map(_simulate_crossover_orientation_chunk, tasks)
            )
    if retain_local_results:
        local = tuple(result for chunk in chunk_results for result in chunk)
        local_echoes = np.asarray(
            [result.echo_amplitudes for result in local], dtype=np.complex128
        )
        local_waveforms = (
            None
            if acquisition_offsets is None
            else np.asarray(
                [result.echo_waveforms for result in local], dtype=np.complex128
            )
        )
    else:
        local = ()
        local_echoes = np.concatenate(
            [chunk[0] for chunk in chunk_results], axis=0
        )
        local_waveforms = (
            None
            if acquisition_offsets is None
            else np.concatenate([chunk[1] for chunk in chunk_results], axis=0)
        )
    weights = np.array([sample.weight for sample in samples], dtype=np.float64)
    coherent_waveforms = None
    unfiltered_waveforms = None
    matched_amplitudes = None
    prefix_matched_amplitudes = None
    if acquisition_offsets is not None:
        unfiltered_waveforms = np.tensordot(
            weights, local_waveforms, axes=(0, 0)
        )
        coherent_waveforms, matched_amplitudes = matched_filter_echo_waveforms(
            unfiltered_waveforms,
            acquisition_offsets,
            receiver_bandwidth_hz=receiver_bandwidth_hz,
        )
        prefix_count = max(1, len(samples) // 2)
        prefix_weights = weights[:prefix_count]
        prefix_weights = prefix_weights / np.sum(prefix_weights)
        prefix_waveform = np.tensordot(
            prefix_weights, local_waveforms[:prefix_count], axes=(0, 0)
        )
        _, prefix_matched_amplitudes = matched_filter_echo_waveforms(
            prefix_waveform,
            acquisition_offsets,
            receiver_bandwidth_hz=receiver_bandwidth_hz,
        )
    ratio = abs(site.gamma_hz_per_t) * field / site.quadrupole_frequency_hz
    return PowderCrossoverSLSEResult(
        echo_times_seconds=(np.arange(int(num_echoes)) + 1.0)
        * float(echo_spacing_seconds),
        echo_amplitudes=weights @ local_echoes,
        local_echo_amplitudes=local_echoes,
        acquisition_offsets_seconds=acquisition_offsets,
        unfiltered_echo_waveforms=unfiltered_waveforms,
        echo_waveforms=coherent_waveforms,
        matched_echo_amplitudes=matched_amplitudes,
        prefix_matched_echo_amplitudes=prefix_matched_amplitudes,
        orientation_weights=weights,
        rf_frequency_hz=carrier,
        b0_tesla=field,
        zeeman_to_quadrupole_ratio=ratio,
        local_results=local,
        site=site,
    )
