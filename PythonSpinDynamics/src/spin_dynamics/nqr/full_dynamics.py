"""Full ``(2I+1)``-level density-matrix NQR dynamics.

The reduced ``SelectivePulse`` path embeds one transition as a fictitious
spin-1/2 rotation, which is only honest when that transition is isolated (see
``docs/python_api/nqr.md`` and the technical note
``References/Pulsed_NQR_Spin_Dynamics_Narrative_Rewrite``). This module keeps the
*entire* energy-level structure and propagates the full density matrix, which is
the correct general model and is required for spin-3/2 -- whose single zero-field
line connects two Kramers doublets, i.e. four states.

Conventions and scope:

* A single rotating frame at the pulse carrier ``rf_frequency_hz`` with the
  rotating-wave approximation (RWA) is used. Each eigenlevel is assigned an
  integer winding number ``n_i = round((nu_i - min nu) / nu_rf)``; the RWA keeps
  RF couplings between levels with ``|n_a - n_b| = 1`` and drops the
  counter-rotating (same-band) terms. This is valid when one carrier addresses
  one transition band -- the spin-3/2 zero-field and weak-Zeeman regime. It is
  *not* a general multi-band higher-spin solver.
* ``nutation_hz`` here is the **bare field nutation** ``gamma * B1 / (2 pi)`` (a
  field property), *not* the per-transition Rabi rate used by the reduced
  ``SelectivePulse``. The realized Rabi rate on a transition ``a-b`` is
  ``2 * nutation_hz * |<a| e1 . I |b>|``, so the matrix element makes the same
  pulse a different flip angle on different transitions -- the physical effect
  the full model is meant to capture.
* Signals are returned demodulated at the carrier (baseband), as a real-time
  acquisition would record them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density, propagator
from spin_dynamics.nqr.hamiltonians import TAU, diagonalize_site
from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.orientations import (
    OrientationSample,
    normalize_orientations,
    powder_average_grid,
)
from spin_dynamics.nqr.relaxation import (
    NQRRelaxationModel,
    liouville_superoperator,
    matrix_exponential,
    propagate_density_liouville,
)
from spin_dynamics.nqr.simulation import equilibrium_density
from spin_dynamics.nqr.systems import NQREigensystem, QuadrupolarSite


OrientationInput = str | tuple[OrientationSample, ...] | list[OrientationSample]


def _as_orientations(orientations: OrientationInput) -> tuple[OrientationSample, ...]:
    if isinstance(orientations, str):
        if orientations == "powder":
            return powder_average_grid()
        if orientations == "single":
            return normalize_orientations(
                [OrientationSample(b1_direction_pas=(1.0, 0.0, 0.0))]
            )
        raise ValueError("orientations string must be 'powder' or 'single'")
    return normalize_orientations(tuple(orientations))


def _orientation_b0_vector(
    orientation: OrientationSample, b0_tesla: float
) -> np.ndarray | None:
    if b0_tesla == 0:
        return None
    direction = orientation.b0_direction_pas
    if direction is None:
        direction = orientation.b1_direction_pas
    return float(b0_tesla) * direction


def _unit(direction) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("direction must be a finite non-zero vector")
    return vec / norm


def rf_operator_eigenbasis(
    eigensystem: NQREigensystem,
    direction,
) -> np.ndarray:
    """Return ``e1 . I`` for unit direction ``e1`` in the energy eigenbasis."""

    ops = spin_matrices(eigensystem.site.spin)
    e1 = _unit(direction)
    lab = e1[0] * ops.ix + e1[1] * ops.iy + e1[2] * ops.iz
    vectors = eigensystem.eigenvectors
    return vectors.conj().T @ lab @ vectors


def rotating_indices(levels_hz: np.ndarray, rf_frequency_hz: float) -> np.ndarray:
    """Return RWA winding numbers ``round((nu_i - min nu) / nu_rf)`` per level."""

    levels_hz = np.asarray(levels_hz, dtype=np.float64).reshape(-1)
    rf_frequency_hz = float(rf_frequency_hz)
    if rf_frequency_hz <= 0:
        raise ValueError("rf_frequency_hz must be positive")
    return np.round((levels_hz - levels_hz.min()) / rf_frequency_hz).astype(np.int64)


def static_hamiltonian_rotating(
    eigensystem: NQREigensystem,
    rf_frequency_hz: float,
) -> np.ndarray:
    """Return the rotating-frame static Hamiltonian (rad/s) in the eigenbasis.

    Diagonal entries are ``2 pi (nu_i - nu_rf * n_i)``; an addressed coherence
    then evolves at ``2 pi (nu_ab - nu_rf)`` -- the detuning -- so on resonance
    it is stationary in this frame.
    """

    indices = rotating_indices(eigensystem.levels_hz, rf_frequency_hz)
    diagonal = TAU * (eigensystem.levels_hz - float(rf_frequency_hz) * indices)
    return np.diag(diagonal).astype(np.complex128)


def pulse_hamiltonian(
    eigensystem: NQREigensystem,
    *,
    nutation_hz: float,
    rf_frequency_hz: float,
    phase: float = 0.0,
    b1_direction_pas=(1.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the rotating-frame RWA pulse Hamiltonian (rad/s) in the eigenbasis."""

    indices = rotating_indices(eigensystem.levels_hz, rf_frequency_hz)
    hamiltonian = static_hamiltonian_rotating(eigensystem, rf_frequency_hz)
    rf_operator = rf_operator_eigenbasis(eigensystem, b1_direction_pas)
    amplitude = TAU * float(nutation_hz)
    phase = float(phase)
    delta = indices[:, None] - indices[None, :]
    # Co-rotating couplings only: a above b (delta=+1) carries exp(-i phi),
    # b above a (delta=-1) carries exp(+i phi); Hermiticity is preserved.
    coupling = np.zeros_like(rf_operator)
    upper = delta == 1
    lower = delta == -1
    coupling[upper] = -amplitude * rf_operator[upper] * np.exp(-1j * phase)
    coupling[lower] = -amplitude * rf_operator[lower] * np.exp(1j * phase)
    return hamiltonian + coupling


def detection_operator(
    eigensystem: NQREigensystem,
    rf_frequency_hz: float,
    rx_direction_pas=(1.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the baseband receive observable ``M`` with ``s = Tr(rho M)``.

    Picks the addressed-band lowering coherences (``n_a - n_b = 1``) weighted by
    the receive matrix element, so ``Tr(rho M)`` is the demodulated complex
    signal radiated near the carrier.
    """

    indices = rotating_indices(eigensystem.levels_hz, rf_frequency_hz)
    rx_operator = rf_operator_eigenbasis(eigensystem, rx_direction_pas)
    detector = np.zeros_like(rx_operator)
    delta = indices[:, None] - indices[None, :]
    raising = delta == 1  # rho[a, b] with a above b
    # s = sum_{a,b} rho[a,b] * rx[b,a]; encode rx[b,a] at detector[b,a].
    detector.T[raising] = rx_operator.T[raising]
    return detector


def _propagate(
    density: np.ndarray,
    hamiltonian: np.ndarray,
    duration: float,
    relaxation: NQRRelaxationModel | None,
) -> np.ndarray:
    if duration <= 0:
        return density
    if relaxation is None:
        return evolve_density(density, hamiltonian, duration)
    return propagate_density_liouville(
        density, hamiltonian, duration, relaxation=relaxation
    )


def _default_carrier_hz(eigensystem: NQREigensystem) -> float:
    if not eigensystem.transitions:
        raise ValueError("site has no RF-active transitions")
    strongest = max(eigensystem.transitions, key=lambda t: t.strength)
    return float(strongest.frequency_hz)


@dataclass(frozen=True)
class FullNQRFIDResult:
    """Complex baseband FID from the full density-matrix model."""

    times_seconds: np.ndarray
    signal: np.ndarray
    rf_frequency_hz: float
    eigensystem: NQREigensystem


@dataclass(frozen=True)
class FullNQREchoResult:
    """Complex baseband echo from a full density-matrix two-pulse sequence."""

    times_seconds: np.ndarray
    signal: np.ndarray
    rf_frequency_hz: float
    eigensystem: NQREigensystem


def simulate_full_fid(
    site: QuadrupolarSite,
    *,
    nutation_hz: float,
    pulse_duration_seconds: float,
    times_seconds: np.ndarray,
    rf_frequency_hz: float | None = None,
    phase: float = 0.0,
    b1_direction_pas=(1.0, 0.0, 0.0),
    rx_direction_pas=None,
    b0_vector_tesla_pas=None,
    relaxation: NQRRelaxationModel | None = None,
    initial_density: np.ndarray | None = None,
) -> FullNQRFIDResult:
    """Simulate a single-pulse full density-matrix NQR FID.

    Works for any supported spin (it is the required model for spin-3/2). The
    sample is excited by one rectangular pulse and the demodulated transverse
    signal is sampled at ``times_seconds`` (measured from the end of the pulse).
    """

    eigensystem = diagonalize_site(site, b0_vector_tesla_pas)
    carrier = _default_carrier_hz(eigensystem) if rf_frequency_hz is None else float(
        rf_frequency_hz
    )
    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times_seconds must not be empty")
    if np.any(np.diff(times) < 0):
        raise ValueError("times_seconds must be non-decreasing")

    rho = (
        equilibrium_density(eigensystem.levels_hz)
        if initial_density is None
        else np.asarray(initial_density, dtype=np.complex128).copy()
    )
    pulse = pulse_hamiltonian(
        eigensystem,
        nutation_hz=nutation_hz,
        rf_frequency_hz=carrier,
        phase=phase,
        b1_direction_pas=b1_direction_pas,
    )
    rho = _propagate(rho, pulse, float(pulse_duration_seconds), relaxation)

    free = static_hamiltonian_rotating(eigensystem, carrier)
    detector = detection_operator(
        eigensystem,
        carrier,
        b1_direction_pas if rx_direction_pas is None else rx_direction_pas,
    )

    signal = np.empty(times.size, dtype=np.complex128)
    current = 0.0
    for idx, sample_time in enumerate(times):
        rho = _propagate(rho, free, float(sample_time) - current, relaxation)
        current = float(sample_time)
        signal[idx] = np.trace(rho @ detector)
    return FullNQRFIDResult(
        times_seconds=times,
        signal=signal,
        rf_frequency_hz=carrier,
        eigensystem=eigensystem,
    )


def simulate_full_echo(
    site: QuadrupolarSite,
    *,
    nutation_hz: float,
    excitation_duration_seconds: float,
    refocus_duration_seconds: float,
    echo_spacing_seconds: float,
    times_seconds: np.ndarray,
    rf_frequency_hz: float | None = None,
    excitation_phase: float = 0.0,
    refocus_phase: float = np.pi / 2,
    b1_direction_pas=(1.0, 0.0, 0.0),
    rx_direction_pas=None,
    b0_vector_tesla_pas=None,
    relaxation: NQRRelaxationModel | None = None,
) -> FullNQREchoResult:
    """Simulate a full density-matrix two-pulse (Hahn-style) NQR echo.

    Sequence: excitation pulse, free evolution ``echo_spacing/2``, refocusing
    pulse, then acquisition sampled at ``times_seconds`` measured from the end of
    the refocusing pulse (so the echo centre sits near ``echo_spacing/2``).
    """

    eigensystem = diagonalize_site(site, b0_vector_tesla_pas)
    carrier = _default_carrier_hz(eigensystem) if rf_frequency_hz is None else float(
        rf_frequency_hz
    )
    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times_seconds must not be empty")
    half = 0.5 * float(echo_spacing_seconds)
    if half <= 0:
        raise ValueError("echo_spacing_seconds must be positive")

    rho = equilibrium_density(eigensystem.levels_hz)
    free = static_hamiltonian_rotating(eigensystem, carrier)
    excite = pulse_hamiltonian(
        eigensystem, nutation_hz=nutation_hz, rf_frequency_hz=carrier,
        phase=excitation_phase, b1_direction_pas=b1_direction_pas,
    )
    refocus = pulse_hamiltonian(
        eigensystem, nutation_hz=nutation_hz, rf_frequency_hz=carrier,
        phase=refocus_phase, b1_direction_pas=b1_direction_pas,
    )
    rho = _propagate(rho, excite, float(excitation_duration_seconds), relaxation)
    rho = _propagate(rho, free, half, relaxation)
    rho = _propagate(rho, refocus, float(refocus_duration_seconds), relaxation)

    detector = detection_operator(
        eigensystem,
        carrier,
        b1_direction_pas if rx_direction_pas is None else rx_direction_pas,
    )
    signal = np.empty(times.size, dtype=np.complex128)
    current = 0.0
    for idx, sample_time in enumerate(times):
        rho = _propagate(rho, free, float(sample_time) - current, relaxation)
        current = float(sample_time)
        signal[idx] = np.trace(rho @ detector)
    return FullNQREchoResult(
        times_seconds=times,
        signal=signal,
        rf_frequency_hz=carrier,
        eigensystem=eigensystem,
    )


@dataclass(frozen=True)
class FullNQRSLSEResult:
    """Spin-lock spin-echo (SLSE) train from the full density-matrix model.

    ``echo_amplitudes`` is the orientation-weighted complex echo amplitude per
    cycle; ``local_echo_amplitudes`` holds the per-orientation trains and
    ``orientation_weights`` their normalized weights. The model is the full
    ``(2I+1)``-level propagation, so it is valid for the spin-3/2 zero-field line
    that connects two Kramers doublets.
    """

    echo_times: np.ndarray
    echo_amplitudes: np.ndarray
    local_echo_amplitudes: np.ndarray
    orientation_weights: np.ndarray
    rf_frequency_hz: float
    eigensystem: NQREigensystem


def _step_propagators(
    eigensystem: NQREigensystem,
    *,
    carrier: float,
    nutation_hz: float,
    excitation_phase: float,
    refocus_phase: float,
    b1_direction_pas,
    excitation_duration: float,
    refocus_duration: float,
    free_half: float,
    relaxation: NQRRelaxationModel | None,
):
    """Pre-compute the excite/refocus/free propagators for one orientation."""

    excite_h = pulse_hamiltonian(
        eigensystem, nutation_hz=nutation_hz, rf_frequency_hz=carrier,
        phase=excitation_phase, b1_direction_pas=b1_direction_pas,
    )
    refocus_h = pulse_hamiltonian(
        eigensystem, nutation_hz=nutation_hz, rf_frequency_hz=carrier,
        phase=refocus_phase, b1_direction_pas=b1_direction_pas,
    )
    free_h = static_hamiltonian_rotating(eigensystem, carrier)
    if relaxation is None:
        return (
            "unitary",
            propagator(excite_h, excitation_duration),
            propagator(refocus_h, refocus_duration),
            propagator(free_h, free_half) if free_half > 0 else None,
        )
    return (
        "liouville",
        matrix_exponential(liouville_superoperator(excite_h, relaxation), excitation_duration),
        matrix_exponential(liouville_superoperator(refocus_h, relaxation), refocus_duration),
        (
            matrix_exponential(liouville_superoperator(free_h, relaxation), free_half)
            if free_half > 0
            else None
        ),
    )


def _apply_step(kind: str, propagator_matrix, density: np.ndarray) -> np.ndarray:
    if propagator_matrix is None:
        return density
    if kind == "unitary":
        return propagator_matrix @ density @ propagator_matrix.conj().T
    vector = density.reshape(-1, order="F")
    return (propagator_matrix @ vector).reshape(density.shape, order="F")


def simulate_full_slse(
    site: QuadrupolarSite,
    *,
    nutation_hz: float,
    excitation_duration_seconds: float,
    refocus_duration_seconds: float,
    echo_spacing_seconds: float,
    num_echoes: int,
    rf_frequency_hz: float | None = None,
    excitation_phase: float = 0.0,
    refocus_phase: float = np.pi / 2.0,
    orientations: OrientationInput = "single",
    b0_tesla: float = 0.0,
    b1_direction_pas=(1.0, 0.0, 0.0),
    rx_direction_pas=None,
    relaxation: NQRRelaxationModel | None = None,
    t2e_seconds: float = np.inf,
) -> FullNQRSLSEResult:
    """Simulate a full density-matrix SLSE echo train (valid for spin-3/2).

    The sequence is an excitation pulse followed by ``num_echoes`` cycles of
    ``[free echo_spacing/2 - refocus/2, refocus pulse, free echo_spacing/2 -
    refocus/2]``, sampling the demodulated coherence at each echo centre. This is
    the spin-lock spin-echo detection train used for chlorine-style spin-3/2 NQR;
    the full ``(2I+1)`` propagation handles the degenerate Kramers doublets that
    the embedded two-level ``simulate_slse`` cannot.

    Pass ``orientations="powder"`` for a powder average and ``b0_tesla > 0`` for a
    weak Zeeman perturbation (the static field direction follows each sample's
    ``b0_direction_pas``, defaulting to its RF direction). ``relaxation`` applies
    Liouville-space ``T1``/``T2``; ``t2e_seconds`` is an optional phenomenological
    envelope. The carrier defaults to the zero-field line of the strongest
    transition.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    excitation_duration = float(excitation_duration_seconds)
    refocus_duration = float(refocus_duration_seconds)
    echo_spacing = float(echo_spacing_seconds)
    if excitation_duration < 0 or refocus_duration < 0:
        raise ValueError("pulse durations must be non-negative")
    if echo_spacing < refocus_duration:
        raise ValueError("echo_spacing_seconds must be at least refocus_duration")
    t2e_seconds = float(t2e_seconds)
    if t2e_seconds <= 0:
        raise ValueError("t2e_seconds must be positive")
    if relaxation is not None and np.isfinite(t2e_seconds):
        warnings.warn(
            "both a finite t2e_seconds envelope and a Liouville relaxation model "
            "were given; their T2 damping composes multiplicatively. Use "
            "t2e_seconds=inf with a relaxation model to avoid double counting.",
            RuntimeWarning,
            stacklevel=2,
        )
    free_half = 0.5 * (echo_spacing - refocus_duration)

    samples = _as_orientations(orientations)
    carrier = (
        _default_carrier_hz(diagonalize_site(site))
        if rf_frequency_hz is None
        else float(rf_frequency_hz)
    )
    echo_times = (np.arange(num_echoes, dtype=np.float64) + 1.0) * echo_spacing
    envelope = (
        np.exp(-echo_times / t2e_seconds) if np.isfinite(t2e_seconds) else 1.0
    )

    local: list[np.ndarray] = []
    first_eigensystem: NQREigensystem | None = None
    for sample in samples:
        eigensystem = diagonalize_site(site, _orientation_b0_vector(sample, b0_tesla))
        kind, excite_u, refocus_u, free_u = _step_propagators(
            eigensystem,
            carrier=carrier,
            nutation_hz=nutation_hz,
            excitation_phase=excitation_phase,
            refocus_phase=refocus_phase,
            b1_direction_pas=sample.b1_direction_pas,
            excitation_duration=excitation_duration,
            refocus_duration=refocus_duration,
            free_half=free_half,
            relaxation=relaxation,
        )
        detector = detection_operator(
            eigensystem,
            carrier,
            sample.b1_direction_pas if rx_direction_pas is None else rx_direction_pas,
        )
        rho = equilibrium_density(eigensystem.levels_hz)
        rho = _apply_step(kind, excite_u, rho)
        echoes = np.empty(num_echoes, dtype=np.complex128)
        for echo_idx in range(num_echoes):
            rho = _apply_step(kind, free_u, rho)
            rho = _apply_step(kind, refocus_u, rho)
            rho = _apply_step(kind, free_u, rho)
            echoes[echo_idx] = np.trace(rho @ detector)
        local.append(echoes * envelope)
        if first_eigensystem is None:
            first_eigensystem = eigensystem

    weights = np.array([sample.weight for sample in samples], dtype=np.float64)
    local_echoes = np.asarray(local, dtype=np.complex128)
    averaged = weights @ local_echoes
    if first_eigensystem is None:
        raise AssertionError("orientation validation should prevent empty samples")
    return FullNQRSLSEResult(
        echo_times=echo_times,
        echo_amplitudes=averaged,
        local_echo_amplitudes=local_echoes,
        orientation_weights=weights,
        rf_frequency_hz=carrier,
        eigensystem=first_eigensystem,
    )
