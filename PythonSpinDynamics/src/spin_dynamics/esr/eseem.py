"""Electron spin echo envelope modulation (ESEEM) for an S=1/2, I=1/2 pair.

ESEEM measures the weak hyperfine coupling between an electron spin and a nearby
nucleus through the modulation it imprints on an electron spin echo. The
modulation arises only from the *anisotropic* (pseudosecular) part of the
hyperfine coupling, which makes the nuclear quantization axis differ between the
two electron manifolds.

For the canonical secular model of an S=1/2 electron coupled to an I=1/2 nucleus
(electron rotating frame, on resonance),

    H = -omega_I I_z + A S_z I_z + B S_z I_x,

with nuclear Larmor frequency ``omega_I`` (``larmor_hz``), secular hyperfine
``A`` (``secular_hz``), and pseudosecular hyperfine ``B`` (``pseudosecular_hz``).
The two nuclear (ENDOR) transition frequencies are

    nu_alpha = sqrt((omega_I + A/2)^2 + (B/2)^2),
    nu_beta  = sqrt((omega_I - A/2)^2 + (B/2)^2),

and the modulation depth is ``k = (omega_I B / (nu_alpha nu_beta))^2``.

This module provides the analytic two- and three-pulse ESEEM expressions, the
ESEEM frequency spectrum, and an independent density-matrix engine (with
electron coherence-pathway selection) that reproduces the analytic traces to
floating-point precision. The density-matrix helpers are reused by
:mod:`spin_dynamics.esr.hyscore` and :mod:`spin_dynamics.esr.endor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density, propagator
from spin_dynamics.esr.hamiltonians import TAU
from spin_dynamics.nqr.operators import spin_matrices


@dataclass(frozen=True)
class HyperfineCoupling:
    """Secular model of one electron (S=1/2) coupled to one nucleus (I=1/2).

    All three parameters are in hertz: ``larmor_hz`` is the (positive) nuclear
    Larmor frequency, ``secular_hz`` is the secular hyperfine ``A``, and
    ``pseudosecular_hz`` is the pseudosecular hyperfine ``B``.
    """

    larmor_hz: float
    secular_hz: float = 0.0
    pseudosecular_hz: float = 0.0

    def __post_init__(self) -> None:
        larmor = float(self.larmor_hz)
        secular = float(self.secular_hz)
        pseudosecular = float(self.pseudosecular_hz)
        if not np.isfinite([larmor, secular, pseudosecular]).all():
            raise ValueError("coupling parameters must be finite")
        if larmor <= 0:
            raise ValueError("larmor_hz must be positive")
        object.__setattr__(self, "larmor_hz", larmor)
        object.__setattr__(self, "secular_hz", secular)
        object.__setattr__(self, "pseudosecular_hz", pseudosecular)


# --- Operators (electron (x) nucleus product basis, ordered |m_S, m_I>) --------

_SM = spin_matrices(0.5)
_SX, _SY, _SZ, _EYE = _SM.ix, _SM.iy, _SM.iz, _SM.identity
SX = np.kron(_SX, _EYE)
SY = np.kron(_SY, _EYE)
SZ = np.kron(_SZ, _EYE)
SPLUS = np.kron(_SX + 1j * _SY, _EYE)
IZ = np.kron(_EYE, _SZ)
IX = np.kron(_EYE, _SX)
_SZIZ = np.kron(_SZ, _SZ)
_SZIX = np.kron(_SZ, _SX)

# Electron coherence order of each density-matrix element (m_S difference).
_M_E = np.array([0.5, 0.5, -0.5, -0.5])
_COHERENCE_ORDER = _M_E[:, None] - _M_E[None, :]


def filter_electron_coherence(density: np.ndarray, order: int) -> np.ndarray:
    """Keep only the requested electron coherence order of a 4x4 density matrix."""

    return np.where(np.isclose(_COHERENCE_ORDER, order), density, 0.0)


def electron_nuclear_hamiltonian(
    coupling: HyperfineCoupling,
    *,
    electron_offset_hz: float = 0.0,
) -> np.ndarray:
    """Return the secular electron-nuclear Hamiltonian in radians per second."""

    return TAU * (
        -coupling.larmor_hz * IZ
        + coupling.secular_hz * _SZIZ
        + coupling.pseudosecular_hz * _SZIX
        + float(electron_offset_hz) * SZ
    )


# --- Analytic frequencies and modulation depth ---------------------------------


def nuclear_frequencies(coupling: HyperfineCoupling) -> tuple[float, float]:
    """Return the two nuclear transition frequencies ``(nu_alpha, nu_beta)`` in Hz."""

    half_a = 0.5 * coupling.secular_hz
    half_b = 0.5 * coupling.pseudosecular_hz
    nu_alpha = float(np.hypot(coupling.larmor_hz + half_a, half_b))
    nu_beta = float(np.hypot(coupling.larmor_hz - half_a, half_b))
    return nu_alpha, nu_beta


def modulation_depth(coupling: HyperfineCoupling) -> float:
    """Return the ESEEM modulation-depth parameter ``k`` in ``[0, 1]``."""

    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    denom = nu_alpha * nu_beta
    if denom <= 0:
        return 0.0
    return float((coupling.larmor_hz * coupling.pseudosecular_hz / denom) ** 2)


# --- Analytic ESEEM traces -----------------------------------------------------


def _times(times_seconds) -> np.ndarray:
    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    return times


def two_pulse_eseem(times_seconds, coupling: HyperfineCoupling) -> np.ndarray:
    """Return the analytic two-pulse ESEEM trace ``V(tau)``.

    ``V(tau) = 1 - (k/4)[2 - 2cos(w_a tau) - 2cos(w_b tau)
    + cos(w_+ tau) + cos(w_- tau)]`` with ``w_+- = w_a +- w_b``.
    """

    tau = _times(times_seconds)
    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    k = modulation_depth(coupling)
    wa, wb = TAU * nu_alpha, TAU * nu_beta
    return 1.0 - (k / 4.0) * (
        2.0
        - 2.0 * np.cos(wa * tau)
        - 2.0 * np.cos(wb * tau)
        + np.cos((wa + wb) * tau)
        + np.cos((wa - wb) * tau)
    )


def three_pulse_eseem(
    times_seconds,
    coupling: HyperfineCoupling,
    *,
    tau_seconds: float,
) -> np.ndarray:
    """Return the analytic three-pulse (stimulated-echo) ESEEM trace ``V(T)``.

    ``V(T) = 1 - (k/4){(1 - cos w_a tau)(1 - cos w_b (tau + T))
    + (1 - cos w_b tau)(1 - cos w_a (tau + T))}`` at fixed ``tau``.
    """

    t = _times(times_seconds)
    tau = float(tau_seconds)
    if tau < 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be non-negative and finite")
    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    k = modulation_depth(coupling)
    wa, wb = TAU * nu_alpha, TAU * nu_beta
    term_a = (1.0 - np.cos(wa * tau)) * (1.0 - np.cos(wb * (tau + t)))
    term_b = (1.0 - np.cos(wb * tau)) * (1.0 - np.cos(wa * (tau + t)))
    return 1.0 - (k / 4.0) * (term_a + term_b)


def eseem_spectrum(
    times_seconds,
    signal,
    *,
    zero_fill: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ESEEM frequency spectrum ``(frequencies_hz, magnitude)``.

    The mean is removed before transforming so the unmodulated baseline does not
    dominate the zero-frequency bin; peaks then mark the nuclear frequencies.
    """

    times = _times(times_seconds)
    if times.size < 2:
        raise ValueError("times must contain at least two points")
    dt = float(times[1] - times[0])
    if dt <= 0 or not np.allclose(np.diff(times), dt):
        raise ValueError("times must be uniformly increasing")
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size != times.size:
        raise ValueError("signal must match times")
    if int(zero_fill) < 1:
        raise ValueError("zero_fill must be at least 1")
    centered = values - float(np.mean(values))
    n_fft = int(zero_fill) * times.size
    spectrum = np.abs(np.fft.rfft(centered, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, d=dt)
    return frequencies, spectrum


# --- Density-matrix ESEEM (independent validation) -----------------------------


def _ideal_pulse(angle_rad: float, axis: str = "y") -> np.ndarray:
    operator = {"x": SX, "y": SY}[axis]
    return propagator(operator, float(angle_rad))


def _phased_pulse(angle_rad: float, phase_rad: float) -> np.ndarray:
    """Return an electron pulse about an axis at ``phase_rad`` in the xy plane."""

    axis = np.cos(phase_rad) * SX + np.sin(phase_rad) * SY
    return propagator(axis, float(angle_rad))


def two_pulse_eseem_quantum(
    times_seconds,
    coupling: HyperfineCoupling,
    *,
    electron_offset_hz: float = 0.0,
) -> np.ndarray:
    """Density-matrix two-pulse ESEEM, normalized to ``V(0) = 1``.

    Independent check on :func:`two_pulse_eseem`: it propagates the electron-
    nuclear Hamiltonian through an ideal ``pi/2 - tau - pi - tau`` echo and reads
    the refocused electron coherence.
    """

    tau = _times(times_seconds)
    hamiltonian = electron_nuclear_hamiltonian(
        coupling, electron_offset_hz=electron_offset_hz
    )
    excite = _ideal_pulse(np.pi / 2.0, "y")
    refocus = _ideal_pulse(np.pi, "x")
    rho0 = excite @ np.kron(_SZ, _EYE) @ excite.conj().T
    out = np.empty(tau.size, dtype=np.float64)
    for idx, value in enumerate(tau):
        rho = evolve_density(rho0, hamiltonian, float(value))
        rho = refocus @ rho @ refocus.conj().T
        rho = evolve_density(rho, hamiltonian, float(value))
        out[idx] = float(np.real(np.trace(rho @ SX)))
    reference = out[0]
    if reference == 0:
        raise ValueError("degenerate echo reference; check coupling parameters")
    return out / reference


def three_pulse_eseem_quantum(
    times_seconds,
    coupling: HyperfineCoupling,
    *,
    tau_seconds: float,
) -> np.ndarray:
    """Density-matrix three-pulse ESEEM, normalized to the unmodulated echo.

    Uses electron coherence-pathway selection (``+1 -> 0 -> -1``) to isolate the
    stimulated echo, and normalizes by the pseudosecular-free amplitude so the
    result matches :func:`three_pulse_eseem`.
    """

    t = _times(times_seconds)
    tau = float(tau_seconds)
    if tau < 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be non-negative and finite")

    def amplitudes(pseudosecular_hz: float) -> np.ndarray:
        local = HyperfineCoupling(
            larmor_hz=coupling.larmor_hz,
            secular_hz=coupling.secular_hz,
            pseudosecular_hz=pseudosecular_hz,
        )
        hamiltonian = electron_nuclear_hamiltonian(local)
        excite = _ideal_pulse(np.pi / 2.0, "y")
        rho0 = np.kron(_SZ, _EYE)
        prepared = filter_electron_coherence(excite @ rho0 @ excite.conj().T, +1)
        prepared = evolve_density(prepared, hamiltonian, tau)
        prepared = filter_electron_coherence(
            excite @ prepared @ excite.conj().T, 0
        )
        out = np.empty(t.size, dtype=np.float64)
        for idx, value in enumerate(t):
            rho = evolve_density(prepared, hamiltonian, float(value))
            rho = filter_electron_coherence(excite @ rho @ excite.conj().T, -1)
            rho = evolve_density(rho, hamiltonian, tau)
            out[idx] = float(np.real(np.trace(rho @ SPLUS)))
        return out

    signal = amplitudes(coupling.pseudosecular_hz)
    reference = float(np.mean(amplitudes(0.0)))
    if reference == 0:
        raise ValueError("degenerate echo reference; check coupling parameters")
    return signal / reference


def three_pulse_eseem_phase_cycled(
    times_seconds,
    coupling: HyperfineCoupling,
    *,
    tau_seconds: float,
    n_phase: int = 4,
) -> np.ndarray:
    """Three-pulse ESEEM selected by an explicit phase cycle.

    This is the experimental counterpart of the coherence-order projection used
    by :func:`three_pulse_eseem_quantum`. It reuses the package phase-cycle
    machinery: :func:`spin_dynamics.phase_cycling.eseem_stimulated_echo_phase_cycle`
    builds the ``n_phase``-per-pulse scan table that selects the stimulated-echo
    pathway ``dp = (+1, -1, -1)``, each branch is simulated by stepping the three
    pulse phases, and :meth:`PhaseCycle.combine` performs the receiver-weighted
    combination. Because the electron coherence order spans only ``{-1, 0, +1}``,
    ``n_phase = 4`` already isolates the pathway exactly, and the result matches
    the coherence-order-filtered simulation to numerical precision -- a direct
    check that the filtering stands in for a phase cycle.
    """

    from spin_dynamics.phase_cycling import eseem_stimulated_echo_phase_cycle

    t = _times(times_seconds)
    tau = float(tau_seconds)
    if tau < 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be non-negative and finite")

    cycle = eseem_stimulated_echo_phase_cycle(n_phase)
    phases1 = cycle.pulse_phases("excitation_90")
    phases2 = cycle.pulse_phases("store_90")
    phases3 = cycle.pulse_phases("read_90")

    def branch_signals(pseudosecular_hz: float) -> list[np.ndarray]:
        local = HyperfineCoupling(
            larmor_hz=coupling.larmor_hz,
            secular_hz=coupling.secular_hz,
            pseudosecular_hz=pseudosecular_hz,
        )
        hamiltonian = electron_nuclear_hamiltonian(local)
        rho0 = np.kron(_SZ, _EYE)
        signals: list[np.ndarray] = []
        for phi1, phi2, phi3 in zip(phases1, phases2, phases3):
            pulse1 = _phased_pulse(np.pi / 2.0, phi1)
            pulse2 = _phased_pulse(np.pi / 2.0, phi2)
            pulse3 = _phased_pulse(np.pi / 2.0, phi3)
            prepared = evolve_density(pulse1 @ rho0 @ pulse1.conj().T, hamiltonian, tau)
            prepared = pulse2 @ prepared @ pulse2.conj().T
            branch = np.empty(t.size, dtype=np.complex128)
            for idx, value in enumerate(t):
                rho = evolve_density(prepared, hamiltonian, float(value))
                rho = pulse3 @ rho @ pulse3.conj().T
                rho = evolve_density(rho, hamiltonian, tau)
                branch[idx] = np.trace(rho @ SPLUS)
            signals.append(branch)
        return signals

    signal = cycle.combine(branch_signals(coupling.pseudosecular_hz))
    # Divide by the complex unmodulated echo so its (axis-dependent) phase
    # cancels, leaving the real modulation.
    reference = complex(np.mean(cycle.combine(branch_signals(0.0))))
    if reference == 0:
        raise ValueError("degenerate echo reference; check coupling parameters")
    return np.real(signal / reference)
