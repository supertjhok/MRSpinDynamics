"""Electron spin echo envelope modulation (ESEEM) for an S=1/2 electron.

ESEEM measures the weak hyperfine coupling between an electron spin and a nearby
nucleus through the modulation it imprints on an electron spin echo. The
modulation arises only when the nuclear quantization axis differs between the two
electron manifolds -- from the *anisotropic* (pseudosecular) hyperfine term and,
for ``I >= 1``, from a nuclear quadrupole interaction.

The secular model (electron rotating frame, on resonance) is

    H = -omega_I I_z + S_z (A I_z + B I_x) + H_Q,

with nuclear Larmor frequency ``omega_I`` (``larmor_hz``), secular hyperfine
``A`` (``secular_hz``), pseudosecular hyperfine ``B`` (``pseudosecular_hz``), and
a nuclear quadrupole term ``H_Q`` for spin ``I >= 1`` (``quadrupole_hz`` ``nu_Q``
and asymmetry ``eta``; the quadrupole principal axis is taken collinear with the
static field). The nucleus may be ``I = 1/2`` (no quadrupole), ``I = 1``, or
``I = 3/2``.

For the special case ``I = 1/2`` the nuclear transition frequencies and
modulation depth have the closed forms

    nu_alpha = sqrt((omega_I + A/2)^2 + (B/2)^2),
    nu_beta  = sqrt((omega_I - A/2)^2 + (B/2)^2),
    k        = (omega_I B / (nu_alpha nu_beta))^2,

and :func:`two_pulse_eseem` / :func:`three_pulse_eseem` give the analytic
envelopes. For any spin, :func:`manifold_frequencies` returns the nuclear
transition frequencies in each electron manifold by diagonalization, and the
density-matrix engine (with electron coherence-pathway selection) simulates the
two- and three-pulse sequences. The density-matrix helpers are reused by
:mod:`spin_dynamics.esr.hyscore` and :mod:`spin_dynamics.esr.endor`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density, propagator
from spin_dynamics.esr.hamiltonians import TAU
from spin_dynamics.nqr.hamiltonians import quadrupole_hamiltonian
from spin_dynamics.nqr.operators import spin_dimension, spin_matrices, validate_spin
from spin_dynamics.nqr.systems import QuadrupolarSite


@dataclass(frozen=True)
class HyperfineCoupling:
    """Secular model of one S=1/2 electron coupled to one nucleus.

    Frequencies are in hertz: ``larmor_hz`` is the (positive) nuclear Larmor
    frequency, ``secular_hz`` the secular hyperfine ``A``, and
    ``pseudosecular_hz`` the pseudosecular hyperfine ``B``. For a quadrupolar
    nucleus (``nuclear_spin >= 1``) ``quadrupole_hz`` is the quadrupole frequency
    ``nu_Q`` and ``eta`` its asymmetry; the quadrupole principal axis is taken
    collinear with the static field. ``nuclear_spin`` defaults to ``1/2``.
    """

    larmor_hz: float
    secular_hz: float = 0.0
    pseudosecular_hz: float = 0.0
    nuclear_spin: float = 0.5
    quadrupole_hz: float = 0.0
    eta: float = 0.0

    def __post_init__(self) -> None:
        larmor = float(self.larmor_hz)
        secular = float(self.secular_hz)
        pseudosecular = float(self.pseudosecular_hz)
        quadrupole = float(self.quadrupole_hz)
        eta = float(self.eta)
        spin = validate_spin(self.nuclear_spin)
        if not np.isfinite([larmor, secular, pseudosecular, quadrupole, eta]).all():
            raise ValueError("coupling parameters must be finite")
        if larmor <= 0:
            raise ValueError("larmor_hz must be positive")
        if quadrupole < 0:
            raise ValueError("quadrupole_hz must be non-negative")
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        if np.isclose(spin, 0.5) and quadrupole != 0.0:
            raise ValueError("spin-1/2 nuclei have no quadrupole interaction")
        object.__setattr__(self, "larmor_hz", larmor)
        object.__setattr__(self, "secular_hz", secular)
        object.__setattr__(self, "pseudosecular_hz", pseudosecular)
        object.__setattr__(self, "nuclear_spin", spin)
        object.__setattr__(self, "quadrupole_hz", quadrupole)
        object.__setattr__(self, "eta", eta)

    @property
    def is_spin_half(self) -> bool:
        """Whether the nucleus is spin-1/2 (closed-form ESEEM available)."""

        return bool(np.isclose(self.nuclear_spin, 0.5))


# --- Spin-parameterized operators (electron (x) nucleus product basis) ----------


@dataclass(frozen=True)
class _Operators:
    nuclear_spin: float
    nucleus_dim: int
    sx: np.ndarray
    sy: np.ndarray
    sz: np.ndarray
    splus: np.ndarray
    iz: np.ndarray
    ix: np.ndarray
    szi_z: np.ndarray
    szi_x: np.ndarray
    electron_eye: np.ndarray
    nucleus_eye: np.ndarray
    thermal: np.ndarray
    coherence_order: np.ndarray


@lru_cache(maxsize=None)
def _operators(nuclear_spin: float) -> _Operators:
    electron = spin_matrices(0.5)
    nucleus = spin_matrices(nuclear_spin)
    dim_n = spin_dimension(nuclear_spin)
    eye_e = electron.identity
    eye_n = nucleus.identity
    m_e = np.array([0.5] * dim_n + [-0.5] * dim_n)
    return _Operators(
        nuclear_spin=float(nuclear_spin),
        nucleus_dim=dim_n,
        sx=np.kron(electron.ix, eye_n),
        sy=np.kron(electron.iy, eye_n),
        sz=np.kron(electron.iz, eye_n),
        splus=np.kron(electron.ix + 1j * electron.iy, eye_n),
        iz=np.kron(eye_e, nucleus.iz),
        ix=np.kron(eye_e, nucleus.ix),
        szi_z=np.kron(electron.iz, nucleus.iz),
        szi_x=np.kron(electron.iz, nucleus.ix),
        electron_eye=eye_e,
        nucleus_eye=eye_n,
        thermal=np.kron(electron.iz, eye_n),
        coherence_order=m_e[:, None] - m_e[None, :],
    )


def filter_electron_coherence(
    density: np.ndarray,
    order: int,
    *,
    nuclear_spin: float = 0.5,
) -> np.ndarray:
    """Keep only the requested electron coherence order of a density matrix."""

    coherence = _operators(nuclear_spin).coherence_order
    return np.where(np.isclose(coherence, order), density, 0.0)


def _quadrupole_nuclear_hamiltonian(coupling: HyperfineCoupling) -> np.ndarray | None:
    """Return the nucleus-space quadrupole Hamiltonian (rad/s), or None for I=1/2."""

    if coupling.is_spin_half or coupling.quadrupole_hz == 0.0:
        return None
    site = QuadrupolarSite(
        spin=coupling.nuclear_spin,
        quadrupole_frequency_hz=coupling.quadrupole_hz,
        eta=coupling.eta,
        gamma_hz_per_t=0.0,
    )
    return quadrupole_hamiltonian(site)


def electron_nuclear_hamiltonian(
    coupling: HyperfineCoupling,
    *,
    electron_offset_hz: float = 0.0,
) -> np.ndarray:
    """Return the secular electron-nuclear Hamiltonian in radians per second."""

    ops = _operators(coupling.nuclear_spin)
    hamiltonian = TAU * (
        -coupling.larmor_hz * ops.iz
        + coupling.secular_hz * ops.szi_z
        + coupling.pseudosecular_hz * ops.szi_x
        + float(electron_offset_hz) * ops.sz
    )
    quadrupole = _quadrupole_nuclear_hamiltonian(coupling)
    if quadrupole is not None:
        hamiltonian = hamiltonian + np.kron(ops.electron_eye, quadrupole)
    return hamiltonian


# --- Nuclear transition frequencies --------------------------------------------


def _manifold_transition_frequencies(coupling: HyperfineCoupling, m_s: float) -> np.ndarray:
    """Sorted positive nuclear transition frequencies in one electron manifold."""

    nucleus = spin_matrices(coupling.nuclear_spin)
    hamiltonian = TAU * (
        -coupling.larmor_hz * nucleus.iz
        + m_s * (coupling.secular_hz * nucleus.iz + coupling.pseudosecular_hz * nucleus.ix)
    )
    quadrupole = _quadrupole_nuclear_hamiltonian(coupling)
    if quadrupole is not None:
        hamiltonian = hamiltonian + quadrupole
    levels = np.sort(np.linalg.eigvalsh(hamiltonian) / TAU)
    diffs = [
        levels[j] - levels[i]
        for i in range(levels.size)
        for j in range(i + 1, levels.size)
    ]
    return np.sort(np.asarray(diffs, dtype=np.float64))


def manifold_frequencies(coupling: HyperfineCoupling) -> tuple[np.ndarray, np.ndarray]:
    """Return nuclear transition frequencies in the two electron manifolds.

    Returns ``(alpha, beta)`` for ``m_S = +1/2`` and ``m_S = -1/2`` respectively,
    each a sorted array of positive transition frequencies in hertz. This works
    for any nuclear spin (it diagonalizes each manifold's nuclear Hamiltonian) and
    is the general counterpart of :func:`nuclear_frequencies`.
    """

    return (
        _manifold_transition_frequencies(coupling, +0.5),
        _manifold_transition_frequencies(coupling, -0.5),
    )


def _require_spin_half(coupling: HyperfineCoupling, what: str) -> None:
    if not coupling.is_spin_half:
        raise ValueError(
            f"{what} has a closed form only for spin-1/2 nuclei; use "
            "manifold_frequencies and the density-matrix helpers for I >= 1"
        )


def nuclear_frequencies(coupling: HyperfineCoupling) -> tuple[float, float]:
    """Return the two spin-1/2 nuclear frequencies ``(nu_alpha, nu_beta)`` in Hz."""

    _require_spin_half(coupling, "nuclear_frequencies")
    half_a = 0.5 * coupling.secular_hz
    half_b = 0.5 * coupling.pseudosecular_hz
    nu_alpha = float(np.hypot(coupling.larmor_hz + half_a, half_b))
    nu_beta = float(np.hypot(coupling.larmor_hz - half_a, half_b))
    return nu_alpha, nu_beta


def modulation_depth(coupling: HyperfineCoupling) -> float:
    """Return the spin-1/2 ESEEM modulation-depth parameter ``k`` in ``[0, 1]``."""

    _require_spin_half(coupling, "modulation_depth")
    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    denom = nu_alpha * nu_beta
    if denom <= 0:
        return 0.0
    return float((coupling.larmor_hz * coupling.pseudosecular_hz / denom) ** 2)


# --- Analytic (spin-1/2) ESEEM traces ------------------------------------------


def _times(times_seconds) -> np.ndarray:
    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    return times


def two_pulse_eseem(times_seconds, coupling: HyperfineCoupling) -> np.ndarray:
    """Return the analytic spin-1/2 two-pulse ESEEM trace ``V(tau)``.

    ``V(tau) = 1 - (k/4)[2 - 2cos(w_a tau) - 2cos(w_b tau)
    + cos(w_+ tau) + cos(w_- tau)]`` with ``w_+- = w_a +- w_b``.
    """

    _require_spin_half(coupling, "two_pulse_eseem")
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
    """Return the analytic spin-1/2 three-pulse (stimulated-echo) trace ``V(T)``.

    ``V(T) = 1 - (k/4){(1 - cos w_a tau)(1 - cos w_b (tau + T))
    + (1 - cos w_b tau)(1 - cos w_a (tau + T))}`` at fixed ``tau``.
    """

    _require_spin_half(coupling, "three_pulse_eseem")
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


# --- Density-matrix ESEEM (any nuclear spin) -----------------------------------


def _ideal_pulse(
    angle_rad: float,
    axis: str = "y",
    *,
    nuclear_spin: float = 0.5,
) -> np.ndarray:
    ops = _operators(nuclear_spin)
    operator = {"x": ops.sx, "y": ops.sy}[axis]
    return propagator(operator, float(angle_rad))


def _phased_pulse(
    angle_rad: float,
    phase_rad: float,
    *,
    nuclear_spin: float = 0.5,
) -> np.ndarray:
    """Return an electron pulse about an axis at ``phase_rad`` in the xy plane."""

    ops = _operators(nuclear_spin)
    axis = np.cos(phase_rad) * ops.sx + np.sin(phase_rad) * ops.sy
    return propagator(axis, float(angle_rad))


def _no_hyperfine(coupling: HyperfineCoupling) -> HyperfineCoupling:
    """Coupling with the hyperfine removed (the modulation-free reference)."""

    return replace(coupling, secular_hz=0.0, pseudosecular_hz=0.0)


def two_pulse_eseem_quantum(
    times_seconds,
    coupling: HyperfineCoupling,
    *,
    electron_offset_hz: float = 0.0,
) -> np.ndarray:
    """Density-matrix two-pulse ESEEM, normalized to ``V(0) = 1``.

    Propagates the electron-nuclear Hamiltonian through an ideal
    ``pi/2 - tau - pi - tau`` echo and reads the refocused electron coherence.
    Works for any nuclear spin; for spin-1/2 it reproduces :func:`two_pulse_eseem`.
    """

    spin = coupling.nuclear_spin
    ops = _operators(spin)
    tau = _times(times_seconds)
    hamiltonian = electron_nuclear_hamiltonian(
        coupling, electron_offset_hz=electron_offset_hz
    )
    excite = _ideal_pulse(np.pi / 2.0, "y", nuclear_spin=spin)
    refocus = _ideal_pulse(np.pi, "x", nuclear_spin=spin)
    rho0 = excite @ ops.thermal @ excite.conj().T
    out = np.empty(tau.size, dtype=np.float64)
    for idx, value in enumerate(tau):
        rho = evolve_density(rho0, hamiltonian, float(value))
        rho = refocus @ rho @ refocus.conj().T
        rho = evolve_density(rho, hamiltonian, float(value))
        out[idx] = float(np.real(np.trace(rho @ ops.sx)))
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
    stimulated echo, and normalizes by the hyperfine-free amplitude. Works for any
    nuclear spin; for spin-1/2 it reproduces :func:`three_pulse_eseem`.
    """

    spin = coupling.nuclear_spin
    ops = _operators(spin)
    t = _times(times_seconds)
    tau = float(tau_seconds)
    if tau < 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be non-negative and finite")
    excite = _ideal_pulse(np.pi / 2.0, "y", nuclear_spin=spin)

    def amplitudes(local: HyperfineCoupling) -> np.ndarray:
        hamiltonian = electron_nuclear_hamiltonian(local)
        prepared = filter_electron_coherence(
            excite @ ops.thermal @ excite.conj().T, +1, nuclear_spin=spin
        )
        prepared = evolve_density(prepared, hamiltonian, tau)
        prepared = filter_electron_coherence(
            excite @ prepared @ excite.conj().T, 0, nuclear_spin=spin
        )
        out = np.empty(t.size, dtype=np.float64)
        for idx, value in enumerate(t):
            rho = evolve_density(prepared, hamiltonian, float(value))
            rho = filter_electron_coherence(
                excite @ rho @ excite.conj().T, -1, nuclear_spin=spin
            )
            rho = evolve_density(rho, hamiltonian, tau)
            out[idx] = float(np.real(np.trace(rho @ ops.splus)))
        return out

    signal = amplitudes(coupling)
    reference = float(np.mean(amplitudes(_no_hyperfine(coupling))))
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

    The experimental counterpart of the coherence-order projection used by
    :func:`three_pulse_eseem_quantum`. It reuses the package phase-cycle
    machinery: :func:`spin_dynamics.phase_cycling.eseem_stimulated_echo_phase_cycle`
    builds the ``n_phase``-per-pulse scan table that selects the stimulated-echo
    pathway ``dp = (+1, -1, -1)``, each branch is simulated by stepping the three
    pulse phases, and :meth:`PhaseCycle.combine` performs the receiver-weighted
    combination. Because the electron coherence order spans only ``{-1, 0, +1}``,
    ``n_phase = 4`` already isolates the pathway exactly, and the result matches
    the coherence-order-filtered simulation to numerical precision.
    """

    from spin_dynamics.phase_cycling import eseem_stimulated_echo_phase_cycle

    spin = coupling.nuclear_spin
    ops = _operators(spin)
    t = _times(times_seconds)
    tau = float(tau_seconds)
    if tau < 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be non-negative and finite")

    cycle = eseem_stimulated_echo_phase_cycle(n_phase)
    phases1 = cycle.pulse_phases("excitation_90")
    phases2 = cycle.pulse_phases("store_90")
    phases3 = cycle.pulse_phases("read_90")

    def branch_signals(local: HyperfineCoupling) -> list[np.ndarray]:
        hamiltonian = electron_nuclear_hamiltonian(local)
        signals: list[np.ndarray] = []
        for phi1, phi2, phi3 in zip(phases1, phases2, phases3):
            pulse1 = _phased_pulse(np.pi / 2.0, phi1, nuclear_spin=spin)
            pulse2 = _phased_pulse(np.pi / 2.0, phi2, nuclear_spin=spin)
            pulse3 = _phased_pulse(np.pi / 2.0, phi3, nuclear_spin=spin)
            prepared = evolve_density(pulse1 @ ops.thermal @ pulse1.conj().T, hamiltonian, tau)
            prepared = pulse2 @ prepared @ pulse2.conj().T
            branch = np.empty(t.size, dtype=np.complex128)
            for idx, value in enumerate(t):
                rho = evolve_density(prepared, hamiltonian, float(value))
                rho = pulse3 @ rho @ pulse3.conj().T
                rho = evolve_density(rho, hamiltonian, tau)
                branch[idx] = np.trace(rho @ ops.splus)
            signals.append(branch)
        return signals

    signal = cycle.combine(branch_signals(coupling))
    # Divide by the complex unmodulated echo so its (axis-dependent) phase
    # cancels, leaving the real modulation.
    reference = complex(np.mean(cycle.combine(branch_signals(_no_hyperfine(coupling)))))
    if reference == 0:
        raise ValueError("degenerate echo reference; check coupling parameters")
    return np.real(signal / reference)
