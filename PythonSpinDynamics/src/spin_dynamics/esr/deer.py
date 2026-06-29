"""Four-pulse DEER/PELDOR forward model and distance-distribution recovery.

DEER (double electron-electron resonance, also PELDOR) measures the dipolar
coupling between two electron spins and therefore the nanometre distance between
them. A pump pulse on the partner ("B") spin inverts the local dipolar field felt
by the observed ("A") spin, modulating the observer echo at the dipolar frequency
``nu_dd(r, theta)``. Summed over an isotropic orientation distribution the result
is the classic dipolar (Pake) evolution kernel.

This module provides

* the single-orientation and powder-averaged dipolar kernel,
* a forward model that turns a distance distribution ``P(r)`` into a DEER form
  factor / trace,
* the dipolar (Pake) spectrum of a trace,
* Tikhonov-regularized recovery of ``P(r)`` from a trace, reusing the inverse
  machinery in :mod:`spin_dynamics.analysis.regularization`, and
* an independent two-electron density-matrix simulation of the full four-pulse
  sequence used to validate the analytic kernel from the spin Hamiltonian.

The form-factor convention is ``F(t) = (1 - lambda) + lambda * <cos(omega_dd t)>``
with modulation depth ``lambda`` (the inverted fraction of partner spins) and the
intramolecular average taken over the orientation distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density, propagator
from spin_dynamics.esr.dipolar import (
    FREE_ELECTRON_G,
    dipolar_angular_frequency_hz,
    dipolar_frequency_hz,
    secular_dipolar_hamiltonian,
)
from spin_dynamics.esr.hamiltonians import TAU
from spin_dynamics.nqr.operators import spin_matrices


@dataclass(frozen=True)
class DeerKernel:
    """Powder-averaged DEER kernel mapping a distance distribution to a trace."""

    times_seconds: np.ndarray
    distances_nm: np.ndarray
    matrix: np.ndarray
    lambda_depth: float


@dataclass(frozen=True)
class DeerDistanceResult:
    """Recovered distance distribution from a DEER form factor."""

    distances_nm: np.ndarray
    distribution: np.ndarray
    fitted_form_factor: np.ndarray
    residual_norm: float
    regularization_strength: float


def _validate_times(times_seconds) -> np.ndarray:
    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times_seconds must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValueError("times_seconds must be finite")
    return times


def _validate_lambda(lambda_depth: float) -> float:
    lambda_depth = float(lambda_depth)
    if not (0.0 <= lambda_depth <= 1.0) or not np.isfinite(lambda_depth):
        raise ValueError("lambda_depth must be in [0, 1]")
    return lambda_depth


def _cos_average_grid(n_theta: int) -> np.ndarray:
    n_theta = int(n_theta)
    if n_theta < 2:
        raise ValueError("n_theta must be at least 2")
    # Isotropic orientation -> uniform in cos(theta). Midpoint samples on [0, 1].
    return (np.arange(n_theta, dtype=np.float64) + 0.5) / n_theta


def deer_pair_trace(
    times_seconds,
    distance_nm: float,
    theta_rad: float,
    *,
    lambda_depth: float = 1.0,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> np.ndarray:
    """Return the DEER form factor for a single spin pair at fixed orientation.

    ``F(t) = 1 - lambda (1 - cos(2 pi nu_dd(r, theta) t))``.
    """

    times = _validate_times(times_seconds)
    lambda_depth = _validate_lambda(lambda_depth)
    nu_dd = float(
        dipolar_angular_frequency_hz(distance_nm, theta_rad, g_a=g_a, g_b=g_b)
    )
    return 1.0 - lambda_depth * (1.0 - np.cos(TAU * nu_dd * times))


def deer_powder_kernel(
    times_seconds,
    distances_nm,
    *,
    lambda_depth: float = 1.0,
    n_theta: int = 2001,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> DeerKernel:
    """Return the powder-averaged DEER kernel ``K[t, r]``.

    Each column maps a unit population at one distance to its powder-averaged
    form-factor contribution ``(1 - lambda) + lambda <cos(omega_dd t)>_theta``.
    A normalized distance distribution ``p`` (summing to one) therefore gives the
    DEER form factor as ``K @ p``.
    """

    times = _validate_times(times_seconds)
    lambda_depth = _validate_lambda(lambda_depth)
    distances = np.asarray(distances_nm, dtype=np.float64).reshape(-1)
    if distances.size == 0:
        raise ValueError("distances_nm must not be empty")
    if np.any(distances <= 0) or not np.all(np.isfinite(distances)):
        raise ValueError("distances_nm must be positive and finite")

    cos_theta = _cos_average_grid(n_theta)
    angular = 1.0 - 3.0 * cos_theta**2  # shape (n_theta,)
    matrix = np.empty((times.size, distances.size), dtype=np.float64)
    for col, distance in enumerate(distances):
        nu_perp = float(dipolar_frequency_hz(distance, g_a=g_a, g_b=g_b))
        # phase[t, theta] = 2 pi nu_perp (1 - 3 cos^2 theta) t
        phase = TAU * nu_perp * np.outer(times, angular)
        cos_avg = np.mean(np.cos(phase), axis=1)
        matrix[:, col] = (1.0 - lambda_depth) + lambda_depth * cos_avg
    return DeerKernel(
        times_seconds=times,
        distances_nm=distances,
        matrix=matrix,
        lambda_depth=lambda_depth,
    )


def gaussian_distance_distribution(
    distances_nm,
    mean_nm: float,
    sigma_nm: float,
) -> np.ndarray:
    """Return a normalized Gaussian distance distribution on a distance grid.

    The distribution sums to one over the grid (it is a discrete probability
    vector, not a density), so it can be fed directly to :func:`simulate_deer`.
    """

    distances = np.asarray(distances_nm, dtype=np.float64).reshape(-1)
    sigma = float(sigma_nm)
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma_nm must be positive and finite")
    weights = np.exp(-0.5 * ((distances - float(mean_nm)) / sigma) ** 2)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("distribution has zero total weight on this grid")
    return weights / total


def simulate_deer(
    times_seconds,
    distances_nm,
    distribution,
    *,
    lambda_depth: float = 1.0,
    n_theta: int = 2001,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> np.ndarray:
    """Return the DEER form factor for a distance distribution.

    ``distribution`` is a non-negative weight per distance; it is normalized to
    sum to one before applying the powder kernel.
    """

    kernel = deer_powder_kernel(
        times_seconds,
        distances_nm,
        lambda_depth=lambda_depth,
        n_theta=n_theta,
        g_a=g_a,
        g_b=g_b,
    )
    weights = np.asarray(distribution, dtype=np.float64).reshape(-1)
    if weights.size != kernel.distances_nm.size:
        raise ValueError("distribution must match distances_nm")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("distribution must be non-negative and finite")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("distribution must have positive total weight")
    return kernel.matrix @ (weights / total)


def deer_dipolar_spectrum(
    times_seconds,
    form_factor,
    *,
    zero_fill: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the dipolar (Pake) spectrum of a DEER form factor.

    The intramolecular ``(1 - lambda)`` offset is removed before transforming, so
    the spectrum is the cosine transform of the modulated part. Returns
    ``(frequencies_hz, spectrum)`` with a one-sided non-negative frequency axis.
    """

    times = _validate_times(times_seconds)
    if times.size < 2:
        raise ValueError("times_seconds must contain at least two points")
    dt = float(times[1] - times[0])
    if dt <= 0 or not np.allclose(np.diff(times), dt):
        raise ValueError("times_seconds must be uniformly increasing")
    signal = np.asarray(form_factor, dtype=np.float64).reshape(-1)
    if signal.size != times.size:
        raise ValueError("form_factor must match times_seconds")
    centered = signal - float(signal[-1])
    n_fft = int(zero_fill) * times.size
    if n_fft < times.size:
        raise ValueError("zero_fill must be at least 1")
    spectrum = np.abs(np.fft.rfft(centered, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, d=dt)
    return frequencies, spectrum


def extract_distance_distribution(
    times_seconds,
    form_factor,
    distances_nm,
    *,
    lambda_depth: float = 1.0,
    snr: float = 100.0,
    n_theta: int = 2001,
    regularization_order: int = 2,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> DeerDistanceResult:
    """Recover a distance distribution ``P(r)`` from a DEER form factor.

    Solves the regularized non-negative least-squares problem ``K p ~= F`` using
    the SNR-informed Tikhonov selector in
    :mod:`spin_dynamics.analysis.regularization`, with the DEER powder kernel as a
    custom design matrix. Requires SciPy (for the non-negative solve).
    """

    from spin_dynamics.analysis.regularization import select_regularization_1d

    kernel = deer_powder_kernel(
        times_seconds,
        distances_nm,
        lambda_depth=lambda_depth,
        n_theta=n_theta,
        g_a=g_a,
        g_b=g_b,
    )
    signal = np.asarray(form_factor, dtype=np.float64).reshape(-1)
    if signal.size != kernel.times_seconds.size:
        raise ValueError("form_factor must match times_seconds")

    selection = select_regularization_1d(
        signal,
        kernel.times_seconds,
        kernel.distances_nm,
        snr=snr,
        kernel=kernel.matrix,
        regularization_order=regularization_order,
        nonnegative=True,
    )
    result = selection.result
    distribution = np.asarray(result.distribution, dtype=np.float64)
    fitted = kernel.matrix @ distribution
    return DeerDistanceResult(
        distances_nm=kernel.distances_nm,
        distribution=distribution,
        fitted_form_factor=fitted,
        residual_norm=float(result.residual_norm),
        regularization_strength=float(selection.selected_strength),
    )


# --- Independent density-matrix validation -------------------------------------


def _two_spin_operators() -> dict[str, np.ndarray]:
    ops = spin_matrices(0.5)
    eye = ops.identity
    return {
        "sxa": np.kron(ops.ix, eye),
        "sya": np.kron(ops.iy, eye),
        "sza": np.kron(ops.iz, eye),
        "sxb": np.kron(eye, ops.ix),
        "szb": np.kron(eye, ops.iz),
    }


def deer_pair_trace_quantum(
    times_seconds,
    distance_nm: float,
    theta_rad: float,
    *,
    pump_flip_rad: float = np.pi,
    tau1_seconds: float = 200e-9,
    tau2_seconds: float = 2.0e-6,
    observer_offset_hz: float = 5.0e6,
    pump_offset_hz: float = 0.0,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> np.ndarray:
    """Simulate the four-pulse DEER form factor from the spin Hamiltonian.

    This is an independent, first-principles check on :func:`deer_pair_trace`: it
    builds the two-electron secular dipolar Hamiltonian, applies ideal
    spin-selective observer (A) and pump (B) pulses for the standard four-pulse
    DEER sequence, and returns the normalized refocused-echo amplitude as a
    function of the pump position ``t``. With an ideal ``pump_flip_rad`` it matches
    ``1 - lambda (1 - cos(2 pi nu_dd t))`` with ``lambda = sin^2(beta / 2)``, and
    the result is independent of the observer offset (the observer echo refocuses
    chemical shift).
    """

    times = _validate_times(times_seconds)
    tau1 = float(tau1_seconds)
    tau2 = float(tau2_seconds)
    if tau1 <= 0 or tau2 <= 0:
        raise ValueError("tau1_seconds and tau2_seconds must be positive")
    if np.any(times < 0) or np.any(times > tau2):
        raise ValueError("pump times must lie within [0, tau2_seconds]")

    ops = _two_spin_operators()
    h_dd = secular_dipolar_hamiltonian(distance_nm, theta_rad, g_a=g_a, g_b=g_b)
    h_offset = TAU * (
        observer_offset_hz * ops["sza"] + pump_offset_hz * ops["szb"]
    )
    h_free = h_dd + h_offset

    # Ideal spin-selective rotations.
    pi_a = propagator(ops["sxa"], np.pi)
    pump = propagator(ops["sxb"], float(pump_flip_rad))
    excite_a = propagator(ops["sya"], np.pi / 2.0)

    detector = 2.0 * (ops["sxa"] - 1j * ops["sya"])  # 2 S_A^- -> reads S_A^+

    # Observer polarized along z, pump spin in a maximally mixed state.
    rho0 = np.kron(spin_matrices(0.5).iz, 0.5 * spin_matrices(0.5).identity)
    rho0 = excite_a @ rho0 @ excite_a.conj().T

    signal = np.empty(times.size, dtype=np.complex128)
    for idx, t_pump in enumerate(times):
        rho = evolve_density(rho0, h_free, tau1)
        rho = pi_a @ rho @ pi_a.conj().T
        rho = evolve_density(rho, h_free, tau1 + float(t_pump))
        rho = pump @ rho @ pump.conj().T
        rho = evolve_density(rho, h_free, tau2 - float(t_pump))
        rho = pi_a @ rho @ pi_a.conj().T
        rho = evolve_density(rho, h_free, tau2)
        signal[idx] = np.trace(rho @ detector)

    reference = np.abs(signal[0]) if np.abs(signal[0]) > 0 else 1.0
    return np.real(signal * np.conj(signal[0]) / reference**2)
