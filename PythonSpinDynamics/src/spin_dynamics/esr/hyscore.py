"""HYSCORE (hyperfine sublevel correlation) 2D spectroscopy for S=1/2, I=1/2.

HYSCORE is the four-pulse experiment ``pi/2 - tau - pi/2 - t1 - pi - t2 - pi/2 -
tau - echo``. The central ``pi`` pulse transfers nuclear coherence between the
two electron manifolds, so coherence that evolves at ``nu_alpha`` during ``t1``
evolves at ``nu_beta`` during ``t2`` (and vice versa). The 2D spectrum therefore
shows cross-peaks at ``(nu_alpha, nu_beta)`` and ``(nu_beta, nu_alpha)`` whose
positions reveal the hyperfine coupling and whose quadrant distinguishes the
weak- from the strong-coupling regime.

This module simulates the sequence with the density-matrix engine and electron
coherence-pathway selection from :mod:`spin_dynamics.esr.eseem`, and provides the
2D spectrum plus the analytic cross-peak positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density
from spin_dynamics.esr.eseem import (
    SPLUS,
    HyperfineCoupling,
    _ideal_pulse,
    electron_nuclear_hamiltonian,
    filter_electron_coherence,
    nuclear_frequencies,
)
from spin_dynamics.nqr.operators import spin_matrices

_SM = spin_matrices(0.5)
_RHO0 = np.kron(_SM.iz, _SM.identity)


@dataclass(frozen=True)
class HyscoreSpectrum:
    """2D HYSCORE spectrum on centered frequency axes."""

    frequencies1_hz: np.ndarray
    frequencies2_hz: np.ndarray
    spectrum: np.ndarray


def hyscore_signal(
    t1_seconds,
    t2_seconds,
    coupling: HyperfineCoupling,
    *,
    tau_seconds: float,
) -> np.ndarray:
    """Return the 2D HYSCORE time-domain signal ``V[t1, t2]``.

    Uses electron coherence-pathway selection ``+1 -> 0 -> 0 -> -1`` across the
    four pulses; the central ``pi`` mixes the nuclear coherences between
    manifolds, which is what produces the HYSCORE cross-peaks.
    """

    t1 = np.asarray(t1_seconds, dtype=np.float64).reshape(-1)
    t2 = np.asarray(t2_seconds, dtype=np.float64).reshape(-1)
    if t1.size == 0 or t2.size == 0:
        raise ValueError("t1_seconds and t2_seconds must not be empty")
    if not (np.all(np.isfinite(t1)) and np.all(np.isfinite(t2))):
        raise ValueError("t1_seconds and t2_seconds must be finite")
    tau = float(tau_seconds)
    if tau < 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be non-negative and finite")

    hamiltonian = electron_nuclear_hamiltonian(coupling)
    p90 = _ideal_pulse(np.pi / 2.0, "y")
    p180 = _ideal_pulse(np.pi, "x")

    # Preparation through the first two pulses and the fixed tau delay.
    prepared = filter_electron_coherence(p90 @ _RHO0 @ p90.conj().T, +1)
    prepared = evolve_density(prepared, hamiltonian, tau)
    prepared = filter_electron_coherence(p90 @ prepared @ p90.conj().T, 0)

    signal = np.empty((t1.size, t2.size), dtype=np.float64)
    for i, first in enumerate(t1):
        mixed = evolve_density(prepared, hamiltonian, float(first))
        mixed = filter_electron_coherence(p180 @ mixed @ p180.conj().T, 0)
        for j, second in enumerate(t2):
            rho = evolve_density(mixed, hamiltonian, float(second))
            rho = filter_electron_coherence(p90 @ rho @ p90.conj().T, -1)
            rho = evolve_density(rho, hamiltonian, tau)
            signal[i, j] = float(np.real(np.trace(rho @ SPLUS)))
    return signal


def cross_peak_positions(coupling: HyperfineCoupling) -> tuple[tuple[float, float], ...]:
    """Return the analytic HYSCORE cross-peak positions in Hz."""

    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    return ((nu_alpha, nu_beta), (nu_beta, nu_alpha))


def hyscore_spectrum(
    t1_seconds,
    t2_seconds,
    signal,
    *,
    zero_fill: int = 4,
) -> HyscoreSpectrum:
    """Return the 2D HYSCORE magnitude spectrum on centered frequency axes."""

    t1 = np.asarray(t1_seconds, dtype=np.float64).reshape(-1)
    t2 = np.asarray(t2_seconds, dtype=np.float64).reshape(-1)
    values = np.asarray(signal, dtype=np.float64)
    if values.shape != (t1.size, t2.size):
        raise ValueError("signal shape must match (t1, t2)")
    if t1.size < 2 or t2.size < 2:
        raise ValueError("t1 and t2 must each contain at least two points")
    dt1 = float(t1[1] - t1[0])
    dt2 = float(t2[1] - t2[0])
    if not (np.allclose(np.diff(t1), dt1) and np.allclose(np.diff(t2), dt2)):
        raise ValueError("t1 and t2 must be uniformly spaced")
    if int(zero_fill) < 1:
        raise ValueError("zero_fill must be at least 1")

    centered = values - float(np.mean(values))
    n1 = int(zero_fill) * t1.size
    n2 = int(zero_fill) * t2.size
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(centered, s=(n1, n2))))
    f1 = np.fft.fftshift(np.fft.fftfreq(n1, d=dt1))
    f2 = np.fft.fftshift(np.fft.fftfreq(n2, d=dt2))
    return HyscoreSpectrum(frequencies1_hz=f1, frequencies2_hz=f2, spectrum=spectrum)
