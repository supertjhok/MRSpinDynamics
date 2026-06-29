"""HYSCORE (hyperfine sublevel correlation) 2D spectroscopy for an S=1/2 electron.

HYSCORE is the four-pulse experiment ``pi/2 - tau - pi/2 - t1 - pi - t2 - pi/2 -
tau - echo``. The central ``pi`` pulse transfers nuclear coherence between the
two electron manifolds, so coherence that evolves at a manifold-alpha frequency
during ``t1`` evolves at a manifold-beta frequency during ``t2`` (and vice
versa). The 2D spectrum therefore shows cross-peaks at every
``(nu_alpha, nu_beta)`` / ``(nu_beta, nu_alpha)`` pair of nuclear frequencies,
whose positions reveal the hyperfine (and, for ``I >= 1``, quadrupole) coupling.

This module simulates the sequence with the density-matrix engine and electron
coherence-pathway selection from :mod:`spin_dynamics.esr.eseem`; it works for any
nuclear spin (``I = 1/2``, ``1``, ``3/2``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density
from spin_dynamics.esr.eseem import (
    HyperfineCoupling,
    _ideal_pulse,
    _operators,
    electron_nuclear_hamiltonian,
    filter_electron_coherence,
    manifold_frequencies,
)


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

    spin = coupling.nuclear_spin
    ops = _operators(spin)
    hamiltonian = electron_nuclear_hamiltonian(coupling)
    p90 = _ideal_pulse(np.pi / 2.0, "y", nuclear_spin=spin)
    p180 = _ideal_pulse(np.pi, "x", nuclear_spin=spin)

    # Preparation through the first two pulses and the fixed tau delay.
    prepared = filter_electron_coherence(p90 @ ops.thermal @ p90.conj().T, +1, nuclear_spin=spin)
    prepared = evolve_density(prepared, hamiltonian, tau)
    prepared = filter_electron_coherence(p90 @ prepared @ p90.conj().T, 0, nuclear_spin=spin)

    signal = np.empty((t1.size, t2.size), dtype=np.float64)
    for i, first in enumerate(t1):
        mixed = evolve_density(prepared, hamiltonian, float(first))
        mixed = filter_electron_coherence(p180 @ mixed @ p180.conj().T, 0, nuclear_spin=spin)
        for j, second in enumerate(t2):
            rho = evolve_density(mixed, hamiltonian, float(second))
            rho = filter_electron_coherence(p90 @ rho @ p90.conj().T, -1, nuclear_spin=spin)
            rho = evolve_density(rho, hamiltonian, tau)
            signal[i, j] = float(np.real(np.trace(rho @ ops.splus)))
    return signal


def cross_peak_positions(coupling: HyperfineCoupling) -> tuple[tuple[float, float], ...]:
    """Return the HYSCORE cross-peak positions in Hz.

    Cross-peaks correlate a manifold-alpha frequency with a manifold-beta
    frequency (and the mirror). For a spin-1/2 nucleus this is the single pair
    ``(nu_alpha, nu_beta)`` and its reflection; for ``I >= 1`` there is one such
    pair for every alpha/beta frequency combination.
    """

    alpha, beta = manifold_frequencies(coupling)
    positions: list[tuple[float, float]] = []
    for nu_a in alpha:
        for nu_b in beta:
            positions.append((float(nu_a), float(nu_b)))
            positions.append((float(nu_b), float(nu_a)))
    return tuple(positions)


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
