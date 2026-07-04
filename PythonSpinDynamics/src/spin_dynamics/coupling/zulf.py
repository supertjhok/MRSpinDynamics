"""Pulse-acquire simulation of zero/ultra-low-field J-coupled spectra (JCS).

A ZULF / Earth's-field ``pi/2``-acquire experiment is modelled here as:

1. a high-temperature equilibrium density ``rho ~ sum_i gamma_i I_iz``;
2. an ideal hard ``pi/2`` pulse tipping the chosen spins onto a transverse axis;
3. free evolution under the lab-frame Zeeman + isotropic-``J`` Hamiltonian and a
   per-spin ``R1``/``R2`` relaxation superoperator; and
4. detection of the (optionally isotope-selective) transverse magnetization,
   Fourier transformed to the J-coupled spectrum.

The free-evolution FID is evaluated by eigendecomposing the (small, dense)
Liouvillian once and summing complex exponentials over the acquisition grid, so
long, high-resolution FIDs are cheap. Relaxation of a fast quadrupolar nucleus
(e.g. ``14N``) coupled to observed spins collapses their J-splittings, the
central effect studied by Altenhof et al., J. Magn. Reson. 355 (2023) 107540.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import propagator
from spin_dynamics.coupling.mixed_operators import embedded_operator, total_operator
from spin_dynamics.coupling.multinuclear import (
    MultinuclearSpinSystem,
    multinuclear_equilibrium_density,
    multinuclear_hamiltonian,
    per_spin_relaxation_superoperator,
)
from spin_dynamics.relaxation import liouville_hamiltonian


@dataclass(frozen=True)
class ZulfSpectrum:
    """Time-domain FID and its J-coupled spectrum."""

    times_seconds: np.ndarray
    fid: np.ndarray
    frequencies_hz: np.ndarray
    spectrum: np.ndarray


def _rate_array(
    rate: float | Sequence[float] | np.ndarray,
    nspin: int,
    name: str,
) -> np.ndarray:
    values = np.broadcast_to(np.asarray(rate, dtype=np.float64), (nspin,)).astype(
        np.float64
    )
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be non-negative and finite")
    return values


def _detection_operator(
    system: MultinuclearSpinSystem,
    detect_indices: Iterable[int] | None,
    gamma_weighted: bool,
) -> np.ndarray:
    spins = system.spins
    selected = tuple(
        range(system.nspin) if detect_indices is None else detect_indices
    )
    if not selected:
        raise ValueError("detect_indices must select at least one spin")
    gammas = system.gammas_hz_per_t
    reference = float(np.max(np.abs(gammas))) or 1.0
    dimension = system.dimension
    operator = np.zeros((dimension, dimension), dtype=np.complex128)
    for idx in selected:
        weight = float(gammas[int(idx)]) / reference if gamma_weighted else 1.0
        operator = operator + weight * (
            embedded_operator(spins, int(idx), "x")
            + 1j * embedded_operator(spins, int(idx), "y")
        )
    return operator


def simulate_zulf_fid(
    system: MultinuclearSpinSystem,
    *,
    r1_per_second: float | Sequence[float] | np.ndarray,
    r2_per_second: float | Sequence[float] | np.ndarray,
    dwell_seconds: float,
    n_points: int,
    excite_indices: Iterable[int] | None = None,
    detect_indices: Iterable[int] | None = None,
    flip_rad: float = np.pi / 2.0,
    phase_rad: float = 0.0,
    gamma_weighted: bool = True,
    coupling: str = "isotropic",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, fid)`` for a ZULF ``pi/2``-acquire experiment.

    ``r1_per_second``/``r2_per_second`` are per-site rates (scalar broadcasts to
    all sites). ``excite_indices``/``detect_indices`` default to all spins; pass
    site indices (e.g. from ``system.indices_for_isotope``) to excite or detect a
    single nuclear species. The pulse is an ideal hard rotation of ``flip_rad``
    about the axis at ``phase_rad`` (0 = x).
    """

    dwell = float(dwell_seconds)
    if not np.isfinite(dwell) or dwell <= 0.0:
        raise ValueError("dwell_seconds must be positive and finite")
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    spins = system.spins
    r1 = _rate_array(r1_per_second, system.nspin, "r1_per_second")
    r2 = _rate_array(r2_per_second, system.nspin, "r2_per_second")

    hamiltonian = multinuclear_hamiltonian(system, coupling=coupling)
    density = multinuclear_equilibrium_density(system, gamma_weighted=gamma_weighted)

    excite = tuple(range(system.nspin) if excite_indices is None else excite_indices)
    if excite:
        pulse_axis = np.cos(phase_rad) * total_operator(
            spins, "y", excite
        ) - np.sin(phase_rad) * total_operator(spins, "x", excite)
        pulse = propagator(pulse_axis, float(flip_rad))
        density = pulse @ density @ pulse.conj().T

    detection = _detection_operator(system, detect_indices, gamma_weighted)

    generator = liouville_hamiltonian(hamiltonian) + per_spin_relaxation_superoperator(
        spins, r1, r2
    )
    eigenvalues, eigenvectors = np.linalg.eig(generator)
    eigenvectors_inv = np.linalg.inv(eigenvectors)

    rho_vec = density.reshape(-1, order="F")
    det_vec = detection.T.reshape(-1, order="F")
    right = eigenvectors_inv @ rho_vec
    left = det_vec @ eigenvectors
    coefficients = left * right

    times = np.arange(n_points, dtype=np.float64) * dwell
    phases = np.exp(np.outer(eigenvalues, times))
    fid = coefficients @ phases
    return times, np.asarray(fid, dtype=np.complex128)


def simulate_zulf_spectrum(
    system: MultinuclearSpinSystem,
    *,
    r1_per_second: float | Sequence[float] | np.ndarray,
    r2_per_second: float | Sequence[float] | np.ndarray,
    dwell_seconds: float,
    n_points: int,
    excite_indices: Iterable[int] | None = None,
    detect_indices: Iterable[int] | None = None,
    flip_rad: float = np.pi / 2.0,
    phase_rad: float = 0.0,
    gamma_weighted: bool = True,
    apodization_hz: float = 0.0,
    coupling: str = "isotropic",
) -> ZulfSpectrum:
    """Return the J-coupled spectrum of a ZULF ``pi/2``-acquire experiment.

    ``apodization_hz`` applies an exponential (Lorentzian) line broadening of the
    given full width at half maximum before the Fourier transform. The returned
    ``frequencies_hz`` and ``spectrum`` are ordered from negative to positive
    frequency; nuclei with positive gyromagnetic ratio appear at their positive
    Larmor frequency.
    """

    times, fid = simulate_zulf_fid(
        system,
        r1_per_second=r1_per_second,
        r2_per_second=r2_per_second,
        dwell_seconds=dwell_seconds,
        n_points=n_points,
        excite_indices=excite_indices,
        detect_indices=detect_indices,
        flip_rad=flip_rad,
        phase_rad=phase_rad,
        gamma_weighted=gamma_weighted,
        coupling=coupling,
    )

    apodization = float(apodization_hz)
    if apodization < 0.0 or not np.isfinite(apodization):
        raise ValueError("apodization_hz must be non-negative and finite")
    windowed = fid
    if apodization > 0.0:
        windowed = fid * np.exp(-np.pi * apodization * times)

    # Detected M+ coherence evolves as exp(+i 2 pi nu t) under H = +2 pi nu I_z,
    # so a plain FFT places a positive-gamma nucleus at its positive Larmor freq.
    spectrum = np.fft.fftshift(np.fft.fft(windowed))
    frequencies = np.fft.fftshift(np.fft.fftfreq(int(n_points), d=float(dwell_seconds)))
    return ZulfSpectrum(
        times_seconds=times,
        fid=fid,
        frequencies_hz=frequencies,
        spectrum=np.asarray(spectrum, dtype=np.complex128),
    )
