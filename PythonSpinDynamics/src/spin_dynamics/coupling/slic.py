"""Spin-lock induced crossing (SLIC) helpers for homonuclear systems."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import equilibrium_density, evolve_density
from spin_dynamics.coupling.hamiltonians import (
    isotropic_j_hamiltonian,
    rf_hamiltonian,
    zeeman_hamiltonian,
)
from spin_dynamics.coupling.operators import total_operator
from spin_dynamics.coupling.systems import CoupledSpinSystem


@dataclass(frozen=True)
class SLICSpectrumResult:
    """Simulated SLIC response as a function of spin-lock nutation frequency."""

    nutation_frequencies_hz: np.ndarray
    normalized_mx: np.ndarray
    dip: np.ndarray
    spin_lock_time: float

    @property
    def strongest_dip_frequency_hz(self) -> float:
        """Nutation frequency at the deepest simulated SLIC dip."""

        return float(self.nutation_frequencies_hz[int(np.argmax(self.dip))])


def two_spin_slic_transfer_time(offset_difference_hz: float) -> float:
    """Return the ideal two-spin SLIC maximum-transfer time."""

    delta = abs(float(offset_difference_hz))
    if delta <= 0:
        raise ValueError("offset_difference_hz must be non-zero")
    return 1.0 / (np.sqrt(2.0) * delta)


def simulate_slic_spectrum(
    system: CoupledSpinSystem,
    nutation_frequencies_hz: Iterable[float] | np.ndarray,
    *,
    spin_lock_time: float,
    initial_axis: str = "x",
    detect_axis: str = "x",
) -> SLICSpectrumResult:
    """Simulate remaining transverse magnetization after a spin-lock pulse."""

    frequencies = np.asarray(nutation_frequencies_hz, dtype=np.float64).reshape(-1)
    if frequencies.size == 0:
        raise ValueError("nutation_frequencies_hz must not be empty")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("nutation_frequencies_hz must be finite")
    if spin_lock_time <= 0:
        raise ValueError("spin_lock_time must be positive")

    initial = equilibrium_density(system, initial_axis)
    detect = total_operator(system.nspin, detect_axis)
    baseline = np.trace(initial @ detect)
    if abs(baseline) == 0:
        raise ValueError("initial and detect axes produce zero baseline signal")

    static_hamiltonian = zeeman_hamiltonian(system) + isotropic_j_hamiltonian(system)
    values = []
    for nutation in frequencies:
        hamiltonian = static_hamiltonian + rf_hamiltonian(system, nutation, phase=0.0)
        density = evolve_density(initial, hamiltonian, spin_lock_time)
        values.append(np.real(np.trace(density @ detect) / baseline))
    normalized = np.asarray(values, dtype=np.float64)
    return SLICSpectrumResult(
        nutation_frequencies_hz=frequencies,
        normalized_mx=normalized,
        dip=1.0 - normalized,
        spin_lock_time=float(spin_lock_time),
    )
