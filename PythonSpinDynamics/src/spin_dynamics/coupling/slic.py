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
        """Nutation frequency at the deepest simulated SLIC dip.

        For an isolated nearly equivalent spin-1/2 pair this approaches the
        magnitude of the intrapair scalar coupling, ``|J|``. The chemical-shift
        difference supplies the singlet-triplet mixing matrix element and thus
        controls the transfer rate rather than the crossing position.
        """

        return float(self.nutation_frequencies_hz[int(np.argmax(self.dip))])


def two_spin_slic_matching_nutation_hz(coupling_hz: float) -> float:
    """Return the ideal two-spin SLIC matching nutation frequency.

    The spin lock shifts a triplet level into resonance with the singlet when
    its nutation frequency is ``|J|``. A nonzero resonance-offset difference is
    still required to mix the two symmetry sectors and produce transfer.
    """

    coupling = abs(float(coupling_hz))
    if not np.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("coupling_hz must be non-zero and finite")
    return coupling


def two_spin_slic_transfer_time(offset_difference_hz: float) -> float:
    """Return the ideal maximum-transfer time at the SLIC crossing.

    For offsets ``(-Delta nu / 2, +Delta nu / 2)`` expressed in hertz, the
    nearly equivalent two-spin result is ``1 / (sqrt(2) |Delta nu|)``. Thus J
    sets the matching spin-lock amplitude while the offset difference sets the
    transfer time.
    """

    delta = abs(float(offset_difference_hz))
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("offset_difference_hz must be non-zero and finite")
    return 1.0 / (np.sqrt(2.0) * delta)


def simulate_slic_spectrum(
    system: CoupledSpinSystem,
    nutation_frequencies_hz: Iterable[float] | np.ndarray,
    *,
    spin_lock_time: float,
    initial_axis: str = "x",
    detect_axis: str = "x",
) -> SLICSpectrumResult:
    """Simulate remaining transverse magnetization after a spin-lock pulse.

    For a nearly equivalent two-spin pair, scan the nutation frequency around
    ``|J|`` and place the RF carrier at the pair's mean resonance (so the two
    stored offsets are approximately symmetric about zero). The offset
    difference mixes singlet and triplet symmetry sectors; if it is exactly
    zero, an ideal nonselective spin lock cannot create a SLIC dip.
    """

    frequencies = np.asarray(nutation_frequencies_hz, dtype=np.float64).reshape(-1)
    if frequencies.size == 0:
        raise ValueError("nutation_frequencies_hz must not be empty")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("nutation_frequencies_hz must be finite")
    if not np.isfinite(spin_lock_time) or spin_lock_time <= 0:
        raise ValueError("spin_lock_time must be positive and finite")

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
