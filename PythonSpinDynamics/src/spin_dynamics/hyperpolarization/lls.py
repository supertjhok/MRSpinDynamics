"""Long-lived singlet preparation, storage, and readout workflows.

The storage model is deliberately phenomenological: a measured singlet
lifetime ``T_S`` damps the selected singlet-order mode.  It does not infer a
singlet lifetime from independent-spin ``T1`` and ``T2`` values.
"""

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
from spin_dynamics.coupling.slic import (
    two_spin_slic_matching_nutation_hz,
    two_spin_slic_transfer_time,
)
from spin_dynamics.coupling.systems import CoupledSpinSystem
from spin_dynamics.hyperpolarization.singlet import (
    singlet_order_amplitude,
    singlet_order_operator,
)


def _validated_operator(operator: np.ndarray) -> tuple[np.ndarray, int]:
    matrix = np.asarray(operator, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density must be a square matrix")
    dimension = matrix.shape[0]
    nspin = int(round(np.log2(dimension)))
    if 2**nspin != dimension:
        raise ValueError("density dimension must be a power of two")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("density must contain only finite values")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
        raise ValueError("density must be Hermitian")
    return matrix, nspin


def store_singlet_order(
    density: np.ndarray,
    duration_seconds: float,
    singlet_lifetime_seconds: float,
    *,
    pair: tuple[int, int] = (0, 1),
    purge_non_singlet: bool = True,
) -> np.ndarray:
    """Store a selected singlet-order mode for a finite duration.

    ``purge_non_singlet=True`` models the gradient/filter stage commonly used
    between preparation and storage: the identity baseline and selected
    singlet order remain, while other deviation operators are removed.  The
    singlet amplitude then decays as ``exp(-duration / T_S)``.  With the purge
    disabled, all orthogonal modes are retained unchanged.

    Unit-trace physical densities and trace-zero deviation densities are both
    supported.  The transformation preserves Hermiticity and trace.
    """

    matrix, nspin = _validated_operator(density)
    duration = float(duration_seconds)
    lifetime = float(singlet_lifetime_seconds)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_seconds must be non-negative and finite")
    if not np.isfinite(lifetime) or lifetime <= 0.0:
        raise ValueError("singlet_lifetime_seconds must be positive and finite")

    order = singlet_order_operator(nspin, pair)
    amplitude = singlet_order_amplitude(matrix, pair=pair)
    singlet_component = amplitude * order
    baseline = np.trace(matrix) * np.eye(2**nspin, dtype=np.complex128) / (2**nspin)
    residual = np.zeros_like(matrix) if purge_non_singlet else matrix - baseline - singlet_component
    attenuation = np.exp(-duration / lifetime)
    stored = baseline + residual + attenuation * singlet_component
    return 0.5 * (stored + stored.conj().T)


@dataclass(frozen=True)
class SLICLongLivedStateResult:
    """Stage-resolved result of a two-spin SLIC/store/SLIC workflow."""

    storage_times_seconds: np.ndarray
    normalized_signal: np.ndarray
    singlet_amplitudes: np.ndarray
    prepared_density: np.ndarray
    prepared_singlet_amplitude: float
    preparation_time_seconds: float
    readout_time_seconds: float
    matching_nutation_hz: float
    singlet_lifetime_seconds: float


def simulate_slic_lls(
    system: CoupledSpinSystem,
    storage_times_seconds: Iterable[float] | np.ndarray,
    *,
    singlet_lifetime_seconds: float,
    preparation_time_seconds: float | None = None,
    readout_time_seconds: float | None = None,
    nutation_hz: float | None = None,
    purge_non_singlet: bool = True,
) -> SLICLongLivedStateResult:
    """Simulate two-spin SLIC preparation, storage, and reconversion.

    The RF carrier is represented by the mean rotating-frame frequency, so the
    two offsets should normally straddle zero.  ``|J|`` sets the SLIC nutation
    match and the offset difference sets the default preparation time.  The
    same spin-lock Hamiltonian reconverts stored singlet order for readout.
    """

    if system.nspin != 2:
        raise ValueError("simulate_slic_lls currently requires exactly two spins")
    times = np.array(storage_times_seconds, dtype=np.float64, copy=True).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("storage_times_seconds must be finite and non-negative")
    coupling = float(system.couplings_hz[0, 1])
    offset_difference = float(system.offsets_hz[1] - system.offsets_hz[0])
    matched_nutation = (
        two_spin_slic_matching_nutation_hz(coupling)
        if nutation_hz is None
        else float(nutation_hz)
    )
    if not np.isfinite(matched_nutation) or matched_nutation <= 0.0:
        raise ValueError("nutation_hz must be positive and finite")
    preparation_time = (
        two_spin_slic_transfer_time(offset_difference)
        if preparation_time_seconds is None
        else float(preparation_time_seconds)
    )
    readout_time = preparation_time if readout_time_seconds is None else float(readout_time_seconds)
    if not np.isfinite(preparation_time) or preparation_time <= 0.0:
        raise ValueError("preparation_time_seconds must be positive and finite")
    if not np.isfinite(readout_time) or readout_time <= 0.0:
        raise ValueError("readout_time_seconds must be positive and finite")

    static = zeeman_hamiltonian(system) + isotropic_j_hamiltonian(system)
    spin_lock = static + rf_hamiltonian(system, matched_nutation, phase=0.0)
    initial = equilibrium_density(system, "x")
    detect = total_operator(system.nspin, "x")
    baseline = np.trace(initial @ detect)
    prepared = evolve_density(initial, spin_lock, preparation_time)
    prepared_amplitude = singlet_order_amplitude(prepared)

    signals: list[float] = []
    amplitudes: list[float] = []
    for storage_time in times:
        stored = store_singlet_order(
            prepared,
            float(storage_time),
            singlet_lifetime_seconds,
            purge_non_singlet=purge_non_singlet,
        )
        amplitudes.append(singlet_order_amplitude(stored))
        readout = evolve_density(stored, spin_lock, readout_time)
        signals.append(float(np.real(np.trace(readout @ detect) / baseline)))

    signal_array = np.asarray(signals, dtype=np.float64)
    amplitude_array = np.asarray(amplitudes, dtype=np.float64)
    for array in (times, signal_array, amplitude_array, prepared):
        array.setflags(write=False)
    return SLICLongLivedStateResult(
        storage_times_seconds=times,
        normalized_signal=signal_array,
        singlet_amplitudes=amplitude_array,
        prepared_density=prepared,
        prepared_singlet_amplitude=prepared_amplitude,
        preparation_time_seconds=preparation_time,
        readout_time_seconds=readout_time,
        matching_nutation_hz=matched_nutation,
        singlet_lifetime_seconds=float(singlet_lifetime_seconds),
    )


__all__ = [
    "SLICLongLivedStateResult",
    "simulate_slic_lls",
    "store_singlet_order",
]
