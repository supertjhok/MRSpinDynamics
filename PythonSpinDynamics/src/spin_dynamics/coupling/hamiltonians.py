"""Hamiltonian builders for dense scalar-coupled spin simulations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from spin_dynamics.coupling.operators import product_operator, spin_operator
from spin_dynamics.coupling.systems import CoupledSpinSystem


TAU = 2.0 * np.pi


def _empty(system: CoupledSpinSystem) -> np.ndarray:
    return np.zeros((system.dimension, system.dimension), dtype=np.complex128)


def zeeman_hamiltonian(system: CoupledSpinSystem) -> np.ndarray:
    """Return the rotating-frame offset Hamiltonian in radians per second."""

    hamiltonian = _empty(system)
    for idx, offset_hz in enumerate(system.offsets_hz):
        hamiltonian = hamiltonian + TAU * offset_hz * spin_operator(
            system.nspin,
            idx,
            "z",
        )
    return hamiltonian


def secular_j_hamiltonian(system: CoupledSpinSystem) -> np.ndarray:
    """Return the weak-coupling secular scalar Hamiltonian."""

    hamiltonian = _empty(system)
    for idx in range(system.nspin):
        for jdx in range(idx + 1, system.nspin):
            coupling_hz = system.couplings_hz[idx, jdx]
            if coupling_hz:
                hamiltonian = hamiltonian + TAU * coupling_hz * product_operator(
                    system.nspin,
                    [(idx, "z"), (jdx, "z")],
                )
    return hamiltonian


def isotropic_j_hamiltonian(system: CoupledSpinSystem) -> np.ndarray:
    """Return the isotropic scalar Hamiltonian for strongly coupled spins."""

    hamiltonian = _empty(system)
    for idx in range(system.nspin):
        for jdx in range(idx + 1, system.nspin):
            coupling_hz = system.couplings_hz[idx, jdx]
            if not coupling_hz:
                continue
            pair = (
                product_operator(system.nspin, [(idx, "x"), (jdx, "x")])
                + product_operator(system.nspin, [(idx, "y"), (jdx, "y")])
                + product_operator(system.nspin, [(idx, "z"), (jdx, "z")])
            )
            hamiltonian = hamiltonian + TAU * coupling_hz * pair
    return hamiltonian


def rf_hamiltonian(
    system: CoupledSpinSystem,
    nutation_hz: float | Iterable[float],
    *,
    phase: float = 0.0,
    indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Return an RF Hamiltonian for selected spins in radians per second."""

    selected = tuple(range(system.nspin) if indices is None else indices)
    amplitudes = np.asarray(nutation_hz, dtype=np.float64)
    if amplitudes.ndim == 0:
        amplitudes = np.full(len(selected), float(amplitudes), dtype=np.float64)
    amplitudes = amplitudes.reshape(-1)
    if amplitudes.size != len(selected):
        raise ValueError("nutation_hz must be scalar or match the selected spins")
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("nutation_hz must be finite")
    cphase = float(np.cos(phase))
    sphase = float(np.sin(phase))
    hamiltonian = _empty(system)
    for amplitude_hz, idx in zip(amplitudes, selected):
        hamiltonian = hamiltonian + TAU * amplitude_hz * (
            cphase * spin_operator(system.nspin, int(idx), "x")
            + sphase * spin_operator(system.nspin, int(idx), "y")
        )
    return hamiltonian
