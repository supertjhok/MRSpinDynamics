"""Control-Hamiltonian models for GRAPE optimal control.

Builds the fixed (non-optimized) operator set for a piecewise-constant-control
Hamiltonian::

    H(t) = H_drift + w1(t) * (cos(phi(t)) * H_x + sin(phi(t)) * H_y)

where ``w1`` (a nutation rate in hertz, matching
``coupling.hamiltonians.rf_hamiltonian``'s convention) and ``phi`` (a phase in
radians) are the per-segment GRAPE control variables. ``H_drift`` and
``H_x``/``H_y`` do not depend on the controls -- they are precomputed once,
host-side, by wrapping the existing ``coupling.hamiltonians`` builders.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.hamiltonians import secular_j_hamiltonian, zeeman_hamiltonian
from spin_dynamics.coupling.operators import total_operator
from spin_dynamics.coupling.systems import CoupledSpinSystem

TAU = 2.0 * np.pi


@dataclass(frozen=True)
class ControlHamiltonianModel:
    """Fixed operator set for a piecewise-constant-control Hamiltonian.

    ``h_drift``, ``h_x``, and ``h_y`` are ``(dimension, dimension)`` complex
    Hermitian matrices in radians/second. They do not depend on the optimized
    control values -- only the per-segment scalar amplitude (hertz) and phase
    (radians) multiplying ``h_x``/``h_y`` are optimized.
    """

    dimension: int
    h_drift: np.ndarray
    h_x: np.ndarray
    h_y: np.ndarray

    def __post_init__(self) -> None:
        dimension = int(self.dimension)
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        object.__setattr__(self, "dimension", dimension)
        for name in ("h_drift", "h_x", "h_y"):
            matrix = np.asarray(getattr(self, name), dtype=np.complex128)
            if matrix.shape != (dimension, dimension):
                raise ValueError(f"{name} must have shape (dimension, dimension)")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError(f"{name} must be Hermitian")
            object.__setattr__(self, name, matrix)


def coupled_spin_control_model(
    system: CoupledSpinSystem,
    *,
    control_indices: Iterable[int] | None = None,
    include_j_coupling: bool = True,
) -> ControlHamiltonianModel:
    """Build a GRAPE control model for a scalar-coupled spin-1/2 system.

    ``h_drift`` is the rotating-frame offset Hamiltonian (plus the secular J
    Hamiltonian when ``include_j_coupling``); ``h_x``/``h_y`` are the summed
    transverse RF-drive operators over ``control_indices`` (default: every
    spin in the system), reusing ``coupling.operators.total_operator`` --
    the same operators ``coupling.hamiltonians.rf_hamiltonian`` sums over a
    fixed amplitude/phase, generalized here to a per-segment control.
    """

    drift = zeeman_hamiltonian(system)
    if include_j_coupling:
        drift = drift + secular_j_hamiltonian(system)
    indices = None if control_indices is None else tuple(int(idx) for idx in control_indices)
    h_x = TAU * total_operator(system.nspin, "x", indices)
    h_y = TAU * total_operator(system.nspin, "y", indices)
    return ControlHamiltonianModel(
        dimension=system.dimension,
        h_drift=drift,
        h_x=h_x,
        h_y=h_y,
    )
