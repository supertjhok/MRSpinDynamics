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

    ``h_grad`` (optional) is the *base* gradient-control operator, an Iz-like
    offset operator (see :func:`gradient_control_operator`). When present it
    enables a gradient control channel ``+ g(t) * h_grad``; for
    position-selective control it is scaled per ensemble member by that member's
    position (:func:`position_gradient_batch`). ``None`` (default) means RF-only
    control, matching Milestone 1 behaviour exactly.
    """

    dimension: int
    h_drift: np.ndarray
    h_x: np.ndarray
    h_y: np.ndarray
    h_grad: np.ndarray | None = None

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
        if self.h_grad is not None:
            h_grad = np.asarray(self.h_grad, dtype=np.complex128)
            if h_grad.shape != (dimension, dimension):
                raise ValueError("h_grad must have shape (dimension, dimension)")
            if not np.allclose(h_grad, h_grad.conj().T):
                raise ValueError("h_grad must be Hermitian")
            object.__setattr__(self, "h_grad", h_grad)


def gradient_control_operator(
    system: CoupledSpinSystem,
    *,
    control_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Return the base gradient-control operator ``TAU * Iz`` for a spin system.

    A magnetic-field gradient couples to a spin as a *position-dependent*
    offset -- an Iz term whose weight is ``gradient * position`` (the
    ``positions @ gradient`` convention of the motion/imaging engine, see
    ``workflows.slice_selective``). This returns the position-independent base
    ``TAU * total_operator(nspin, "z", indices)`` (radians/second per unit of
    ``gradient * position``, matching how ``h_x``/``h_y`` are TAU-scaled in
    :func:`coupled_spin_control_model`); scale it per ensemble member with
    :func:`position_gradient_batch`.
    """

    indices = None if control_indices is None else tuple(int(idx) for idx in control_indices)
    return TAU * total_operator(system.nspin, "z", indices)


def position_gradient_batch(
    h_grad_base: np.ndarray, positions: Iterable[float]
) -> list[np.ndarray]:
    """Scale a base gradient operator by each ensemble member's position.

    Returns ``[position_k * h_grad_base for position_k in positions]`` -- the
    per-member gradient operators the batched propagators (and
    ``objectives.make_grape_objective(gradient_operator_batch=...)``) consume.
    A single shared gradient waveform ``g(t)`` then produces the local offset
    ``g(t) * position_k`` at member ``k``.
    """

    base = np.asarray(h_grad_base, dtype=np.complex128)
    return [float(position) * base for position in positions]


def coupled_spin_control_model(
    system: CoupledSpinSystem,
    *,
    control_indices: Iterable[int] | None = None,
    include_j_coupling: bool = True,
    include_gradient: bool = False,
) -> ControlHamiltonianModel:
    """Build a GRAPE control model for a scalar-coupled spin-1/2 system.

    ``h_drift`` is the rotating-frame offset Hamiltonian (plus the secular J
    Hamiltonian when ``include_j_coupling``); ``h_x``/``h_y`` are the summed
    transverse RF-drive operators over ``control_indices`` (default: every
    spin in the system), reusing ``coupling.operators.total_operator`` --
    the same operators ``coupling.hamiltonians.rf_hamiltonian`` sums over a
    fixed amplitude/phase, generalized here to a per-segment control. When
    ``include_gradient`` is set, ``h_grad`` is populated with the base gradient
    operator (:func:`gradient_control_operator`), enabling the gradient control
    channel.
    """

    drift = zeeman_hamiltonian(system)
    if include_j_coupling:
        drift = drift + secular_j_hamiltonian(system)
    indices = None if control_indices is None else tuple(int(idx) for idx in control_indices)
    h_x = TAU * total_operator(system.nspin, "x", indices)
    h_y = TAU * total_operator(system.nspin, "y", indices)
    h_grad = (
        gradient_control_operator(system, control_indices=control_indices)
        if include_gradient
        else None
    )
    return ControlHamiltonianModel(
        dimension=system.dimension,
        h_drift=drift,
        h_x=h_x,
        h_y=h_y,
        h_grad=h_grad,
    )
