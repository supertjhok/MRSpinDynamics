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

from collections.abc import Iterable, Sequence
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


def nqr_site_control_model(
    site,
    *,
    rf_frequency_hz: float | None = None,
    b1_direction_pas: Sequence[float] = (1.0, 0.0, 0.0),
    b0_vector_tesla_pas: Sequence[float] | np.ndarray | None = None,
) -> ControlHamiltonianModel:
    """Build a GRAPE control model for a quadrupolar (NQR) site.

    Works in the **rotating frame at the RF carrier**, in the energy eigenbasis,
    exactly matching the NQR density-matrix engine (``nqr.full_dynamics``): a
    piecewise-constant lab/PAS field cannot drive an MHz quadrupolar transition,
    so control must live in this frame. ``h_drift`` is the rotating-frame static
    Hamiltonian (``static_hamiltonian_rotating``, diagonal detunings in the
    eigenbasis) and ``h_x``/``h_y`` are the co-rotating RWA drive operators,
    extracted from ``pulse_hamiltonian`` so the amplitude/phase convention is
    identical to :func:`coupled_spin_control_model` -- namely ``pulse_hamiltonian
    == h_drift + w1 * (cos(phi) * h_x + sin(phi) * h_y)``. Consequently a pulse
    optimized on this model round-trips exactly through the real NQR engine.

    ``rf_frequency_hz`` defaults to the fundamental (strongest) transition; a
    warning is issued (via the engine's guard) if the carrier does not address a
    single-quantum coherence, since the single-carrier RWA is faithful only near
    the fundamental line. ``b1_direction_pas`` is the RF field direction in the
    EFG principal-axis system (the axis that varies across a powder). Pass
    ``b0_vector_tesla_pas`` to include a weak static Zeeman perturbation.
    """

    from spin_dynamics.nqr.full_dynamics import (
        _default_carrier_hz,
        _warn_if_rwa_carrier_invalid,
        pulse_hamiltonian,
        static_hamiltonian_rotating,
    )
    from spin_dynamics.nqr.hamiltonians import diagonalize_site

    eigensystem = diagonalize_site(site, b0_vector_tesla_pas)
    rf = float(rf_frequency_hz) if rf_frequency_hz is not None else _default_carrier_hz(eigensystem)
    _warn_if_rwa_carrier_invalid(eigensystem, rf)

    h_drift = static_hamiltonian_rotating(eigensystem, rf)
    # h_x/h_y are the pure drive quadratures: pulse_hamiltonian already includes
    # h_drift, and is linear in nutation, so subtracting h_drift and evaluating
    # at phi=0 / phi=pi/2 isolates cos/sin quadratures at unit (1 Hz) nutation.
    pulse_x = pulse_hamiltonian(
        eigensystem, nutation_hz=1.0, rf_frequency_hz=rf, phase=0.0,
        b1_direction_pas=b1_direction_pas,
    )
    pulse_y = pulse_hamiltonian(
        eigensystem, nutation_hz=1.0, rf_frequency_hz=rf, phase=np.pi / 2.0,
        b1_direction_pas=b1_direction_pas,
    )
    h_x = pulse_x - h_drift
    h_y = pulse_y - h_drift
    return ControlHamiltonianModel(
        dimension=eigensystem.site.dimension,
        h_drift=h_drift,
        h_x=h_x,
        h_y=h_y,
    )


def nqr_fundamental_states(
    site,
    *,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray | None = None,
) -> tuple[int, int]:
    """Return the ``(lower, upper)`` eigen-indices of the fundamental NQR line.

    In the eigenbasis the model operates in (see :func:`nqr_site_control_model`),
    eigenstate ``k`` is the standard basis vector ``e_k``, so these indices give
    the natural GRAPE inversion target ``psi0 = e_lower``, ``target = e_upper``
    for the strongest (fundamental) transition.
    """

    from spin_dynamics.nqr.hamiltonians import diagonalize_site

    eigensystem = diagonalize_site(site, b0_vector_tesla_pas)
    if not eigensystem.transitions:
        raise ValueError("site has no RF-active transitions")
    fundamental = max(eigensystem.transitions, key=lambda t: t.strength)
    return int(fundamental.lower), int(fundamental.upper)


def nqr_powder_control_batch(
    site,
    orientations,
    *,
    rf_frequency_hz: float | None = None,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-orientation ``(h_x, h_y)`` drive operators for a powder ensemble.

    A crystallite's RF-to-EFG coupling depends on the RF field direction in the
    PAS, so ``h_x``/``h_y`` vary across a powder while ``h_drift`` (the
    quadrupolar interaction) is shared. Each ``orientations`` entry is an
    ``nqr.orientations.OrientationSample`` (its ``b1_direction_pas`` sets the RF
    direction). Feed the returned list to
    ``objectives.make_grape_objective(control_operator_batch=...)`` /
    ``solvers.grape_optimize(control_operator_batch=...)`` for powder-robust
    optimization.
    """

    batch: list[tuple[np.ndarray, np.ndarray]] = []
    for sample in orientations:
        model = nqr_site_control_model(
            site,
            rf_frequency_hz=rf_frequency_hz,
            b1_direction_pas=sample.b1_direction_pas,
            b0_vector_tesla_pas=b0_vector_tesla_pas,
        )
        batch.append((model.h_x, model.h_y))
    return batch
