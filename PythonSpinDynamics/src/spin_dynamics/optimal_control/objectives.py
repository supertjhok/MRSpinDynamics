"""Fidelity objectives and the GRAPE value-and-grad factory.

Implements the roadmap's two target fidelity notions -- state-to-state
transfer and full propagator/gate fidelity -- plus a robust/ensemble wrapper
that reduces fidelity over a batch of Hamiltonian variants (B0 offset, B1
scale, orientation, ...) to a scalar score. :func:`make_grape_objective` wires
these to :mod:`optimal_control._jax_propagation` and returns a
``jax.value_and_grad`` closure, the direct analogue of
``optimization._jax_objectives.make_ideal_v0crit_objective``.

Control-vector layout: phase-only (the default, and the primary mode for
switching-power-amplifier hardware that cannot vary amplitude) is a flat
``(n_segments,)`` phase array. Joint amplitude+phase control is
``concat([amplitude (n_segments,), phase (n_segments,)])``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import (
        propagate_batched,
        propagate_state_scan,
        propagate_unitary_batched,
        propagate_unitary_scan,
    )


def state_transfer_fidelity(psi_final, psi_target):
    """``|<target|final>|^2`` for normalized state vectors."""

    overlap = jnp.vdot(psi_target, psi_final)
    return jnp.real(overlap * jnp.conj(overlap))


def average_gate_fidelity(u_final, u_target):
    """Nielsen average-gate fidelity ``(|Tr(U_target^H U_final)|^2 + d) / (d(d+1))``."""

    dimension = u_target.shape[-1]
    overlap = jnp.trace(u_target.conj().T @ u_final)
    return (jnp.real(overlap * jnp.conj(overlap)) + dimension) / (dimension * (dimension + 1))


_SU2_AXIS_ANGLE_EPS = 1e-12


def su2_effective_axis(u):
    """Extract the effective rotation (axis, angle) of a 2x2 SU(2) unitary.

    Any single-spin-1/2 propagator built from a traceless Hamiltonian (as
    ``optimal_control.hamiltonians`` always builds) has ``det(U) == 1``, so
    ``U = cos(alpha/2) I - i sin(alpha/2) (nx sx + ny sy + nz sz)`` for a real
    unit vector ``n`` and angle ``alpha`` -- this is the same effective axis
    ``core.rotations.calc_rot_axis_arba4`` computes for a composite pulse
    (validated to agree with it to ~1e-15 for a single-spin cycle), extracted
    here directly from the propagator instead of the semiclassical
    axis-angle composition. Used to build axis-matched (AMEX) excitation
    targets: the state aligned with ``axis`` is the fixed point of repeated
    application of ``U`` (up to a global phase), so exciting onto it gives an
    asymptotic CPMG echo train with no transient.

    Returns ``(axis, angle)`` with ``axis`` shape ``(3,)`` (nx, ny, nz) and
    ``angle`` a scalar in ``[0, 2*pi)``. ``axis`` defaults to ``+z`` at
    ``angle == 0`` (identity, axis undefined) rather than dividing by zero.
    """

    cos_half = jnp.clip(jnp.real(u[0, 0]), -1.0, 1.0)
    angle = 2.0 * jnp.arccos(cos_half)
    sin_half = jnp.sin(angle / 2.0)
    degenerate = jnp.abs(sin_half) < _SU2_AXIS_ANGLE_EPS
    sin_half_safe = jnp.where(degenerate, 1.0, sin_half)
    nx = -jnp.imag(u[0, 1]) / sin_half_safe
    ny = -jnp.real(u[0, 1]) / sin_half_safe
    nz = jnp.imag(u[1, 1]) / sin_half_safe
    axis = jnp.where(degenerate, jnp.array([0.0, 0.0, 1.0]), jnp.stack([nx, ny, nz]))
    return axis, angle


def bloch_vector_to_ket(axis):
    """Return the spin-1/2 ket aligned with a Bloch-sphere unit vector.

    Standard spherical-to-spinor map: ``|n> = [cos(theta/2), sin(theta/2)
    exp(i*phi)]`` with ``theta = arccos(nz)``, ``phi = atan2(ny, nx)`` --
    ``U|n> = exp(-i*angle/2)|n>`` for the ``su2_effective_axis`` that produced
    ``axis``, i.e. ``|n>`` is the fixed point (up to global phase) of the
    corresponding rotation.
    """

    nx, ny, nz = axis[0], axis[1], axis[2]
    theta = jnp.arccos(jnp.clip(nz, -1.0, 1.0))
    phi = jnp.arctan2(ny, nx)
    return jnp.stack(
        [jnp.cos(theta / 2.0) + 0j, jnp.sin(theta / 2.0) * jnp.exp(1j * phi)]
    )


def robust_ensemble_fidelity(
    fidelity_per_case,
    *,
    reduction: Literal["mean", "worst_case"] = "mean",
):
    """Reduce a ``(batch,)`` array of per-case fidelities to a scalar score.

    ``"mean"`` (default) matches the roadmap's "averaging fidelity over
    B1-inhomogeneity and offset distributions" wording and has a well-behaved
    gradient everywhere. ``"worst_case"`` (``jnp.min``) has a gradient that
    flows only through the single worst ensemble member and can flip
    discontinuously between optimizer steps -- documented here, not hidden.
    """

    if reduction == "mean":
        return jnp.mean(fidelity_per_case)
    if reduction == "worst_case":
        return jnp.min(fidelity_per_case)
    raise ValueError("reduction must be 'mean' or 'worst_case'")


def _require_jax() -> None:
    if not JAX_AVAILABLE:
        raise ImportError(
            "GRAPE optimal control requires the optional 'jax' extra. "
            "Install it with `python -m pip install -e .[jax]` (or `.[perf]`)."
        )


def make_grape_objective(
    model,
    *,
    n_segments: int,
    dt,
    target,
    psi0=None,
    mode: Literal["state_transfer", "gate"] = "state_transfer",
    optimize_amplitude: bool = False,
    fixed_amplitude: float | np.ndarray | None = None,
    hamiltonian_batch: Sequence[np.ndarray] | None = None,
    ensemble_reduction: Literal["mean", "worst_case"] = "mean",
    phase_smoothness_weight: float = 0.0,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Build a ``value_and_grad`` callable for a GRAPE fidelity objective.

    Returns ``vg(x: np.ndarray) -> (fidelity, grad)`` using reverse-mode
    autodiff through :mod:`optimal_control._jax_propagation`. Phase-only by
    default (``x`` is the phase vector, amplitude fixed at ``fixed_amplitude``);
    pass ``optimize_amplitude=True`` for joint amplitude+phase control (``x``
    is then ``concat([amplitude, phase])``). ``hamiltonian_batch``, when
    given, turns this into a robust/ensemble objective: each entry is an
    ``H_drift`` variant (e.g. a different B0 offset or B1-scaled ``h_x``/
    ``h_y`` pairing folded into a per-case drift), propagated with the SAME
    shared controls, then reduced via ``ensemble_reduction``.
    """

    _require_jax()
    if mode not in ("state_transfer", "gate"):
        raise ValueError("mode must be 'state_transfer' or 'gate'")
    if mode == "state_transfer" and psi0 is None:
        raise ValueError("psi0 is required for mode='state_transfer'")
    n_segments = int(n_segments)
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")

    amp_fixed = None
    if not optimize_amplitude:
        if fixed_amplitude is None:
            raise ValueError("fixed_amplitude is required when optimize_amplitude=False")
        amp_fixed = jnp.broadcast_to(
            jnp.asarray(fixed_amplitude, dtype=jnp.float64), (n_segments,)
        )

    h_drift = jnp.asarray(model.h_drift, dtype=jnp.complex128)
    h_x = jnp.asarray(model.h_x, dtype=jnp.complex128)
    h_y = jnp.asarray(model.h_y, dtype=jnp.complex128)
    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    target_j = jnp.asarray(target, dtype=jnp.complex128)

    batch = None
    if hamiltonian_batch is not None:
        batch = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in hamiltonian_batch])

    def _split(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        if optimize_amplitude:
            return x[:n_segments], x[n_segments:]
        return amp_fixed, x

    if mode == "state_transfer":
        psi0_j = jnp.asarray(psi0, dtype=jnp.complex128)
        psi0_batch = (
            jnp.broadcast_to(psi0_j, (batch.shape[0],) + psi0_j.shape)
            if batch is not None
            else None
        )

        def score(x):
            amplitude, phase = _split(x)
            if batch is not None:
                psi_finals = propagate_batched(
                    batch, h_x, h_y, amplitude, phase, dt_j, psi0_batch
                )
                fids = jax.vmap(state_transfer_fidelity, in_axes=(0, None))(
                    psi_finals, target_j
                )
                fidelity = robust_ensemble_fidelity(fids, reduction=ensemble_reduction)
            else:
                psi_final = propagate_state_scan(
                    h_drift, h_x, h_y, amplitude, phase, dt_j, psi0_j
                )
                fidelity = state_transfer_fidelity(psi_final, target_j)
            penalty = phase_smoothness_weight * jnp.sum(jnp.diff(phase) ** 2)
            return fidelity - penalty

    else:  # mode == "gate"

        def score(x):
            amplitude, phase = _split(x)
            if batch is not None:
                u_finals = propagate_unitary_batched(batch, h_x, h_y, amplitude, phase, dt_j)
                fids = jax.vmap(average_gate_fidelity, in_axes=(0, None))(u_finals, target_j)
                fidelity = robust_ensemble_fidelity(fids, reduction=ensemble_reduction)
            else:
                u_final = propagate_unitary_scan(h_drift, h_x, h_y, amplitude, phase, dt_j)
                fidelity = average_gate_fidelity(u_final, target_j)
            penalty = phase_smoothness_weight * jnp.sum(jnp.diff(phase) ** 2)
            return fidelity - penalty

    _vg = jax.jit(jax.value_and_grad(score))

    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = _vg(jnp.asarray(x, dtype=jnp.float64))
        return float(value), np.asarray(grad)

    return value_and_grad
