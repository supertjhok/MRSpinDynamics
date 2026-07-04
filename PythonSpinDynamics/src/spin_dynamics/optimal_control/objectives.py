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
from spin_dynamics.optimal_control.control_response import build_control_delivery

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import (
        propagate_batched,
        propagate_batched_controls,
        propagate_batched_grad,
        propagate_state_scan,
        propagate_unitary_batched,
        propagate_unitary_batched_grad,
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
    optimize_gradient: bool = False,
    fixed_gradient: float | np.ndarray | None = None,
    gradient_operator_batch: Sequence[np.ndarray] | None = None,
    control_operator_batch: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    hamiltonian_batch: Sequence[np.ndarray] | None = None,
    ensemble_reduction: Literal["mean", "worst_case"] = "mean",
    phase_smoothness_weight: float = 0.0,
    gradient_smoothness_weight: float = 0.0,
    propagator: Literal["eigh", "expm"] = "eigh",
    rf_response=None,
    gradient_response=None,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Build a ``value_and_grad`` callable for a GRAPE fidelity objective.

    Returns ``vg(x: np.ndarray) -> (fidelity, grad)`` using reverse-mode
    autodiff through :mod:`optimal_control._jax_propagation`. Phase-only by
    default (``x`` is the phase vector, amplitude fixed at ``fixed_amplitude``);
    pass ``optimize_amplitude=True`` for joint amplitude+phase control.
    ``hamiltonian_batch``, when given, turns this into a robust/ensemble
    objective: each entry is an ``H_drift`` variant (e.g. a different B0 offset
    or B1-scaled ``h_x``/``h_y`` pairing folded into a per-case drift),
    propagated with the SAME shared controls, then reduced via
    ``ensemble_reduction``.

    **Gradient control channel.** Passing ``gradient_operator_batch`` (a
    sequence of per-case, already position-scaled ``h_grad`` operators, e.g.
    from ``hamiltonians.position_gradient_batch``) activates a gradient control
    channel: every case is driven by ONE shared gradient waveform ``g(t)``
    whose position-dependent effect distinguishes the cases. Set
    ``optimize_gradient=True`` to optimize ``g(t)`` (appended to the control
    vector) or pass ``fixed_gradient`` to hold it constant. ``target`` may then
    be per-case, shape ``(batch, dim)`` for state transfer (e.g. inverted
    in-slice, untouched out-of-slice) as well as the shared ``(dim,)`` form.

    **Per-case control operators (e.g. NQR powder).** Passing
    ``control_operator_batch`` (a sequence of per-case ``(h_x, h_y)`` drive-
    operator pairs, e.g. from ``hamiltonians.nqr_powder_control_batch``) makes
    the RF drive operators vary per ensemble member while the scalar
    ``amplitude``/``phase`` waveform stays shared -- the powder-averaging case,
    where each crystallite's RF-to-EFG coupling differs. State-transfer mode
    only; not combinable with the gradient channel in this increment.

    Control-vector layout: ``concat([amplitude?, phase, gradient?])`` (each
    block ``n_segments``); the amplitude block is present only when
    ``optimize_amplitude``, the gradient block only when ``optimize_gradient``.

    **Hardware response (probe / gradient driver).** ``rf_response`` (an
    :class:`optimal_control.control_response.EnvelopeResponse`, e.g. a tuned or
    matched probe) and ``gradient_response`` (a
    :class:`~optimal_control.control_response.GradientDriverResponse`) insert an
    LTI actuator between the *commanded* control and the Hamiltonian: the command
    is upsampled by the response's ``oversample`` factor, filtered (RF: FFT
    multiply on the complex envelope; gradient: causal state-space cascade), and
    a ring-down/eddy tail is appended so its rotation of the spins is scored. Box
    constraints stay on the command, so the optimizer pre-emphasizes only up to
    real hardware limits. Both are optional and default to the ideal (unfiltered)
    behaviour bit-for-bit. Filters require a uniform (scalar) ``dt``.
    """

    _require_jax()
    if mode not in ("state_transfer", "gate"):
        raise ValueError("mode must be 'state_transfer' or 'gate'")
    if mode == "state_transfer" and psi0 is None:
        raise ValueError("psi0 is required for mode='state_transfer'")
    n_segments = int(n_segments)
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")

    gradient_active = gradient_operator_batch is not None
    if optimize_gradient and not gradient_active:
        raise ValueError(
            "optimize_gradient=True requires gradient_operator_batch "
            "(the per-case gradient operators to drive)"
        )

    controls_active = control_operator_batch is not None
    if controls_active and gradient_active:
        raise ValueError(
            "control_operator_batch and the gradient channel cannot be combined "
            "in this increment"
        )
    if controls_active and mode != "state_transfer":
        raise ValueError("control_operator_batch is supported only for mode='state_transfer'")

    if propagator not in ("eigh", "expm"):
        raise ValueError("propagator must be 'eigh' or 'expm'")
    if propagator == "expm" and (gradient_active or mode != "state_transfer"):
        raise ValueError(
            "propagator='expm' (for degenerate NQR spectra) is currently supported "
            "only for mode='state_transfer' without the gradient channel"
        )

    amp_fixed = None
    if not optimize_amplitude:
        if fixed_amplitude is None:
            raise ValueError("fixed_amplitude is required when optimize_amplitude=False")
        amp_fixed = jnp.broadcast_to(
            jnp.asarray(fixed_amplitude, dtype=jnp.float64), (n_segments,)
        )

    grad_fixed = None
    if gradient_active and not optimize_gradient:
        if fixed_gradient is None:
            raise ValueError(
                "fixed_gradient is required when a gradient channel is active "
                "but optimize_gradient=False"
            )
        grad_fixed = jnp.broadcast_to(
            jnp.asarray(fixed_gradient, dtype=jnp.float64), (n_segments,)
        )

    if gradient_response is not None and not gradient_active:
        raise ValueError(
            "gradient_response requires an active gradient channel "
            "(pass gradient_operator_batch)"
        )

    h_drift = jnp.asarray(model.h_drift, dtype=jnp.complex128)
    h_x = jnp.asarray(model.h_x, dtype=jnp.complex128)
    h_y = jnp.asarray(model.h_y, dtype=jnp.complex128)
    target_j = jnp.asarray(target, dtype=jnp.complex128)

    # Hardware-response actuator layer (probe / gradient driver). When active,
    # controls are upsampled, filtered and tail-extended onto a fine grid before
    # propagation; the propagators are agnostic to segment count, so only the
    # per-segment arrays and dt fed to them change. The transform is shared with
    # the PGSE objective via control_response.build_control_delivery.
    _deliver, _n_fine, _dt_fine = build_control_delivery(
        n_segments, dt, rf_response=rf_response, gradient_response=gradient_response
    )

    batch = None
    if hamiltonian_batch is not None:
        batch = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in hamiltonian_batch])

    hgrad_batch = None
    if gradient_active:
        hgrad_batch = jnp.stack(
            [jnp.asarray(h, dtype=jnp.complex128) for h in gradient_operator_batch]
        )
        grad_batch_size = hgrad_batch.shape[0]
        if batch is None:
            # No per-case drift supplied: every position shares the model drift.
            drift_batch = jnp.broadcast_to(
                h_drift, (grad_batch_size,) + h_drift.shape
            )
        else:
            if batch.shape[0] != grad_batch_size:
                raise ValueError(
                    "hamiltonian_batch and gradient_operator_batch must have "
                    "the same length"
                )
            drift_batch = batch
    else:
        drift_batch = None
        grad_batch_size = None

    hx_batch = None
    hy_batch = None
    controls_drift_batch = None
    controls_batch_size = None
    if controls_active:
        hx_batch = jnp.stack(
            [jnp.asarray(hx, dtype=jnp.complex128) for hx, _hy in control_operator_batch]
        )
        hy_batch = jnp.stack(
            [jnp.asarray(hy, dtype=jnp.complex128) for _hx, hy in control_operator_batch]
        )
        controls_batch_size = hx_batch.shape[0]
        if batch is None:
            # Shared drift (the usual powder case: H_Q is orientation-independent
            # in the PAS; only the RF direction, hence h_x/h_y, varies).
            controls_drift_batch = jnp.broadcast_to(
                h_drift, (controls_batch_size,) + h_drift.shape
            )
        else:
            if batch.shape[0] != controls_batch_size:
                raise ValueError(
                    "hamiltonian_batch and control_operator_batch must have the "
                    "same length"
                )
            controls_drift_batch = batch

    # Per-case (vs shared) target: an extra leading batch axis beyond the
    # base operand rank (1 for a state ket, 2 for a gate unitary).
    base_target_rank = 1 if mode == "state_transfer" else 2
    per_case_target = target_j.ndim > base_target_rank
    target_in_axis = 0 if per_case_target else None

    def _split(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        offset = 0
        if optimize_amplitude:
            amplitude = x[offset : offset + n_segments]
            offset += n_segments
        else:
            amplitude = amp_fixed
        phase = x[offset : offset + n_segments]
        offset += n_segments
        if optimize_gradient:
            gradient = x[offset : offset + n_segments]
        else:
            gradient = grad_fixed
        return amplitude, phase, gradient

    def _regularization(phase, gradient):
        penalty = phase_smoothness_weight * jnp.sum(jnp.diff(phase) ** 2)
        if gradient_active and gradient_smoothness_weight != 0.0:
            penalty = penalty + gradient_smoothness_weight * jnp.sum(jnp.diff(gradient) ** 2)
        return penalty

    if mode == "state_transfer":
        psi0_j = jnp.asarray(psi0, dtype=jnp.complex128)
        if gradient_active:
            ensemble_size = grad_batch_size
        elif controls_active:
            ensemble_size = controls_batch_size
        elif batch is not None:
            ensemble_size = batch.shape[0]
        else:
            ensemble_size = None
        psi0_batch = (
            jnp.broadcast_to(psi0_j, (ensemble_size,) + psi0_j.shape)
            if ensemble_size is not None
            else None
        )

        def score(x):
            amplitude_c, phase_c, gradient_c = _split(x)
            amplitude, phase, gradient, dt_eff = _deliver(amplitude_c, phase_c, gradient_c)
            if gradient_active:
                psi_finals = propagate_batched_grad(
                    drift_batch, h_x, h_y, hgrad_batch,
                    amplitude, phase, gradient, dt_eff, psi0_batch,
                )
                fids = jax.vmap(state_transfer_fidelity, in_axes=(0, target_in_axis))(
                    psi_finals, target_j
                )
                fidelity = robust_ensemble_fidelity(fids, reduction=ensemble_reduction)
            elif controls_active:
                psi_finals = propagate_batched_controls(
                    controls_drift_batch, hx_batch, hy_batch,
                    amplitude, phase, dt_eff, psi0_batch, method=propagator,
                )
                fids = jax.vmap(state_transfer_fidelity, in_axes=(0, target_in_axis))(
                    psi_finals, target_j
                )
                fidelity = robust_ensemble_fidelity(fids, reduction=ensemble_reduction)
            elif batch is not None:
                psi_finals = propagate_batched(
                    batch, h_x, h_y, amplitude, phase, dt_eff, psi0_batch, method=propagator
                )
                fids = jax.vmap(state_transfer_fidelity, in_axes=(0, target_in_axis))(
                    psi_finals, target_j
                )
                fidelity = robust_ensemble_fidelity(fids, reduction=ensemble_reduction)
            else:
                psi_final = propagate_state_scan(
                    h_drift, h_x, h_y, amplitude, phase, dt_eff, psi0_j, method=propagator
                )
                fidelity = state_transfer_fidelity(psi_final, target_j)
            return fidelity - _regularization(phase_c, gradient_c)

    else:  # mode == "gate"

        def score(x):
            amplitude_c, phase_c, gradient_c = _split(x)
            amplitude, phase, gradient, dt_eff = _deliver(amplitude_c, phase_c, gradient_c)
            if gradient_active:
                u_finals = propagate_unitary_batched_grad(
                    drift_batch, h_x, h_y, hgrad_batch, amplitude, phase, gradient, dt_eff
                )
                fids = jax.vmap(average_gate_fidelity, in_axes=(0, target_in_axis))(
                    u_finals, target_j
                )
                fidelity = robust_ensemble_fidelity(fids, reduction=ensemble_reduction)
            elif batch is not None:
                u_finals = propagate_unitary_batched(batch, h_x, h_y, amplitude, phase, dt_eff)
                fids = jax.vmap(average_gate_fidelity, in_axes=(0, target_in_axis))(
                    u_finals, target_j
                )
                fidelity = robust_ensemble_fidelity(fids, reduction=ensemble_reduction)
            else:
                u_final = propagate_unitary_scan(h_drift, h_x, h_y, amplitude, phase, dt_eff)
                fidelity = average_gate_fidelity(u_final, target_j)
            return fidelity - _regularization(phase_c, gradient_c)

    _vg = jax.jit(jax.value_and_grad(score))

    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = _vg(jnp.asarray(x, dtype=jnp.float64))
        return float(value), np.asarray(grad)

    return value_and_grad
