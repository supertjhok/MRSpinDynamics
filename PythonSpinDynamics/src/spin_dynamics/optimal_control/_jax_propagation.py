"""JAX-differentiable piecewise-constant control-Hamiltonian propagation.

Mirrors the ``core._jax_kernels`` ``lax.scan`` convention: the scan body is
traced once, so compiled program size and compile time are independent of the
number of control segments (an unrolled Python loop over ``propagator()``
calls would reproduce the multi-minute compile blowup documented in
``core._jax_kernels`` for long echo trains -- GRAPE waveforms routinely run
50-500+ segments).

Each segment applies ``expm(-i H_seg dt_seg)`` via Hermitian eigendecomposition
(``jnp.linalg.eigh``), the same construction ``coupling.evolution.propagator``
uses -- ported to ``jnp`` here so amplitude and phase become differentiable
(traced) leaves instead of the baked-in host-side constants they are in the
production ``core.kernels``/``core._jax_kernels`` arb10 path.

x64 is enabled on import, matching every other JAX module in this package.
"""

from __future__ import annotations

from functools import partial

import numpy as np

try:  # pragma: no cover - exercised by environment, not logic
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.linalg import expm as _jax_expm

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    JAX_AVAILABLE = False


if JAX_AVAILABLE:

    def _segment_propagator(h_drift, h_x, h_y, amplitude, phase, dt, method="eigh"):
        """``exp(-i H_seg dt)`` for one piecewise-constant control segment.

        ``method="eigh"`` (default) diagonalizes the Hermitian ``H_seg`` -- fast
        and machine-precise for non-degenerate spectra, and the mode M1/M2 use.
        ``method="expm"`` uses ``jax.scipy.linalg.expm``, whose reverse-mode
        gradient is well-defined even when eigenvalues are **degenerate**; the
        Hermitian-eigh gradient carries ``1/(lambda_i - lambda_j)`` terms that
        blow up (NaN) for degenerate spectra. Quadrupolar/NQR Hamiltonians have
        exact Kramers degeneracy, so NQR control models must use ``"expm"``.
        """

        h_seg = h_drift + amplitude * (jnp.cos(phase) * h_x + jnp.sin(phase) * h_y)
        if method == "expm":
            return _jax_expm(-1j * h_seg * dt)
        values, vectors = jnp.linalg.eigh(h_seg)
        seg_phases = jnp.exp(-1j * values * dt)
        return (vectors * seg_phases[jnp.newaxis, :]) @ vectors.conj().T

    def _broadcast_dt(dt, n_segments):
        return jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (n_segments,))

    @partial(jax.jit, static_argnames=("method",))
    def propagate_state_scan(h_drift, h_x, h_y, amplitude, phase, dt, psi0, *, method="eigh"):
        """Propagate a state vector through the control sequence.

        ``amplitude``/``phase`` have shape ``(n_segments,)``; ``dt`` is a
        scalar (uniform segments) or ``(n_segments,)``. Returns the final
        state, same shape as ``psi0``. ``method`` selects the segment
        propagator (see :func:`_segment_propagator`; ``"expm"`` for degenerate
        NQR spectra).
        """

        dt_arr = _broadcast_dt(dt, amplitude.shape[0])

        def body(psi, xs):
            amp_j, phase_j, dt_j = xs
            u_seg = _segment_propagator(h_drift, h_x, h_y, amp_j, phase_j, dt_j, method)
            return u_seg @ psi, None

        psi_final, _ = lax.scan(body, psi0, (amplitude, phase, dt_arr))
        return psi_final

    @jax.jit
    def propagate_unitary_scan(h_drift, h_x, h_y, amplitude, phase, dt):
        """Propagate the identity through the control sequence to a full unitary."""

        dt_arr = _broadcast_dt(dt, amplitude.shape[0])
        u0 = jnp.eye(h_drift.shape[0], dtype=jnp.complex128)

        def body(u, xs):
            amp_j, phase_j, dt_j = xs
            u_seg = _segment_propagator(h_drift, h_x, h_y, amp_j, phase_j, dt_j)
            return u_seg @ u, None

        u_final, _ = lax.scan(body, u0, (amplitude, phase, dt_arr))
        return u_final

    @partial(jax.jit, static_argnums=(2,))
    def iterate_unitary(u, psi0, n_iterations):
        """Repeatedly apply a fixed unitary ``u`` to ``psi0``.

        Returns the full trajectory, shape ``(n_iterations + 1, dim)``
        (``trajectory[0] == psi0``), via ``lax.scan``. This is the
        steady-state/echo-train primitive: for a refocusing-cycle unitary,
        repeated application models the CPMG echo train, and its convergence
        (or lack of it) as a function of the initial state is exactly what
        distinguishes axis-matched from conventional excitation.
        """

        def body(psi, _):
            psi_next = u @ psi
            return psi_next, psi_next

        _, rest = lax.scan(body, psi0, None, length=n_iterations)
        return jnp.concatenate([psi0[jnp.newaxis, :], rest], axis=0)

    @partial(jax.jit, static_argnames=("method",))
    def propagate_batched(h_drift_batch, h_x, h_y, amplitude, phase, dt, psi0_batch, *, method="eigh"):
        """vmap ``propagate_state_scan`` over a leading batch axis.

        ``h_drift_batch``/``psi0_batch`` carry the batch axis (ensemble
        members: B0 offset, B1 scale, orientation, ...); ``h_x``, ``h_y``,
        ``amplitude``, ``phase``, ``dt`` are shared controls, identical for
        every ensemble member. This is the robustness/ensemble primitive."""

        def one(h_drift_b, psi0_b):
            return propagate_state_scan(
                h_drift_b, h_x, h_y, amplitude, phase, dt, psi0_b, method=method
            )

        return jax.vmap(one)(h_drift_batch, psi0_batch)

    @jax.jit
    def propagate_unitary_batched(h_drift_batch, h_x, h_y, amplitude, phase, dt):
        """vmap ``propagate_unitary_scan`` over a leading batch axis of ``h_drift_batch``."""

        def one(h_drift_b):
            return propagate_unitary_scan(h_drift_b, h_x, h_y, amplitude, phase, dt)

        return jax.vmap(one)(h_drift_batch)

    def _segment_propagator_grad(h_drift, h_x, h_y, h_grad, amplitude, phase, grad, dt, method="eigh"):
        """``expm(-i H_seg dt)`` for a segment with an added gradient-control term.

        Extends :func:`_segment_propagator` with a gradient channel:
        ``H_seg += grad * h_grad``. For slice-selective control ``h_grad`` is a
        position-scaled offset operator (``position * (TAU * Iz)``) and ``grad``
        the shared per-segment gradient amplitude (hertz per unit position), so
        ``grad * h_grad`` is the local off-resonance ``grad * position`` in the
        same rad/s units the drift uses -- the exact ``positions @ gradient``
        coupling the motion/imaging engine applies. ``method="expm"`` uses the
        degeneracy-safe matrix exponential (needed when a segment's ``H_seg`` is
        degenerate -- e.g. an on-resonance spin during an RF/gradient gap has
        ``H_seg == 0``, whose ``eigh`` gradient is NaN)."""

        h_seg = (
            h_drift
            + amplitude * (jnp.cos(phase) * h_x + jnp.sin(phase) * h_y)
            + grad * h_grad
        )
        if method == "expm":
            return _jax_expm(-1j * h_seg * dt)
        values, vectors = jnp.linalg.eigh(h_seg)
        seg_phases = jnp.exp(-1j * values * dt)
        return (vectors * seg_phases[jnp.newaxis, :]) @ vectors.conj().T

    @partial(jax.jit, static_argnames=("method",))
    def propagate_state_scan_grad(
        h_drift, h_x, h_y, h_grad, amplitude, phase, gradient, dt, psi0, *, method="eigh"
    ):
        """Propagate a state through an RF+gradient control sequence.

        Like :func:`propagate_state_scan` with an extra ``gradient`` control
        channel, shape ``(n_segments,)``, coupling through the (already
        position-scaled) ``h_grad`` operator. Returns the final state.
        ``method="expm"`` selects the degeneracy-safe segment propagator (see
        :func:`_segment_propagator_grad`)."""

        dt_arr = _broadcast_dt(dt, amplitude.shape[0])

        def body(psi, xs):
            amp_j, phase_j, grad_j, dt_j = xs
            u_seg = _segment_propagator_grad(
                h_drift, h_x, h_y, h_grad, amp_j, phase_j, grad_j, dt_j, method
            )
            return u_seg @ psi, None

        psi_final, _ = lax.scan(body, psi0, (amplitude, phase, gradient, dt_arr))
        return psi_final

    @jax.jit
    def propagate_unitary_scan_grad(h_drift, h_x, h_y, h_grad, amplitude, phase, gradient, dt):
        """Propagate the identity through an RF+gradient control sequence."""

        dt_arr = _broadcast_dt(dt, amplitude.shape[0])
        u0 = jnp.eye(h_drift.shape[0], dtype=jnp.complex128)

        def body(u, xs):
            amp_j, phase_j, grad_j, dt_j = xs
            u_seg = _segment_propagator_grad(
                h_drift, h_x, h_y, h_grad, amp_j, phase_j, grad_j, dt_j
            )
            return u_seg @ u, None

        u_final, _ = lax.scan(body, u0, (amplitude, phase, gradient, dt_arr))
        return u_final

    @jax.jit
    def propagate_batched_grad(
        h_drift_batch, h_x, h_y, h_grad_batch, amplitude, phase, gradient, dt, psi0_batch
    ):
        """vmap ``propagate_state_scan_grad`` over a position/ensemble axis.

        ``h_drift_batch``, ``h_grad_batch`` and ``psi0_batch`` carry the batch
        axis; ``h_grad_batch[k] = position_k * h_grad_base`` gives each ensemble
        member its own gradient-operator scale. The RF (``h_x``, ``h_y``,
        ``amplitude``, ``phase``) AND the ``gradient`` waveform are shared,
        identical for every member -- the single control waveform whose
        position-dependent effect is exactly what the gradient channel models."""

        def one(h_drift_b, h_grad_b, psi0_b):
            return propagate_state_scan_grad(
                h_drift_b, h_x, h_y, h_grad_b, amplitude, phase, gradient, dt, psi0_b
            )

        return jax.vmap(one)(h_drift_batch, h_grad_batch, psi0_batch)

    @jax.jit
    def propagate_unitary_batched_grad(
        h_drift_batch, h_x, h_y, h_grad_batch, amplitude, phase, gradient, dt
    ):
        """vmap ``propagate_unitary_scan_grad`` over ``(h_drift_batch, h_grad_batch)``."""

        def one(h_drift_b, h_grad_b):
            return propagate_unitary_scan_grad(
                h_drift_b, h_x, h_y, h_grad_b, amplitude, phase, gradient, dt
            )

        return jax.vmap(one)(h_drift_batch, h_grad_batch)

    @partial(jax.jit, static_argnames=("method",))
    def propagate_batched_grad_controls(
        h_drift_batch, h_x_batch, h_y_batch, h_grad_batch,
        amplitude, phase, gradient, dt, psi0_batch, *, method="eigh",
    ):
        """vmap ``propagate_state_scan_grad`` over per-case drift, drive AND gradient operators.

        The combined-ensemble primitive for PGSE-style optimal control over an
        inhomogeneous ``(B0, B1, position)`` grid: unlike
        :func:`propagate_batched_grad` (shared ``h_x``/``h_y``), every operator
        carries the batch axis -- ``h_drift_batch`` (B0 offset), ``h_x_batch``/
        ``h_y_batch`` (per-case RF drive, e.g. a B1 scale ``beta_k*(h_x,h_y)``)
        and ``h_grad_batch`` (position-scaled gradient operator ``r_k*h_grad``).
        The RF ``amplitude``/``phase`` and the ``gradient`` waveform are the
        shared controls. Returns the per-case final states, ``(batch, dim)``."""

        def one(h_drift_b, h_x_b, h_y_b, h_grad_b, psi0_b):
            return propagate_state_scan_grad(
                h_drift_b, h_x_b, h_y_b, h_grad_b,
                amplitude, phase, gradient, dt, psi0_b, method=method,
            )

        return jax.vmap(one)(
            h_drift_batch, h_x_batch, h_y_batch, h_grad_batch, psi0_batch
        )

    @partial(jax.jit, static_argnames=("method",))
    def propagate_batched_controls(
        h_drift_batch, h_x_batch, h_y_batch, amplitude, phase, dt, psi0_batch, *, method="eigh"
    ):
        """vmap ``propagate_state_scan`` over per-case drift AND drive operators.

        Unlike :func:`propagate_batched` (which shares ``h_x``/``h_y`` across the
        batch), here ``h_x_batch``/``h_y_batch`` carry the batch axis too: each
        ensemble member has its own RF drive operators. This is the NQR
        powder-averaging primitive -- a crystallite's RF-to-EFG coupling depends
        on the RF direction in the principal-axis system, so ``h_x``/``h_y`` vary
        per orientation while the (scalar) RF ``amplitude``/``phase`` waveform is
        the one shared control. ``h_drift_batch`` and ``psi0_batch`` also carry
        the batch axis (``h_drift`` is typically shared -- broadcast by the
        caller -- but per-case is supported for combined offset+orientation
        ensembles)."""

        def one(h_drift_b, h_x_b, h_y_b, psi0_b):
            return propagate_state_scan(
                h_drift_b, h_x_b, h_y_b, amplitude, phase, dt, psi0_b, method=method
            )

        return jax.vmap(one)(h_drift_batch, h_x_batch, h_y_batch, psi0_batch)


def propagate_state_numpy(h_drift, h_x, h_y, amplitude, phase, dt, psi0):
    """Plain-NumPy reference implementation (no JAX required).

    Segment-by-segment ``expm(-i H_seg dt)`` via ``numpy.linalg.eigh``,
    algorithmically identical to :func:`propagate_state_scan` -- used to
    validate the JAX path without requiring the optional ``jax`` extra, and
    as the ground truth in parity tests.
    """

    h_drift = np.asarray(h_drift, dtype=np.complex128)
    h_x = np.asarray(h_x, dtype=np.complex128)
    h_y = np.asarray(h_y, dtype=np.complex128)
    amplitude = np.asarray(amplitude, dtype=np.float64).reshape(-1)
    phase = np.asarray(phase, dtype=np.float64).reshape(-1)
    n_segments = amplitude.size
    dt_arr = np.broadcast_to(np.asarray(dt, dtype=np.float64), (n_segments,))
    psi = np.asarray(psi0, dtype=np.complex128).copy()
    for amp_j, phase_j, dt_j in zip(amplitude, phase, dt_arr):
        h_seg = h_drift + amp_j * (np.cos(phase_j) * h_x + np.sin(phase_j) * h_y)
        values, vectors = np.linalg.eigh(h_seg)
        seg_phases = np.exp(-1j * values * dt_j)
        u_seg = (vectors * seg_phases[np.newaxis, :]) @ vectors.conj().T
        psi = u_seg @ psi
    return psi


def propagate_state_numpy_grad(
    h_drift, h_x, h_y, h_grad, amplitude, phase, gradient, dt, psi0
):
    """Plain-NumPy reference for the RF+gradient propagator.

    Algorithmically identical to :func:`propagate_state_scan_grad`; the ground
    truth for the gradient-channel parity tests, and usable without the ``jax``
    extra (e.g. to evaluate an optimized waveform across a position grid).
    """

    h_drift = np.asarray(h_drift, dtype=np.complex128)
    h_x = np.asarray(h_x, dtype=np.complex128)
    h_y = np.asarray(h_y, dtype=np.complex128)
    h_grad = np.asarray(h_grad, dtype=np.complex128)
    amplitude = np.asarray(amplitude, dtype=np.float64).reshape(-1)
    phase = np.asarray(phase, dtype=np.float64).reshape(-1)
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    n_segments = amplitude.size
    dt_arr = np.broadcast_to(np.asarray(dt, dtype=np.float64), (n_segments,))
    psi = np.asarray(psi0, dtype=np.complex128).copy()
    for amp_j, phase_j, grad_j, dt_j in zip(amplitude, phase, gradient, dt_arr):
        h_seg = (
            h_drift
            + amp_j * (np.cos(phase_j) * h_x + np.sin(phase_j) * h_y)
            + grad_j * h_grad
        )
        values, vectors = np.linalg.eigh(h_seg)
        seg_phases = np.exp(-1j * values * dt_j)
        u_seg = (vectors * seg_phases[np.newaxis, :]) @ vectors.conj().T
        psi = u_seg @ psi
    return psi


def propagate_batched_grad_controls_numpy(
    h_drift_batch, h_x_batch, h_y_batch, h_grad_batch,
    amplitude, phase, gradient, dt, psi0_batch,
):
    """Plain-NumPy reference for :func:`propagate_batched_grad_controls`.

    Loops the single-case NumPy gradient propagator over the batch axis; the
    ground truth for the combined (B0 x B1 x position) parity tests and usable
    without the ``jax`` extra.
    """

    return np.stack([
        propagate_state_numpy_grad(
            h_drift_batch[k], h_x_batch[k], h_y_batch[k], h_grad_batch[k],
            amplitude, phase, gradient, dt, psi0_batch[k],
        )
        for k in range(len(psi0_batch))
    ])
