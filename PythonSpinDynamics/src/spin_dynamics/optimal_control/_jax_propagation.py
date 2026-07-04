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

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    JAX_AVAILABLE = False


if JAX_AVAILABLE:

    def _segment_propagator(h_drift, h_x, h_y, amplitude, phase, dt):
        """``expm(-i H_seg dt)`` for one piecewise-constant control segment."""

        h_seg = h_drift + amplitude * (jnp.cos(phase) * h_x + jnp.sin(phase) * h_y)
        values, vectors = jnp.linalg.eigh(h_seg)
        seg_phases = jnp.exp(-1j * values * dt)
        return (vectors * seg_phases[jnp.newaxis, :]) @ vectors.conj().T

    def _broadcast_dt(dt, n_segments):
        return jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (n_segments,))

    @jax.jit
    def propagate_state_scan(h_drift, h_x, h_y, amplitude, phase, dt, psi0):
        """Propagate a state vector through the control sequence.

        ``amplitude``/``phase`` have shape ``(n_segments,)``; ``dt`` is a
        scalar (uniform segments) or ``(n_segments,)``. Returns the final
        state, same shape as ``psi0``.
        """

        dt_arr = _broadcast_dt(dt, amplitude.shape[0])

        def body(psi, xs):
            amp_j, phase_j, dt_j = xs
            u_seg = _segment_propagator(h_drift, h_x, h_y, amp_j, phase_j, dt_j)
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

    @jax.jit
    def propagate_batched(h_drift_batch, h_x, h_y, amplitude, phase, dt, psi0_batch):
        """vmap ``propagate_state_scan`` over a leading batch axis.

        ``h_drift_batch``/``psi0_batch`` carry the batch axis (ensemble
        members: B0 offset, B1 scale, orientation, ...); ``h_x``, ``h_y``,
        ``amplitude``, ``phase``, ``dt`` are shared controls, identical for
        every ensemble member. This is the robustness/ensemble primitive."""

        def one(h_drift_b, psi0_b):
            return propagate_state_scan(h_drift_b, h_x, h_y, amplitude, phase, dt, psi0_b)

        return jax.vmap(one)(h_drift_batch, psi0_batch)

    @jax.jit
    def propagate_unitary_batched(h_drift_batch, h_x, h_y, amplitude, phase, dt):
        """vmap ``propagate_unitary_scan`` over a leading batch axis of ``h_drift_batch``."""

        def one(h_drift_b):
            return propagate_unitary_scan(h_drift_b, h_x, h_y, amplitude, phase, dt)

        return jax.vmap(one)(h_drift_batch)


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
