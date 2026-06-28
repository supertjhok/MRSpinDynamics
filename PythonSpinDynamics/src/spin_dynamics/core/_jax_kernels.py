"""JAX-compiled spin-dynamics kernels (Phase 2 of the acceleration plan).

The arb10 segment loop is expressed with ``jax.lax.scan`` so the compiled
program is **independent of the train length**: the loop body is traced once and
XLA compiles a single fused kernel regardless of how many segments the sequence
has. (An earlier unrolled-trace version compiled the whole train into one giant
graph, which made compilation time and memory blow up for long echo trains.)

The sequence *structure* — which segments are pulses vs free precession, the
pulse index, the acquisition mask, the per-segment time and gradient — is passed
as the scan's per-step inputs (``xs``), i.e. as data rather than control flow.
The per-isochromat fields and the stacked pulse matrices are closed-over
constants. This keeps the kernel fully generic (one compilation serves every
sequence), ``vmap``-able over parameter sweeps, and differentiable for the
autodiff optimizer work (Phase 3).

x64 is enabled on import so complex128 matches the NumPy reference; without it
JAX would silently downcast to complex64 and fail the parity gate.

If ``jax`` is not installed, ``JAX_AVAILABLE`` is ``False`` and the entry points
raise a helpful error when called. See ``docs/performance.md``.
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

    def _arb10_single(
        rstack,
        is_pulse,
        pul_idx,
        acq_flag,
        tp,
        grad,
        del_w0,
        del_wg,
        t1n,
        t2n,
        m0,
        mth,
        n_acq,
    ):
        """Pure arb10 scan for one simulation (no jit; jit/vmap applied below)."""

        numpts = m0.shape[0]
        macq0 = jnp.zeros((n_acq, numpts), dtype=jnp.complex128)

        def body(carry, xs):
            m0r, mmr, mpr, macq, cnt = carry
            is_pulse_j, pul_j, acq_j, tf_j, grad_j = xs

            def pulse_branch(_):
                mat = rstack[pul_j]
                a0, am, ap = m0r, mmr, mpr
                return (
                    mat[0, 0] * a0 + mat[0, 1] * am + mat[0, 2] * ap,
                    mat[1, 0] * a0 + mat[1, 1] * am + mat[1, 2] * ap,
                    mat[2, 0] * a0 + mat[2, 1] * am + mat[2, 2] * ap,
                )

            def free_branch(_):
                dw = del_w0 + grad_j * del_wg
                lon = jnp.exp(-tf_j / t1n)
                tr = jnp.exp(-tf_j / t2n) * jnp.exp(1j * dw * tf_j)
                return (lon * m0r + mth * (1.0 - lon), jnp.conj(tr) * mmr, tr * mpr)

            nm0, nmm, nmp = lax.cond(is_pulse_j, pulse_branch, free_branch, None)

            def do_write(operands):
                macq_in, cnt_in = operands
                macq_out = lax.dynamic_update_slice(macq_in, nmm[None, :], (cnt_in, 0))
                return macq_out, cnt_in + 1

            macq, cnt = lax.cond(acq_j, do_write, lambda o: o, (macq, cnt))
            return (nm0, nmm, nmp, macq, cnt), None

        init = (
            m0,
            jnp.zeros(numpts, dtype=jnp.complex128),
            jnp.zeros(numpts, dtype=jnp.complex128),
            macq0,
            jnp.int64(0),
        )
        (_, _, _, macq, _), _ = lax.scan(
            body, init, (is_pulse, pul_idx, acq_flag, tp, grad)
        )
        return macq

    _arb10_scan = jax.jit(_arb10_single, static_argnums=(12,))

    def _arb10_branchless(
        rstack,
        is_pulse,
        pul_idx,
        acq_idx,
        tp,
        grad,
        del_w0,
        del_wg,
        t1n,
        t2n,
        m0,
        mth,
    ):
        """Branchless single-sim scan for the batched/GPU path.

        Differs from ``_arb10_single`` in two GPU-critical ways: the pulse/free
        choice uses ``jnp.where`` (both branches computed, no ``lax.cond``), and
        every step's transverse component is emitted as the scan output, then the
        acquired rows are gathered with ``acq_idx`` — avoiding the per-step
        ``dynamic_update_slice`` scatter that crippled the GPU. Both changes are
        numerically identical to the reference selection; the cost is a larger
        ``(nseg, numpts)`` transient, which is the right trade for throughput.
        """

        def body(carry, xs):
            m0r, mmr, mpr = carry
            is_pulse_j, pul_j, tf_j, grad_j = xs

            mat = rstack[pul_j]
            a0, am, ap = m0r, mmr, mpr
            p0 = mat[0, 0] * a0 + mat[0, 1] * am + mat[0, 2] * ap
            p1 = mat[1, 0] * a0 + mat[1, 1] * am + mat[1, 2] * ap
            p2 = mat[2, 0] * a0 + mat[2, 1] * am + mat[2, 2] * ap

            dw = del_w0 + grad_j * del_wg
            lon = jnp.exp(-tf_j / t1n)
            tr = jnp.exp(-tf_j / t2n) * jnp.exp(1j * dw * tf_j)
            f0 = lon * m0r + mth * (1.0 - lon)
            f1 = jnp.conj(tr) * mmr
            f2 = tr * mpr

            n0 = jnp.where(is_pulse_j, p0, f0)
            n1 = jnp.where(is_pulse_j, p1, f1)
            n2 = jnp.where(is_pulse_j, p2, f2)
            return (n0, n1, n2), n1

        init = (
            m0,
            jnp.zeros_like(m0),
            jnp.zeros_like(m0),
        )
        (_, _, _), all_mminus = lax.scan(
            body, init, (is_pulse, pul_idx, tp, grad)
        )
        return all_mminus[acq_idx]

    @partial(jax.jit, static_argnums=(12,))
    def _arb10_batched_cond(
        rstack,
        is_pulse,
        pul_idx,
        acq_flag,
        tp,
        grad,
        del_w0,
        del_wg,
        t1n,
        t2n,
        m0,
        mth,
        n_acq,
    ):
        """vmap of the memory-light ``cond`` single kernel (best on CPU)."""

        def one(rstack_b, del_w0_b, t1n_b, t2n_b, m0_b, mth_b):
            return _arb10_single(
                rstack_b, is_pulse, pul_idx, acq_flag, tp, grad,
                del_w0_b, del_wg, t1n_b, t2n_b, m0_b, mth_b, n_acq,
            )

        return jax.vmap(one)(rstack, del_w0, t1n, t2n, m0, mth)

    @jax.jit
    def _arb10_batched_branchless(
        rstack,
        is_pulse,
        pul_idx,
        acq_idx,
        tp,
        grad,
        del_w0,
        del_wg,
        t1n,
        t2n,
        m0,
        mth,
    ):
        """vmap of the branchless kernel over a leading batch axis (best on GPU).

        Batched (axis 0): ``rstack, del_w0, t1n, t2n, m0, mth``. Shared across
        the batch: the sequence structure (``is_pulse, pul_idx, acq_idx``), the
        per-segment timing (``tp, grad``) and ``del_wg``. This covers the common
        sweeps — coil Q (varies ``rstack``), relaxation (``t1n/t2n``), offset
        (``del_w0``), thermal (``m0/mth``) — all sharing one pulse program.
        """

        def one(rstack_b, del_w0_b, t1n_b, t2n_b, m0_b, mth_b):
            return _arb10_branchless(
                rstack_b,
                is_pulse,
                pul_idx,
                acq_idx,
                tp,
                grad,
                del_w0_b,
                del_wg,
                t1n_b,
                t2n_b,
                m0_b,
                mth_b,
            )

        return jax.vmap(one)(rstack, del_w0, t1n, t2n, m0, mth)


def run_arb10(tp, pul, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth, rstack):
    """Run the JAX arb10 kernel and return a NumPy ``(n_acq, numpts)`` array."""

    if not JAX_AVAILABLE:  # pragma: no cover - guarded by caller
        raise ImportError("jax is not installed")

    amp = np.asarray(amp)
    pul = np.asarray(pul)
    acq_np = np.asarray(acq).astype(bool)
    is_pulse = amp > 0
    # Zero-based pulse index; free-precession steps get -1 (never gathered).
    pul_idx = np.where(is_pulse, pul - 1, -1).astype(np.int64)
    n_acq = int(np.count_nonzero(acq_np))

    result = _arb10_scan(
        jnp.asarray(rstack, dtype=jnp.complex128),
        jnp.asarray(is_pulse),
        jnp.asarray(pul_idx),
        jnp.asarray(acq_np),
        jnp.asarray(tp, dtype=jnp.float64),
        jnp.asarray(grad, dtype=jnp.float64),
        jnp.asarray(del_w0, dtype=jnp.float64),
        jnp.asarray(del_wg, dtype=jnp.float64),
        jnp.asarray(t1n, dtype=jnp.float64),
        jnp.asarray(t2n, dtype=jnp.float64),
        jnp.asarray(m0, dtype=jnp.complex128),
        jnp.asarray(mth, dtype=jnp.complex128),
        n_acq,
    )
    return np.asarray(result)


def run_arb10_batched(
    tp, pul, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth, rstack
):
    """Run a batch of arb10 simulations in one vmapped call.

    The sequence structure (``tp, pul, amp, acq, grad, del_wg``) is shared; the
    batched per-case arrays carry a leading batch axis: ``del_w0, t1n, t2n, m0,
    mth`` with shape ``(batch, numpts)`` and ``rstack`` with shape
    ``(batch, num_pulses, 3, 3, numpts)``. Returns ``(batch, n_acq, numpts)``.

    The kernel is chosen by the default JAX device: the branchless,
    scatter-free variant on GPU (where ``cond``/``dynamic_update_slice`` are
    expensive), and the memory-light ``cond`` variant on CPU.
    """

    if not JAX_AVAILABLE:  # pragma: no cover - guarded by caller
        raise ImportError("jax is not installed")

    amp = np.asarray(amp)
    pul = np.asarray(pul)
    acq_np = np.asarray(acq).astype(bool)
    is_pulse = amp > 0

    rstack_j = jnp.asarray(rstack, dtype=jnp.complex128)
    is_pulse_j = jnp.asarray(is_pulse)
    tp_j = jnp.asarray(tp, dtype=jnp.float64)
    grad_j = jnp.asarray(grad, dtype=jnp.float64)
    del_w0_j = jnp.asarray(del_w0, dtype=jnp.float64)
    del_wg_j = jnp.asarray(del_wg, dtype=jnp.float64)
    t1n_j = jnp.asarray(t1n, dtype=jnp.float64)
    t2n_j = jnp.asarray(t2n, dtype=jnp.float64)
    m0_j = jnp.asarray(m0, dtype=jnp.complex128)
    mth_j = jnp.asarray(mth, dtype=jnp.complex128)

    on_gpu = jax.devices()[0].platform == "gpu"
    if on_gpu:
        pul_idx = np.where(is_pulse, pul - 1, 0).astype(np.int64)
        acq_idx = np.flatnonzero(acq_np).astype(np.int64)
        result = _arb10_batched_branchless(
            rstack_j, is_pulse_j, jnp.asarray(pul_idx), jnp.asarray(acq_idx),
            tp_j, grad_j, del_w0_j, del_wg_j, t1n_j, t2n_j, m0_j, mth_j,
        )
    else:
        pul_idx = np.where(is_pulse, pul - 1, -1).astype(np.int64)
        n_acq = int(np.count_nonzero(acq_np))
        result = _arb10_batched_cond(
            rstack_j, is_pulse_j, jnp.asarray(pul_idx), jnp.asarray(acq_np),
            tp_j, grad_j, del_w0_j, del_wg_j, t1n_j, t2n_j, m0_j, mth_j, n_acq,
        )
    return np.asarray(result)
