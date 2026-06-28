"""Numba-compiled spin-dynamics kernels (Phase 1 of the acceleration plan).

The compiled kernels are written as explicit scalar loops over segments and
isochromats: this is where Numba beats vectorized NumPy, because it fuses the
nine complex multiplies of each coherence-basis rotation into a single pass with
no temporary arrays. The loop is released from the GIL (``nogil=True``) so the
existing isochromat-chunking ``ThreadPoolExecutor`` in ``core.kernels`` actually
runs in parallel.

If ``numba`` is not installed, ``NUMBA_AVAILABLE`` is ``False`` and ``njit``
degrades to a no-op decorator. The functions then run as ordinary (slow) Python,
which keeps them importable and unit-testable, but callers should gate on
``NUMBA_AVAILABLE`` before using them on production-sized grids.

See ``docs/performance.md``.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised by environment, not logic
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op fallback when numba is absent (runs pure-Python)."""

        def _decorate(func):
            return func

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _decorate


@njit(cache=True, nogil=True, fastmath=False)
def arb10_core(
    tp,
    pul,
    amp,
    acq,
    grad,
    del_w0,
    del_wg,
    t1n,
    t2n,
    m0,
    mth,
    rstack,
    n_acq,
):
    """Core segment loop for ``sim_spin_dynamics_arb10``.

    Parameters mirror the validated NumPy kernel. ``acq`` is a ``uint8`` mask
    and ``rstack`` has shape ``(num_pulses, 3, 3, numpts)`` (output component as
    the first matrix axis, input component as the second). Returns the acquired
    spectra of shape ``(n_acq, numpts)``.
    """

    numpts = del_w0.shape[0]
    nseg = tp.shape[0]

    m0row = np.empty(numpts, dtype=np.complex128)
    mmrow = np.zeros(numpts, dtype=np.complex128)
    mprow = np.zeros(numpts, dtype=np.complex128)
    for k in range(numpts):
        m0row[k] = m0[k]

    macq = np.zeros((n_acq, numpts), dtype=np.complex128)
    acq_cnt = 0

    for j in range(nseg):
        if amp[j] > 0.0:
            p = pul[j] - 1
            for k in range(numpts):
                a0 = m0row[k]
                am = mmrow[k]
                ap = mprow[k]
                m0row[k] = (
                    rstack[p, 0, 0, k] * a0
                    + rstack[p, 0, 1, k] * am
                    + rstack[p, 0, 2, k] * ap
                )
                mmrow[k] = (
                    rstack[p, 1, 0, k] * a0
                    + rstack[p, 1, 1, k] * am
                    + rstack[p, 1, 2, k] * ap
                )
                mprow[k] = (
                    rstack[p, 2, 0, k] * a0
                    + rstack[p, 2, 1, k] * am
                    + rstack[p, 2, 2, k] * ap
                )
        else:
            tf = tp[j]
            g = grad[j]
            for k in range(numpts):
                dw = del_w0[k] + g * del_wg[k]
                lon = np.exp(-tf / t1n[k])
                tr = np.exp(-tf / t2n[k]) * np.exp(1j * dw * tf)
                m0row[k] = lon * m0row[k] + mth[k] * (1.0 - lon)
                mmrow[k] = tr.conjugate() * mmrow[k]
                mprow[k] = tr * mprow[k]

        if acq[j] != 0:
            for k in range(numpts):
                macq[acq_cnt, k] = mmrow[k]
            acq_cnt += 1

    return macq
