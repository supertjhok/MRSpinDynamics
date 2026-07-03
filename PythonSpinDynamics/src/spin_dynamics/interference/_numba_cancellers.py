"""Numba-compiled inner loops for the adaptive RFI cancellers.

The LMS/NLMS and RLS cancellers are inherently sequential -- each sample's
prediction depends on the coefficient state left by the previous one -- so there
is no vectorized-across-samples NumPy form. The per-sample work is a handful of
small complex vector (LMS) or matrix (RLS) operations, and the Python-level
per-sample overhead dominates for long records. These kernels rewrite the loops
as explicit scalar passes that Numba compiles to native code.

If ``numba`` is not installed, ``NUMBA_AVAILABLE`` is ``False`` and ``njit``
degrades to a no-op decorator, so the kernels stay importable and run as ordinary
(slow) Python -- which keeps them unit-testable for parity against the NumPy
reference paths in :mod:`spin_dynamics.interference.cancellers`. Callers gate on
``NUMBA_AVAILABLE`` and fall back to the NumPy loop otherwise.

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
def lms_kernel(design, y, update, step, normalized, epsilon, leak, coeff):
    """Mask-aware LMS/NLMS pass. Mirrors the NumPy loop element-for-element.

    ``design`` is the ``(N, D)`` tapped reference matrix, ``update`` a ``uint8``
    mask, and ``coeff`` the initial ``(D,)`` coefficient vector (modified in
    place). Returns ``(predicted, history)`` with shapes ``(N,)`` and ``(N, D)``.
    """

    num_samples = design.shape[0]
    dim = design.shape[1]
    predicted = np.zeros(num_samples, dtype=np.complex128)
    history = np.zeros((num_samples, dim), dtype=np.complex128)

    for idx in range(num_samples):
        acc = 0.0 + 0.0j
        for d in range(dim):
            acc += design[idx, d] * coeff[d]
        predicted[idx] = acc
        error = y[idx] - acc
        if update[idx]:
            if normalized:
                power = 0.0
                for d in range(dim):
                    phi = design[idx, d]
                    power += phi.real * phi.real + phi.imag * phi.imag
                scale = epsilon + power
            else:
                scale = 1.0
            gain = step / scale
            for d in range(dim):
                coeff[d] = (1.0 - leak) * coeff[d] + gain * np.conj(design[idx, d]) * error
        for d in range(dim):
            history[idx, d] = coeff[d]

    return predicted, history


@njit(cache=True, nogil=True, fastmath=False)
def rls_kernel(design, y, update, forgetting, coeff, covariance):
    """Mask-aware RLS pass. Mirrors the NumPy loop element-for-element.

    ``coeff`` (``(D,)``) and ``covariance`` (``(D, D)``) are the initial state and
    are modified in place. Returns ``(predicted, history)``.
    """

    num_samples = design.shape[0]
    dim = design.shape[1]
    predicted = np.zeros(num_samples, dtype=np.complex128)
    history = np.zeros((num_samples, dim), dtype=np.complex128)
    p_phi = np.zeros(dim, dtype=np.complex128)
    phi_cov = np.zeros(dim, dtype=np.complex128)

    for idx in range(num_samples):
        acc = 0.0 + 0.0j
        for d in range(dim):
            acc += design[idx, d] * coeff[d]
        predicted[idx] = acc
        error = y[idx] - acc
        if update[idx]:
            # p_phi = covariance @ conj(phi)
            for i in range(dim):
                acc_i = 0.0 + 0.0j
                for j in range(dim):
                    acc_i += covariance[i, j] * np.conj(design[idx, j])
                p_phi[i] = acc_i
            # denom = forgetting + phi @ p_phi
            denom = forgetting + 0.0j
            for i in range(dim):
                denom += design[idx, i] * p_phi[i]
            # phi_cov = phi @ covariance  (row vector)
            for j in range(dim):
                acc_j = 0.0 + 0.0j
                for i in range(dim):
                    acc_j += design[idx, i] * covariance[i, j]
                phi_cov[j] = acc_j
            for i in range(dim):
                gain_i = p_phi[i] / denom
                coeff[i] = coeff[i] + gain_i * error
                for j in range(dim):
                    covariance[i, j] = (covariance[i, j] - gain_i * phi_cov[j]) / forgetting
        for d in range(dim):
            history[idx, d] = coeff[d]

    return predicted, history
