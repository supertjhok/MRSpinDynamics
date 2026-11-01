"""Small numerical compatibility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(y: Any, x: Any | None = None, axis: int = -1) -> np.ndarray:
    """Integrate with NumPy's trapezoid rule across NumPy versions.

    NumPy 2.x exposes `np.trapezoid`; older NumPy releases expose `np.trapz`.
    Keep this wrapper so public examples work in common Anaconda environments.
    """

    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return integrator(y, x=x, axis=axis)


def expm(matrix: Any) -> np.ndarray:
    """Return ``exp(matrix)`` by scaling and squaring a Taylor series.

    The matrix is scaled by ``2**s`` so its 1-norm is at most one, an 18-term
    Taylor series is summed (truncation below ``1/19! ~ 1e-17``), and the
    result is squared ``s`` times. Unlike an eigendecomposition this stays
    accurate for defective or near-defective (non-normal) generators such as
    relaxation and exchange Liouvillians at exceptional points. This is the
    package's shared dense matrix exponential.
    """

    a = np.asarray(matrix, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    dim = a.shape[0]
    norm = float(np.max(np.sum(np.abs(a), axis=0))) if a.size else 0.0
    if not np.isfinite(norm):
        raise ValueError("matrix must contain finite values")
    if norm == 0.0:
        return np.eye(dim, dtype=np.complex128)
    squarings = max(0, int(np.ceil(np.log2(norm))))
    scaled = a / (2.0**squarings)
    result = np.eye(dim, dtype=np.complex128)
    term = np.eye(dim, dtype=np.complex128)
    for k in range(1, 19):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result
