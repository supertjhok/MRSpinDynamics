"""Small numerical compatibility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(y: Any, x: Any | None = None, axis: int = -1) -> np.ndarray:
    """Integrate with NumPy's trapezoid rule across NumPy versions.

    NumPy 2.x exposes `np.trapezoid`; older NumPy releases expose `np.trapz`.
    Keep this wrapper so public examples work in common Anaconda environments.
    """

    integrator = np.trapezoid if hasattr(np, "trapezoid") else getattr(np, "trapz")
    return integrator(y, x=x, axis=axis)


def expm(matrix: Any) -> np.ndarray:
    """Return ``exp(matrix)``; the package's shared dense matrix exponential.

    Uses :func:`scipy.linalg.expm` (Al-Mohy & Higham scaling and squaring with
    Pade approximants) when SciPy is installed, and otherwise the dependency-
    free Pade-13 scaling-and-squaring method of Higham (2005). Both stay
    accurate for defective or near-defective (non-normal) generators such as
    relaxation and exchange Liouvillians at exceptional points, where an
    eigendecomposition fails. For large-norm Liouvillians (``||A||_1 ~ 1e5``)
    the SciPy path agrees with itself to rounding and preserves density-matrix
    traces to ~2e-13; the fallback to a few 1e-12 (it needs more squarings).
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
    try:
        from scipy.linalg import expm as scipy_expm
    except ImportError:  # pragma: no cover - exercised without the opt extra
        return _expm_pade13(a)
    return np.asarray(scipy_expm(a), dtype=np.complex128)


# Higham (2005), "The scaling and squaring method for the matrix exponential
# revisited", SIAM J. Matrix Anal. Appl. 26(4):1179-1193: Pade coefficients
# b_k and the 1-norm bounds theta_m for degrees m = 3, 5, 7, 9, 13 (double).
_PADE_COEFFICIENTS = {
    3: (120.0, 60.0, 12.0, 1.0),
    5: (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0),
    7: (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0),
    9: (
        17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
        2162160.0, 110880.0, 3960.0, 90.0, 1.0,
    ),
    13: (
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
        33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0,
    ),
}
_PADE_THETA = {
    3: 1.495585217958292e-2,
    5: 2.539398330063230e-1,
    7: 9.504178996162932e-1,
    9: 2.097847961257068e0,
    13: 5.371920351148152e0,
}


def _expm_pade13(a: np.ndarray) -> np.ndarray:
    """Higham (2005) Algorithm 2.3: Pade degree from the 1-norm, then squaring."""

    ident = np.eye(a.shape[0], dtype=np.complex128)
    norm = float(np.max(np.sum(np.abs(a), axis=0)))
    for degree in (3, 5, 7, 9):
        if norm <= _PADE_THETA[degree]:
            return _pade_approximant(a, degree, ident)
    squarings = max(0, int(np.ceil(np.log2(norm / _PADE_THETA[13]))))
    result = _pade_approximant(a / 2.0**squarings, 13, ident)
    for _ in range(squarings):
        result = result @ result
    return result


def _pade_approximant(a: np.ndarray, degree: int, ident: np.ndarray) -> np.ndarray:
    b = _PADE_COEFFICIENTS[degree]
    a2 = a @ a
    if degree == 13:
        a4 = a2 @ a2
        a6 = a4 @ a2
        u = a @ (
            a6 @ (b[13] * a6 + b[11] * a4 + b[9] * a2)
            + b[7] * a6 + b[5] * a4 + b[3] * a2 + b[1] * ident
        )
        v = (
            a6 @ (b[12] * a6 + b[10] * a4 + b[8] * a2)
            + b[6] * a6 + b[4] * a4 + b[2] * a2 + b[0] * ident
        )
    else:
        powers = [ident, a2]
        while len(powers) < (degree + 1) // 2:
            powers.append(powers[-1] @ a2)
        u = a @ sum(b[2 * k + 1] * power for k, power in enumerate(powers))
        v = sum(b[2 * k] * power for k, power in enumerate(powers))
    return np.linalg.solve(v - u, v + u)
