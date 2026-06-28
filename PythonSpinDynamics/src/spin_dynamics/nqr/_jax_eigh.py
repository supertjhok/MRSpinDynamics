"""JAX batched Hermitian eigensolve for NQR/ESR dense scans (Phase 4).

``jnp.linalg.eigh`` broadcasts over the leading axis and runs on GPU, so a whole
powder grid (or field sweep) of small Hamiltonians diagonalizes in one call.
x64 is enabled so eigenvalues match the NumPy reference. Requires the optional
``jax`` extra. See ``docs/performance.md``.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised by environment, not logic
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    JAX_AVAILABLE = True

    _eigh = jax.jit(jnp.linalg.eigh)
except Exception:  # pragma: no cover - import guard
    JAX_AVAILABLE = False


def batched_eigh(hamiltonians: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, eigenvectors)`` for a stack of Hermitian matrices."""

    if not JAX_AVAILABLE:  # pragma: no cover - guarded by caller
        raise ImportError("jax is not installed")
    values, vectors = _eigh(jnp.asarray(hamiltonians, dtype=jnp.complex128))
    return np.asarray(values), np.asarray(vectors)
