"""JAX batched Hermitian eigensolve for NQR/ESR dense scans (Phase 4).

``jnp.linalg.eigh`` broadcasts over the leading axis and runs on GPU, so a whole
powder grid (or field sweep) of small Hamiltonians diagonalizes in one call.
The solve runs with JAX x64 enabled only for its own duration
(``core._jax_kernels.x64_scope``) so eigenvalues match the NumPy reference
without changing the process-wide ``jax_enable_x64``. Requires the optional
``jax`` extra. See ``docs/performance.md``.
"""

from __future__ import annotations

import numpy as np

from spin_dynamics.core._jax_kernels import x64_enabled

try:  # pragma: no cover - exercised by environment, not logic
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True

    _eigh = jax.jit(jnp.linalg.eigh)
except Exception:  # pragma: no cover - import guard
    JAX_AVAILABLE = False


@x64_enabled
def batched_eigh(hamiltonians: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, eigenvectors)`` for a stack of Hermitian matrices."""

    if not JAX_AVAILABLE:  # pragma: no cover - guarded by caller
        raise ImportError("jax is not installed")
    values, vectors = _eigh(jnp.asarray(hamiltonians, dtype=jnp.complex128))
    return np.asarray(values), np.asarray(vectors)
