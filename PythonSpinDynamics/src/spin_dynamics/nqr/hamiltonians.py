"""Hamiltonian builders and transition analysis for NQR."""

from __future__ import annotations

import numpy as np

from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.systems import NQREigensystem, NQRTransition, QuadrupolarSite


TAU = 2.0 * np.pi
_AXES = ("x", "y", "z")


def quadrupole_frequency_scale_hz(site: QuadrupolarSite) -> float:
    """Return the Hamiltonian scale matching the public frequency parameter."""

    spin = float(site.spin)
    if np.isclose(spin, 1.0):
        denominator = 3.0
    elif np.isclose(spin, 1.5):
        denominator = 6.0
    else:
        raise ValueError(
            "quadrupole_frequency_hz is currently calibrated for spin=1 and spin=3/2"
        )
    return site.quadrupole_frequency_hz / denominator


def quadrupole_hamiltonian(site: QuadrupolarSite) -> np.ndarray:
    """Return the zero-field quadrupole Hamiltonian in radians per second."""

    ops = spin_matrices(site.spin)
    spin = site.spin
    quadrupole_operator = (
        3.0 * (ops.iz @ ops.iz)
        - spin * (spin + 1.0) * ops.identity
        + site.eta * (ops.ix @ ops.ix - ops.iy @ ops.iy)
    )
    scale_hz = quadrupole_frequency_scale_hz(site)
    return TAU * scale_hz * quadrupole_operator


def zeeman_hamiltonian(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float],
) -> np.ndarray:
    """Return the Zeeman Hamiltonian in radians per second."""

    b0 = np.asarray(b0_vector_tesla_pas, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_pas must be finite")
    ops = spin_matrices(site.spin)
    hz_operator = (
        b0[0] * ops.ix
        + b0[1] * ops.iy
        + b0[2] * ops.iz
    )
    return -TAU * site.gamma_hz_per_t * hz_operator


def nqr_hamiltonian(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Return the quadrupole plus optional Zeeman Hamiltonian."""

    hamiltonian = quadrupole_hamiltonian(site)
    if b0_vector_tesla_pas is not None:
        hamiltonian = hamiltonian + zeeman_hamiltonian(site, b0_vector_tesla_pas)
    return hamiltonian


def _transitions_from_eigensystem(
    site: QuadrupolarSite,
    levels_hz: np.ndarray,
    vectors: np.ndarray,
    operator_components: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    strength_tolerance: float,
    frequency_tolerance_hz: float,
) -> tuple[NQRTransition, ...]:
    """Build sorted transition metadata from one site's eigenpairs.

    Shared by :func:`diagonalize_site` and the batched powder path so the
    transition-selection logic has a single definition.
    """

    used_labels: set[str] = set()
    transitions: list[NQRTransition] = []
    for lower in range(site.dimension):
        for upper in range(lower + 1, site.dimension):
            frequency_hz = float(levels_hz[upper] - levels_hz[lower])
            if frequency_hz <= frequency_tolerance_hz:
                continue
            dipole = np.array(
                [
                    vectors[:, lower].conj().T @ op @ vectors[:, upper]
                    for op in operator_components
                ],
                dtype=np.complex128,
            )
            strength = float(np.linalg.norm(dipole))
            if strength <= strength_tolerance:
                continue
            axis_label = _AXES[int(np.argmax(np.abs(dipole)))]
            label = axis_label
            if label in used_labels:
                label = f"{axis_label}{len(used_labels) + 1}"
            used_labels.add(label)
            transitions.append(
                NQRTransition(
                    label=label,
                    lower=lower,
                    upper=upper,
                    frequency_hz=frequency_hz,
                    dipole_vector=dipole,
                    strength=strength,
                )
            )

    transitions.sort(key=lambda item: item.frequency_hz)
    return tuple(transitions)


def diagonalize_site(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    *,
    strength_tolerance: float = 1e-12,
    frequency_tolerance_hz: float = 1e-9,
) -> NQREigensystem:
    """Diagonalize a site Hamiltonian and return transition metadata."""

    hamiltonian = nqr_hamiltonian(site, b0_vector_tesla_pas)
    values, vectors = np.linalg.eigh(hamiltonian)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    levels_hz = values / TAU

    ops = spin_matrices(site.spin)
    transitions = _transitions_from_eigensystem(
        site,
        levels_hz,
        vectors,
        (ops.ix, ops.iy, ops.iz),
        strength_tolerance=strength_tolerance,
        frequency_tolerance_hz=frequency_tolerance_hz,
    )
    return NQREigensystem(
        site=site,
        levels_hz=levels_hz,
        eigenvectors=vectors,
        transitions=transitions,
    )


def batched_nqr_hamiltonians(
    site: QuadrupolarSite,
    b0_vectors_tesla_pas: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Return one Hamiltonian per static-field vector, shape ``(N, dim, dim)``.

    The quadrupole term is shared; the Zeeman term is formed for all ``N`` field
    vectors at once via a single contraction. This is the stacked input the
    batched diagonalizer consumes.
    """

    b0 = np.asarray(b0_vectors_tesla_pas, dtype=np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vectors_tesla_pas must be finite")
    hq = quadrupole_hamiltonian(site)
    ops = spin_matrices(site.spin)
    axis_stack = np.stack([ops.ix, ops.iy, ops.iz], axis=0)  # (3, dim, dim)
    hz = np.einsum("ni,ijk->njk", b0, axis_stack)
    hz = (-TAU * site.gamma_hz_per_t) * hz
    return hq[np.newaxis, :, :] + hz


def _batched_eigh(hamiltonians: np.ndarray, backend: str) -> tuple[np.ndarray, np.ndarray]:
    if backend == "numpy":
        return np.linalg.eigh(hamiltonians)
    if backend == "jax":
        from spin_dynamics.nqr import _jax_eigh as je

        if not je.JAX_AVAILABLE:
            raise ImportError(
                "backend='jax' requires the optional 'jax' extra. Install it "
                "with `python -m pip install -e .[jax]` (or `.[perf]`)."
            )
        return je.batched_eigh(hamiltonians)
    raise ValueError("backend must be 'numpy' or 'jax'")


def diagonalize_sites_over_b0(
    site: QuadrupolarSite,
    b0_vectors_tesla_pas: np.ndarray | list[list[float]],
    *,
    backend: str = "numpy",
    strength_tolerance: float = 1e-12,
    frequency_tolerance_hz: float = 1e-9,
) -> tuple[NQREigensystem, ...]:
    """Diagonalize one site across many static-field vectors with one ``eigh``.

    Builds every Hamiltonian, runs a single batched Hermitian eigensolve
    (NumPy's ``eigh`` broadcasts over the leading axis; the ``"jax"`` backend
    adds GPU execution), then reuses the per-orientation transition extraction.
    This replaces the Python ``diagonalize_site`` loop in powder/field scans.
    """

    b0 = np.asarray(b0_vectors_tesla_pas, dtype=np.float64).reshape(-1, 3)
    hamiltonians = batched_nqr_hamiltonians(site, b0)
    values, vectors = _batched_eigh(hamiltonians, backend)
    values = np.asarray(values)
    vectors = np.asarray(vectors)

    ops = spin_matrices(site.spin)
    components = (ops.ix, ops.iy, ops.iz)
    results: list[NQREigensystem] = []
    for index in range(b0.shape[0]):
        levels_hz = values[index] / TAU
        site_vectors = vectors[index]
        transitions = _transitions_from_eigensystem(
            site,
            levels_hz,
            site_vectors,
            components,
            strength_tolerance=strength_tolerance,
            frequency_tolerance_hz=frequency_tolerance_hz,
        )
        results.append(
            NQREigensystem(
                site=site,
                levels_hz=levels_hz,
                eigenvectors=site_vectors,
                transitions=transitions,
            )
        )
    return tuple(results)
