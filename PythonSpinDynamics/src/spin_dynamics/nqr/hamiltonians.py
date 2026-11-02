"""Hamiltonian builders and transition analysis for NQR."""

from __future__ import annotations

from typing import cast

import numpy as np

from spin_dynamics.nqr.operators import fundamental_operator_gap, spin_matrices
from spin_dynamics.nqr.systems import NQREigensystem, NQRTransition, QuadrupolarSite


TAU = 2.0 * np.pi
_AXES = ("x", "y", "z")


def quadrupole_frequency_scale_hz(site: QuadrupolarSite) -> float:
    """Return the Hamiltonian prefactor matching the public frequency parameter.

    ``quadrupole_frequency_hz`` (nu_Q) is the ``eta = 0`` *fundamental* transition
    frequency: the ``0 <-> 1`` line for integer spin and the lowest ``1/2 <-> 3/2``
    line for half-integer spin. In the operator units of
    ``3 Iz^2 - I(I+1) + eta (Ix^2 - Iy^2)`` that fundamental gap is ``d = 3`` for
    integer spin and ``d = 6`` for half-integer spin, so the prefactor is
    ``nu_Q / d``. This reproduces the historical spin-1 (``d = 3``) and spin-3/2
    (``d = 6``) values exactly and extends unchanged to spin-5/2, 7/2, 9/2, where
    the higher axial lines fall at ``2 nu_Q``, ``3 nu_Q``, ..., and to integer
    spin >= 2, where they fall at the odd multiples ``3 nu_Q``, ``5 nu_Q``, ...
    (see ``mr_integration.conversions`` for the matching ``C_Q <-> nu_Q`` mapping).
    """

    return site.quadrupole_frequency_hz / fundamental_operator_gap(site.spin)


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


# Levels closer than this fraction of the largest |level| are exactly degenerate
# (eigensolver noise is ~1e-15 relative); physical splittings are far larger.
_DEGENERACY_RELATIVE_TOLERANCE = 1.0e-10
# Eigenvalues of the O(1)-O(I^2) symmetry-breaking operators below closer than
# this are treated as equal when refining a degenerate subspace.
_REFINE_TOLERANCE = 1.0e-8


def _refine_degenerate_subspace(
    subspace: np.ndarray, operators: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Rotate a degenerate eigenspace onto the eigenbasis of ``operators``.

    Each operator is diagonalized inside the subspace; where it is (numerically)
    a multiple of the identity there it is skipped, so solver noise never picks
    the basis. Remaining degeneracies are refined by the next operator.
    """

    if subspace.shape[1] == 1 or not operators:
        return subspace
    operator, rest = operators[0], operators[1:]
    projected = subspace.conj().T @ operator @ subspace
    projected = 0.5 * (projected + projected.conj().T)
    values, rotation = np.linalg.eigh(projected)
    if float(np.ptp(values)) <= _REFINE_TOLERANCE:
        return _refine_degenerate_subspace(subspace, rest)
    rotated = subspace @ rotation
    blocks: list[list[int]] = [[0]]
    for index in range(1, values.size):
        if values[index] - values[blocks[-1][0]] <= _REFINE_TOLERANCE:
            blocks[-1].append(index)
        else:
            blocks.append([index])
    return np.hstack(
        [_refine_degenerate_subspace(rotated[:, block], rest) for block in blocks]
    )


def _canonical_degenerate_eigenvectors(
    site: QuadrupolarSite, values: np.ndarray, vectors: np.ndarray
) -> np.ndarray:
    """Return ``vectors`` with every exactly degenerate eigenspace made canonical.

    ``eigh`` returns an arbitrary basis inside a degenerate eigenspace (the
    spin-1 ``m = +-1`` pair at ``eta = 0``, every Kramers doublet of a
    half-integer spin at zero field), which made transition labels and dipole
    vectors arbitrary. Each such eigenspace is rotated onto the eigenbasis of
    ``Ix^2 - Iy^2`` (the ``eta -> 0+`` limit, giving the spin-1 ``|x>, |y>``
    states), then ``Iz`` and ``Ix`` (the weak-field limit along z, then x), and
    each vector's largest component is made real and positive. ``values`` must
    be sorted ascending. Non-degenerate eigenvectors are returned unchanged.
    """

    scale = float(np.max(np.abs(values))) if values.size else 0.0
    tolerance = _DEGENERACY_RELATIVE_TOLERANCE * max(scale, 1.0)
    groups: list[list[int]] = [[0]]
    for index in range(1, values.size):
        if values[index] - values[groups[-1][0]] <= tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
    if all(len(group) == 1 for group in groups):
        return vectors
    ops = spin_matrices(site.spin)
    operators = (ops.ix @ ops.ix - ops.iy @ ops.iy, ops.iz, ops.ix)
    out = np.array(vectors, dtype=np.complex128, copy=True)
    for group in groups:
        if len(group) == 1:
            continue
        refined = _refine_degenerate_subspace(out[:, group], operators)
        for column in range(refined.shape[1]):
            vector = refined[:, column]
            pivot = vector[int(np.argmax(np.abs(vector)))]
            refined[:, column] = vector * (abs(pivot) / pivot)
        out[:, group] = refined
    return out


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

    # Exactly-degenerate levels (the Kramers doublets of half-integer spin) are
    # split only by eigensolver noise (~1e-15 of the largest level), which would
    # otherwise surface as spurious near-zero-frequency "transitions". Floor the
    # frequency cut at the same relative tolerance that defines exact degeneracy
    # in ``_canonical_degenerate_eigenvectors``. The floor used to be 1e-6 of the
    # level span, which for heavy nuclei (127I, 209Bi: span ~1 GHz) also dropped
    # real weak-field Zeeman transitions of a few hundred Hz.
    level_scale = float(np.max(np.abs(levels_hz))) if np.size(levels_hz) else 0.0
    frequency_floor_hz = max(
        float(frequency_tolerance_hz),
        _DEGENERACY_RELATIVE_TOLERANCE * max(level_scale, 1.0),
    )

    used_labels: set[str] = set()
    transitions: list[NQRTransition] = []
    for lower in range(site.dimension):
        for upper in range(lower + 1, site.dimension):
            frequency_hz = float(levels_hz[upper] - levels_hz[lower])
            if frequency_hz <= frequency_floor_hz:
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
    vectors = _canonical_degenerate_eigenvectors(site, values, vectors[:, order])
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
        return cast(tuple[np.ndarray, np.ndarray], np.linalg.eigh(hamiltonians))
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
        site_vectors = _canonical_degenerate_eigenvectors(
            site, values[index], vectors[index]
        )
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
