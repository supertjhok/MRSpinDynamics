"""Hamiltonian builders and transition analysis for NQR."""

from __future__ import annotations

import numpy as np

from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.systems import NQREigensystem, NQRTransition, QuadrupolarSite


TAU = 2.0 * np.pi
_AXES = ("x", "y", "z")


def quadrupole_hamiltonian(site: QuadrupolarSite) -> np.ndarray:
    """Return the zero-field quadrupole Hamiltonian in radians per second."""

    ops = spin_matrices(site.spin)
    spin = site.spin
    quadrupole_operator = (
        3.0 * (ops.iz @ ops.iz)
        - spin * (spin + 1.0) * ops.identity
        + site.eta * (ops.ix @ ops.ix - ops.iy @ ops.iy)
    )
    scale_hz = site.quadrupole_frequency_hz / 3.0
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


def diagonalize_site(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    *,
    strength_tolerance: float = 1e-12,
) -> NQREigensystem:
    """Diagonalize a site Hamiltonian and return transition metadata."""

    hamiltonian = nqr_hamiltonian(site, b0_vector_tesla_pas)
    values, vectors = np.linalg.eigh(hamiltonian)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    levels_hz = values / TAU

    ops = spin_matrices(site.spin)
    operator_components = (ops.ix, ops.iy, ops.iz)
    used_labels: set[str] = set()
    transitions: list[NQRTransition] = []
    for lower in range(site.dimension):
        for upper in range(lower + 1, site.dimension):
            frequency_hz = float(levels_hz[upper] - levels_hz[lower])
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
    return NQREigensystem(
        site=site,
        levels_hz=levels_hz,
        eigenvectors=vectors,
        transitions=tuple(transitions),
    )
