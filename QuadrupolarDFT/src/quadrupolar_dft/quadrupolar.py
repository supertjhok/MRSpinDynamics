"""Quadrupolar coupling and zero-field transition calculations."""

from __future__ import annotations

import numpy as np

from .constants import BARN_M2, ELEMENTARY_CHARGE_C, PLANCK_CONSTANT_J_S


def coupling_constant_hz(vzz_si: float, quadrupole_moment_barns: float) -> float:
    """Return ``C_Q = e Q Vzz / h`` in Hz.

    ``vzz_si`` is in V/m^2 and ``quadrupole_moment_barns`` is in barns.
    """

    return (
        ELEMENTARY_CHARGE_C
        * quadrupole_moment_barns
        * BARN_M2
        * vzz_si
        / PLANCK_CONSTANT_J_S
    )


def nqr_frequencies_hz(
    *,
    spin: float,
    cq_hz: float,
    eta: float,
    tolerance_hz: float = 1e-6,
    include_forbidden: bool = False,
    relative_strength_tolerance: float = 1e-8,
) -> np.ndarray:
    """Return unique zero-field NQR transition frequencies.

    The Hamiltonian is expressed in frequency units as

    ``C_Q / (4 I (2 I - 1)) * [3 Iz^2 - I(I+1) + eta/2 (I+^2 + I-^2)]``.

    Only magnetic-dipole-allowed lines are returned: a pair of (Kramers-)
    degenerate levels is kept when its summed ``|<a|I|b>|^2`` exceeds
    ``relative_strength_tolerance`` times the strongest pair. For spin >= 5/2 at
    ``eta = 0`` this drops the forbidden ``|Delta m| >= 2`` level differences
    (e.g. the ``+-1/2 <-> +-5/2`` "sum" line); for ``eta > 0`` those lines are
    weakly allowed and are kept. Pass ``include_forbidden=True`` for every level
    difference (the behaviour before this selection rule was applied).
    """

    if spin < 1:
        raise ValueError("Quadrupolar NQR transitions require spin >= 1.")
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be between 0 and 1.")

    matrices = _spin_matrices(spin)
    iz = matrices["iz"]
    ip = matrices["ip"]
    im = matrices["im"]
    identity = np.eye(iz.shape[0])
    prefactor = cq_hz / (4.0 * spin * (2.0 * spin - 1.0))
    h_over_h = prefactor * (
        3.0 * (iz @ iz)
        - spin * (spin + 1.0) * identity
        + 0.5 * eta * (ip @ ip + im @ im)
    )
    energies, vectors = np.linalg.eigh(h_over_h)
    span = float(np.ptp(energies))
    degeneracy_hz = max(float(tolerance_hz), 1e-9 * span)
    # Group (Kramers-)degenerate levels so the strength is basis-independent.
    groups: list[list[int]] = []
    for index in np.argsort(energies):
        if groups and abs(energies[index] - energies[groups[-1][0]]) <= degeneracy_hz:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    levels = [float(np.mean(energies[group])) for group in groups]
    # |<a|I|b>|^2 summed over x, y, z: |Iz|^2 + (|I+|^2 + |I-|^2) / 2.
    iz_e = vectors.T @ iz @ vectors
    ip_e = vectors.T @ ip @ vectors
    im_e = vectors.T @ im @ vectors
    strength_matrix = np.abs(iz_e) ** 2 + 0.5 * (np.abs(ip_e) ** 2 + np.abs(im_e) ** 2)
    pairs = []
    for lower in range(len(groups)):
        for upper in range(lower + 1, len(groups)):
            value = abs(levels[upper] - levels[lower])
            if value <= tolerance_hz:
                continue
            strength = float(np.sum(strength_matrix[np.ix_(groups[lower], groups[upper])]))
            pairs.append((value, strength))
    if not include_forbidden and pairs:
        strongest = max(strength for _, strength in pairs)
        pairs = [
            pair for pair in pairs
            if pair[1] > relative_strength_tolerance * strongest
        ]
    positive = sorted(value for value, _ in pairs)
    return np.asarray(_unique_with_tolerance(positive, tolerance_hz), dtype=float)


def _spin_matrices(spin: float) -> dict[str, np.ndarray]:
    twice_spin = round(2.0 * spin)
    if not np.isclose(twice_spin, 2.0 * spin):
        raise ValueError("spin must be an integer or half-integer.")

    m_values = np.arange(spin, -spin - 1.0, -1.0)
    dim = len(m_values)
    iz = np.diag(m_values)
    ip = np.zeros((dim, dim), dtype=float)
    im = np.zeros((dim, dim), dtype=float)
    index = {m: i for i, m in enumerate(m_values)}

    for m in m_values:
        raised = m + 1.0
        lowered = m - 1.0
        if raised in index:
            ip[index[raised], index[m]] = np.sqrt(spin * (spin + 1.0) - m * raised)
        if lowered in index:
            im[index[lowered], index[m]] = np.sqrt(
                spin * (spin + 1.0) - m * lowered
            )

    return {"iz": iz, "ip": ip, "im": im}


def _unique_with_tolerance(values: list[float], tolerance: float) -> list[float]:
    unique: list[float] = []
    for value in values:
        if not any(abs(value - previous) <= tolerance for previous in unique):
            unique.append(value)
    return unique
