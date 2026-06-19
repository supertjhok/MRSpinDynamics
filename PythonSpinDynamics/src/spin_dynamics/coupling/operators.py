"""Dense spin-1/2 product operators."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


IDENTITY_2 = np.eye(2, dtype=np.complex128)
IX = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
IY = 0.5 * np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
IZ = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

_AXIS = {
    "x": IX,
    "y": IY,
    "z": IZ,
}


def _check_nspin(nspin: int) -> int:
    nspin = int(nspin)
    if nspin <= 0:
        raise ValueError("nspin must be positive")
    return nspin


def _kron_all(factors: Iterable[np.ndarray]) -> np.ndarray:
    out: np.ndarray | None = None
    for factor in factors:
        out = factor if out is None else np.kron(out, factor)
    if out is None:
        raise ValueError("at least one factor is required")
    return out


def spin_operator(nspin: int, index: int, axis: str) -> np.ndarray:
    """Return a single-spin operator embedded in the full Hilbert space."""

    nspin = _check_nspin(nspin)
    index = int(index)
    if index < 0 or index >= nspin:
        raise ValueError("index must select an existing spin")
    try:
        op = _AXIS[axis.lower()]
    except KeyError as exc:
        raise ValueError("axis must be 'x', 'y', or 'z'") from exc
    return _kron_all(op if idx == index else IDENTITY_2 for idx in range(nspin))


def total_operator(nspin: int, axis: str, indices: Iterable[int] | None = None) -> np.ndarray:
    """Return the sum of selected spin operators along one axis."""

    nspin = _check_nspin(nspin)
    selected = range(nspin) if indices is None else tuple(int(idx) for idx in indices)
    out = np.zeros((2**nspin, 2**nspin), dtype=np.complex128)
    for idx in selected:
        out = out + spin_operator(nspin, idx, axis)
    return out


def product_operator(nspin: int, terms: Iterable[tuple[int, str]]) -> np.ndarray:
    """Return a product operator such as ``I1z I2z``."""

    nspin = _check_nspin(nspin)
    by_index: dict[int, np.ndarray] = {}
    for index, axis in terms:
        index = int(index)
        if index < 0 or index >= nspin:
            raise ValueError("term index must select an existing spin")
        try:
            op = _AXIS[axis.lower()]
        except KeyError as exc:
            raise ValueError("term axis must be 'x', 'y', or 'z'") from exc
        if index in by_index:
            raise ValueError("product_operator accepts at most one term per spin")
        by_index[index] = op
    return _kron_all(by_index.get(idx, IDENTITY_2) for idx in range(nspin))
