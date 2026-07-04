"""Dense product operators for heteronuclear (mixed spin quantum number) systems.

The existing :mod:`spin_dynamics.coupling.operators` hardcodes spin-1/2 Pauli
matrices and a ``2**nspin`` Hilbert space. Zero/ultra-low-field J-coupled spin
networks mix nuclei of different spin (e.g. spin-1/2 ``1H``/``19F`` with spin-1
``14N``), so the Hilbert space is a tensor product of factors with dimension
``2 I_k + 1``. These helpers build single-spin and product operators embedded in
that mixed space, reusing :func:`spin_dynamics.relaxation.single_spin_matrices`
for the per-spin angular-momentum matrices.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from spin_dynamics.relaxation import single_spin_matrices, spin_dimension

_AXIS_ATTR = {
    "x": "ix",
    "y": "iy",
    "z": "iz",
    "+": "i_plus",
    "-": "i_minus",
}


def _validate_spins(spins: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(spin) for spin in spins)
    if not values:
        raise ValueError("at least one spin is required")
    return values


def hilbert_dimension(spins: Sequence[float] | Iterable[float]) -> int:
    """Return the tensor-product Hilbert dimension ``prod(2 I_k + 1)``."""

    dimension = 1
    for spin in _validate_spins(spins):
        dimension *= spin_dimension(spin)
    return int(dimension)


def _kron_all(factors: Iterable[np.ndarray]) -> np.ndarray:
    out: np.ndarray | None = None
    for factor in factors:
        out = factor if out is None else np.kron(out, factor)
    if out is None:
        raise ValueError("at least one factor is required")
    return out


def _single_spin_axis(spin: float, axis: str) -> np.ndarray:
    matrices = single_spin_matrices(float(spin))
    try:
        return getattr(matrices, _AXIS_ATTR[axis])
    except KeyError as exc:
        raise ValueError("axis must be one of 'x', 'y', 'z', '+', '-'") from exc


def embedded_operator(
    spins: Sequence[float] | Iterable[float],
    index: int,
    axis: str,
) -> np.ndarray:
    """Return a single-spin operator embedded in the full mixed Hilbert space."""

    spin_values = _validate_spins(spins)
    index = int(index)
    if index < 0 or index >= len(spin_values):
        raise ValueError("index must select an existing spin")
    factors = [
        _single_spin_axis(spin, axis)
        if idx == index
        else single_spin_matrices(spin).identity
        for idx, spin in enumerate(spin_values)
    ]
    return _kron_all(factors)


def product_operator(
    spins: Sequence[float] | Iterable[float],
    terms: Iterable[tuple[int, str]],
) -> np.ndarray:
    """Return a product operator such as ``I1z I2z`` for mixed spins."""

    spin_values = _validate_spins(spins)
    by_index: dict[int, str] = {}
    for index, axis in terms:
        index = int(index)
        if index < 0 or index >= len(spin_values):
            raise ValueError("term index must select an existing spin")
        if index in by_index:
            raise ValueError("product_operator accepts at most one term per spin")
        by_index[index] = axis
    factors = [
        _single_spin_axis(spin, by_index[idx])
        if idx in by_index
        else single_spin_matrices(spin).identity
        for idx, spin in enumerate(spin_values)
    ]
    return _kron_all(factors)


def total_operator(
    spins: Sequence[float] | Iterable[float],
    axis: str,
    indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Return the sum of selected single-spin operators along one axis."""

    spin_values = _validate_spins(spins)
    selected = (
        range(len(spin_values))
        if indices is None
        else tuple(int(idx) for idx in indices)
    )
    dimension = hilbert_dimension(spin_values)
    out = np.zeros((dimension, dimension), dtype=np.complex128)
    for index in selected:
        out = out + embedded_operator(spin_values, index, axis)
    return out


def dot_product_operator(
    spins: Sequence[float] | Iterable[float],
    index_a: int,
    index_b: int,
) -> np.ndarray:
    """Return the scalar ``I_a . I_b`` operator for two distinct spins."""

    if int(index_a) == int(index_b):
        raise ValueError("dot_product_operator requires two distinct spins")
    return (
        product_operator(spins, [(index_a, "x"), (index_b, "x")])
        + product_operator(spins, [(index_a, "y"), (index_b, "y")])
        + product_operator(spins, [(index_a, "z"), (index_b, "z")])
    )
