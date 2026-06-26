"""Electric-field-gradient tensor conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .constants import EFG_AU_TO_SI


@dataclass(frozen=True)
class EFGTensor:
    """Symmetric traceless electric-field-gradient tensor.

    Principal components are returned in the conventional order
    ``|Vzz| >= |Vyy| >= |Vxx|`` as ``(Vxx, Vyy, Vzz)``.
    """

    matrix_si: np.ndarray

    @classmethod
    def from_components(
        cls,
        components: Sequence[Sequence[float]],
        *,
        unit: str = "au",
        symmetrize: bool = True,
        trace_tolerance: float = 1e-8,
    ) -> "EFGTensor":
        matrix = np.asarray(components, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("EFG tensor must be a 3x3 matrix.")
        if symmetrize:
            matrix = 0.5 * (matrix + matrix.T)
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=trace_tolerance):
            raise ValueError("EFG tensor must be symmetric.")

        factor = _unit_factor(unit)
        matrix_si = matrix * factor
        trace_scale = max(1.0, float(np.max(np.abs(matrix_si))))
        if abs(float(np.trace(matrix_si))) > trace_tolerance * trace_scale:
            raise ValueError("EFG tensor must be traceless within tolerance.")
        return cls(matrix_si=matrix_si)

    @property
    def principal_components_si(self) -> np.ndarray:
        eigvals = np.linalg.eigvalsh(self.matrix_si)
        order = np.argsort(np.abs(eigvals))
        return eigvals[order]

    @property
    def principal_axes(self) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(self.matrix_si)
        order = np.argsort(np.abs(eigvals))
        return eigvecs[:, order]

    @property
    def vzz_si(self) -> float:
        return float(self.principal_components_si[2])

    @property
    def eta(self) -> float:
        vxx, vyy, vzz = self.principal_components_si
        if np.isclose(vzz, 0.0):
            return 0.0
        eta = float((vxx - vyy) / vzz)
        return abs(eta)

    def as_unit(self, unit: str) -> np.ndarray:
        return self.matrix_si / _unit_factor(unit)


def average_tensors(
    tensors: Iterable[EFGTensor],
    *,
    weights: Iterable[float] | None = None,
) -> EFGTensor:
    """Average EFG tensors before diagonalizing the result."""

    matrices = [tensor.matrix_si for tensor in tensors]
    if not matrices:
        raise ValueError("At least one tensor is required.")

    if weights is None:
        average = np.mean(matrices, axis=0)
    else:
        weight_array = np.asarray(list(weights), dtype=float)
        if weight_array.shape != (len(matrices),):
            raise ValueError("Weights must match the number of tensors.")
        if np.any(weight_array < 0.0):
            raise ValueError("Weights must be nonnegative.")
        total = float(np.sum(weight_array))
        if total <= 0.0:
            raise ValueError("Weights must have positive sum.")
        average = np.tensordot(weight_array / total, matrices, axes=(0, 0))

    return EFGTensor.from_components(average, unit="si")


def _unit_factor(unit: str) -> float:
    normalized = unit.lower()
    if normalized in {"au", "atomic", "atomic_unit", "atomic_units"}:
        return EFG_AU_TO_SI
    if normalized in {"si", "v/m^2", "v/m2"}:
        return 1.0
    if normalized in {"1e21_v/m^2", "1e21_v/m2", "10^21_v/m^2"}:
        return 1e21
    raise ValueError(f"Unsupported EFG unit: {unit!r}")
