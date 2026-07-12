"""Singlet, triplet, and parahydrogen states for spin-1/2 systems.

Physical density matrices in this module have unit trace. Trace-zero deviation
densities are returned separately so hyperpolarization calculations do not mix
population fractions with high-temperature NMR signal conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.operators import product_operator


def _validated_pair(nspin: int, pair: tuple[int, int]) -> tuple[int, int, int]:
    count = int(nspin)
    if count < 2:
        raise ValueError("nspin must be at least 2")
    if len(pair) != 2:
        raise ValueError("pair must contain exactly two spin indices")
    first, second = (int(pair[0]), int(pair[1]))
    if first == second:
        raise ValueError("pair indices must be distinct")
    if not (0 <= first < count and 0 <= second < count):
        raise ValueError("pair indices must lie in range(nspin)")
    return count, first, second


def _pair_dot_operator(nspin: int, pair: tuple[int, int]) -> np.ndarray:
    count, first, second = _validated_pair(nspin, pair)
    return sum(
        (
            product_operator(count, [(first, axis), (second, axis)])
            for axis in ("x", "y", "z")
        ),
        start=np.zeros((2**count, 2**count), dtype=np.complex128),
    )


def singlet_projector(
    nspin: int = 2,
    pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Return the selected pair's singlet projector embedded in ``nspin`` spins.

    Other spins carry identity operators, so the projector rank is
    ``2 ** (nspin - 2)`` and its expectation value in a unit-trace density
    matrix is the reduced pair's singlet population.
    """

    count, _, _ = _validated_pair(nspin, pair)
    identity = np.eye(2**count, dtype=np.complex128)
    return 0.25 * identity - _pair_dot_operator(count, pair)


def triplet_projector(
    nspin: int = 2,
    pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Return the selected pair's total-triplet projector."""

    count, _, _ = _validated_pair(nspin, pair)
    identity = np.eye(2**count, dtype=np.complex128)
    return identity - singlet_projector(count, pair)


def singlet_order_operator(
    nspin: int = 2,
    pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Return trace-zero pair singlet order ``P_S - P_T / 3``.

    For systems larger than the selected pair, the result is normalized by the
    spectator identity so its partial trace over spectators is the conventional
    two-spin singlet-order operator.
    """

    count, _, _ = _validated_pair(nspin, pair)
    spectator_dimension = 2 ** (count - 2)
    return (
        singlet_projector(count, pair) - triplet_projector(count, pair) / 3.0
    ) / spectator_dimension


def spin_pair_swap_operator(
    nspin: int = 2,
    pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Return the unitary operator that swaps the selected spin-1/2 sites."""

    count, _, _ = _validated_pair(nspin, pair)
    identity = np.eye(2**count, dtype=np.complex128)
    return 0.5 * identity + 2.0 * _pair_dot_operator(count, pair)


def _validated_density(density: np.ndarray, nspin: int) -> np.ndarray:
    matrix = np.asarray(density, dtype=np.complex128)
    dimension = 2**int(nspin)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"density must have shape ({dimension}, {dimension}) for {nspin} spins"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("density must contain only finite values")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
        raise ValueError("density must be Hermitian")
    return matrix


def singlet_population(
    density: np.ndarray,
    *,
    pair: tuple[int, int] = (0, 1),
) -> float:
    """Return the selected pair's singlet population from a physical density.

    The input must have unit trace. Use :func:`singlet_order_amplitude` for a
    trace-zero deviation density.
    """

    matrix = np.asarray(density, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density must be square")
    dimension = matrix.shape[0]
    nspin = int(round(np.log2(dimension)))
    if 2**nspin != dimension:
        raise ValueError("density dimension must be a power of two")
    matrix = _validated_density(matrix, nspin)
    if not np.isclose(np.trace(matrix), 1.0, atol=1e-10):
        raise ValueError("singlet_population requires a unit-trace density")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-10:
        raise ValueError("singlet_population requires a positive density")
    value = np.trace(matrix @ singlet_projector(nspin, pair))
    if abs(float(np.imag(value))) > 1e-10:
        raise ValueError("singlet population has an unexpected imaginary part")
    return float(np.real(value))


def triplet_population(
    density: np.ndarray,
    *,
    pair: tuple[int, int] = (0, 1),
) -> float:
    """Return the selected pair's total triplet population."""

    return 1.0 - singlet_population(density, pair=pair)


def singlet_order_amplitude(
    density: np.ndarray,
    *,
    pair: tuple[int, int] = (0, 1),
) -> float:
    """Project a physical or deviation density onto normalized singlet order."""

    matrix = np.asarray(density, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density must be square")
    dimension = matrix.shape[0]
    nspin = int(round(np.log2(dimension)))
    if 2**nspin != dimension:
        raise ValueError("density dimension must be a power of two")
    matrix = _validated_density(matrix, nspin)
    order = singlet_order_operator(nspin, pair)
    numerator = np.vdot(order, matrix)
    denominator = float(np.real(np.vdot(order, order)))
    if abs(float(np.imag(numerator))) > 1e-10:
        raise ValueError("singlet-order amplitude has an unexpected imaginary part")
    return float(np.real(numerator)) / denominator


@dataclass(frozen=True)
class ParahydrogenState:
    """Physical and deviation density matrices for a para/ortho H2 mixture."""

    para_fraction: float
    density_matrix: np.ndarray
    deviation_density: np.ndarray

    @property
    def para_excess(self) -> float:
        """Return para fraction above the unpolarized statistical value 1/4."""

        return self.para_fraction - 0.25

    @property
    def singlet_population(self) -> float:
        """Return the physical singlet population, equal to ``para_fraction``."""

        return singlet_population(self.density_matrix)


def parahydrogen_state(para_fraction: float) -> ParahydrogenState:
    """Return the two-proton state for a specified parahydrogen fraction.

    The orthohydrogen population is distributed equally over the three triplet
    substates. Fractions below ``0.25`` represent ortho-enriched mixtures and
    consequently negative singlet order relative to the statistical mixture.
    """

    fraction = float(para_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("para_fraction must be finite and lie between 0 and 1")
    singlet = singlet_projector()
    triplet = triplet_projector()
    order = singlet_order_operator()
    density = fraction * singlet + (1.0 - fraction) * triplet / 3.0
    deviation = (fraction - 0.25) * order
    density.setflags(write=False)
    deviation.setflags(write=False)
    return ParahydrogenState(
        para_fraction=fraction,
        density_matrix=density,
        deviation_density=deviation,
    )


__all__ = [
    "ParahydrogenState",
    "parahydrogen_state",
    "singlet_order_amplitude",
    "singlet_order_operator",
    "singlet_population",
    "singlet_projector",
    "spin_pair_swap_operator",
    "triplet_population",
    "triplet_projector",
]
