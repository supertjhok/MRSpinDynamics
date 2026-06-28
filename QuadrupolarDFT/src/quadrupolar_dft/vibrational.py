"""Finite-temperature EFG from harmonic vibrational averaging.

The measured EFG is the static tensor averaged over the nuclear motion.  To
second order in the (mass-weighted) phonon normal coordinates ``Q_k``,

    <V_ij>(T) = V_ij^eq + (1/2) sum_k (d^2 V_ij / dQ_k^2) <Q_k^2>(T)

with ``<Q_k^2>(T)`` from :func:`quadrupolar_dft.thermal.
mean_square_normal_coordinate`.  Crucially the *tensor* is averaged in a fixed
crystal frame and only then diagonalized -- librational motion reorients the
principal-axis system, and averaging ``V_zz`` or ``eta`` directly would discard
exactly that effect.

The mode curvatures ``d^2 V_ij / dQ_k^2`` come from a finite-displacement DFT
workflow: displace the structure along each phonon eigenvector, recompute the
EFG, and central-difference (:func:`efg_curvature_central_difference`).  This
module is backend-agnostic -- it consumes curvatures, however they were
produced -- so the thermal physics is unit-testable without running DFT.

A lightweight analytic limit (:func:`fit_bayer_single_mode`) is provided for
validating the temperature *functional form* directly against measured line
frequencies, where per-mode EFG curvatures are not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .quadrupolar import coupling_constant_hz, nqr_frequencies_hz
from .tensors import EFGTensor
from .thermal import (
    mean_square_normal_coordinate,
    thermal_quantum_factor,
    wavenumber_to_angular_frequency,
)


def _as_matrix(value: EFGTensor | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(value, EFGTensor):
        return value.matrix_si
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class VibrationalMode:
    """One harmonic mode and the EFG curvature along it.

    ``efg_curvature_si`` is ``d^2 V_ij / dQ^2`` in SI EFG units (V/m^2) per
    mass-weighted normal coordinate squared (kg m^2), matching the units of
    :func:`quadrupolar_dft.thermal.mean_square_normal_coordinate`.
    """

    wavenumber_cm_inv: float
    efg_curvature_si: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        curvature = np.asarray(self.efg_curvature_si, dtype=float)
        if curvature.shape != (3, 3):
            raise ValueError("efg_curvature_si must be a 3x3 matrix")
        if float(self.wavenumber_cm_inv) <= 0.0:
            raise ValueError("wavenumber_cm_inv must be positive")
        object.__setattr__(self, "efg_curvature_si", 0.5 * (curvature + curvature.T))

    @property
    def angular_frequency_rad_s(self) -> float:
        return wavenumber_to_angular_frequency(self.wavenumber_cm_inv)


def efg_curvature_central_difference(
    efg_minus: EFGTensor | np.ndarray,
    efg_zero: EFGTensor | np.ndarray,
    efg_plus: EFGTensor | np.ndarray,
    *,
    delta_q: float,
) -> np.ndarray:
    """Second derivative of the EFG along a mode by central difference.

    ``efg_minus``/``efg_zero``/``efg_plus`` are the EFG tensors at displacement
    ``-delta_q``, ``0`` and ``+delta_q`` along the (mass-weighted) normal
    coordinate.  Returns ``(V_+ - 2 V_0 + V_-) / delta_q^2``.
    """

    if delta_q <= 0.0:
        raise ValueError("delta_q must be positive")
    minus = _as_matrix(efg_minus)
    zero = _as_matrix(efg_zero)
    plus = _as_matrix(efg_plus)
    return (plus - 2.0 * zero + minus) / (delta_q * delta_q)


def thermally_averaged_efg(
    equilibrium: EFGTensor,
    modes: Sequence[VibrationalMode],
    temperature_k: float,
) -> EFGTensor:
    """Return the harmonically averaged EFG tensor at ``temperature_k``."""

    matrix = np.array(equilibrium.matrix_si, dtype=float)
    for mode in modes:
        amplitude = mean_square_normal_coordinate(
            mode.angular_frequency_rad_s, temperature_k
        )
        matrix = matrix + 0.5 * mode.efg_curvature_si * amplitude
    # The EFG is traceless by Laplace's equation; remove any residual trace from
    # finite-difference rounding (amplified by the 1/delta_q^2 in the curvature)
    # before rebuilding the tensor.
    matrix = 0.5 * (matrix + matrix.T)
    matrix = matrix - (np.trace(matrix) / 3.0) * np.eye(3)
    return EFGTensor.from_components(matrix, unit="si")


@dataclass(frozen=True)
class ThermalEFGPoint:
    """EFG-derived quadrupolar quantities at one temperature."""

    temperature_k: float
    vzz_si: float
    eta: float
    cq_hz: float
    frequencies_hz: np.ndarray


def efg_temperature_sweep(
    equilibrium: EFGTensor,
    modes: Sequence[VibrationalMode],
    temperatures_k: Sequence[float],
    *,
    spin: float,
    quadrupole_moment_barns: float,
) -> list[ThermalEFGPoint]:
    """Average the EFG at each temperature and report C_Q, eta, and lines."""

    points: list[ThermalEFGPoint] = []
    for temperature in temperatures_k:
        tensor = thermally_averaged_efg(equilibrium, modes, temperature)
        cq_hz = coupling_constant_hz(tensor.vzz_si, quadrupole_moment_barns)
        frequencies = nqr_frequencies_hz(
            spin=spin, cq_hz=cq_hz, eta=tensor.eta
        )
        points.append(
            ThermalEFGPoint(
                temperature_k=float(temperature),
                vzz_si=tensor.vzz_si,
                eta=tensor.eta,
                cq_hz=cq_hz,
                frequencies_hz=frequencies,
            )
        )
    return points


# --------------------------------------------------------------------------
# Analytic Bayer-Kushida limit (scalar, for validation against measured lines)
# --------------------------------------------------------------------------


def bayer_frequency(
    nu0_hz: float,
    amplitudes: Sequence[float],
    wavenumbers_cm_inv: Sequence[float],
    temperature_k: float,
) -> float:
    """Bayer-Kushida librational averaging of a single NQR line.

    ``nu(T) = nu0 * (1 - sum_i a_i coth(hbar omega_i / 2 k_B T))``.  Each ``a_i``
    is the small dimensionless librational amplitude of mode ``i`` (in the
    rigid-libration theory ``a_i = 3 hbar / (4 I_i omega_i)``).
    """

    amplitudes = np.asarray(amplitudes, dtype=float)
    wavenumbers = np.asarray(wavenumbers_cm_inv, dtype=float)
    if amplitudes.shape != wavenumbers.shape:
        raise ValueError("amplitudes and wavenumbers must have the same length")
    reduction = 0.0
    for amplitude, wavenumber in zip(amplitudes, wavenumbers):
        omega = wavenumber_to_angular_frequency(wavenumber)
        reduction += amplitude * thermal_quantum_factor(omega, temperature_k)
    return float(nu0_hz) * (1.0 - reduction)


def bayer_slope_hz_per_k(
    nu0_hz: float,
    amplitude: float,
    wavenumber_cm_inv: float,
    temperature_k: float,
    *,
    delta_t: float = 0.5,
) -> float:
    """Local ``dnu/dT`` of the single-mode Bayer model (central difference)."""

    high = bayer_frequency(
        nu0_hz, [amplitude], [wavenumber_cm_inv], temperature_k + delta_t
    )
    low = bayer_frequency(
        nu0_hz, [amplitude], [wavenumber_cm_inv], temperature_k - delta_t
    )
    return (high - low) / (2.0 * delta_t)


@dataclass(frozen=True)
class BayerFit:
    """Single-mode Bayer fit to a measured ``nu(T)`` series."""

    nu0_hz: float
    amplitude: float
    wavenumber_cm_inv: float
    rms_hz: float

    def frequency(self, temperature_k: float) -> float:
        return bayer_frequency(
            self.nu0_hz, [self.amplitude], [self.wavenumber_cm_inv], temperature_k
        )

    def slope_hz_per_k(self, temperature_k: float) -> float:
        return bayer_slope_hz_per_k(
            self.nu0_hz, self.amplitude, self.wavenumber_cm_inv, temperature_k
        )


def fit_bayer_single_mode(
    temperatures_k: Sequence[float],
    frequencies_hz: Sequence[float],
    *,
    wavenumber_grid_cm_inv: Sequence[float] | None = None,
) -> BayerFit:
    """Fit ``nu(T) = nu0 (1 - a coth(hbar omega / 2 k_B T))`` to measured lines.

    For a fixed ``omega`` the model is linear in ``A = nu0`` and ``B = nu0 a``
    (``nu = A - B coth``), so each grid frequency is solved by linear least
    squares and the best ``omega`` is the one with the smallest residual.
    Numpy-only; no nonlinear optimizer required.
    """

    temperatures = np.asarray(temperatures_k, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if temperatures.shape != frequencies.shape or temperatures.size < 3:
        raise ValueError("need at least three matching (T, nu) points")
    if wavenumber_grid_cm_inv is None:
        wavenumber_grid_cm_inv = np.linspace(20.0, 400.0, 1521)
    grid = np.asarray(wavenumber_grid_cm_inv, dtype=float)

    best: BayerFit | None = None
    for wavenumber in grid:
        omega = wavenumber_to_angular_frequency(wavenumber)
        coth = np.array(
            [thermal_quantum_factor(omega, t) for t in temperatures]
        )
        # Solve [A, B] for nu = A - B * coth.
        design = np.column_stack([np.ones_like(coth), -coth])
        solution, *_ = np.linalg.lstsq(design, frequencies, rcond=None)
        a_param, b_param = float(solution[0]), float(solution[1])
        if a_param <= 0.0:
            continue
        residual = frequencies - (a_param - b_param * coth)
        rms = float(np.sqrt(np.mean(residual**2)))
        if best is None or rms < best.rms_hz:
            best = BayerFit(
                nu0_hz=a_param,
                amplitude=b_param / a_param,
                wavenumber_cm_inv=float(wavenumber),
                rms_hz=rms,
            )
    if best is None:
        raise ValueError("no positive-nu0 fit found on the wavenumber grid")
    return best
