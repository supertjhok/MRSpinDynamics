"""Prepolarization helpers for low-field and transport NMR workflows.

The core simulators use normalized magnetization units: ``m0`` is the initial
longitudinal magnetization and ``mth`` is the equilibrium magnetization during
the sequence. This module computes prepared ``m0`` values for experiments where
the sample first relaxes in a different polarizing field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PrepolarizedMagnetization:
    """Prepared longitudinal magnetization and sequence equilibrium arrays."""

    m0: np.ndarray
    mth: np.ndarray
    enhancement: np.ndarray

    def as_parameters(self) -> dict[str, np.ndarray]:
        """Return ``m0`` and ``mth`` fields suitable for kernel parameters."""

        return {"m0": self.m0.copy(), "mth": self.mth.copy()}


def longitudinal_recovery(
    initial_magnetization: float | Iterable[float] | np.ndarray,
    equilibrium_magnetization: float | Iterable[float] | np.ndarray,
    duration_seconds: float | Iterable[float] | np.ndarray,
    t1_seconds: float | Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return longitudinal magnetization after finite-time T1 recovery."""

    initial, equilibrium, duration, t1 = np.broadcast_arrays(
        np.asarray(initial_magnetization, dtype=np.float64),
        np.asarray(equilibrium_magnetization, dtype=np.float64),
        np.asarray(duration_seconds, dtype=np.float64),
        np.asarray(t1_seconds, dtype=np.float64),
    )
    _require_finite(initial, "initial_magnetization")
    _require_finite(equilibrium, "equilibrium_magnetization")
    if np.any(duration < 0.0):
        raise ValueError("duration_seconds must be non-negative")
    if np.any(t1 <= 0.0):
        raise ValueError("t1_seconds must be positive")
    if np.any(np.isnan(duration)) or np.any(np.isnan(t1)):
        raise ValueError("duration_seconds and t1_seconds must not contain NaN")

    recovery = np.exp(-duration / t1)
    return equilibrium + (initial - equilibrium) * recovery


def field_ratio_equilibrium(
    polarizing_field_tesla: float | Iterable[float] | np.ndarray,
    detection_field_tesla: float,
    *,
    detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0,
) -> np.ndarray:
    """Return polarizing-field equilibrium in detection-field units.

    The result is proportional to ``B_pol / B_det`` and is therefore signed.
    This convention makes an antiparallel prepolarizer produce negative prepared
    magnetization relative to the detection-field thermal equilibrium.
    """

    b_pol = np.asarray(polarizing_field_tesla, dtype=np.float64)
    m_det = np.asarray(detection_equilibrium_magnetization, dtype=np.float64)
    _require_finite(b_pol, "polarizing_field_tesla")
    _require_finite(m_det, "detection_equilibrium_magnetization")
    b_det = float(detection_field_tesla)
    if not np.isfinite(b_det) or b_det == 0.0:
        raise ValueError("detection_field_tesla must be finite and non-zero")
    return m_det * (b_pol / b_det)


def prepolarized_magnetization(
    polarizing_field_tesla: float | Iterable[float] | np.ndarray,
    detection_field_tesla: float,
    prepolarization_time_seconds: float | Iterable[float] | np.ndarray,
    t1_seconds: float | Iterable[float] | np.ndarray,
    *,
    initial_magnetization: float | Iterable[float] | np.ndarray = 0.0,
    detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0,
) -> np.ndarray:
    """Return prepared ``m0`` after relaxing in a polarizing field.

    The returned magnetization is in the same units as
    ``detection_equilibrium_magnetization``. With the default normalization,
    ``m0=1`` is thermal equilibrium in the detection field and
    ``m0=B_pol/B_det`` is full equilibrium in the polarizing field.
    """

    equilibrium = field_ratio_equilibrium(
        polarizing_field_tesla,
        detection_field_tesla,
        detection_equilibrium_magnetization=detection_equilibrium_magnetization,
    )
    return longitudinal_recovery(
        initial_magnetization,
        equilibrium,
        prepolarization_time_seconds,
        t1_seconds,
    )


def prepolarized_state(
    polarizing_field_tesla: float | Iterable[float] | np.ndarray,
    detection_field_tesla: float,
    prepolarization_time_seconds: float | Iterable[float] | np.ndarray,
    t1_seconds: float | Iterable[float] | np.ndarray,
    *,
    initial_magnetization: float | Iterable[float] | np.ndarray = 0.0,
    detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0,
) -> PrepolarizedMagnetization:
    """Return prepared ``m0``, sequence ``mth``, and enhancement arrays."""

    m0 = prepolarized_magnetization(
        polarizing_field_tesla,
        detection_field_tesla,
        prepolarization_time_seconds,
        t1_seconds,
        initial_magnetization=initial_magnetization,
        detection_equilibrium_magnetization=detection_equilibrium_magnetization,
    )
    mth = np.broadcast_to(
        np.asarray(detection_equilibrium_magnetization, dtype=np.float64),
        np.shape(m0),
    ).copy()
    enhancement = np.divide(
        m0,
        mth,
        out=np.full_like(m0, np.nan, dtype=np.float64),
        where=mth != 0.0,
    )
    return PrepolarizedMagnetization(
        m0=np.asarray(m0, dtype=np.float64),
        mth=mth,
        enhancement=enhancement,
    )


def residence_time_seconds(
    path_length_meters: float | Iterable[float] | np.ndarray,
    speed_meters_per_second: float | Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return residence time for transport through a prepolarizing region."""

    length, speed = np.broadcast_arrays(
        np.asarray(path_length_meters, dtype=np.float64),
        np.asarray(speed_meters_per_second, dtype=np.float64),
    )
    _require_finite(length, "path_length_meters")
    if np.any(np.isnan(speed)):
        raise ValueError("speed_meters_per_second must not contain NaN")
    if np.any(length < 0.0):
        raise ValueError("path_length_meters must be non-negative")
    if np.any(speed < 0.0):
        raise ValueError("speed_meters_per_second must be non-negative")
    return np.divide(
        length,
        speed,
        out=np.full_like(length, np.inf, dtype=np.float64),
        where=speed > 0.0,
    )


def prepolarized_flow_state(
    polarizing_field_tesla: float | Iterable[float] | np.ndarray,
    detection_field_tesla: float,
    path_length_meters: float | Iterable[float] | np.ndarray,
    speed_meters_per_second: float | Iterable[float] | np.ndarray,
    t1_seconds: float | Iterable[float] | np.ndarray,
    *,
    initial_magnetization: float | Iterable[float] | np.ndarray = 0.0,
    detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0,
) -> PrepolarizedMagnetization:
    """Return a prepolarized state for flow through a finite polarizer."""

    return prepolarized_state(
        polarizing_field_tesla,
        detection_field_tesla,
        residence_time_seconds(path_length_meters, speed_meters_per_second),
        t1_seconds,
        initial_magnetization=initial_magnetization,
        detection_equilibrium_magnetization=detection_equilibrium_magnetization,
    )


def apply_prepolarization_to_parameters(
    params: Mapping[str, Any] | Any,
    prepared: PrepolarizedMagnetization,
) -> dict[str, Any]:
    """Return a shallow parameter copy with ``m0`` and ``mth`` replaced."""

    if isinstance(params, Mapping):
        updated = dict(params)
    else:
        updated = {
            name: getattr(params, name)
            for name in dir(params)
            if not name.startswith("_") and not callable(getattr(params, name))
        }
    updated["m0"] = prepared.m0.copy()
    updated["mth"] = prepared.mth.copy()
    return updated


def _require_finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
