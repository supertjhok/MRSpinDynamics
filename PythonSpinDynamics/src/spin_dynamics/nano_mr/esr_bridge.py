"""Conversions between compatible pure-ESR and defect-sensor models."""

from __future__ import annotations

import numpy as np

from spin_dynamics.esr.systems import ESRSpinSystem
from spin_dynamics.nano_mr.frames import CoordinateFrame
from spin_dynamics.nano_mr.sensors import DefectSpinSensor


def defect_sensor_from_esr(
    system: ESRSpinSystem,
    *,
    frame: CoordinateFrame | None = None,
    depth_nm: float = 1.0,
    material: str = "generic",
    defect: str = "spin-1/2 ESR center",
) -> DefectSpinSensor:
    """Promote a pure spin-1/2 ESR model to a zero-ZFS defect sensor."""

    if not isinstance(system, ESRSpinSystem):
        raise TypeError("system must be an ESRSpinSystem")
    return DefectSpinSensor(
        spin=system.spin,
        g_tensor=system.g_tensor,
        zfs_tensor_hz=np.zeros((3, 3)),
        frame=CoordinateFrame(np.eye(3), name="ESR principal axes")
        if frame is None
        else frame,
        depth_nm=depth_nm,
        material=material,
        defect=defect,
        label=system.label,
    )


def esr_system_from_defect(
    sensor: DefectSpinSensor,
    *,
    zfs_tolerance_hz: float = 1.0e-9,
) -> ESRSpinSystem:
    """Return the equivalent pure ESR model when spin and ZFS permit it.

    NV, PL6, and other higher-spin/ZFS sensors intentionally fail this
    conversion; they require the nano-MR defect Hamiltonian.
    """

    if not isinstance(sensor, DefectSpinSensor):
        raise TypeError("sensor must be a DefectSpinSensor")
    if not np.isclose(sensor.spin, 0.5):
        raise ValueError("pure ESR conversion requires a spin-1/2 sensor")
    if np.linalg.norm(sensor.zfs_tensor_hz) > float(zfs_tolerance_hz):
        raise ValueError("pure ESR conversion requires negligible zero-field splitting")
    return ESRSpinSystem(
        g_tensor=sensor.g_tensor,
        spin=sensor.spin,
        label=sensor.label,
    )


__all__ = [
    "defect_sensor_from_esr",
    "esr_system_from_defect",
]
