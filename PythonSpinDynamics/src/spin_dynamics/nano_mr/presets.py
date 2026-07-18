"""Material presets for common diamond and silicon-carbide defect sensors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spin_dynamics.nano_mr.frames import CoordinateFrame
from spin_dynamics.nano_mr.sensors import DefectSpinSensor, zfs_tensor_from_d_e


DIAMOND_NV_ZFS_HZ = 2.87e9
"""Room-temperature axial ground-state splitting of diamond NV-minus."""

SIC_PL6_ZFS_HZ = 1.35e9
"""Approximate axial ground-state splitting of the 4H-SiC PL6 center."""


def _frame(axis_lab: Sequence[float], name: str) -> CoordinateFrame:
    return CoordinateFrame.from_z_axis(axis_lab, name=name)


def diamond_nv_minus(
    *,
    depth_nm: float = 5.0,
    axis_lab: Sequence[float] = (0.0, 0.0, 1.0),
    d_hz: float = DIAMOND_NV_ZFS_HZ,
    e_hz: float = 0.0,
    g_tensor: float | Sequence[float] | np.ndarray = 2.0028,
    label: str = "NV-",
) -> DefectSpinSensor:
    """Return a spin-1 diamond NV-minus ground-state sensor preset."""

    return DefectSpinSensor(
        spin=1.0,
        g_tensor=g_tensor,
        zfs_tensor_hz=zfs_tensor_from_d_e(d_hz, e_hz),
        frame=_frame(axis_lab, "diamond NV"),
        depth_nm=depth_nm,
        material="diamond",
        defect="NV-",
        label=label,
    )


def sic_pl6(
    *,
    depth_nm: float = 2.0,
    axis_lab: Sequence[float] = (0.0, 0.0, 1.0),
    d_hz: float = SIC_PL6_ZFS_HZ,
    e_hz: float = 0.0,
    g_tensor: float | Sequence[float] | np.ndarray = 2.0,
    label: str = "PL6",
) -> DefectSpinSensor:
    """Return a spin-1 4H-SiC PL6/divacancy-related sensor preset.

    The PL6 microscopic assignment and calibrated tensor parameters remain an
    active research topic.  For that reason all preset constants are public
    arguments rather than hidden immutable material data.
    """

    return DefectSpinSensor(
        spin=1.0,
        g_tensor=g_tensor,
        zfs_tensor_hz=zfs_tensor_from_d_e(d_hz, e_hz),
        frame=_frame(axis_lab, "4H-SiC PL6"),
        depth_nm=depth_nm,
        material="4H-SiC",
        defect="PL6",
        label=label,
    )


__all__ = [
    "DIAMOND_NV_ZFS_HZ",
    "SIC_PL6_ZFS_HZ",
    "diamond_nv_minus",
    "sic_pl6",
]
