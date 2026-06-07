"""CPMG workflow entry points.

MATLAB reference folder:
    SpinDynamicsUpdated/Version_2/code/CPMG_Asymp_Examples
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from spin_dynamics.core.rotations import (
    calc_rot_axis_arba3,
    sim_spin_dynamics_asymp_mag3,
)


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def calc_masy_ideal(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any) -> np.ndarray:
    """Calculate ideal CPMG asymptotic magnetization.

    Mirrors MATLAB `calc_masy/calc_masy_ideal.m`, with plotting removed.
    """

    T_90 = _field(pp, "T_90")
    del_w = np.asarray(_field(sp, "del_w"), dtype=np.float64).reshape(-1)

    tacq = (np.pi / 2) * np.asarray(_field(pp, "tacq"), dtype=np.float64) / T_90
    tacq_scalar = float(np.ravel(tacq)[0])

    tref = (np.pi / 2) * np.asarray(_field(pp, "tref"), dtype=np.float64) / T_90
    pref = np.asarray(_field(pp, "pref"), dtype=np.float64)
    aref = np.asarray(_field(pp, "aref"), dtype=np.float64)
    neff = calc_rot_axis_arba3(tref, pref, aref, del_w)

    texc = (np.pi / 2) * np.asarray(_field(pp, "texc"), dtype=np.float64) / T_90
    texc = np.concatenate([texc.reshape(-1), np.array([-1.0])])
    pexc = np.concatenate([
        np.asarray(_field(pp, "pexc"), dtype=np.float64).reshape(-1),
        np.array([0.0]),
    ])
    aexc = np.concatenate([
        np.asarray(_field(pp, "aexc"), dtype=np.float64).reshape(-1),
        np.array([0.0]),
    ])

    masy1 = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff, del_w, tacq_scalar)
    masy2 = sim_spin_dynamics_asymp_mag3(texc, pexc + np.pi, aexc, neff, del_w, tacq_scalar)
    return (masy1 - masy2) / 2
