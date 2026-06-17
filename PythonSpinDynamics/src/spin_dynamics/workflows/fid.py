"""FID workflow entry points.

MATLAB reference folder:
    SpinDynamicsUpdated/Version_2/code/FID_Example
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from spin_dynamics.core.echo import calc_fid_time_domain
from spin_dynamics.core.kernels import sim_spin_dynamics_arb7


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def calc_macq_fid(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    params: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, float]:
    """Calculate acquired ideal FID magnetization.

    Mirrors MATLAB `calc_macq/calc_macq_fid.m`, with plotting removed.
    """

    T_90 = float(_field(pp, "T_90"))
    normalized = {
        "tp": (np.pi / 2) * np.asarray(_field(params, "tp"), dtype=np.float64) / T_90,
        "phi": np.asarray(_field(params, "phi"), dtype=np.float64),
        "amp": np.asarray(_field(params, "amp"), dtype=np.float64),
        "acq": np.asarray(_field(params, "acq"), dtype=bool),
        "grad": np.asarray(_field(params, "grad"), dtype=np.float64),
        "len_acq": (np.pi / 2) * float(_field(pp, "tacq")) / T_90,
        "del_w": np.asarray(_field(params, "del_w"), dtype=np.float64),
        "w_1": np.asarray(_field(params, "w_1"), dtype=np.float64),
        "m0": np.asarray(_field(params, "m0"), dtype=np.complex128)
        * np.ones_like(np.asarray(_field(params, "del_w"), dtype=np.float64)),
        "T1n": (np.pi / 2) * np.asarray(_field(sp, "T1"), dtype=np.float64) / T_90,
        "T2n": (np.pi / 2) * np.asarray(_field(sp, "T2"), dtype=np.float64) / T_90,
        "mth": np.asarray(_field(sp, "mth"), dtype=np.complex128)
        * np.ones_like(np.asarray(_field(params, "del_w"), dtype=np.float64)),
    }
    tacq_normalized = normalized["len_acq"]
    return sim_spin_dynamics_arb7(normalized), float(tacq_normalized)


def sim_fid_ideal(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the ideal no-probe FID workflow.

    Mirrors MATLAB `Sim_FID/simFID_ideal.m`, returning the acquired spectrum,
    time-domain FID, and normalized acquisition time vector.
    """

    T_90 = float(_field(pp, "T_90"))
    params = {
        "tp": np.array(
            [
                T_90,
                _field(pp, "acqDelay"),
                _field(pp, "tacq"),
                _field(pp, "acqDelay"),
            ],
            dtype=np.float64,
        ),
        "phi": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "amp": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "acq": np.array([0, 0, 1, 0], dtype=bool),
        "grad": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "len_acq": float(_field(pp, "tacq")),
        "del_w": np.asarray(_field(sp, "del_w"), dtype=np.float64),
        "w_1": np.asarray(_field(sp, "w_1"), dtype=np.float64),
        "m0": _field(sp, "m0"),
        "T1n": np.asarray(_field(sp, "T1"), dtype=np.float64),
        "T2n": np.asarray(_field(sp, "T2"), dtype=np.float64),
        "mth": _field(sp, "mth"),
    }
    macq, tacq = calc_macq_fid(sp, pp, params)
    echo, tvect = calc_fid_time_domain(
        macq[0, :],
        np.asarray(_field(sp, "del_w"), dtype=np.float64),
        tacq,
        (np.pi / 2) * float(_field(pp, "tdw")) / T_90,
    )
    return macq, echo, tvect
