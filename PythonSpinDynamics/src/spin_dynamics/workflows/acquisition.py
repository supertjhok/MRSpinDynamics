"""Finite-acquisition workflow helpers.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/calc_macq/calc_macq_ideal_probe_relax4.m
    SpinDynamicsUpdated/Version_2/code/calc_macq/calc_macq_tuned_probe_relax4.m
    SpinDynamicsUpdated/Version_2/code/calc_macq/calc_macq_matched_probe_relax4.m
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from spin_dynamics.core.kernels import sim_spin_dynamics_arb10


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _as_vector(value: Any, dtype: Any) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def calc_macq_ideal_probe_relax4(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> np.ndarray:
    """Calculate acquired spectra for an ideal-probe arbitrary sequence.

    Mirrors MATLAB `calc_macq/calc_macq_ideal_probe_relax4.m`. Pulse sequence
    segment durations in `pp.tp` are already normalized to the nominal
    `w1 = 1` convention. Physical `sp.T1` and `sp.T2` values are normalized
    here using `pp.T_90`, matching the MATLAB wrapper.
    """

    t_90 = float(_field(pp, "T_90"))
    t1n = (np.pi / 2) * _as_vector(_field(sp, "T1"), np.float64) / t_90
    t2n = (np.pi / 2) * _as_vector(_field(sp, "T2"), np.float64) / t_90

    params = {
        "tp": _field(pp, "tp"),
        "pul": _field(pp, "pul"),
        "amp": _field(pp, "amp"),
        "acq": _field(pp, "acq"),
        "grad": _field(pp, "grad"),
        "Rtot": _field(pp, "Rtot"),
        "del_w": _field(sp, "del_w"),
        "del_wg": _field(sp, "del_wg"),
        "w_1": _field(sp, "w_1"),
        "T1n": t1n,
        "T2n": t2n,
        "m0": _field(sp, "m0"),
        "mth": _field(sp, "mth"),
    }
    return sim_spin_dynamics_arb10(params)


def _calc_macq_relax4(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> np.ndarray:
    t_90 = float(_field(pp, "T_90"))
    t1n = (np.pi / 2) * _as_vector(_field(sp, "T1"), np.float64) / t_90
    t2n = (np.pi / 2) * _as_vector(_field(sp, "T2"), np.float64) / t_90

    return sim_spin_dynamics_arb10(
        {
            "tp": _field(pp, "tp"),
            "pul": _field(pp, "pul"),
            "amp": _field(pp, "amp"),
            "acq": _field(pp, "acq"),
            "grad": _field(pp, "grad"),
            "Rtot": _field(pp, "Rtot"),
            "del_w": _field(sp, "del_w"),
            "del_wg": _field(sp, "del_wg"),
            "w_1": _field(sp, "w_1"),
            "T1n": t1n,
            "T2n": t2n,
            "m0": _field(sp, "m0"),
            "mth": _field(sp, "mth"),
        }
    )


def _apply_receiver(
    macq: np.ndarray,
    transfer: Any,
    sensitivity: Any,
) -> np.ndarray:
    transfer_v = _as_vector(transfer, np.complex128)
    sensitivity_v = _as_vector(sensitivity, np.complex128)
    if transfer_v.size != macq.shape[1]:
        raise ValueError("receiver transfer function must match acquisition width")
    if sensitivity_v.size != macq.shape[1]:
        raise ValueError("receiver sensitivity must match acquisition width")
    return macq * transfer_v[np.newaxis, :] * sensitivity_v[np.newaxis, :]


def calc_macq_tuned_probe_relax4(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate finite acquisition for a tuned probe.

    Mirrors MATLAB `calc_macq/calc_macq_tuned_probe_relax4.m`. The receiver
    transfer function must be supplied as `sp.tf`; receive sensitivity as
    `sp.w_1r`.
    """

    macq = _calc_macq_relax4(sp, pp)
    mrx = _apply_receiver(macq, _field(sp, "tf"), _field(sp, "w_1r"))
    return macq, mrx


def calc_macq_untuned_probe_relax4(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate finite acquisition for an untuned probe.

    This is the Python analogue of the tuned `relax4` wrapper. Version 2 does
    not contain a separate MATLAB `calc_macq_untuned_probe_relax4.m`; callers
    provide the untuned receiver transfer function as `sp.tf`.
    """

    macq = _calc_macq_relax4(sp, pp)
    mrx = _apply_receiver(macq, _field(sp, "tf"), _field(sp, "w_1r"))
    return macq, mrx


def calc_macq_matched_probe_relax4(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate finite acquisition for a tuned-and-matched probe.

    Mirrors MATLAB `calc_macq/calc_macq_matched_probe_relax4.m`. The receiver
    transfer function must be supplied as `sp.tf2`; receive sensitivity as
    `sp.w_1r`.
    """

    macq = _calc_macq_relax4(sp, pp)
    mrx = _apply_receiver(macq, _field(sp, "tf2"), _field(sp, "w_1r"))
    return macq, mrx
