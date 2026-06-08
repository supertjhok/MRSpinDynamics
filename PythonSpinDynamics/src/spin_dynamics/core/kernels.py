"""Arbitrary-pulse spin-dynamics kernels.

Primary MATLAB references:
    SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m
    SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb9.m
    SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.core.rotations import MatrixElements, rf_matrix_elements


@dataclass(frozen=True)
class Arb10Parameters:
    """Parameters for `sim_spin_dynamics_arb10`."""

    tp: np.ndarray
    pul: np.ndarray
    Rtot: Sequence[MatrixElements]
    amp: np.ndarray
    acq: np.ndarray
    grad: np.ndarray
    del_w: np.ndarray
    del_wg: np.ndarray
    T1n: np.ndarray
    T2n: np.ndarray
    m0: np.ndarray
    mth: np.ndarray


@dataclass(frozen=True)
class Arb7Parameters:
    """Parameters for `sim_spin_dynamics_arb7`."""

    tp: np.ndarray
    phi: np.ndarray
    amp: np.ndarray
    acq: np.ndarray
    grad: np.ndarray
    len_acq: float
    del_w: np.ndarray
    w_1: np.ndarray
    T1n: np.ndarray
    T2n: np.ndarray
    m0: np.ndarray
    mth: np.ndarray


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _field_or_default(obj: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_vector(value: Any, dtype: Any) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _free_precession_matrix_elements(
    del_w: np.ndarray,
    tf: float,
    T1n: np.ndarray,
    T2n: np.ndarray,
) -> MatrixElements:
    numpts = del_w.size
    zeros = np.zeros(numpts, dtype=np.complex128)
    R_00 = np.exp(-tf / T1n).astype(np.complex128)
    R_pp = np.exp(-tf / T2n) * np.exp(1j * del_w * tf)
    return MatrixElements(
        R_00=R_00,
        R_0p=zeros.copy(),
        R_0m=zeros.copy(),
        R_p0=zeros.copy(),
        R_m0=zeros.copy(),
        R_pp=R_pp,
        R_mm=np.conj(R_pp),
        R_pm=zeros.copy(),
        R_mp=zeros.copy(),
    )


def sim_spin_dynamics_arb10(params: Mapping[str, Any] | Arb10Parameters | Any) -> np.ndarray:
    """Simulate arbitrary-pulse spin dynamics with precomputed pulse matrices.

    Mirrors MATLAB `sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m`.
    `Rtot` uses MATLAB-style pulse numbering in `pul`, so `pul=1` selects the
    first Python sequence entry. Free-precession segments should have `amp=0`.
    """

    tp = _as_vector(_field(params, "tp"), np.float64)
    pul = _as_vector(_field(params, "pul"), np.int64)
    rtot = _field(params, "Rtot")
    amp = _as_vector(_field(params, "amp"), np.float64)
    acq = _as_vector(_field(params, "acq"), bool)
    grad = _as_vector(_field(params, "grad"), np.float64)
    del_w0 = _as_vector(_field(params, "del_w"), np.float64)
    del_wg = _as_vector(_field(params, "del_wg"), np.float64)
    T1n = _as_vector(_field(params, "T1n"), np.float64)
    T2n = _as_vector(_field(params, "T2n"), np.float64)
    m0 = _as_vector(_field(params, "m0"), np.complex128)
    mth = _as_vector(_field(params, "mth"), np.complex128)

    numpts = del_w0.size
    if not (tp.size == pul.size == amp.size == acq.size == grad.size):
        raise ValueError("tp, pul, amp, acq, and grad must have the same length")
    for name, arr in {
        "del_wg": del_wg,
        "T1n": T1n,
        "T2n": T2n,
        "m0": m0,
        "mth": mth,
    }.items():
        if arr.size != numpts:
            raise ValueError(f"{name} must have length len(del_w)")

    mvect = np.zeros((3, numpts), dtype=np.complex128)
    mvect[0, :] = m0

    macq = np.zeros((int(np.sum(acq)), numpts), dtype=np.complex128)
    acq_cnt = 0

    for tp_j, pul_j, amp_j, acq_j, grad_j in zip(tp, pul, amp, acq, grad):
        del_w = del_w0 + grad_j * del_wg

        if amp_j > 0:
            mat = rtot[int(pul_j) - 1]
            mlong = np.zeros(numpts, dtype=np.complex128)
        else:
            mat = _free_precession_matrix_elements(del_w, float(tp_j), T1n, T2n)
            mlong = mth * (1 - np.exp(-tp_j / T1n))

        tmp = mvect.copy()
        mvect[0, :] = mat.R_00 * tmp[0, :] + mat.R_0m * tmp[1, :] + mat.R_0p * tmp[2, :] + mlong
        mvect[1, :] = mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
        mvect[2, :] = mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]

        if acq_j:
            macq[acq_cnt, :] = mvect[1, :]
            acq_cnt += 1

    return macq


def sim_spin_dynamics_arb7(params: Mapping[str, Any] | Arb7Parameters | Any) -> np.ndarray:
    """Simulate arbitrary-pulse dynamics with acquisition-window convolution.

    Mirrors MATLAB `sim_spin_dynamics_arb/sim_spin_dynamics_arb7.m`. This older
    compatibility kernel is still used by the ideal FID workflow.
    """

    tp = _as_vector(_field(params, "tp"), np.float64)
    phi = _as_vector(_field(params, "phi"), np.float64)
    amp = _as_vector(_field(params, "amp"), np.float64)
    acq = _as_vector(_field(params, "acq"), bool)
    grad = _as_vector(_field(params, "grad"), np.float64)
    del_w0 = _as_vector(_field(params, "del_w"), np.float64)
    del_wg = _as_vector(_field_or_default(params, "del_wg", del_w0), np.float64)
    w_1 = _as_vector(_field(params, "w_1"), np.float64)
    T1n = _as_vector(_field(params, "T1n"), np.float64)
    T2n = _as_vector(_field(params, "T2n"), np.float64)
    m0 = _as_vector(_field(params, "m0"), np.complex128)
    mth = _as_vector(_field(params, "mth"), np.complex128)

    numpts = del_w0.size
    if not (tp.size == phi.size == amp.size == acq.size == grad.size):
        raise ValueError("tp, phi, amp, acq, and grad must have the same length")
    for name, arr in {
        "del_wg": del_wg,
        "w_1": w_1,
        "T1n": T1n,
        "T2n": T2n,
        "m0": m0,
        "mth": mth,
    }.items():
        if arr.size != numpts:
            raise ValueError(f"{name} must have length len(del_w)")

    window = np.sinc(del_w0 / (2 * np.pi))
    window = window / np.sum(window)

    mvect = np.zeros((3, numpts), dtype=np.complex128)
    mvect[0, :] = m0

    macq = np.zeros((int(np.sum(acq)), numpts), dtype=np.complex128)
    acq_cnt = 0

    for tp_j, phi_j, amp_j, acq_j, grad_j in zip(tp, phi, amp, acq, grad):
        del_w = del_w0 + grad_j * del_wg

        if amp_j > 0:
            mat = rf_matrix_elements(del_w, amp_j * w_1, float(tp_j), float(phi_j))
            mlong = np.zeros(numpts, dtype=np.complex128)
        else:
            mat = _free_precession_matrix_elements(del_w, float(tp_j), T1n, T2n)
            mlong = mth * (1 - np.exp(-tp_j / T1n))

        tmp = mvect.copy()
        mvect[0, :] = (
            mat.R_00 * tmp[0, :]
            + mat.R_0m * tmp[1, :]
            + mat.R_0p * tmp[2, :]
            + mlong
        )
        mvect[1, :] = (
            mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
        )
        mvect[2, :] = (
            mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]
        )

        if acq_j:
            macq[acq_cnt, :] = np.convolve(mvect[1, :], window, mode="same")
            acq_cnt += 1

    return macq
