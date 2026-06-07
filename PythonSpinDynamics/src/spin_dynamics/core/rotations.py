"""Rotation matrix and effective-axis calculations.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/calc_rot
    SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_asymp/sim_spin_dynamics_asymp_mag3.m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatrixElements:
    """Rotation matrix elements in MATLAB's `M0`, `M-`, `M+` coherence basis."""

    R_00: np.ndarray
    R_0p: np.ndarray
    R_0m: np.ndarray
    R_p0: np.ndarray
    R_m0: np.ndarray
    R_pp: np.ndarray
    R_mm: np.ndarray
    R_pm: np.ndarray
    R_mp: np.ndarray


def rf_matrix_elements(
    del_w: np.ndarray,
    w1: float,
    tp: float,
    phi: float,
) -> MatrixElements:
    """Calculate RF-pulse matrix elements without relaxation.

    Mirrors the nested `calc_matrix_elements` helper in
    `sim_spin_dynamics_asymp_mag3.m`.
    """

    del_w = np.asarray(del_w, dtype=np.float64).reshape(-1)
    omega = np.sqrt(w1**2 + del_w**2)
    dw = del_w / omega
    w1n = w1 / omega
    ph = np.exp(1j * phi)
    sn = np.sin(omega * tp)
    cs = np.cos(omega * tp)

    R_00 = dw**2 + w1n**2 * cs
    R_0p = 0.5 * w1n * (dw * (1 - cs) - 1j * sn) / ph
    R_p0 = w1n * (dw * (1 - cs) - 1j * sn) * ph
    R_pp = 0.5 * (w1n**2 + (1 + dw**2) * cs) + 1j * dw * sn
    R_pm = 0.5 * w1n**2 * (1 - cs) * ph**2

    return MatrixElements(
        R_00=R_00,
        R_0p=R_0p,
        R_0m=np.conj(R_0p),
        R_p0=R_p0,
        R_m0=np.conj(R_p0),
        R_pp=R_pp,
        R_mm=np.conj(R_pp),
        R_pm=R_pm,
        R_mp=np.conj(R_pm),
    )


def free_precession_matrix_elements(del_w: np.ndarray, tf: float) -> MatrixElements:
    """Calculate free-precession matrix elements without relaxation."""

    del_w = np.asarray(del_w, dtype=np.float64).reshape(-1)
    numpts = del_w.size
    zeros = np.zeros(numpts, dtype=np.complex128)
    R_pp = np.cos(del_w * tf) + 1j * np.sin(del_w * tf)
    return MatrixElements(
        R_00=np.ones(numpts, dtype=np.complex128),
        R_0p=zeros.copy(),
        R_0m=zeros.copy(),
        R_p0=zeros.copy(),
        R_m0=zeros.copy(),
        R_pp=R_pp,
        R_mm=np.conj(R_pp),
        R_pm=zeros.copy(),
        R_mp=zeros.copy(),
    )


def sim_spin_dynamics_asymp_mag3(
    tp: np.ndarray,
    phi: np.ndarray,
    amp: np.ndarray,
    neff: np.ndarray,
    del_w: np.ndarray,
    t_acq: float,
) -> np.ndarray:
    """Calculate asymptotic magnetization for a small-pulse sequence.

    Mirrors MATLAB `sim_spin_dynamics_asymp_mag3.m`.
    """

    tp = np.asarray(tp, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    amp = np.asarray(amp, dtype=np.float64).reshape(-1)
    del_w = np.asarray(del_w, dtype=np.float64).reshape(-1)
    neff = np.asarray(neff, dtype=np.complex128)

    if not (tp.size == phi.size == amp.size):
        raise ValueError("tp, phi, and amp must have the same length")
    if neff.shape != (3, del_w.size):
        raise ValueError("neff must have shape (3, len(del_w))")

    window = np.sinc(del_w * t_acq / (2 * np.pi))
    window = window / np.sum(window)

    mvect = np.zeros((3, del_w.size), dtype=np.complex128)
    mvect[0, :] = 1.0

    for tp_j, phi_j, amp_j in zip(tp, phi, amp):
        if amp_j > 0:
            mat = rf_matrix_elements(del_w, amp_j, tp_j, phi_j)
        else:
            mat = free_precession_matrix_elements(del_w, tp_j)

        tmp = mvect.copy()
        mvect[0, :] = mat.R_00 * tmp[0, :] + mat.R_0m * tmp[1, :] + mat.R_0p * tmp[2, :]
        mvect[1, :] = mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
        mvect[2, :] = mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]

    tmp = mvect.copy()
    mvect[0, :] = 0.5 * (tmp[2, :] + tmp[1, :])
    mvect[1, :] = -0.5j * (tmp[2, :] - tmp[1, :])
    mvect[2, :] = tmp[0, :]

    trans = np.sum(mvect * neff, axis=0) * (neff[0, :] - 1j * neff[1, :])
    return np.convolve(trans, window, mode="same")


def calc_rot_axis_arba4(
    tp: np.ndarray,
    phi: np.ndarray,
    amp: np.ndarray,
    del_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate effective rotation axis and angle for arbitrary amplitudes.

    Mirrors MATLAB `calc_rot_axis_arba4.m`, with plotting removed.
    """

    tp = np.asarray(tp, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    amp = np.asarray(amp, dtype=np.float64).reshape(-1)
    del_w = np.asarray(del_w, dtype=np.float64).reshape(-1)
    if not (tp.size == phi.size == amp.size):
        raise ValueError("tp, phi, and amp must have the same length")
    if tp.size == 0:
        raise ValueError("pulse sequence must be non-empty")

    numpts = del_w.size
    zero_tol = 1e-12

    n = np.zeros((3, numpts), dtype=np.float64)
    ncurr = np.zeros_like(n)
    tmp = np.zeros_like(n)

    if amp[0] > 0:
        w1 = amp[0]
        omega = np.sqrt(w1**2 + del_w**2)
        alpha = omega * tp[0]
        sn = np.sin(alpha / 2)
        n[0, :] = sn * w1 * np.cos(phi[0]) / omega
        n[1, :] = sn * w1 * np.sin(phi[0]) / omega
        n[2, :] = sn * del_w / omega
    else:
        alpha = del_w * tp[0]
        sn = np.sin(alpha / 2)
        n[2, :] = sn
    cs = np.cos(alpha / 2)

    for j in range(1, phi.size):
        if amp[j] > 0:
            w1 = amp[j]
            omega = np.sqrt(w1**2 + del_w**2)
            alpha_curr = omega * tp[j]
            ncurr[0, :] = w1 * np.cos(phi[j]) / omega
            ncurr[1, :] = w1 * np.sin(phi[j]) / omega
            ncurr[2, :] = del_w / omega

            crs = np.cross(n.T, ncurr.T).T
            sn_c = np.sin(alpha_curr / 2)
            cs_c = np.cos(alpha_curr / 2)

            tmp[0, :] = cs_c * n[0, :] + sn_c * (cs * ncurr[0, :] - crs[0, :])
            tmp[1, :] = cs_c * n[1, :] + sn_c * (cs * ncurr[1, :] - crs[1, :])
            tmp[2, :] = cs_c * n[2, :] + sn_c * (cs * ncurr[2, :] - crs[2, :])
            cs = cs * cs_c - sn_c * np.sum(n * ncurr, axis=0)
        else:
            alpha_curr = del_w * tp[j]
            sn_c = np.sin(alpha_curr / 2)
            cs_c = np.cos(alpha_curr / 2)

            tmp[0, :] = cs_c * n[0, :] - sn_c * n[1, :]
            tmp[1, :] = cs_c * n[1, :] + sn_c * n[0, :]
            tmp[2, :] = cs_c * n[2, :] + cs * sn_c
            cs = cs * cs_c - sn_c * n[2, :]

        n = tmp.copy()

    alpha = 2 * np.arccos(cs)
    sn = np.sin(alpha / 2)
    sn = np.where(sn == 0, zero_tol, sn)
    n[0, :] = n[0, :] / sn
    n[1, :] = n[1, :] / sn
    n[2, :] = n[2, :] / sn
    return n, alpha


def calc_rot_axis_arba3(
    tp: np.ndarray,
    phi: np.ndarray,
    amp: np.ndarray,
    del_w: np.ndarray,
) -> np.ndarray:
    """Calculate effective rotation axis for arbitrary-amplitude cycles."""

    n, _alpha = calc_rot_axis_arba4(tp, phi, amp, del_w)
    return n


def calc_rotation_matrix(
    del_w: np.ndarray,
    w_1: np.ndarray | float,
    tp: np.ndarray,
    phi: np.ndarray,
    amp: np.ndarray,
) -> MatrixElements:
    """Calculate the equivalent rotation matrix of a composite pulse.

    Mirrors MATLAB `calc_rot/calc_rotation_matrix.m`.
    """

    del_w = np.asarray(del_w, dtype=np.float64).reshape(-1)
    w_1 = np.asarray(w_1, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    amp = np.asarray(amp, dtype=np.float64).reshape(-1)
    if not (tp.size == phi.size == amp.size):
        raise ValueError("tp, phi, and amp must have the same length")
    if tp.size == 0:
        raise ValueError("pulse sequence must be non-empty")

    rtot: MatrixElements | None = None
    for tp_j, phi_j, amp_j in zip(tp, phi, amp):
        mat = rf_matrix_elements(del_w, amp_j * w_1, tp_j, phi_j)
        if rtot is None:
            rtot = mat
            continue

        tmp = rtot
        rtot = MatrixElements(
            R_00=mat.R_00 * tmp.R_00 + mat.R_0m * tmp.R_m0 + mat.R_0p * tmp.R_p0,
            R_0m=mat.R_00 * tmp.R_0m + mat.R_0m * tmp.R_mm + mat.R_0p * tmp.R_pm,
            R_0p=mat.R_00 * tmp.R_0p + mat.R_0m * tmp.R_mp + mat.R_0p * tmp.R_pp,
            R_m0=mat.R_m0 * tmp.R_00 + mat.R_mm * tmp.R_m0 + mat.R_mp * tmp.R_p0,
            R_mm=mat.R_m0 * tmp.R_0m + mat.R_mm * tmp.R_mm + mat.R_mp * tmp.R_pm,
            R_mp=mat.R_m0 * tmp.R_0p + mat.R_mm * tmp.R_mp + mat.R_mp * tmp.R_pp,
            R_p0=mat.R_p0 * tmp.R_00 + mat.R_pm * tmp.R_m0 + mat.R_pp * tmp.R_p0,
            R_pm=mat.R_p0 * tmp.R_0m + mat.R_pm * tmp.R_mm + mat.R_pp * tmp.R_pm,
            R_pp=mat.R_p0 * tmp.R_0p + mat.R_pm * tmp.R_mp + mat.R_pp * tmp.R_pp,
        )

    if rtot is None:
        raise AssertionError("unreachable: empty sequence checked above")
    return rtot
