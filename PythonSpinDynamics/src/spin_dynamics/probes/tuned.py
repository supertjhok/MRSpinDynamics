"""Tuned-probe transmit/receive models.

MATLAB reference folder:
    SpinDynamicsUpdated/Version_2/code/circuit_simulation/tuned_probe
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import calc_rot_axis_arba3, sim_spin_dynamics_asymp_mag3


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        if name == "in_":
            return obj["in"]
    return getattr(obj, name)


def _with_fields(obj: Mapping[str, Any] | Any, **updates: Any) -> Any:
    if isinstance(obj, Mapping):
        out = dict(obj)
        out.update(updates)
        return out
    return replace(obj, **updates)


def _as_vector(value: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def tuned_probe_lp_orig(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate tuned-probe coil current for the original/reference path.

    Mirrors MATLAB `circuit_simulation/tuned_probe/tuned_probe_lp_Orig.m`.
    """

    L = float(_field(sp, "L"))
    R = float(_field(sp, "R"))
    C = float(_field(sp, "C"))
    wp = 1 / np.sqrt(L * C)
    Z0 = np.sqrt(L / C)

    w0 = float(_field(sp, "w0")) / wp
    tp = wp * _as_vector(_field(pp, "tref"))
    phi = _as_vector(_field(pp, "pref"))
    amp = _as_vector(_field(pp, "aref"))
    Rs = _as_vector(_field(pp, "Rsref"))
    w = float(_field(pp, "w")) / wp
    N = int(_field(pp, "N"))

    np_seg = np.rint(tp * N * w / (2 * np.pi)).astype(int)
    ncyc = int(np.sum(np_seg))
    if ncyc <= 0:
        raise ValueError("pulse sequence must contain at least one simulation cycle")
    tvect = 2 * np.pi * np.linspace(1, ncyc, ncyc) / (w * N)

    ic0 = np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    ic = np.zeros(ncyc, dtype=np.complex128)
    cnt = 0

    for np_j, phi_j, amp_j, Rs_j in zip(np_seg, phi, amp, Rs):
        if np_j <= 0:
            continue
        gamma = 0.5 * (R / Z0 + Z0 / Rs_j)
        wn = np.sqrt(1 + R / Rs_j)
        alpha = np.sqrt((gamma**2 - wn**2) + 0j)
        lam = -gamma + np.array([1.0, -1.0]) * alpha
        lmat = np.array([[1.0 + 0j, 1.0 + 0j], [lam[0], lam[1]]])

        tv = 2 * np.pi * np.linspace(1, np_j, np_j) / (w * N)
        tve = tv[-1]
        phi_eff = w * tvect[cnt] - 2 * np.pi / N + phi_j

        ch = np.linalg.solve(lmat, ic0)
        ich = ch[0] * np.exp(lam[0] * tv) + ch[1] * np.exp(lam[1] * tv)
        dich = ch[0] * lam[0] * np.exp(lam[0] * tve) + ch[1] * lam[1] * np.exp(lam[1] * tve)

        if amp_j == 0:
            icd = 0.0
            dicd = 0.0
        else:
            icd1 = (
                (amp_j / Rs_j)
                * np.exp(1j * phi_eff)
                * (
                    np.exp(1j * w * tv)
                    + (
                        (lam[1] - 1j * w) * np.exp(lam[0] * tv) / (2 * alpha)
                        - (lam[0] - 1j * w) * np.exp(lam[1] * tv)
                    )
                    / (2 * alpha)
                )
                / (wn**2 - w**2 + 2j * gamma * w)
            )
            icd2 = (
                (amp_j / Rs_j)
                * np.exp(-1j * phi_eff)
                * (
                    np.exp(-1j * w * tv)
                    + (
                        (lam[1] + 1j * w) * np.exp(lam[0] * tv) / (2 * alpha)
                        - (lam[0] + 1j * w) * np.exp(lam[1] * tv)
                    )
                    / (2 * alpha)
                )
                / (wn**2 - w**2 - 2j * gamma * w)
            )
            icd = 0.5 * (icd1 + icd2)

            dicd1 = (
                (amp_j / Rs_j)
                * np.exp(1j * phi_eff)
                * (
                    1j * w * np.exp(1j * w * tve)
                    + (
                        lam[0] * (lam[1] - 1j * w) * np.exp(lam[0] * tve) / (2 * alpha)
                        - lam[1] * (lam[0] - 1j * w) * np.exp(lam[1] * tve)
                    )
                    / (2 * alpha)
                )
                / (wn**2 - w**2 + 2j * gamma * w)
            )
            dicd2 = (
                (amp_j / Rs_j)
                * np.exp(-1j * phi_eff)
                * (
                    -1j * w * np.exp(-1j * w * tve)
                    + (
                        lam[0] * (lam[1] + 1j * w) * np.exp(lam[0] * tve) / (2 * alpha)
                        - lam[1] * (lam[0] + 1j * w) * np.exp(lam[1] * tve)
                    )
                    / (2 * alpha)
                )
                / (wn**2 - w**2 - 2j * gamma * w)
            )
            dicd = 0.5 * (dicd1 + dicd2)

        ic[cnt : cnt + np_j] = ich + icd
        ic0[0] = ic[cnt + np_j - 1]
        ic0[1] = dich + dicd
        cnt += np_j

    icr = ic * np.exp(-1j * w0 * tvect)

    numpts = int(np.floor(ncyc / N))
    tvect2 = np.zeros(numpts, dtype=np.float64)
    icr2 = np.zeros(numpts, dtype=np.complex128)
    for idx in range(numpts):
        ind = slice(idx * N, (idx + 1) * N)
        tvect2[idx] = np.mean(tvect[ind])
        icr2[idx] = np.mean(icr[ind])

    return tvect2 / wp, icr2, tvect / wp, ic


def tuned_probe_rx(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    macq: np.ndarray,
) -> tuple[np.ndarray, float, complex]:
    """Apply tuned-probe receiver filtering and matched-filter SNR.

    Mirrors MATLAB `circuit_simulation/tuned_probe/tuned_probe_rx.m`.
    """

    k = float(_field(sp, "k"))
    T = float(_field(sp, "T"))
    L = float(_field(sp, "L"))
    R = float(_field(sp, "R"))
    C = float(_field(sp, "C"))
    Cin = float(_field(sp, "Cin"))
    Rin = float(_field(sp, "Rin"))
    Rd = float(_field(sp, "Rd"))
    vn = float(_field(sp, "vn"))
    inn = float(_field(sp, "in_"))
    mf_type = int(_field(sp, "mf_type"))
    w0 = float(_field(sp, "w0"))
    del_w = _as_vector(_field(sp, "del_w"))
    macq = np.asarray(macq, dtype=np.complex128).reshape(-1)

    w1_max = (np.pi / 2) / float(_field(pp, "T_90"))
    s = 1j * (w0 + del_w * w1_max)
    f = np.imag(s) / (2 * np.pi)
    Yin = s * Cin + 1 / Rin
    Yp = s * C + 1 / Rd + 1 / (s * L + R)
    tf = 1 / (1 + (s * L + R) * (s * C + 1 / Rd + Yin))
    Zs = 1 / (Yin + Yp)

    mrx = macq * tf * (2 * np.pi * f / w0) ** 2
    vni2 = 4 * k * T * R * np.abs(tf) ** 2
    pnoise = vn**2 + inn**2 * np.abs(Zs) ** 2 + vni2

    if mf_type == 0:
        theta = np.arctan2(np.sum(np.imag(mrx)), np.sum(np.real(mrx)))
        T_W = 0.8 * np.pi
        mf = np.sinc(del_w * T_W / (2 * np.pi)) * np.exp(-1j * theta)
    elif mf_type == 1:
        mf = np.conj(mrx)
    elif mf_type == 2:
        mf = np.conj(mrx) / pnoise
    else:
        raise ValueError("mf_type must be 0, 1, or 2")

    mf = mf / np.sqrt(trapezoid(np.abs(mf) ** 2, del_w))
    vsig = trapezoid(mrx * mf, del_w)
    vnoise = np.sqrt(trapezoid(pnoise * np.abs(mf) ** 2, f))
    snr = float(np.real(vsig) / vnoise)
    return mrx, snr, vsig


def calc_rot_axis_tuned_probe_lp_orig2(
    params: Mapping[str, Any] | Any,
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> np.ndarray:
    """Calculate tuned-probe refocusing rotation axis for the Orig path."""

    T_90 = float(_field(pp, "T_90"))
    B1max = (np.pi / 2) / (T_90 * float(_field(sp, "gamma")))
    sens = float(_field(sp, "sens"))

    params_tref = _as_vector(_field(params, "tref"))
    params_pref = _as_vector(_field(params, "pref"))
    params_aref = _as_vector(_field(params, "aref"))
    params_Rs = _as_vector(_field(params, "Rs"))
    tfp = float(_field(params, "tfp"))
    tqs = float(_field(params, "tqs"))

    pp_ref = _with_fields(
        pp,
        tref=np.concatenate([[tfp], params_tref, [tqs, tfp - tqs]]),
        pref=np.concatenate([[0.0], params_pref, [0.0, 0.0]]),
        aref=np.concatenate([[0.0], params_aref, [0.0, 0.0]]),
        Rsref=np.concatenate(
            [
                [params_Rs[0]],
                params_Rs[1] * np.ones(params_tref.size),
                [params_Rs[2], params_Rs[0]],
            ]
        ),
    )
    tvect, icr, _tvect_raw, _ic = tuned_probe_lp_orig(sp, pp_ref)

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    trefc = delt * np.ones(tvect.size)
    prefc = np.arctan2(np.imag(icr), np.real(icr))
    arefc = np.abs(icr) * sens / B1max
    arefc[arefc < float(_field(pp, "amp_zero"))] = 0
    prefc[arefc == 0] = 0

    return calc_rot_axis_arba3(trefc, prefc, arefc, _as_vector(_field(sp, "del_w")))


def calc_masy_tuned_probe_lp_orig(
    params: Mapping[str, Any] | Any,
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate tuned-probe CPMG asymptotic and received spectra.

    Mirrors MATLAB `calc_masy/calc_masy_tuned_probe_lp_Orig.m`.
    """

    T_90 = float(_field(pp, "T_90"))
    B1max = (np.pi / 2) / (T_90 * float(_field(sp, "gamma")))
    amp_zero = float(_field(pp, "amp_zero"))
    tacq = (np.pi / 2) * _as_vector(_field(pp, "tacq")) / T_90
    tacq_scalar = float(tacq[0])
    neff = calc_rot_axis_tuned_probe_lp_orig2(params, sp, pp)

    texc_params = _as_vector(_field(params, "texc"))
    pexc_params = _as_vector(_field(params, "pexc"))
    aexc_params = _as_vector(_field(params, "aexc"))
    params_Rs = _as_vector(_field(params, "Rs"))

    pp_exc = _with_fields(
        pp,
        tref=np.concatenate([texc_params, [_field(params, "tqs"), _field(params, "trd")]]),
        pref=np.concatenate([pexc_params, [0.0, 0.0]]),
        aref=np.concatenate([aexc_params, [0.0, 0.0]]),
        Rsref=np.concatenate([
            params_Rs[1] * np.ones(texc_params.size),
            [params_Rs[2], params_Rs[0]],
        ]),
    )
    tvect, icr, _tvect_raw, _ic = tuned_probe_lp_orig(sp, pp_exc)

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    texc = delt * np.ones(tvect.size)
    pexc = np.arctan2(np.imag(icr), np.real(icr))
    aexc = np.abs(icr) * float(_field(sp, "sens")) / B1max
    aexc[aexc < amp_zero] = 0
    pexc[aexc == 0] = 0

    if texc_params.size == 1:
        tail = -(
            float(_field(params, "tqs"))
            + float(_field(params, "trd"))
            - float(_field(pp, "tcorr"))
        ) * (np.pi / 2) / T_90
    else:
        tail = -(
            float(_field(params, "tqs")) + float(_field(params, "trd"))
        ) * (np.pi / 2) / T_90

    texc = np.concatenate([texc, [tail]])
    pexc = np.concatenate([pexc, [0.0]])
    aexc = np.concatenate([aexc, [0.0]])
    del_w = _as_vector(_field(sp, "del_w"))

    pcycle = int(_field(params, "pcycle"))
    if pcycle == 0:
        masy = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff, del_w, tacq_scalar)
    elif pcycle == 1:
        masy1 = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff, del_w, tacq_scalar)
        masy2 = sim_spin_dynamics_asymp_mag3(texc, pexc + np.pi, aexc, neff, del_w, tacq_scalar)
        masy = (masy1 - masy2) / 2
    elif pcycle == 2:
        masy1 = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff, del_w, tacq_scalar)
        masy2 = sim_spin_dynamics_asymp_mag3(texc, -pexc, aexc, neff, del_w, tacq_scalar)
        masy = (masy1 - masy2) / 2
    else:
        raise ValueError("pcycle must be 0, 1, or 2")

    mrx, snr, _vsig = tuned_probe_rx(sp, pp, masy)
    return mrx, masy, snr / 1e8
