"""Arbitrary-pulse spin-dynamics kernels.

Primary MATLAB references:
    SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m
    SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb9.m
    SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
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
class Arb10DiffusionParameters(Arb10Parameters):
    """Parameters for `sim_spin_dynamics_arb10_diffusion`."""

    gamma: float
    gradient: float
    diffusion_coefficient: float
    diffusion_time: float


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


def _free_precession_matrix_elements_diffusion(
    del_w: np.ndarray,
    tf: float,
    T1n: np.ndarray,
    T2n: np.ndarray,
    gamma: float,
    gradient: float,
    diffusion_coefficient: float,
    diffusion_time: float,
) -> MatrixElements:
    mat = _free_precession_matrix_elements(del_w, tf, T1n, T2n)
    attenuation = np.exp(
        -(1.0 / 12.0)
        * float(gamma) ** 2
        * float(gradient) ** 2
        * float(diffusion_coefficient)
        * float(diffusion_time) ** 3
    )
    return MatrixElements(
        R_00=mat.R_00,
        R_0p=mat.R_0p,
        R_0m=mat.R_0m,
        R_p0=mat.R_p0,
        R_m0=mat.R_m0,
        R_pp=attenuation * mat.R_pp,
        R_mm=attenuation * mat.R_mm,
        R_pm=mat.R_pm,
        R_mp=mat.R_mp,
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


def sim_spin_dynamics_arb10_diffusion(
    params: Mapping[str, Any] | Arb10DiffusionParameters | Any,
) -> np.ndarray:
    """Simulate arbitrary-pulse dynamics with a diffusion free-precession term.

    This is an `arb10`-style modernization of MATLAB
    `sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m`: RF pulse
    matrices are precomputed and acquisitions are returned as spectra without
    the older sinc-window convolution.
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
    gamma = float(_field(params, "gamma"))
    gradient = float(_field(params, "gradient"))
    diffusion_coefficient = float(_field(params, "diffusion_coefficient"))
    diffusion_time = float(_field(params, "diffusion_time"))

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
            mat = _free_precession_matrix_elements_diffusion(
                del_w,
                float(tp_j),
                T1n,
                T2n,
                gamma,
                gradient,
                diffusion_coefficient,
                diffusion_time,
            )
            mlong = mth * (1 - np.exp(-tp_j / T1n))

        tmp = mvect.copy()
        mvect[0, :] = mat.R_00 * tmp[0, :] + mat.R_0m * tmp[1, :] + mat.R_0p * tmp[2, :] + mlong
        mvect[1, :] = mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
        mvect[2, :] = mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]

        if acq_j:
            macq[acq_cnt, :] = mvect[1, :]
            acq_cnt += 1

    return macq


def _slice_matrix_elements(mat: MatrixElements, slc: slice) -> MatrixElements:
    return MatrixElements(
        R_00=mat.R_00[slc],
        R_0p=mat.R_0p[slc],
        R_0m=mat.R_0m[slc],
        R_p0=mat.R_p0[slc],
        R_m0=mat.R_m0[slc],
        R_pp=mat.R_pp[slc],
        R_mm=mat.R_mm[slc],
        R_pm=mat.R_pm[slc],
        R_mp=mat.R_mp[slc],
    )


def _slice_arb10_params(
    params: Mapping[str, Any] | Arb10Parameters | Any,
    slc: slice,
) -> Arb10Parameters:
    rtot = tuple(_slice_matrix_elements(mat, slc) for mat in _field(params, "Rtot"))
    return Arb10Parameters(
        tp=_field(params, "tp"),
        pul=_field(params, "pul"),
        Rtot=rtot,
        amp=_field(params, "amp"),
        acq=_field(params, "acq"),
        grad=_field(params, "grad"),
        del_w=_as_vector(_field(params, "del_w"), np.float64)[slc],
        del_wg=_as_vector(_field(params, "del_wg"), np.float64)[slc],
        T1n=_as_vector(_field(params, "T1n"), np.float64)[slc],
        T2n=_as_vector(_field(params, "T2n"), np.float64)[slc],
        m0=_as_vector(_field(params, "m0"), np.complex128)[slc],
        mth=_as_vector(_field(params, "mth"), np.complex128)[slc],
    )


def _slice_arb10_diffusion_params(
    params: Mapping[str, Any] | Arb10DiffusionParameters | Any,
    slc: slice,
) -> Arb10DiffusionParameters:
    base = _slice_arb10_params(params, slc)
    return Arb10DiffusionParameters(
        **base.__dict__,
        gamma=float(_field(params, "gamma")),
        gradient=float(_field(params, "gradient")),
        diffusion_coefficient=float(_field(params, "diffusion_coefficient")),
        diffusion_time=float(_field(params, "diffusion_time")),
    )


def _chunk_slices(numpts: int, chunks: int) -> list[slice]:
    bounds = np.linspace(0, numpts, chunks + 1, dtype=np.int64)
    return [
        slice(int(start), int(stop))
        for start, stop in zip(bounds[:-1], bounds[1:])
        if stop > start
    ]


def sim_spin_dynamics_arb10_chunked(
    params: Mapping[str, Any] | Arb10Parameters | Any,
    num_workers: int | None = None,
    min_chunk_size: int = 256,
) -> np.ndarray:
    """Run `sim_spin_dynamics_arb10` on contiguous isochromat chunks.

    The serial kernel is already vectorized over isochromats. This helper
    splits that vector into core-sized chunks and uses threads to avoid copying
    the full state through process boundaries.
    """

    del_w = _as_vector(_field(params, "del_w"), np.float64)
    numpts = del_w.size
    if numpts == 0:
        return sim_spin_dynamics_arb10(params)

    if num_workers is None:
        workers = os.cpu_count() or 1
    else:
        workers = int(num_workers)
    if workers <= 1:
        return sim_spin_dynamics_arb10(params)

    max_useful_workers = max(1, int(np.ceil(numpts / max(1, int(min_chunk_size)))))
    workers = min(workers, numpts, max_useful_workers)
    if workers <= 1:
        return sim_spin_dynamics_arb10(params)

    slices = _chunk_slices(numpts, workers)
    chunk_params = [_slice_arb10_params(params, slc) for slc in slices]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(executor.map(sim_spin_dynamics_arb10, chunk_params))
    return np.concatenate(chunks, axis=1)


def sim_spin_dynamics_arb10_diffusion_chunked(
    params: Mapping[str, Any] | Arb10DiffusionParameters | Any,
    num_workers: int | None = None,
    min_chunk_size: int = 256,
) -> np.ndarray:
    """Run `sim_spin_dynamics_arb10_diffusion` on isochromat chunks."""

    del_w = _as_vector(_field(params, "del_w"), np.float64)
    numpts = del_w.size
    if numpts == 0:
        return sim_spin_dynamics_arb10_diffusion(params)

    if num_workers is None:
        workers = os.cpu_count() or 1
    else:
        workers = int(num_workers)
    if workers <= 1:
        return sim_spin_dynamics_arb10_diffusion(params)

    max_useful_workers = max(1, int(np.ceil(numpts / max(1, int(min_chunk_size)))))
    workers = min(workers, numpts, max_useful_workers)
    if workers <= 1:
        return sim_spin_dynamics_arb10_diffusion(params)

    slices = _chunk_slices(numpts, workers)
    chunk_params = [_slice_arb10_diffusion_params(params, slc) for slc in slices]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(executor.map(sim_spin_dynamics_arb10_diffusion, chunk_params))
    return np.concatenate(chunks, axis=1)


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
