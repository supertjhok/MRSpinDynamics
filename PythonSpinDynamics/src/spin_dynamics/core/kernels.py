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
from spin_dynamics.radiation_damping import RadiationDampingSpec


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
    """Parameters for `sim_spin_dynamics_arb10_diffusion`.

    ``time_scale`` is the number of physical seconds per unit of normalized
    pulse time (``2 * T_90 / pi``). It converts each normalized free-precession
    interval ``tf`` to seconds so the constant-gradient diffusion attenuation
    is evaluated on the interval's own duration.
    """

    gamma: float
    gradient: float
    diffusion_coefficient: float
    time_scale: float = 1.0


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
    time_scale: float,
) -> MatrixElements:
    """Free-precession matrix with constant-gradient diffusion attenuation.

    Each free-precession interval of physical duration ``t = |tf| * time_scale``
    attenuates the transverse coherence by
    ``exp(-(1/3) * gamma**2 * gradient**2 * D * t**3)``. This is the unrefocused
    (FID) free-diffusion result in a constant gradient; diffusion is
    irreversible, so the refocusing pulses between intervals do not recover it
    and applying the factor independently per interval is exact for a constant
    background gradient. A CPMG train of ``N`` echoes with echo spacing ``t_E``
    (two intervals of ``t_E/2`` per echo) thus reproduces the textbook
    ``exp(-(1/12) * gamma**2 * gradient**2 * D * t_E**3 * N)``.

    ``tf`` is the normalized interval time and ``time_scale`` (``2 * T_90 / pi``)
    converts it to seconds; ``gamma``, ``gradient``, and ``diffusion_coefficient``
    are SI, so the exponent is dimensionless. The interval duration is taken in
    magnitude so rewind/negative-time segments still attenuate rather than grow.
    """

    mat = _free_precession_matrix_elements(del_w, tf, T1n, T2n)
    t_seconds = abs(float(tf)) * float(time_scale)
    attenuation = np.exp(
        -(1.0 / 3.0)
        * float(gamma) ** 2
        * float(gradient) ** 2
        * float(diffusion_coefficient)
        * t_seconds ** 3
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


def _validate_arb10_inputs(
    params: Mapping[str, Any] | Arb10Parameters | Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Sequence[MatrixElements],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tp = _as_vector(_field(params, "tp"), np.float64)
    pul = _as_vector(_field(params, "pul"), np.int64)
    rtot = _field(params, "Rtot")
    amp = _as_vector(_field(params, "amp"), np.float64)
    acq = _as_vector(_field(params, "acq"), bool)
    grad = _as_vector(_field(params, "grad"), np.float64)
    del_w0 = _as_vector(_field(params, "del_w"), np.float64)
    del_wg = _as_vector(_field(params, "del_wg"), np.float64)
    t1n = _as_vector(_field(params, "T1n"), np.float64)
    t2n = _as_vector(_field(params, "T2n"), np.float64)
    m0 = _as_vector(_field(params, "m0"), np.complex128)
    mth = _as_vector(_field(params, "mth"), np.complex128)

    numpts = del_w0.size
    if not (tp.size == pul.size == amp.size == acq.size == grad.size):
        raise ValueError("tp, pul, amp, acq, and grad must have the same length")
    for name, arr in {
        "del_wg": del_wg,
        "T1n": t1n,
        "T2n": t2n,
        "m0": m0,
        "mth": mth,
    }.items():
        if arr.size != numpts:
            raise ValueError(f"{name} must have length len(del_w)")
    return tp, pul, rtot, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth


def _apply_matrix_step(
    mvect: np.ndarray,
    mat: MatrixElements,
    mlong: np.ndarray,
) -> np.ndarray:
    tmp = mvect.copy()
    out = np.empty_like(mvect)
    out[0, :] = mat.R_00 * tmp[0, :] + mat.R_0m * tmp[1, :] + mat.R_0p * tmp[2, :] + mlong
    out[1, :] = mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
    out[2, :] = mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]
    return out


def _matrix_elements_power(mat: MatrixElements, exponent: float) -> MatrixElements:
    size = mat.R_00.size
    arrays = {
        "R_00": np.empty(size, dtype=np.complex128),
        "R_0m": np.empty(size, dtype=np.complex128),
        "R_0p": np.empty(size, dtype=np.complex128),
        "R_m0": np.empty(size, dtype=np.complex128),
        "R_mm": np.empty(size, dtype=np.complex128),
        "R_mp": np.empty(size, dtype=np.complex128),
        "R_p0": np.empty(size, dtype=np.complex128),
        "R_pm": np.empty(size, dtype=np.complex128),
        "R_pp": np.empty(size, dtype=np.complex128),
    }
    for idx in range(size):
        full = np.array(
            [
                [mat.R_00[idx], mat.R_0m[idx], mat.R_0p[idx]],
                [mat.R_m0[idx], mat.R_mm[idx], mat.R_mp[idx]],
                [mat.R_p0[idx], mat.R_pm[idx], mat.R_pp[idx]],
            ],
            dtype=np.complex128,
        )
        vals, vecs = np.linalg.eig(full)
        powered = vecs @ np.diag(vals**exponent) @ np.linalg.inv(vecs)
        arrays["R_00"][idx] = powered[0, 0]
        arrays["R_0m"][idx] = powered[0, 1]
        arrays["R_0p"][idx] = powered[0, 2]
        arrays["R_m0"][idx] = powered[1, 0]
        arrays["R_mm"][idx] = powered[1, 1]
        arrays["R_mp"][idx] = powered[1, 2]
        arrays["R_p0"][idx] = powered[2, 0]
        arrays["R_pm"][idx] = powered[2, 1]
        arrays["R_pp"][idx] = powered[2, 2]
    return MatrixElements(**arrays)


def _radiation_damping_weights(
    spec: RadiationDampingSpec,
    numpts: int,
) -> np.ndarray:
    if spec.weights is None:
        return np.full(numpts, 1.0 / max(1, numpts), dtype=np.float64)
    weights = _as_vector(spec.weights, np.float64)
    if weights.size != numpts:
        raise ValueError("radiation damping weights must match len(del_w)")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("radiation damping weights must sum to a positive value")
    return weights / total


def _radiation_damping_step_limit(
    spec: RadiationDampingSpec,
    t1n: np.ndarray,
    t2n: np.ndarray,
) -> float:
    candidates = [spec.trd / 50.0]
    if spec.model == "circuit":
        candidates.append(spec.resonator_time_constant / 20.0)
    finite_t1 = t1n[np.isfinite(t1n) & (t1n > 0)]
    finite_t2 = t2n[np.isfinite(t2n) & (t2n > 0)]
    if finite_t1.size:
        candidates.append(float(np.min(finite_t1)) / 50.0)
    if finite_t2.size:
        candidates.append(float(np.min(finite_t2)) / 50.0)
    if spec.max_step is not None:
        if spec.max_step <= 0:
            raise ValueError("radiation damping max_step must be positive")
        candidates.append(float(spec.max_step))
    step = min(candidates)
    if step <= 0 or not np.isfinite(step):
        raise ValueError("radiation damping step size must be finite and positive")
    return step


def _rd_free_rhs(
    mvect: np.ndarray,
    feedback: complex,
    del_w: np.ndarray,
    t1n: np.ndarray,
    t2n: np.ndarray,
    mth: np.ndarray,
) -> np.ndarray:
    deriv = np.empty_like(mvect)
    mz = mvect[0, :]
    mminus = mvect[1, :]
    mplus = mvect[2, :]
    deriv[0, :] = (mth - mz) / t1n - np.real(np.conj(mplus) * feedback)
    deriv[1, :] = (-1j * del_w - 1.0 / t2n) * mminus + mz * np.conj(feedback)
    deriv[2, :] = (1j * del_w - 1.0 / t2n) * mplus + mz * feedback
    return deriv


def _rd_feedback_rhs(
    mvect: np.ndarray,
    feedback: complex,
) -> np.ndarray:
    deriv = np.empty_like(mvect)
    mz = mvect[0, :]
    mplus = mvect[2, :]
    deriv[0, :] = -np.real(np.conj(mplus) * feedback)
    deriv[1, :] = mz * np.conj(feedback)
    deriv[2, :] = mz * feedback
    return deriv


def _radiation_damping_target(
    mvect: np.ndarray,
    spec: RadiationDampingSpec,
    weights: np.ndarray,
) -> complex:
    return (
        np.exp(1j * spec.probe.phase)
        * np.conj(np.sum(weights * mvect[2, :]))
        / spec.trd
    )


def _feedback_rhs_pair(
    mvect: np.ndarray,
    feedback: complex,
    spec: RadiationDampingSpec,
    weights: np.ndarray,
) -> tuple[np.ndarray, complex]:
    if spec.model == "instant":
        use_fb = _radiation_damping_target(mvect, spec, weights)
        dfb = 0.0 + 0.0j
    elif spec.model == "circuit":
        use_fb = feedback
        dfb = (
            (_radiation_damping_target(mvect, spec, weights) - feedback)
            / spec.resonator_time_constant
            - 1j * spec.detuning * feedback
        )
    else:
        raise ValueError("radiation damping model must be 'instant' or 'circuit'")
    return _rd_feedback_rhs(mvect, use_fb), dfb


def _advance_feedback_only_radiation_damping(
    mvect: np.ndarray,
    duration: float,
    spec: RadiationDampingSpec,
    weights: np.ndarray,
    feedback: complex,
    step_limit: float,
) -> tuple[np.ndarray, complex]:
    if duration == 0:
        return mvect, feedback
    steps = max(1, int(np.ceil(abs(duration) / step_limit)))
    h = duration / steps
    state = mvect
    fb = complex(feedback)
    for _ in range(steps):
        k1, f1 = _feedback_rhs_pair(state, fb, spec, weights)
        k2, f2 = _feedback_rhs_pair(state + h * k1 / 2.0, fb + h * f1 / 2.0, spec, weights)
        k3, f3 = _feedback_rhs_pair(state + h * k2 / 2.0, fb + h * f2 / 2.0, spec, weights)
        k4, f4 = _feedback_rhs_pair(state + h * k3, fb + h * f3, spec, weights)
        state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        fb = fb + h * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
        if spec.model == "instant":
            fb = _radiation_damping_target(state, spec, weights)
    return state, fb


def _advance_free_radiation_damping(
    mvect: np.ndarray,
    duration: float,
    del_w: np.ndarray,
    t1n: np.ndarray,
    t2n: np.ndarray,
    mth: np.ndarray,
    spec: RadiationDampingSpec,
    weights: np.ndarray,
    feedback: complex,
    step_limit: float,
) -> tuple[np.ndarray, complex]:
    if duration == 0:
        return mvect, feedback
    steps = max(1, int(np.ceil(abs(duration) / step_limit)))
    h = duration / steps
    state = mvect
    fb = complex(feedback)

    def rhs_pair(current: np.ndarray, current_fb: complex) -> tuple[np.ndarray, complex]:
        if spec.model == "instant":
            use_fb = _radiation_damping_target(current, spec, weights)
            dfb = 0.0 + 0.0j
        elif spec.model == "circuit":
            use_fb = current_fb
            dfb = (
                (_radiation_damping_target(current, spec, weights) - current_fb)
                / spec.resonator_time_constant
                - 1j * spec.detuning * current_fb
            )
        else:
            raise ValueError("radiation damping model must be 'instant' or 'circuit'")
        return _rd_free_rhs(current, use_fb, del_w, t1n, t2n, mth), dfb

    for _ in range(steps):
        k1, f1 = rhs_pair(state, fb)
        k2, f2 = rhs_pair(state + h * k1 / 2.0, fb + h * f1 / 2.0)
        k3, f3 = rhs_pair(state + h * k2 / 2.0, fb + h * f2 / 2.0)
        k4, f4 = rhs_pair(state + h * k3, fb + h * f3)
        state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        fb = fb + h * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
        if spec.model == "instant":
            fb = _radiation_damping_target(state, spec, weights)
    return state, fb


def _advance_pulse_matrix_with_radiation_damping(
    mvect: np.ndarray,
    mat: MatrixElements,
    duration: float,
    spec: RadiationDampingSpec,
    weights: np.ndarray,
    feedback: complex,
    step_limit: float,
) -> tuple[np.ndarray, complex]:
    if duration == 0:
        return _apply_matrix_step(
            mvect,
            mat,
            np.zeros(mvect.shape[1], dtype=np.complex128),
        ), feedback
    steps = max(1, int(np.ceil(abs(duration) / step_limit)))
    submat = _matrix_elements_power(mat, 1.0 / steps)
    state = mvect
    fb = feedback
    zeros = np.zeros(mvect.shape[1], dtype=np.complex128)
    h = duration / steps
    for _ in range(steps):
        state = _apply_matrix_step(state, submat, zeros)
        state, fb = _advance_feedback_only_radiation_damping(
            state,
            h,
            spec,
            weights,
            fb,
            step_limit,
        )
    return state, fb


def sim_spin_dynamics_arb10_radiation_damping(
    params: Mapping[str, Any] | Arb10Parameters | Any,
    radiation_damping: RadiationDampingSpec,
) -> np.ndarray:
    """Simulate `arb10` with ensemble radiation damping during free intervals.

    RF pulse blocks continue to use the supplied precomputed matrices. This
    first nonlinear bridge therefore captures receive-window and inter-pulse
    back-action without perturbing the validated pulse-shape machinery.
    """

    tp, pul, rtot, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth = (
        _validate_arb10_inputs(params)
    )
    numpts = del_w0.size
    weights = _radiation_damping_weights(radiation_damping, numpts)
    step_limit = _radiation_damping_step_limit(radiation_damping, t1n, t2n)

    mvect = np.zeros((3, numpts), dtype=np.complex128)
    mvect[0, :] = m0
    macq = np.zeros((int(np.sum(acq)), numpts), dtype=np.complex128)
    acq_cnt = 0
    feedback = complex(radiation_damping.initial_feedback)

    for tp_j, pul_j, amp_j, acq_j, grad_j in zip(tp, pul, amp, acq, grad):
        del_w = del_w0 + grad_j * del_wg
        if amp_j > 0 and not radiation_damping.apply_during_pulses:
            mat = rtot[int(pul_j) - 1]
            mlong = np.zeros(numpts, dtype=np.complex128)
            mvect = _apply_matrix_step(mvect, mat, mlong)
        elif amp_j > 0:
            mvect, feedback = _advance_pulse_matrix_with_radiation_damping(
                mvect,
                rtot[int(pul_j) - 1],
                float(tp_j),
                radiation_damping,
                weights,
                feedback,
                step_limit,
            )
        else:
            mvect, feedback = _advance_free_radiation_damping(
                mvect,
                float(tp_j),
                del_w,
                t1n,
                t2n,
                mth,
                radiation_damping,
                weights,
                feedback,
                step_limit,
            )

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
    time_scale = float(_field_or_default(params, "time_scale", 1.0))

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
                time_scale,
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
        time_scale=float(_field_or_default(params, "time_scale", 1.0)),
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
