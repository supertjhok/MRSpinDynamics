"""Arbitrary-pulse spin-dynamics kernels.

Primary MATLAB references:
    MATLABSpinDynamics/Version_3/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m
    MATLABSpinDynamics/Version_3/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb9.m
    MATLABSpinDynamics/Version_3/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Any
import warnings

import numpy as np

from spin_dynamics.core.rotations import (
    MatrixElements,
    calc_rotation_matrix,
    rf_matrix_elements,
)
from spin_dynamics.radiation_damping import RadiationDampingSpec


_VALID_BACKENDS = ("numpy", "numba", "jax")
_DEFAULT_BACKEND = "numpy"


def set_arb10_backend(name: str) -> None:
    """Select the default backend for ``sim_spin_dynamics_arb10``.

    ``"numpy"`` (default) is the always-available reference. ``"numba"`` uses
    the JIT-compiled segment-loop kernel (optional ``numba`` extra). ``"jax"``
    uses the x64 jit-compiled kernel (optional ``jax`` extra), which also
    supports GPU and ``vmap`` and is the basis for the autodiff optimizer work.
    The serial and chunked acquisition paths both honor this default.
    """

    if name not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}")
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = name


def get_arb10_backend() -> str:
    """Return the current default ``sim_spin_dynamics_arb10`` backend."""

    return _DEFAULT_BACKEND


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
    is evaluated on the interval's own duration. This is the reciprocal of
    :attr:`spin_dynamics.radiation_damping.RadiationDampingSpec.time_scale`
    (normalized units per second) for the same ``T_90``.
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


def _apply_free_precession_step(
    mvect: np.ndarray,
    del_w: np.ndarray,
    tf: float,
    t1n: np.ndarray,
    t2n: np.ndarray,
    mth: np.ndarray,
    transverse_attenuation: float = 1.0,
) -> None:
    longitudinal = np.exp(-tf / t1n)
    transverse = np.exp(-tf / t2n) * np.exp(1j * del_w * tf)
    if transverse_attenuation != 1.0:
        transverse = float(transverse_attenuation) * transverse
    mvect[0, :] = longitudinal * mvect[0, :] + mth * (1.0 - longitudinal)
    mvect[1, :] = np.conj(transverse) * mvect[1, :]
    mvect[2, :] = transverse * mvect[2, :]


def sim_spin_dynamics_arb10(
    params: Mapping[str, Any] | Arb10Parameters | Any,
    *,
    backend: str | None = None,
) -> np.ndarray:
    """Simulate arbitrary-pulse spin dynamics with precomputed pulse matrices.

    Mirrors MATLAB `sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m`.
    `Rtot` uses MATLAB-style pulse numbering in `pul`, so `pul=1` selects the
    first Python sequence entry. Free-precession segments should have `amp=0`.
    Acquired rows are ``M- = Mx - i My``; combine phase-cycled branches with
    ``PhaseCycle.combine(..., detection="m_minus")``.

    `backend` selects the compute path: ``"numpy"`` (the reference) or
    ``"numba"`` (the JIT-compiled segment loop, requires the ``numba`` extra).
    When ``None`` the module default from :func:`set_arb10_backend` is used.
    """

    resolved = _DEFAULT_BACKEND if backend is None else backend
    if resolved == "numba":
        return _sim_spin_dynamics_arb10_numba(params)
    if resolved == "jax":
        return _sim_spin_dynamics_arb10_jax(params)
    if resolved != "numpy":
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}")

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
        if amp_j > 0:
            mat = rtot[int(pul_j) - 1]
            mlong = np.zeros(numpts, dtype=np.complex128)
            tmp = mvect.copy()
            mvect[0, :] = mat.R_00 * tmp[0, :] + mat.R_0m * tmp[1, :] + mat.R_0p * tmp[2, :] + mlong
            mvect[1, :] = mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
            mvect[2, :] = mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]
        else:
            del_w = del_w0 + grad_j * del_wg
            _apply_free_precession_step(
                mvect,
                del_w,
                float(tp_j),
                T1n,
                T2n,
                mth,
            )

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


def _sim_spin_dynamics_arb10_numba(
    params: Mapping[str, Any] | Arb10Parameters | Any,
) -> np.ndarray:
    """Numba backend for :func:`sim_spin_dynamics_arb10`.

    Packs the precomputed pulse matrices into the stacked layout and runs the
    JIT-compiled segment loop. Semantically identical to the NumPy path.
    """

    from spin_dynamics.core import _numba_kernels as nk

    if not nk.NUMBA_AVAILABLE:
        raise ImportError(
            "backend='numba' requires the optional 'numba' extra. Install it "
            "with `python -m pip install -e .[numba]` (or `.[perf]`)."
        )

    tp, pul, rtot, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth = (
        _validate_arb10_inputs(params)
    )
    rstack = _stack_matrix_elements(rtot, del_w0.size)
    n_acq = int(np.sum(acq))
    return nk.arb10_core(
        tp,
        pul,
        amp,
        acq.astype(np.uint8),
        grad,
        del_w0,
        del_wg,
        t1n,
        t2n,
        m0,
        mth,
        rstack,
        n_acq,
    )


def _sim_spin_dynamics_arb10_jax(
    params: Mapping[str, Any] | Arb10Parameters | Any,
) -> np.ndarray:
    """JAX backend for :func:`sim_spin_dynamics_arb10`.

    Packs the precomputed pulse matrices into the stacked layout and runs the
    x64 jit-compiled segment kernel. Semantically identical to the NumPy path;
    the compiled kernel is cached per sequence structure and isochromat shape.
    """

    from spin_dynamics.core import _jax_kernels as jk

    if not jk.JAX_AVAILABLE:
        raise ImportError(
            "backend='jax' requires the optional 'jax' extra. Install it with "
            "`python -m pip install -e .[jax]` (or `.[perf]`)."
        )

    tp, pul, rtot, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth = (
        _validate_arb10_inputs(params)
    )
    rstack = _stack_matrix_elements(rtot, del_w0.size)
    return jk.run_arb10(
        tp, pul, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth, rstack
    )


def sim_spin_dynamics_arb10_batched(
    params_list: Sequence[Mapping[str, Any] | Arb10Parameters | Any],
) -> np.ndarray:
    """Run many same-structured arb10 simulations in one vmapped JAX call.

    All cases must share the pulse program and segment timing
    (``tp, pul, amp, acq, grad, del_wg``); they may differ in the per-isochromat
    fields (``del_w, T1n, T2n, m0, mth``) and the pulse matrices (``Rtot``). This
    is the batched primitive behind GPU-accelerated parameter sweeps and
    optimizer multistarts: a single wide program instead of a Python/thread loop.

    Returns an array of shape ``(len(params_list), n_acq, numpts)``. Requires the
    optional ``jax`` extra.
    """

    from spin_dynamics.core import _jax_kernels as jk

    if not jk.JAX_AVAILABLE:
        raise ImportError(
            "sim_spin_dynamics_arb10_batched requires the optional 'jax' extra. "
            "Install it with `python -m pip install -e .[jax]` (or `.[perf]`)."
        )
    if len(params_list) == 0:
        raise ValueError("params_list must not be empty")

    base = None
    del_w0_b, t1n_b, t2n_b, m0_b, mth_b, rstack_b = [], [], [], [], [], []
    for params in params_list:
        tp, pul, rtot, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth = (
            _validate_arb10_inputs(params)
        )
        structure = (
            tp,
            pul,
            amp,
            acq.astype(np.int64),
            grad,
            del_wg,
        )
        if base is None:
            base = structure
            shared = {"tp": tp, "pul": pul, "amp": amp, "acq": acq, "grad": grad, "del_wg": del_wg}
        else:
            for got, want, name in zip(
                structure, base, ("tp", "pul", "amp", "acq", "grad", "del_wg")
            ):
                if got.shape != want.shape or not np.array_equal(got, want):
                    raise ValueError(
                        "all batched cases must share the same pulse program and "
                        f"segment timing; '{name}' differs between cases"
                    )
        del_w0_b.append(del_w0)
        t1n_b.append(t1n)
        t2n_b.append(t2n)
        m0_b.append(m0)
        mth_b.append(mth)
        rstack_b.append(_stack_matrix_elements(rtot, del_w0.size))

    return jk.run_arb10_batched(
        shared["tp"],
        shared["pul"],
        shared["amp"],
        shared["acq"],
        shared["grad"],
        np.stack(del_w0_b, axis=0),
        shared["del_wg"],
        np.stack(t1n_b, axis=0),
        np.stack(t2n_b, axis=0),
        np.stack(m0_b, axis=0),
        np.stack(mth_b, axis=0),
        np.stack(rstack_b, axis=0),
    )


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


def _stack_matrix_elements(
    rtot: Sequence[MatrixElements],
    numpts: int,
) -> np.ndarray:
    """Pack a pulse-matrix sequence into a ``(num_pulses, 3, 3, numpts)`` array.

    Rows index the output coherence component ``(0, -, +)`` and columns the
    input component, matching the application order in ``sim_spin_dynamics_arb10``.
    This stacked layout is what the Numba (and later JAX) backends consume.
    """

    num_pulses = len(rtot)
    stack = np.empty((num_pulses, 3, 3, numpts), dtype=np.complex128)
    for k, mat in enumerate(rtot):
        stack[k, 0, 0] = mat.R_00
        stack[k, 0, 1] = mat.R_0m
        stack[k, 0, 2] = mat.R_0p
        stack[k, 1, 0] = mat.R_m0
        stack[k, 1, 1] = mat.R_mm
        stack[k, 1, 2] = mat.R_mp
        stack[k, 2, 0] = mat.R_p0
        stack[k, 2, 1] = mat.R_pm
        stack[k, 2, 2] = mat.R_pp
    return stack


# Cartesian (Mx, My, Mz) <- coherence (Mz, M-, M+) basis change and its inverse,
# matching the conversion at the end of ``sim_spin_dynamics_exc``.
_COH_TO_CART = np.array(
    [[0.0, 0.5, 0.5], [0.0, 0.5j, -0.5j], [1.0, 0.0, 0.0]], dtype=np.complex128
)
_CART_TO_COH = np.array(
    [[0.0, 0.0, 1.0], [1.0, -1.0j, 0.0], [1.0, 1.0j, 0.0]], dtype=np.complex128
)
_ROTATION_TOL = 1e-8
_ROTATION_PI_TOL = 1e-6


def _matrix_elements_power(mat: MatrixElements, exponent: float) -> MatrixElements:
    """Raise each isochromat's pulse rotation to the real power ``exponent``.

    Each 3x3 coherence-basis matrix must be a proper rotation (no relaxation).
    The result is the rotation about the same effective axis by
    ``exponent * theta``, where ``theta`` in ``[0, pi]`` is the net rotation
    angle recovered from the matrix (Rodrigues form, vectorized over
    isochromats). A matrix alone cannot tell a rotation by ``theta > pi`` from
    one by ``2*pi - theta`` about the opposite axis, so this follows the
    shortest path; it raises ``ValueError`` when ``theta`` is within
    ``1e-6`` rad of ``pi``, where the half-step direction is undetermined.
    Callers that need the physical in-pulse trajectory should build sub-steps
    from the pulse generator instead (see ``pulse_segments`` in
    :func:`sim_spin_dynamics_arb10_radiation_damping`).
    """

    size = mat.R_00.size
    full = np.empty((size, 3, 3), dtype=np.complex128)
    full[:, 0, 0] = mat.R_00
    full[:, 0, 1] = mat.R_0m
    full[:, 0, 2] = mat.R_0p
    full[:, 1, 0] = mat.R_m0
    full[:, 1, 1] = mat.R_mm
    full[:, 1, 2] = mat.R_mp
    full[:, 2, 0] = mat.R_p0
    full[:, 2, 1] = mat.R_pm
    full[:, 2, 2] = mat.R_pp

    cart_c = _COH_TO_CART @ full @ _CART_TO_COH
    if np.max(np.abs(cart_c.imag), initial=0.0) > _ROTATION_TOL:
        raise ValueError("pulse matrix is not a real rotation; cannot take its power")
    cart = cart_c.real
    gram = np.swapaxes(cart, 1, 2) @ cart
    if (
        np.max(np.abs(gram - np.eye(3)), initial=0.0) > _ROTATION_TOL
        or np.any(np.linalg.det(cart) <= 0.0)
    ):
        raise ValueError("pulse matrix is not a proper rotation; cannot take its power")

    trace = np.trace(cart, axis1=1, axis2=2)
    theta = np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    if np.any(np.pi - theta < _ROTATION_PI_TOL):
        raise ValueError(
            "pulse rotation angle is pi to within tolerance: its fractional power "
            "is ambiguous; supply pulse generators (pulse_segments) instead"
        )
    # R = I + sin(theta) K + (1 - cos(theta)) K^2 with K the unit-axis cross
    # matrix: skew = sin(theta) K and sym - I = (1 - cos(theta)) K^2.
    skew = 0.5 * (cart - np.swapaxes(cart, 1, 2))
    sym_minus_i = 0.5 * (cart + np.swapaxes(cart, 1, 2)) - np.eye(3)
    t = float(exponent)
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)
    sin_ratio = np.where(small, t, np.sin(t * safe) / np.sin(safe))
    cos_ratio = np.where(small, t * t, (1.0 - np.cos(t * safe)) / (1.0 - np.cos(safe)))
    powered_cart = (
        np.eye(3)
        + sin_ratio[:, np.newaxis, np.newaxis] * skew
        + cos_ratio[:, np.newaxis, np.newaxis] * sym_minus_i
    )
    powered = _CART_TO_COH @ powered_cart.astype(np.complex128) @ _COH_TO_CART
    return MatrixElements(
        R_00=powered[:, 0, 0].copy(),
        R_0m=powered[:, 0, 1].copy(),
        R_0p=powered[:, 0, 2].copy(),
        R_m0=powered[:, 1, 0].copy(),
        R_mm=powered[:, 1, 1].copy(),
        R_mp=powered[:, 1, 2].copy(),
        R_p0=powered[:, 2, 0].copy(),
        R_pm=powered[:, 2, 1].copy(),
        R_pp=powered[:, 2, 2].copy(),
    )


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
    # Phase-independent (Bloom) feedback target -m+/Trd; see the convention
    # note in spin_dynamics.radiation_damping._rhs and docs/radiation_damping.md.
    return (
        -np.exp(1j * spec.probe.phase)
        * np.sum(weights * mvect[2, :])
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


@dataclass(frozen=True)
class _PulseSegments:
    """Normalized generator of one ``Rtot`` pulse (see ``pulse_segments``)."""

    tp: np.ndarray
    phi: np.ndarray
    amp: np.ndarray
    w_1: np.ndarray
    post_rotation: MatrixElements | None


def _pulse_segments_for(
    raw_segments: Sequence[Any] | None,
    num_pulses: int,
) -> list[_PulseSegments | None]:
    """Normalize optional per-pulse generators."""

    if raw_segments is None:
        return [None] * num_pulses
    if len(raw_segments) != num_pulses:
        raise ValueError("pulse_segments must have one entry per Rtot pulse")
    out: list[_PulseSegments | None] = []
    for entry in raw_segments:
        if entry is None:
            out.append(None)
            continue
        tp = _as_vector(_field(entry, "tp"), np.float64)
        phi = _as_vector(_field(entry, "phi"), np.float64)
        amp = _as_vector(_field(entry, "amp"), np.float64)
        w_1 = np.asarray(_field_or_default(entry, "w_1", 1.0), dtype=np.float64)
        if not (tp.size == phi.size == amp.size) or tp.size == 0:
            raise ValueError("pulse_segments tp, phi, and amp must be non-empty and equal length")
        post = _field_or_default(entry, "post_rotation", None)
        out.append(_PulseSegments(tp, phi, amp, w_1, post))
    return out


def _check_pulse_segments(
    segments: _PulseSegments,
    rtot_mat: MatrixElements,
    del_w: np.ndarray,
) -> None:
    """Raise unless the segment generators reproduce the ``Rtot`` matrix."""

    composed = calc_rotation_matrix(
        del_w, segments.w_1, segments.tp, segments.phi, segments.amp
    )
    if segments.post_rotation is not None:
        # Column c of (post @ composed) is post applied to column c of composed.
        zeros = np.zeros(del_w.size, dtype=np.complex128)
        columns = {
            col: _apply_matrix_step(
                np.stack(
                    [
                        getattr(composed, f"R_0{col}"),
                        getattr(composed, f"R_m{col}"),
                        getattr(composed, f"R_p{col}"),
                    ]
                ),
                segments.post_rotation,
                zeros,
            )
            for col in ("0", "m", "p")
        }
        composed = MatrixElements(
            **{
                f"R_{row}{col}": columns[col][index]
                for index, row in enumerate(("0", "m", "p"))
                for col in ("0", "m", "p")
            }
        )
    for name in MatrixElements.__dataclass_fields__:
        if not np.allclose(
            getattr(composed, name), getattr(rtot_mat, name), rtol=0.0, atol=1e-8
        ):
            raise ValueError("pulse_segments do not reproduce the corresponding Rtot matrix")


def _advance_pulse_segments_with_radiation_damping(
    mvect: np.ndarray,
    segments: _PulseSegments,
    del_w: np.ndarray,
    spec: RadiationDampingSpec,
    weights: np.ndarray,
    feedback: complex,
    step_limit: float,
) -> tuple[np.ndarray, complex]:
    """Sub-step a pulse from its generator, interleaving RD feedback.

    Each rectangular segment (nutation ``amp * w_1``, phase ``phi``, duration
    ``tp``) is split into equal sub-pulses built with ``rf_matrix_elements``,
    so the in-pulse trajectory is the physical one even for rotations of
    ``pi`` or more. Negative-duration (rewind) segments integrate the feedback
    backward, matching the free-interval convention. An optional
    zero-duration ``post_rotation`` is applied last.
    """

    tp, phi, amp, w_1 = segments.tp, segments.phi, segments.amp, segments.w_1
    state = mvect
    fb = feedback
    zeros = np.zeros(mvect.shape[1], dtype=np.complex128)
    for tp_k, phi_k, amp_k in zip(tp, phi, amp):
        if tp_k == 0:
            continue
        steps = max(1, int(np.ceil(abs(float(tp_k)) / step_limit)))
        h = float(tp_k) / steps
        submat = rf_matrix_elements(del_w, amp_k * w_1, h, float(phi_k))
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
    if segments.post_rotation is not None:
        state = _apply_matrix_step(state, segments.post_rotation, zeros)
    return state, fb


def sim_spin_dynamics_arb10_radiation_damping(
    params: Mapping[str, Any] | Arb10Parameters | Any,
    radiation_damping: RadiationDampingSpec,
) -> np.ndarray:
    """Simulate `arb10` with ensemble radiation damping during free intervals.

    RF pulse blocks continue to use the supplied precomputed matrices. This
    first nonlinear bridge therefore captures receive-window and inter-pulse
    back-action without perturbing the validated pulse-shape machinery.

    With ``radiation_damping.apply_during_pulses`` the pulses are sub-stepped
    so feedback also acts inside them. ``params`` may then carry an optional
    ``pulse_segments`` entry: one item per ``Rtot`` matrix, either ``None`` or a
    mapping/object with segment arrays ``tp``, ``phi``, ``amp`` and optional
    per-isochromat ``w_1`` (the :func:`calc_rotation_matrix` convention), plus
    an optional zero-duration ``post_rotation`` matrix applied after the
    segments. Those generators give the exact in-pulse path and must reproduce
    the matrix; the in-pulse feedback then acts over the segments' own
    durations (including any negative rewind segment) rather than ``tp``.
    Without them the pulse is split by :func:`_matrix_elements_power`, which
    follows the shortest rotation path (exact for net rotations below ``pi``),
    emits a ``RuntimeWarning``, and raises for rotations of ``pi`` such as
    on-resonance refocusing pulses.
    """

    tp, pul, rtot, amp, acq, grad, del_w0, del_wg, t1n, t2n, m0, mth = (
        _validate_arb10_inputs(params)
    )
    segments = _pulse_segments_for(
        _field_or_default(params, "pulse_segments", None),
        len(rtot),
    )
    numpts = del_w0.size
    weights = _radiation_damping_weights(radiation_damping, numpts)
    step_limit = _radiation_damping_step_limit(radiation_damping, t1n, t2n)

    mvect = np.zeros((3, numpts), dtype=np.complex128)
    mvect[0, :] = m0
    macq = np.zeros((int(np.sum(acq)), numpts), dtype=np.complex128)
    acq_cnt = 0
    feedback = complex(radiation_damping.initial_feedback)
    verified: set[tuple[int, bytes]] = set()
    warned: set[int] = set()

    for tp_j, pul_j, amp_j, acq_j, grad_j in zip(tp, pul, amp, acq, grad):
        del_w = del_w0 + grad_j * del_wg
        if amp_j > 0 and not radiation_damping.apply_during_pulses:
            mat = rtot[int(pul_j) - 1]
            mlong = np.zeros(numpts, dtype=np.complex128)
            mvect = _apply_matrix_step(mvect, mat, mlong)
        elif amp_j > 0 and segments[int(pul_j) - 1] is not None:
            key = (int(pul_j), del_w.tobytes())
            if key not in verified:
                _check_pulse_segments(segments[int(pul_j) - 1], rtot[int(pul_j) - 1], del_w)
                verified.add(key)
            mvect, feedback = _advance_pulse_segments_with_radiation_damping(
                mvect,
                segments[int(pul_j) - 1],
                del_w,
                radiation_damping,
                weights,
                feedback,
                step_limit,
            )
        elif amp_j > 0:
            if int(pul_j) not in warned:
                warnings.warn(
                    f"radiation damping during pulse Rtot[{int(pul_j) - 1}] has no "
                    "pulse_segments; the in-pulse trajectory follows the shortest "
                    "rotation path, which is exact only for net rotations below pi",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned.add(int(pul_j))
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
        if amp_j > 0:
            mat = rtot[int(pul_j) - 1]
            mlong = np.zeros(numpts, dtype=np.complex128)
            tmp = mvect.copy()
            mvect[0, :] = mat.R_00 * tmp[0, :] + mat.R_0m * tmp[1, :] + mat.R_0p * tmp[2, :] + mlong
            mvect[1, :] = mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
            mvect[2, :] = mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]
        else:
            del_w = del_w0 + grad_j * del_wg
            t_seconds = abs(float(tp_j)) * time_scale
            attenuation = np.exp(
                -(1.0 / 3.0)
                * gamma**2
                * gradient**2
                * diffusion_coefficient
                * t_seconds**3
            )
            _apply_free_precession_step(
                mvect,
                del_w,
                float(tp_j),
                T1n,
                T2n,
                mth,
                transverse_attenuation=float(attenuation),
            )

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
    min_chunk_size: int = 8192,
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
    min_chunk_size: int = 8192,
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
        else:
            _apply_free_precession_step(
                mvect,
                del_w,
                float(tp_j),
                T1n,
                T2n,
                mth,
            )

        if acq_j:
            macq[acq_cnt, :] = np.convolve(mvect[1, :], window, mode="same")
            acq_cnt += 1

    return macq
