"""JAX-differentiable pulse-optimization objectives (Phase 3).

The headline of the acceleration plan: replace the optimizer's
finite-difference gradients (N+1 forward simulations per gradient step) with a
single reverse-mode pass via ``jax.value_and_grad``.

This module ports the ideal-probe v0crit refocusing objective
(`spin_dynamics.optimization.refocusing.evaluate_ideal_v0crit_refocusing_pulse`)
to JAX: the effective rotation-axis composition (`calc_rot_axis_arba4`), the
critical-velocity term (`calc_v0crit`), and the windowed score. The pulse
*structure* (which segments are free precession vs RF, all amplitudes/times) is
host-known and baked in; only the segment phases — the optimization variable —
are traced, so the whole objective is differentiable with respect to them.

x64 is enabled so the score matches the NumPy reference. Requires the optional
``jax`` extra. See ``docs/performance.md``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from spin_dynamics.parameters import set_params_ideal

try:  # pragma: no cover - exercised by environment, not logic
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    JAX_AVAILABLE = False


_ACOS_EPS = 1e-12


def _trapezoid(y, x):
    return jnp.sum((y[1:] + y[:-1]) * 0.5 * (x[1:] - x[:-1]))


def _interp_extrap(x, xp, fp):
    """Linear interpolation with MATLAB-style end extrapolation (jnp)."""

    out = jnp.interp(x, xp, fp)
    slope_l = (fp[1] - fp[0]) / (xp[1] - xp[0])
    slope_r = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    out = jnp.where(x < xp[0], fp[0] + slope_l * (x - xp[0]), out)
    out = jnp.where(x > xp[-1], fp[-1] + slope_r * (x - xp[-1]), out)
    return out


def _rot_axis_arba4(tp, phi, amp_host, del_w):
    """JAX port of ``rotations.calc_rot_axis_arba4`` (returns axis + angle).

    ``amp_host`` is a concrete NumPy array (the pulse/free structure is fixed),
    so the per-segment branch is taken on the host. ``phi`` is the traced
    optimization variable.
    """

    if amp_host[0] > 0:
        w1 = amp_host[0]
        omega = jnp.sqrt(w1**2 + del_w**2)
        alpha = omega * tp[0]
        sn = jnp.sin(alpha / 2)
        n0 = sn * w1 * jnp.cos(phi[0]) / omega
        n1 = sn * w1 * jnp.sin(phi[0]) / omega
        n2 = sn * del_w / omega
    else:
        alpha = del_w * tp[0]
        sn = jnp.sin(alpha / 2)
        n0 = jnp.zeros_like(del_w)
        n1 = jnp.zeros_like(del_w)
        n2 = sn
    cs = jnp.cos(alpha / 2)

    for j in range(1, amp_host.shape[0]):
        if amp_host[j] > 0:
            w1 = amp_host[j]
            omega = jnp.sqrt(w1**2 + del_w**2)
            alpha_curr = omega * tp[j]
            c0 = w1 * jnp.cos(phi[j]) / omega
            c1 = w1 * jnp.sin(phi[j]) / omega
            c2 = del_w / omega
            crs0 = n1 * c2 - n2 * c1
            crs1 = n2 * c0 - n0 * c2
            crs2 = n0 * c1 - n1 * c0
            sn_c = jnp.sin(alpha_curr / 2)
            cs_c = jnp.cos(alpha_curr / 2)
            t0 = cs_c * n0 + sn_c * (cs * c0 - crs0)
            t1 = cs_c * n1 + sn_c * (cs * c1 - crs1)
            t2 = cs_c * n2 + sn_c * (cs * c2 - crs2)
            cs = cs * cs_c - sn_c * (n0 * c0 + n1 * c1 + n2 * c2)
            n0, n1, n2 = t0, t1, t2
        else:
            alpha_curr = del_w * tp[j]
            sn_c = jnp.sin(alpha_curr / 2)
            cs_c = jnp.cos(alpha_curr / 2)
            t0 = cs_c * n0 - sn_c * n1
            t1 = cs_c * n1 + sn_c * n0
            t2 = cs_c * n2 + cs * sn_c
            cs = cs * cs_c - sn_c * n2
            n0, n1, n2 = t0, t1, t2

    cs = jnp.clip(cs, -1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS)
    alpha = 2.0 * jnp.arccos(cs)
    sn = jnp.sin(alpha / 2)
    sn = jnp.where(sn == 0, 1e-12, sn)
    return n0 / sn, n1 / sn, n2 / sn, alpha


def _v0crit(del_w, n0, n1, n2, alpha):
    step = del_w[1:] - del_w[:-1]
    center = 0.5 * (del_w[:-1] + del_w[1:])
    alpha_c = 0.5 * (alpha[:-1] + alpha[1:])
    a0, a1, a2 = n0[:-1], n1[:-1], n2[:-1]
    b0, b1, b2 = n0[1:], n1[1:], n2[1:]
    c0 = a1 * b2 - a2 * b1
    c1 = a2 * b0 - a0 * b2
    c2 = a0 * b1 - a1 * b0
    ncross = jnp.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
    v0crit_center = alpha_c * step / ncross
    return _interp_extrap(del_w, center, v0crit_center)


@lru_cache(maxsize=64)
def make_ideal_v0crit_objective(
    num_segments: int,
    *,
    segment_fraction: float = 0.1,
    free_precession_t180: float = 1.5,
    numpts: int = 101,
    maxoffs: float | None = None,
    acquisition_time_normalized: float | None = None,
    v0crit_weight: float = 100.0,
):
    """Build a ``value_and_grad`` callable for the ideal v0crit objective.

    Returns ``vg(phases: np.ndarray) -> (score: float, grad: np.ndarray)`` using
    reverse-mode autodiff. The returned callable is jit-compiled and baked with
    all non-phase parameters. Cached by configuration so repeated optimizer runs
    (e.g. multistart) reuse the same compiled program instead of recompiling.
    """

    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX-differentiable objectives require the optional 'jax' extra. "
            "Install it with `python -m pip install -e .[jax]` (or `.[perf]`)."
        )

    sp, pp = set_params_ideal(numpts=numpts)
    mo = float(sp.maxoffs if maxoffs is None else maxoffs)
    del_w = np.linspace(-mo, mo, int(numpts))
    t180 = np.pi
    tfp = float(free_precession_t180) * t180
    seg = float(segment_fraction) * t180
    tp = np.concatenate([[tfp], seg * np.ones(num_segments), [tfp]])
    amp_host = np.concatenate([[0.0], np.ones(num_segments), [0.0]])
    if acquisition_time_normalized is None:
        w1n = (np.pi / 2) / pp.T_90
        tacq = float(w1n * np.ravel(pp.tacq)[0])
    else:
        tacq = float(acquisition_time_normalized)

    del_w_j = jnp.asarray(del_w, dtype=jnp.float64)
    tp_j = jnp.asarray(tp, dtype=jnp.float64)
    window = jnp.sinc(del_w_j * tacq / (2 * np.pi))
    window = window / jnp.sum(window)
    weight = float(v0crit_weight)

    def score(phases):
        phi = jnp.concatenate([jnp.zeros(1, dtype=phases.dtype), phases, jnp.zeros(1, dtype=phases.dtype)])
        n0, n1, n2, alpha = _rot_axis_arba4(tp_j, phi, amp_host, del_w_j)
        v0 = _v0crit(del_w_j, n0, n1, n2, alpha)
        transverse = n0 + 1j * n1
        masy = jnp.convolve(transverse, window, mode="same")
        axis_rms = jnp.real(_trapezoid(jnp.abs(masy) ** 2, del_w_j))
        denom = jnp.real(_trapezoid(1.0 / v0, del_w_j))
        v0crit_average = weight / denom
        return axis_rms + v0crit_average

    _vg = jax.jit(jax.value_and_grad(score))

    def value_and_grad(phases: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = _vg(jnp.asarray(phases, dtype=jnp.float64))
        return float(value), np.asarray(grad)

    return value_and_grad
