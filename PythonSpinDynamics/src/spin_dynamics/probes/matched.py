"""Matched-probe transmit/receive models.

MATLAB reference folder:
    SpinDynamicsUpdated/Version_2/code/circuit_simulation/matched_probe
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import (
    calc_rot_axis_arba3,
    rf_matrix_elements,
    sim_spin_dynamics_asymp_mag3,
)


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _with_fields(obj: Mapping[str, Any] | Any, **updates: Any) -> Any:
    if isinstance(obj, Mapping):
        out = dict(obj)
        out.update(updates)
        return out
    out = dict(vars(obj))
    out.update(updates)
    return out


def _as_vector(value: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def matching_network_design2(
    L: float,
    Q: float,
    f0: float,
    R0: float,
) -> tuple[float, float]:
    """Design the two-capacitor matching network.

    Mirrors the objective in MATLAB `matching_network_design2.m`.

    The MATLAB routine minimizes the scalar input-match error for positive
    dimensionless capacitor parameters. Solving the same match equations
    analytically avoids the false Newton root that can appear at the higher
    imaging frequency.
    """

    w0 = 2 * np.pi * f0
    Rs = w0 * L / Q
    Ld = L * w0 / R0
    Rsd = Rs / R0
    coil_z = Rsd + 1j * Ld

    # With dimensionless c1=x and c2=y,
    # zin = A / (1 + i*x*A) + 1/(i*y).  The real-part match gives a
    # quadratic in x; y then cancels the remaining positive reactance.
    coeff = np.array([Rsd**2 + Ld**2, -2 * Ld, 1 - Rsd], dtype=np.float64)
    roots = np.roots(coeff)
    candidates: list[tuple[float, float, float]] = []
    for root in roots:
        if abs(np.imag(root)) > 1e-10:
            continue
        c1 = float(np.real(root))
        if c1 <= 0:
            continue
        zin_without_c2 = coil_z / (1 + 1j * c1 * coil_z)
        imag_part = float(np.imag(zin_without_c2))
        if imag_part <= 0:
            continue
        c2 = 1 / imag_part
        zin = zin_without_c2 + 1 / (1j * c2)
        candidates.append((abs(zin - 1), c1, c2))

    if candidates:
        _err, c1, c2 = min(candidates, key=lambda item: item[0])
        return float(c1 / (w0 * R0)), float(c2 / (w0 * R0))

    # Defensive fallback for unusual parameters where the direct match has no
    # positive-reactance root.
    x = np.array(
        [
            (w0 * R0) / (L * w0**2),
            ((w0 * R0) / (L * w0**2)) * np.sqrt(Rs / R0),
        ],
        dtype=np.float64,
    )

    def residual(v: np.ndarray) -> np.ndarray:
        c1, c2 = v
        s = 1j
        zin = (s * Ld + Rsd) / ((s * Ld + Rsd) * s * c1 + 1) + 1 / (s * c2)
        return np.array([np.real(zin) - 1, np.imag(zin)], dtype=np.float64)

    for _ in range(50):
        r = residual(x)
        if np.linalg.norm(r, ord=2) < 1e-14:
            break
        jac = np.zeros((2, 2), dtype=np.float64)
        for idx in range(2):
            step = 1e-6 * max(1.0, abs(x[idx]))
            xp = x.copy()
            xp[idx] += step
            xm = x.copy()
            xm[idx] = max(xm[idx] - step, 1e-12)
            jac[:, idx] = (residual(xp) - residual(xm)) / (xp[idx] - xm[idx])
        try:
            delta = np.linalg.solve(jac, r)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(jac, r, rcond=None)[0]
        current = np.linalg.norm(r, ord=2)
        scale = 1.0
        for _line in range(30):
            candidate = np.maximum(x - scale * delta, 1e-12)
            if np.linalg.norm(residual(candidate), ord=2) < current:
                x = candidate
                break
            scale *= 0.5
        else:
            break

    return float(x[0] / (w0 * R0)), float(x[1] / (w0 * R0))


def _rk4_step(state: np.ndarray, t: float, h: float, rhs: Any) -> np.ndarray:
    k1 = rhs(t, state)
    k2 = rhs(t + h / 2, state + h * k1 / 2)
    k3 = rhs(t + h / 2, state + h * k2 / 2)
    k4 = rhs(t + h, state + h * k3)
    return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _rk4_advance(state: np.ndarray, t: float, h: float, rhs: Any, max_step: float) -> np.ndarray:
    steps = max(1, int(np.ceil(abs(h) / max_step)))
    dt = h / steps
    out = state
    t_curr = t
    for _ in range(steps):
        out = _rk4_step(out, t_curr, dt, rhs)
        t_curr += dt
    return out


def _impulse_response(c1: float, c2: float, c3: float, Vs0: float, tvect: np.ndarray) -> np.ndarray:
    amat = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-c1, -c2, -c3]],
        dtype=np.complex128,
    )
    bvec = np.array([0.0, 0.0, Vs0], dtype=np.complex128)
    vals, vecs = np.linalg.eig(amat)
    coeff = np.linalg.solve(vecs, bvec)
    response = vecs @ (np.exp(vals[:, np.newaxis] * tvect[np.newaxis, :]) * coeff[:, np.newaxis])
    return np.real(response[0, :])


def find_coil_current(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find rotating-frame current in a tuned-and-matched probe.

    Mirrors MATLAB `find_coil_current.m`. The ODE is integrated on the same
    quantized RF grid with RK4 instead of MATLAB's adaptive `ode45`.
    """

    L = float(_field(sp, "L"))
    Q = float(_field(sp, "Q"))
    f0 = float(_field(sp, "f0"))
    Rs = float(_field(sp, "Rs"))
    C1 = float(_field(sp, "C1"))
    C2 = float(_field(sp, "C2"))
    fin = float(_field(sp, "fin"))

    n = C1 / C2
    Z0 = np.sqrt(L / C1)
    wp = 1 / np.sqrt(L * C1)
    wn = (2 * np.pi * fin) / wp
    Rc = (2 * np.pi * f0 * L) / Q
    c1 = n * Z0 / Rs
    c2 = (Rc / Rs) * (n + 1) + 1
    c3 = Rc / Z0 + (Z0 / Rs) * (n + 1)
    Vs0 = 2 * np.sqrt(Rc / Rs)

    tp = wp * _as_vector(_field(pp, "tp"))
    phi = _as_vector(_field(pp, "phi"))
    amp = _as_vector(_field(pp, "amp"))
    N = int(_field(pp, "N"))
    psi = float(_field(pp, "psi"))
    del_w = _as_vector(_field(sp, "del_w"))
    segment_fraction = 1.0
    if isinstance(pp, Mapping):
        segment_fraction = float(pp.get("segment_fraction", segment_fraction))
    elif hasattr(pp, "segment_fraction"):
        segment_fraction = float(getattr(pp, "segment_fraction"))
    if segment_fraction <= 0:
        raise ValueError("segment_fraction must be positive")

    ttot = float(np.sum(tp))
    delt = 2 * np.pi / (wn * N)
    ntot = int(np.floor(ttot / delt)) + 1
    tvec = np.linspace(0, ttot, ntot)
    h = tvec[1] - tvec[0]
    y_imp = _impulse_response(c1, c2, c3, Vs0, np.arange(ntot) * h)

    ysin = np.zeros((ntot, 3), dtype=np.float64)
    ycos = np.zeros((ntot, 3), dtype=np.float64)
    ysin_imp = np.zeros(ntot, dtype=np.float64)
    ycos_imp = np.zeros(ntot, dtype=np.float64)
    max_rk4_step = min(h, 0.05)

    time_el = 0.0
    ind_last = 0
    ind2 = 0
    ycos_state = np.zeros(3, dtype=np.float64)
    ysin_state = np.zeros(3, dtype=np.float64)

    for idx, (tp_i, phi_i, amp_i) in enumerate(zip(tp, phi, amp)):
        if idx > 0:
            ind_last = ind2
            ycos_state = ycos[ind_last, :].copy()
            ysin_state = ysin[ind_last, :].copy()
        ind = np.nonzero((tvec >= time_el) & (tvec <= time_el + tp_i))[0]
        ind1 = ind_last
        ind2 = int(ind[-1])

        def rhs_cos(t: float, y: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    y[1],
                    y[2],
                    -c3 * y[2]
                    - c2 * y[1]
                    - c1 * y[0]
                    - amp_i * Vs0 * wn * np.sin(wn * t + phi_i + psi),
                ],
                dtype=np.float64,
            )

        def rhs_sin(t: float, y: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    y[1],
                    y[2],
                    -c3 * y[2]
                    - c2 * y[1]
                    - c1 * y[0]
                    + amp_i * Vs0 * wn * np.cos(wn * t + phi_i + psi),
                ],
                dtype=np.float64,
            )

        ycos[ind1, :] = ycos_state
        ysin[ind1, :] = ysin_state
        for pos in range(ind1 + 1, ind2 + 1):
            ycos[pos, :] = _rk4_advance(
                ycos[pos - 1, :],
                tvec[pos - 1],
                h,
                rhs_cos,
                max_rk4_step,
            )
            ysin[pos, :] = _rk4_advance(
                ysin[pos - 1, :],
                tvec[pos - 1],
                h,
                rhs_sin,
                max_rk4_step,
            )

        ysin_imp[ind1:ntot] += amp_i * np.sin(wn * tvec[ind1] + phi_i + psi) * y_imp[: ntot - ind1]
        ysin_imp[ind2:ntot] -= amp_i * np.sin(wn * tvec[ind2] + phi_i + psi) * y_imp[: ntot - ind2]
        ycos_imp[ind1:ntot] += amp_i * np.cos(wn * tvec[ind1] + phi_i + psi) * y_imp[: ntot - ind1]
        ycos_imp[ind2:ntot] -= amp_i * np.cos(wn * tvec[ind2] + phi_i + psi) * y_imp[: ntot - ind2]

        time_el += tp_i

    y = ycos[:, 0] + 1j * ysin[:, 0] + ycos_imp + 1j * ysin_imp
    yr = y * np.exp(-1j * wn * tvec) * np.exp(-1j * psi)

    window = max(1, int(round(N * segment_fraction)))
    ntot2 = int(np.floor(ntot / window))
    tvec2 = np.zeros(ntot2, dtype=np.float64)
    yr2 = np.zeros(ntot2, dtype=np.complex128)
    for idx in range(ntot2):
        ind = slice(idx * window, (idx + 1) * window)
        tvec2[idx] = np.mean(tvec[ind])
        yr2[idx] = np.mean(yr[ind])

    wv = (2 * np.pi * f0 + (np.pi / (2 * float(_field(pp, "T_90")))) * del_w) / wp
    s = 1j * wv
    den = s**3 + c3 * s**2 + c2 * s + c1
    tf1 = s / den
    tf2 = -1j * s**3 / den

    return tvec2 / wp, yr2, tf1, tf2


def find_coil_current_wurst(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find matched-probe current for a frequency-swept WURST RF block.

    This ports the non-plotting core of MATLAB `find_coil_current_WURST.m`.
    Segment frequencies are supplied as angular offsets in `pp.freq`, added to
    the input carrier before solving the matched-network transient response.
    """

    L = float(_field(sp, "L"))
    Q = float(_field(sp, "Q"))
    f0 = float(_field(sp, "f0"))
    Rs = float(_field(sp, "Rs"))
    C1 = float(_field(sp, "C1"))
    C2 = float(_field(sp, "C2"))
    fin = float(_field(sp, "fin"))

    n = C1 / C2
    Z0 = np.sqrt(L / C1)
    wp = 1 / np.sqrt(L * C1)
    wR = (2 * np.pi * fin) / wp
    Rc = (2 * np.pi * f0 * L) / Q
    c1 = n * Z0 / Rs
    c2 = (Rc / Rs) * (n + 1) + 1
    c3 = Rc / Z0 + (Z0 / Rs) * (n + 1)
    Vs0_base = 2 * np.sqrt(Rc / Rs)

    tp = wp * _as_vector(_field(pp, "tp"))
    phi = _as_vector(_field(pp, "phi"))
    amp = _as_vector(_field(pp, "amp"))
    freq = _as_vector(_field(pp, "freq"))
    if not (tp.size == phi.size == amp.size == freq.size):
        raise ValueError("tp, phi, amp, and freq must have the same length")
    N = int(_field(pp, "N"))
    if N <= 0:
        raise ValueError("pp.N must be positive")
    psi = float(_field(pp, "psi"))
    del_w = _as_vector(_field(sp, "del_w"))
    wn = (2 * np.pi * fin + freq) / wp

    ttot = float(np.sum(tp))
    if ttot <= 0:
        raise ValueError("total pulse duration must be positive")
    delt = 2 * np.pi / (wR * N)
    ntot = int(np.floor(ttot / delt)) + 1
    if ntot < 2:
        raise ValueError("WURST pulse is too short for the RF quantization grid")
    tvec = np.linspace(0, ttot, ntot)
    h = tvec[1] - tvec[0]
    y_imp = _impulse_response(c1, c2, c3, Vs0_base, np.arange(ntot) * h)

    ysin = np.zeros((ntot, 3), dtype=np.float64)
    ycos = np.zeros((ntot, 3), dtype=np.float64)
    ysin_imp = np.zeros(ntot, dtype=np.float64)
    ycos_imp = np.zeros(ntot, dtype=np.float64)
    max_rk4_step = min(h, 0.05)

    time_el = 0.0
    ind_last = 0
    ind2 = 0
    ycos_state = np.zeros(3, dtype=np.float64)
    ysin_state = np.zeros(3, dtype=np.float64)

    for idx, (tp_i, phi_i, amp_i, wn_i) in enumerate(zip(tp, phi, amp, wn)):
        if idx > 0:
            ind_last = ind2
            ycos_state = ycos[ind_last, :].copy()
            ysin_state = ysin[ind_last, :].copy()
        ind = np.nonzero((tvec >= time_el) & (tvec <= time_el + tp_i))[0]
        if ind.size == 0:
            time_el += tp_i
            continue
        ind1 = ind_last
        ind2 = int(ind[-1])
        Vs0 = float(amp_i) * Vs0_base

        def rhs_cos(t: float, y: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    y[1],
                    y[2],
                    -c3 * y[2]
                    - c2 * y[1]
                    - c1 * y[0]
                    - Vs0 * wn_i * np.sin(wn_i * t + phi_i + psi),
                ],
                dtype=np.float64,
            )

        def rhs_sin(t: float, y: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    y[1],
                    y[2],
                    -c3 * y[2]
                    - c2 * y[1]
                    - c1 * y[0]
                    + Vs0 * wn_i * np.cos(wn_i * t + phi_i + psi),
                ],
                dtype=np.float64,
            )

        ycos[ind1, :] = ycos_state
        ysin[ind1, :] = ysin_state
        for pos in range(ind1 + 1, ind2 + 1):
            ycos[pos, :] = _rk4_advance(
                ycos[pos - 1, :],
                tvec[pos - 1],
                h,
                rhs_cos,
                max_rk4_step,
            )
            ysin[pos, :] = _rk4_advance(
                ysin[pos - 1, :],
                tvec[pos - 1],
                h,
                rhs_sin,
                max_rk4_step,
            )

        ysin_imp[ind1:ntot] += amp_i * np.sin(wn_i * tvec[ind1] + phi_i + psi) * y_imp[
            : ntot - ind1
        ]
        ysin_imp[ind2:ntot] -= amp_i * np.sin(wn_i * tvec[ind2] + phi_i + psi) * y_imp[
            : ntot - ind2
        ]
        ycos_imp[ind1:ntot] += amp_i * np.cos(wn_i * tvec[ind1] + phi_i + psi) * y_imp[
            : ntot - ind1
        ]
        ycos_imp[ind2:ntot] -= amp_i * np.cos(wn_i * tvec[ind2] + phi_i + psi) * y_imp[
            : ntot - ind2
        ]

        time_el += tp_i

    y = ycos[:, 0] + 1j * ysin[:, 0] + ycos_imp + 1j * ysin_imp
    yr = y * np.exp(-1j * wR * tvec) * np.exp(-1j * psi)

    window = max(1, N // 2)
    ntot2 = int(np.floor(ntot / window))
    tvec2 = np.zeros(ntot2, dtype=np.float64)
    yr2 = np.zeros(ntot2, dtype=np.complex128)
    for idx in range(ntot2):
        ind = slice(idx * window, (idx + 1) * window)
        tvec2[idx] = np.mean(tvec[ind])
        yr2[idx] = np.mean(yr[ind])

    wv = (2 * np.pi * f0 + (np.pi / (2 * float(_field(pp, "T_90")))) * del_w) / wp
    s = 1j * wv
    den = s**3 + c3 * s**2 + c2 * s + c1
    tf1 = s / den
    tf2 = -1j * s**3 / den

    return tvec2 / wp, yr2, tf1, tf2


def calc_rot_axis_matched_probe(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> np.ndarray:
    """Calculate matched-probe refocusing rotation axis."""

    T_90 = float(_field(pp, "T_90"))
    win = 2 * np.pi * float(_field(sp, "fin"))
    texc_tot = np.sum(_as_vector(_field(pp, "texc"))) + float(_field(pp, "tcorr"))
    pp_curr = _with_fields(
        pp,
        tp=_as_vector(_field(pp, "tref")),
        phi=_as_vector(_field(pp, "pref")),
        amp=_as_vector(_field(pp, "aref")),
        psi=float(_field(pp, "psi")) + np.mod(win * texc_tot, 2 * np.pi),
    )
    tvect, icr, _tf1, _tf2 = find_coil_current(sp, pp_curr)
    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    trefc = delt * np.ones(tvect.size)
    prefc = np.arctan2(np.imag(icr), np.real(icr))
    arefc = np.abs(icr)
    arefc[arefc < float(_field(pp, "amp_zero"))] = 0
    return calc_rot_axis_arba3(trefc, prefc, arefc, _as_vector(_field(sp, "del_w")))


def matched_probe_rx(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    macq: np.ndarray,
    tf1: np.ndarray,
    tf2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply matched-probe receiver filtering and estimate SNR."""

    k = float(_field(sp, "k"))
    T = float(_field(sp, "T"))
    L = float(_field(sp, "L"))
    f0 = float(_field(sp, "f0"))
    Q = float(_field(sp, "Q"))
    Rc = (2 * np.pi * f0 * L) / Q
    del_w = _as_vector(_field(sp, "del_w"))
    macq = np.asarray(macq, dtype=np.complex128).reshape(-1)
    tf1 = np.asarray(tf1, dtype=np.complex128).reshape(-1)
    tf2 = np.asarray(tf2, dtype=np.complex128).reshape(-1)
    w1_max = (np.pi / 2) / float(_field(pp, "T_90"))
    f = (2 * np.pi * f0 + del_w * w1_max) / (2 * np.pi)

    mrx = macq * tf2
    # Coil thermal noise (open-circuit 4*k*T*Rc) referred through the loaded
    # matched transfer tf1, plus the amplifier excess noise k*T*Rin*(F-1). The
    # latter has no factor of 4 because it is on the matched available-power
    # basis: tf1's matched load halves the EMF (1/2 voltage, 1/4 power), so the
    # coil term also reduces to k*T*Rin at the input node. Both are consistent
    # with the signal mrx, which is referred through the same network.
    vni2 = 4 * k * T * Rc * np.abs(tf1) ** 2
    Fn = 10 ** (float(_field(sp, "NF")) / 10)
    vn2 = k * T * float(_field(sp, "Rin")) * (Fn - 1) * np.ones(f.size)
    pnoise = vni2 + vn2

    if int(_field(sp, "mf_type")) == 1:
        mf = np.conj(mrx)
    elif int(_field(sp, "mf_type")) == 2:
        mf = np.conj(mrx) / pnoise
    else:
        raise ValueError("matched probe mf_type must be 1 or 2")

    mf = mf / np.sqrt(trapezoid(np.abs(mf) ** 2, del_w))
    vsig = trapezoid(mrx * mf, del_w)
    vnoise = np.sqrt(trapezoid(pnoise * np.abs(mf) ** 2, f))
    snr = float(np.real(vsig) / vnoise) / 1e8
    echo, tvect = calc_time_domain_echo(mrx, del_w)
    return mrx, echo, tvect, snr


def calc_masy_matched_probe_orig(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate matched-probe CPMG asymptotic and received spectra."""

    c1, c2 = matching_network_design2(
        float(_field(sp, "L")),
        float(_field(sp, "Q")),
        float(_field(sp, "f0")),
        float(_field(sp, "Rs")),
    )
    sp_match = _with_fields(sp, C1=c1, C2=c2)
    T_90 = float(_field(pp, "T_90"))
    tacq = (np.pi / 2) * _as_vector(_field(pp, "tacq")) / T_90
    neff = calc_rot_axis_matched_probe(sp_match, pp)

    pp_curr = _with_fields(
        pp,
        tp=np.concatenate([_as_vector(_field(pp, "texc")), [float(_field(pp, "trd"))]]),
        phi=np.concatenate([_as_vector(_field(pp, "pexc")), [0.0]]),
        amp=np.concatenate([_as_vector(_field(pp, "aexc")), [0.0]]),
    )
    tvect, icr, tf1, tf2 = find_coil_current(sp_match, pp_curr)
    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    texc = delt * np.ones(tvect.size)
    tdeln = (np.pi / 2) * float(_field(pp, "trd")) / T_90
    tcorrn = (np.pi / 2) * float(_field(pp, "tcorr")) / T_90
    pexc = np.arctan2(np.imag(icr), np.real(icr))
    aexc = np.abs(icr)
    aexc[aexc < float(_field(pp, "amp_zero"))] = 0
    texc = np.concatenate([texc, [-tdeln, tcorrn]])
    pexc = np.concatenate([pexc, [0.0, 0.0]])
    aexc = np.concatenate([aexc, [0.0, 0.0]])
    del_w = _as_vector(_field(sp_match, "del_w"))

    masy1 = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff, del_w, float(tacq[0]))
    masy2 = sim_spin_dynamics_asymp_mag3(texc, pexc + np.pi, aexc, neff, del_w, float(tacq[0]))
    masy = (masy1 - masy2) / 2
    mrx, _echo, _tvect, snr = matched_probe_rx(sp_match, pp, masy, tf1, tf2)
    return mrx, masy, snr


def _sim_spin_dynamics_asymp_nut(
    pulse_length: np.ndarray,
    phi: np.ndarray,
    amp: np.ndarray,
    del_w: np.ndarray,
) -> np.ndarray:
    """Propagate excitation/nutation dynamics and return final z magnetization."""

    tp = _as_vector(pulse_length)
    phi = _as_vector(phi)
    amp = _as_vector(amp)
    del_w = _as_vector(del_w)
    if not (tp.size == phi.size == amp.size):
        raise ValueError("pulse_length, phi, and amp must have the same length")

    mvect = np.zeros((3, del_w.size), dtype=np.complex128)
    mvect[0, :] = 1.0
    for tp_j, phi_j, amp_j in zip(tp, phi, amp):
        mat = rf_matrix_elements(del_w, float(amp_j), float(tp_j), float(phi_j))
        tmp = mvect.copy()
        mvect[0, :] = mat.R_00 * tmp[0, :] + mat.R_0m * tmp[1, :] + mat.R_0p * tmp[2, :]
        mvect[1, :] = mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
        mvect[2, :] = mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]

    return mvect[0, :]


def calc_masy_matched_nut(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate matched-probe excitation/nutation z magnetization.

    Mirrors MATLAB `calc_masy/calc_masy_matched_nut.m`, with plotting removed.
    """

    c1, c2 = matching_network_design2(
        float(_field(sp, "L")),
        float(_field(sp, "Q")),
        float(_field(sp, "f0")),
        float(_field(sp, "Rs")),
    )
    sp_match = _with_fields(sp, C1=c1, C2=c2)
    t_90 = float(_field(pp, "T_90"))
    amp_zero = float(_field(pp, "amp_zero"))
    pp_curr = _with_fields(
        pp,
        tp=np.concatenate([_as_vector(_field(pp, "texc")), [2 * t_90]]),
        phi=np.concatenate([_as_vector(_field(pp, "pexc")), [0.0]]),
        amp=np.concatenate([_as_vector(_field(pp, "aexc")), [0.0]]),
    )
    tvect, icr, _tf1, _tf2 = find_coil_current(sp_match, pp_curr)
    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / t_90
    texc = delt * np.ones(tvect.size, dtype=np.float64)
    pexc = np.arctan2(np.imag(icr), np.real(icr))
    aexc = np.abs(icr)
    aexc[aexc < amp_zero] = 0
    mz = _sim_spin_dynamics_asymp_nut(texc, pexc, aexc, _field(sp_match, "del_w"))
    return mz, tvect
