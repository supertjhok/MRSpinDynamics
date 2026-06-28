"""Optional Numba kernels for moving-walker simulations."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on optional local dependency
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def _decorate(func):
            return func

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _decorate

    prange = range  # type: ignore[assignment]


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def cpmg_voxel_walkers_core(
    positions0,
    weights,
    diffusion,
    t1,
    t2,
    b0_map,
    pore_mask,
    axis0,
    axis1,
    axis2,
    dt_steps,
    rf_amp_steps,
    rf_phase_steps,
    acquire_steps,
    normals,
):
    """Run a CPMG-like walker sequence on a regular 3D voxel pore map.

    This specialized kernel fuses motion, solid-voxel rejection, B0 sampling,
    RF/free-precession updates, and receive summation for the porous-rock
    example. It intentionally avoids the generic callback/interpolation stack.
    """

    nsteps = dt_steps.shape[0]
    nparticles = positions0.shape[0]
    nx = axis0.shape[0]
    ny = axis1.shape[0]
    nz = axis2.shape[0]
    n_acq = 0
    for s in range(nsteps):
        if acquire_steps[s] != 0:
            n_acq += 1

    x0 = axis0[0]
    y0 = axis1[0]
    z0 = axis2[0]
    dx = (axis0[nx - 1] - axis0[0]) / (nx - 1)
    dy = (axis1[ny - 1] - axis1[0]) / (ny - 1)
    dz = (axis2[nz - 1] - axis2[0]) / (nz - 1)
    x1 = axis0[nx - 1]
    y1 = axis1[ny - 1]
    z1 = axis2[nz - 1]

    positions = positions0.copy()
    m0 = np.ones(nparticles, dtype=np.complex128)
    mm = np.zeros(nparticles, dtype=np.complex128)
    mp = np.zeros(nparticles, dtype=np.complex128)
    signal = np.zeros(n_acq, dtype=np.complex128)
    acq_index = 0

    for s in range(nsteps):
        dt = dt_steps[s]
        rf_amp = rf_amp_steps[s]
        rf_phase = rf_phase_steps[s]

        for k in prange(nparticles):
            old_x = positions[k, 0]
            old_y = positions[k, 1]
            old_z = positions[k, 2]

            sigma = np.sqrt(2.0 * diffusion[k] * dt)
            x = _reflect_coordinate(old_x + sigma * normals[s, k, 0], x0, x1)
            y = _reflect_coordinate(old_y + sigma * normals[s, k, 1], y0, y1)
            z = _reflect_coordinate(old_z + sigma * normals[s, k, 2], z0, z1)

            ixn = _nearest_index(x, x0, dx, nx)
            iyn = _nearest_index(y, y0, dy, ny)
            izn = _nearest_index(z, z0, dz, nz)
            if not pore_mask[ixn, iyn, izn]:
                x = old_x
                y = old_y
                z = old_z

            positions[k, 0] = x
            positions[k, 1] = y
            positions[k, 2] = z

            off = _trilinear_sample(b0_map, x, y, z, x0, y0, z0, dx, dy, dz)
            if rf_amp == 0.0:
                e1 = np.exp(-dt / t1[k])
                e2 = np.exp(-dt / t2[k])
                ph = off * dt
                tr = e2 * (np.cos(ph) + 1j * np.sin(ph))
                m0[k] = e1 * m0[k] + (1.0 - e1)
                mm[k] = np.conjugate(tr) * mm[k]
                mp[k] = tr * mp[k]
            else:
                _apply_rf_scalar(m0, mm, mp, k, off, rf_amp, dt, rf_phase)

        if acquire_steps[s] != 0:
            total = 0.0 + 0.0j
            for k in range(nparticles):
                total += weights[k] * mm[k]
            signal[acq_index] = total
            acq_index += 1

    return signal, positions


@njit(cache=True, nogil=True, fastmath=False)
def _reflect_coordinate(value, lower, upper):
    if upper == lower:
        return lower
    width = upper - lower
    folded = (value - lower) % (2.0 * width)
    if folded <= width:
        return lower + folded
    return lower + 2.0 * width - folded


@njit(cache=True, nogil=True, fastmath=False)
def _nearest_index(value, lower, spacing, size):
    idx = int(np.floor((value - lower) / spacing + 0.5))
    if idx < 0:
        return 0
    if idx >= size:
        return size - 1
    return idx


@njit(cache=True, nogil=True, fastmath=False)
def _lower_index_and_fraction(value, lower, spacing, size):
    u = (value - lower) / spacing
    idx = int(np.floor(u))
    if idx < 0:
        return 0, 0.0
    if idx >= size - 1:
        return size - 2, 1.0
    return idx, u - idx


@njit(cache=True, nogil=True, fastmath=False)
def _trilinear_sample(values, x, y, z, x0, y0, z0, dx, dy, dz):
    nx = values.shape[0]
    ny = values.shape[1]
    nz = values.shape[2]
    ix, fx = _lower_index_and_fraction(x, x0, dx, nx)
    iy, fy = _lower_index_and_fraction(y, y0, dy, ny)
    iz, fz = _lower_index_and_fraction(z, z0, dz, nz)
    out = 0.0
    for cx in range(2):
        wx = fx if cx == 1 else 1.0 - fx
        for cy in range(2):
            wy = fy if cy == 1 else 1.0 - fy
            for cz in range(2):
                wz = fz if cz == 1 else 1.0 - fz
                out += wx * wy * wz * values[ix + cx, iy + cy, iz + cz]
    return out


@njit(cache=True, nogil=True, fastmath=False)
def _apply_rf_scalar(m0, mm, mp, k, off, w1, dt, phi):
    omega = np.sqrt(w1 * w1 + off * off)
    if omega == 0.0:
        return
    dw = off / omega
    w1n = w1 / omega
    sn = np.sin(omega * dt)
    cs = np.cos(omega * dt)
    ph = np.cos(phi) + 1j * np.sin(phi)

    r00 = dw * dw + w1n * w1n * cs
    r0p = 0.5 * w1n * (dw * (1.0 - cs) - 1j * sn) / ph
    rp0 = w1n * (dw * (1.0 - cs) - 1j * sn) * ph
    rpp = 0.5 * (w1n * w1n + (1.0 + dw * dw) * cs) + 1j * dw * sn
    rpm = 0.5 * w1n * w1n * (1.0 - cs) * ph * ph

    a0 = m0[k]
    am = mm[k]
    ap = mp[k]
    m0[k] = r00 * a0 + np.conjugate(r0p) * am + r0p * ap
    mm[k] = np.conjugate(rp0) * a0 + np.conjugate(rpp) * am + np.conjugate(rpm) * ap
    mp[k] = rp0 * a0 + rpm * am + rpp * ap
