"""Optional JAX kernels for moving-walker simulations."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on optional local dependency
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    lax = None


def devices() -> list[str]:
    """Return visible JAX device descriptions."""

    if not JAX_AVAILABLE:
        return []
    return [str(device) for device in jax.devices()]


def cpmg_voxel_walkers_core(
    positions0: np.ndarray,
    weights: np.ndarray,
    diffusion: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    b0_map: np.ndarray,
    pore_mask: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    dt_steps: np.ndarray,
    rf_amp_steps: np.ndarray,
    rf_phase_steps: np.ndarray,
    acquire_steps: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the voxel-walker CPMG kernel through JAX ``lax.scan``."""

    if not JAX_AVAILABLE:
        raise ImportError("jax is not installed")
    full_signal, positions = _cpmg_voxel_walkers_jit(
        jnp.asarray(positions0),
        jnp.asarray(weights),
        jnp.asarray(diffusion),
        jnp.asarray(t1),
        jnp.asarray(t2),
        jnp.asarray(b0_map),
        jnp.asarray(pore_mask),
        jnp.asarray(axis0),
        jnp.asarray(axis1),
        jnp.asarray(axis2),
        jnp.asarray(dt_steps),
        jnp.asarray(rf_amp_steps),
        jnp.asarray(rf_phase_steps),
        jnp.asarray(normals),
    )
    signal = full_signal[np.asarray(acquire_steps, dtype=bool)]
    return np.asarray(signal), np.asarray(positions)


def cpmg_voxel_walkers_core_prng(
    positions0: np.ndarray,
    weights: np.ndarray,
    diffusion: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    b0_map: np.ndarray,
    pore_mask: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    dt_steps: np.ndarray,
    rf_amp_steps: np.ndarray,
    rf_phase_steps: np.ndarray,
    acquire_steps: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the voxel-walker CPMG kernel with JAX-generated Brownian increments."""

    if not JAX_AVAILABLE:
        raise ImportError("jax is not installed")
    full_signal, positions = _cpmg_voxel_walkers_prng_jit(
        jnp.asarray(positions0),
        jnp.asarray(weights),
        jnp.asarray(diffusion),
        jnp.asarray(t1),
        jnp.asarray(t2),
        jnp.asarray(b0_map),
        jnp.asarray(pore_mask),
        jnp.asarray(axis0),
        jnp.asarray(axis1),
        jnp.asarray(axis2),
        jnp.asarray(dt_steps),
        jnp.asarray(rf_amp_steps),
        jnp.asarray(rf_phase_steps),
        jax.random.PRNGKey(int(seed)),
    )
    signal = full_signal[np.asarray(acquire_steps, dtype=bool)]
    return np.asarray(signal), np.asarray(positions)


@jax.jit if JAX_AVAILABLE else (lambda f: f)
def _cpmg_voxel_walkers_jit(
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
    normals,
):
    nx = axis0.shape[0]
    ny = axis1.shape[0]
    nz = axis2.shape[0]
    lower = jnp.array([axis0[0], axis1[0], axis2[0]])
    upper = jnp.array([axis0[-1], axis1[-1], axis2[-1]])
    spacing = (upper - lower) / jnp.array([nx - 1, ny - 1, nz - 1])

    m0 = jnp.ones(weights.shape[0], dtype=jnp.complex128)
    mm = jnp.zeros(weights.shape[0], dtype=jnp.complex128)
    mp = jnp.zeros(weights.shape[0], dtype=jnp.complex128)

    def step(carry, inputs):
        positions, m0, mm, mp = carry
        dt, rf_amp, rf_phase, normal = inputs
        sigma = jnp.sqrt(2.0 * diffusion * dt)
        proposed = _reflect_positions(
            positions + sigma[:, None] * normal,
            lower,
            upper,
        )
        nearest = jnp.clip(
            jnp.floor((proposed - lower) / spacing + 0.5).astype(jnp.int32),
            0,
            jnp.array([nx - 1, ny - 1, nz - 1]),
        )
        in_pore = pore_mask[nearest[:, 0], nearest[:, 1], nearest[:, 2]]
        positions = jnp.where(in_pore[:, None], proposed, positions)
        off = _trilinear_sample(b0_map, positions, lower, spacing)

        free_m0, free_mm, free_mp = _free_update(m0, mm, mp, off, t1, t2, dt)
        rf_m0, rf_mm, rf_mp = _rf_update(m0, mm, mp, off, rf_amp, dt, rf_phase)
        use_rf = rf_amp != 0.0
        m0 = jnp.where(use_rf, rf_m0, free_m0)
        mm = jnp.where(use_rf, rf_mm, free_mm)
        mp = jnp.where(use_rf, rf_mp, free_mp)
        signal = jnp.sum(weights * mm)
        return (positions, m0, mm, mp), signal

    (positions, _m0, _mm, _mp), full_signal = lax.scan(
        step,
        (positions0, m0, mm, mp),
        (dt_steps, rf_amp_steps, rf_phase_steps, normals),
    )
    return full_signal, positions


@jax.jit if JAX_AVAILABLE else (lambda f: f)
def _cpmg_voxel_walkers_prng_jit(
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
    key0,
):
    nx = axis0.shape[0]
    ny = axis1.shape[0]
    nz = axis2.shape[0]
    lower = jnp.array([axis0[0], axis1[0], axis2[0]])
    upper = jnp.array([axis0[-1], axis1[-1], axis2[-1]])
    spacing = (upper - lower) / jnp.array([nx - 1, ny - 1, nz - 1])

    m0 = jnp.ones(weights.shape[0], dtype=jnp.complex128)
    mm = jnp.zeros(weights.shape[0], dtype=jnp.complex128)
    mp = jnp.zeros(weights.shape[0], dtype=jnp.complex128)

    def step(carry, inputs):
        positions, m0, mm, mp, key = carry
        dt, rf_amp, rf_phase = inputs
        key, normal_key = jax.random.split(key)
        normal = jax.random.normal(
            normal_key,
            positions.shape,
            dtype=positions.dtype,
        )
        positions, m0, mm, mp, signal = _voxel_step(
            positions,
            m0,
            mm,
            mp,
            weights,
            diffusion,
            t1,
            t2,
            b0_map,
            pore_mask,
            lower,
            upper,
            spacing,
            dt,
            rf_amp,
            rf_phase,
            normal,
        )
        return (positions, m0, mm, mp, key), signal

    (positions, _m0, _mm, _mp, _key), full_signal = lax.scan(
        step,
        (positions0, m0, mm, mp, key0),
        (dt_steps, rf_amp_steps, rf_phase_steps),
    )
    return full_signal, positions


def _voxel_step(
    positions,
    m0,
    mm,
    mp,
    weights,
    diffusion,
    t1,
    t2,
    b0_map,
    pore_mask,
    lower,
    upper,
    spacing,
    dt,
    rf_amp,
    rf_phase,
    normal,
):
    shape_max = jnp.array(pore_mask.shape) - 1
    sigma = jnp.sqrt(2.0 * diffusion * dt)
    proposed = _reflect_positions(
        positions + sigma[:, None] * normal,
        lower,
        upper,
    )
    nearest = jnp.clip(
        jnp.floor((proposed - lower) / spacing + 0.5).astype(jnp.int32),
        0,
        shape_max,
    )
    in_pore = pore_mask[nearest[:, 0], nearest[:, 1], nearest[:, 2]]
    positions = jnp.where(in_pore[:, None], proposed, positions)
    off = _trilinear_sample(b0_map, positions, lower, spacing)

    free_m0, free_mm, free_mp = _free_update(m0, mm, mp, off, t1, t2, dt)
    rf_m0, rf_mm, rf_mp = _rf_update(m0, mm, mp, off, rf_amp, dt, rf_phase)
    use_rf = rf_amp != 0.0
    m0 = jnp.where(use_rf, rf_m0, free_m0)
    mm = jnp.where(use_rf, rf_mm, free_mm)
    mp = jnp.where(use_rf, rf_mp, free_mp)
    signal = jnp.sum(weights * mm)
    return positions, m0, mm, mp, signal


def _reflect_positions(positions, lower, upper):
    width = upper - lower
    folded = jnp.mod(positions - lower, 2.0 * width)
    return lower + jnp.where(folded <= width, folded, 2.0 * width - folded)


def _trilinear_sample(values, positions, lower, spacing):
    shape = jnp.array(values.shape)
    u = (positions - lower) / spacing
    idx = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, shape - 2)
    frac = jnp.clip(u - idx, 0.0, 1.0)
    out = jnp.zeros(positions.shape[0], dtype=jnp.float64)
    for cx in range(2):
        wx = jnp.where(cx == 1, frac[:, 0], 1.0 - frac[:, 0])
        for cy in range(2):
            wy = jnp.where(cy == 1, frac[:, 1], 1.0 - frac[:, 1])
            for cz in range(2):
                wz = jnp.where(cz == 1, frac[:, 2], 1.0 - frac[:, 2])
                out = out + wx * wy * wz * values[
                    idx[:, 0] + cx,
                    idx[:, 1] + cy,
                    idx[:, 2] + cz,
                ]
    return out


def _free_update(m0, mm, mp, off, t1, t2, dt):
    e1 = jnp.exp(-dt / t1)
    e2 = jnp.exp(-dt / t2)
    tr = e2 * jnp.exp(1j * off * dt)
    return e1 * m0 + (1.0 - e1), jnp.conj(tr) * mm, tr * mp


def _rf_update(m0, mm, mp, off, w1, dt, phi):
    omega = jnp.sqrt(w1 * w1 + off * off)
    omega_safe = jnp.where(omega == 0.0, 1.0, omega)
    dw = off / omega_safe
    w1n = w1 / omega_safe
    sn = jnp.sin(omega * dt)
    cs = jnp.cos(omega * dt)
    ph = jnp.exp(1j * phi)

    r00 = dw * dw + w1n * w1n * cs
    r0p = 0.5 * w1n * (dw * (1.0 - cs) - 1j * sn) / ph
    rp0 = w1n * (dw * (1.0 - cs) - 1j * sn) * ph
    rpp = 0.5 * (w1n * w1n + (1.0 + dw * dw) * cs) + 1j * dw * sn
    rpm = 0.5 * w1n * w1n * (1.0 - cs) * ph * ph

    out0 = r00 * m0 + jnp.conj(r0p) * mm + r0p * mp
    outm = jnp.conj(rp0) * m0 + jnp.conj(rpp) * mm + jnp.conj(rpm) * mp
    outp = rp0 * m0 + rpm * mm + rpp * mp
    return out0, outm, outp
