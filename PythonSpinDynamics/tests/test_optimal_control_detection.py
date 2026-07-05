"""Tests for the detector-aware GRAPE objective (PR-4).

Covers the field-referred detected-SNR helpers and the objective factory: the
flat-noise reduction to ``detected_echo_snr``, the detector noise grid, autodiff
vs finite difference, that the detector's noise shape actually enters the score,
and the per-RF-power efficiency mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="requires the optional 'jax' extra")

if JAX_AVAILABLE:
    import jax.numpy as jnp

    from spin_dynamics.coupling.systems import coupled_spin_system
    from spin_dynamics.detection import OPMMagnetometer, SQUIDMagnetometer
    from spin_dynamics.optimal_control.detection_objective import (
        detected_field_snr_jax,
        detector_noise_grid,
        make_detected_snr_objective,
    )
    from spin_dynamics.optimal_control.diffusion import detected_echo_snr
    from spin_dynamics.optimal_control.hamiltonians import coupled_spin_control_model

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_I_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)


def _ensemble(offsets, b1s):
    base = coupled_spin_control_model(coupled_spin_system([0.0], [[0.0]]))
    hx0, hy0 = base.h_x, base.h_y
    drift_b, hx_b, hy_b, w, offs = [], [], [], [], []
    for off in offsets:
        drift = coupled_spin_control_model(coupled_spin_system([off], [[0.0]])).h_drift
        for b1 in b1s:
            drift_b.append(drift)
            hx_b.append(b1 * hx0)
            hy_b.append(b1 * hy0)
            w.append(1.0)
            offs.append(off)
    return drift_b, hx_b, hy_b, np.array(w), np.array(offs)


def _common(n=8):
    drift_b, hx_b, hy_b, w, offs = _ensemble([-2000.0, 0.0, 2000.0], [0.9, 1.1])
    # acquisition_dt must RESOLVE the offset band (bins finer than the offsets),
    # else all offsets pile into the DC bin and the detector shape cannot act.
    return dict(
        drift_batch=drift_b, hx_batch=hx_b, hy_batch=hy_b, psi0=_PSI_UP,
        detection_operator=_I_PLUS, weights=w, offsets_hz=offs, dt=5e-6,
        n_segments=n, amplitude_template=np.full(n, 2000.0),
        acquisition_points=41, acquisition_dt=1.0 / (4.0 * 2000.0), optimize_amplitude=True,
    )


def test_flat_grid_reduces_to_detected_echo_snr():
    rng = np.random.default_rng(0)
    n_acq, dt_acq, sigma = 33, 2e-6, 0.7
    sig = rng.standard_normal(n_acq) + 1j * rng.standard_normal(n_acq)
    flat = np.full(n_acq, sigma**2)
    a = float(detected_field_snr_jax(jnp.asarray(sig), dt_acq, jnp.asarray(flat)))
    b = float(detected_echo_snr(jnp.asarray(sig), dt_acq, noise=sigma))
    assert a == pytest.approx(b, rel=1e-10)


def test_detector_noise_grid_squid_is_flat():
    grid = detector_noise_grid(SQUIDMagnetometer.berkeley_2007(), 33, 2e-6)
    assert np.allclose(grid, grid[0])
    assert np.all(grid > 0)


def test_detector_noise_grid_serf_rises_off_dc():
    serf = OPMMagnetometer.serf(field_noise_T_per_rtHz=0.16e-15, bandwidth_hz=800.0)
    grid = detector_noise_grid(serf, 64, 2e-6)  # fftfreq -> index 0 is DC
    assert grid[0] == pytest.approx((0.16e-15) ** 2)
    assert grid[1] > grid[0]  # neighbouring bin is noisier (rolled off)


def test_objective_gradient_matches_finite_difference():
    vg = make_detected_snr_objective(detector=SQUIDMagnetometer.berkeley_2007(), **_common())
    rng = np.random.default_rng(1)
    n = 8
    x = np.concatenate([rng.uniform(500, 2500, n), rng.uniform(-1, 1, n)])
    _v, g = vg(x)
    eps = 1e-2
    for i in (2, 11):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        fd = (vg(xp)[0] - vg(xm)[0]) / (2 * eps)
        assert g[i] == pytest.approx(fd, rel=2e-3, abs=1e-3 * abs(g[i]) + 1e-4 * abs(g).max())


def test_detector_noise_shape_changes_score():
    common = _common()
    x = np.concatenate([np.full(8, 2000.0), np.zeros(8)])
    squid = SQUIDMagnetometer.berkeley_2007()
    serf = OPMMagnetometer.serf(field_noise_T_per_rtHz=1.0e-15, bandwidth_hz=500.0)
    v_squid = make_detected_snr_objective(detector=squid, **common)(x)[0]
    v_serf = make_detected_snr_objective(detector=serf, **common)(x)[0]
    # different frequency weighting -> different detected SNR for the same pulse
    assert not np.isclose(v_squid, v_serf, rtol=1e-3)


def test_per_rf_power_changes_score_and_is_finite():
    common = _common()
    x = np.concatenate([np.full(8, 2000.0), np.zeros(8)])
    raw = make_detected_snr_objective(detector=SQUIDMagnetometer.berkeley_2007(), **common)(x)[0]
    perp = make_detected_snr_objective(
        detector=SQUIDMagnetometer.berkeley_2007(), per_rf_power=True, **common
    )(x)[0]
    assert np.isfinite(raw) and np.isfinite(perp)
    assert perp < raw  # divided by the RF norm


def test_explicit_noise_grid_overrides_detector():
    common = _common()
    grid = np.linspace(1.0, 4.0, common["acquisition_points"]) ** 2
    vg = make_detected_snr_objective(noise_psd_grid=grid, **common)
    x = np.concatenate([np.full(8, 2000.0), np.zeros(8)])
    assert np.isfinite(vg(x)[0])
    with pytest.raises(ValueError):
        make_detected_snr_objective(noise_psd_grid=np.ones(3), **common)
