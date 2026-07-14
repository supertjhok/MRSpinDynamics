"""Tests for tri-axial (multi-coil) NQR SLSE control.

Covers the multi-coil composite-pulse propagator (numpy/jax parity, single-coil
reduction to the existing single-axis path) and the parametric SLSE objective
(value matches the NumPy reference, finite gradients that match finite
differences -- the expm propagator must avoid the Kramers-degeneracy NaN).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr.hamiltonians import diagonalize_site  # noqa: E402
from spin_dynamics.nqr.isotopes import quadrupolar_site  # noqa: E402
from spin_dynamics.nqr.orientations import powder_frame_grid  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import (  # noqa: E402
    JAX_AVAILABLE,
    propagate_unitary_scan_multi_numpy,
)
from spin_dynamics.optimal_control.hamiltonians import nqr_site_control_model  # noqa: E402
from spin_dynamics.optimal_control.multi_axis import (  # noqa: E402
    MultiAxisSLSEConfig,
    control_bounds,
    make_multi_axis_slse_objective,
    optimize_multi_axis_slse,
    simultaneous_seed,
    slse_train_amplitudes,
)

if JAX_AVAILABLE:
    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import (
        propagate_unitary_scan,
        propagate_unitary_scan_multi,
    )
    from spin_dynamics.optimal_control.multi_axis import _make_energy_fn


def _site():
    return quadrupolar_site("14N", nu_q_hz=3.3e6, eta=0.3)


def _coil_operators(site, n_coils):
    eig = diagonalize_site(site, None)
    rf = max(t.frequency_hz for t in eig.transitions)
    axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][:n_coils]
    models = [nqr_site_control_model(site, rf_frequency_hz=rf, b1_direction_pas=a)
              for a in axes]
    h_drift = models[0].h_drift
    hx = np.stack([m.h_x for m in models])
    hy = np.stack([m.h_y for m in models])
    return h_drift, hx, hy


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class MultiCoilPropagatorTests(unittest.TestCase):
    def setUp(self):
        self.h_drift, self.hx, self.hy = _coil_operators(_site(), 3)
        rng = np.random.default_rng(0)
        self.amp = rng.uniform(0, 3e4, size=(5, 3))
        self.phase = rng.uniform(-np.pi, np.pi, size=(5, 3))
        self.dt = np.full(5, 2e-6)

    def test_numpy_jax_parity(self):
        u_np = propagate_unitary_scan_multi_numpy(
            self.h_drift, self.hx, self.hy, self.amp, self.phase, self.dt)
        u_jx = np.asarray(propagate_unitary_scan_multi(
            jnp.asarray(self.h_drift), jnp.asarray(self.hx), jnp.asarray(self.hy),
            jnp.asarray(self.amp), jnp.asarray(self.phase), jnp.asarray(self.dt),
            method="expm"))
        self.assertLess(np.max(np.abs(u_np - u_jx)), 1e-7)

    def test_result_is_unitary(self):
        u = propagate_unitary_scan_multi_numpy(
            self.h_drift, self.hx, self.hy, self.amp, self.phase, self.dt)
        np.testing.assert_allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=1e-9)

    def test_single_coil_matches_single_axis_scan(self):
        u_multi = np.asarray(propagate_unitary_scan_multi(
            jnp.asarray(self.h_drift), jnp.asarray(self.hx[:1]), jnp.asarray(self.hy[:1]),
            jnp.asarray(self.amp[:, :1]), jnp.asarray(self.phase[:, :1]),
            jnp.asarray(self.dt), method="expm"))
        u_single = np.asarray(propagate_unitary_scan(
            jnp.asarray(self.h_drift), jnp.asarray(self.hx[0]), jnp.asarray(self.hy[0]),
            jnp.asarray(self.amp[:, 0]), jnp.asarray(self.phase[:, 0]),
            jnp.asarray(self.dt)))
        self.assertLess(np.max(np.abs(u_multi - u_single)), 1e-7)


class ControlLayoutTests(unittest.TestCase):
    def test_bounds_length(self):
        site = _site()
        frames = powder_frame_grid(2, 2, 1)
        cfg = MultiAxisSLSEConfig(site=site, frames=frames, n_coils=3)
        self.assertEqual(len(control_bounds(cfg)), 2 * 3 * 4)

    def test_quadrature_requires_two_coils(self):
        with self.assertRaises(ValueError):
            MultiAxisSLSEConfig(site=_site(), frames=powder_frame_grid(2, 2, 1),
                                n_coils=1, rx_scheme="quadrature")

    def test_config_rejects_invalid_physical_and_grid_parameters(self):
        frames = powder_frame_grid(2, 2, 1)
        cases = (
            {"frames": ()},
            {"frames": frames, "nutation_scale_hz": 0.0},
            {"frames": frames, "window_ex_seconds": 0.0},
            {"frames": frames, "n_fine": 0},
            {"frames": frames, "num_echoes": 0},
            {"frames": frames, "ramp_segments": -1.0},
            {"frames": frames, "helicity": 0},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                MultiAxisSLSEConfig(site=_site(), n_coils=2, **overrides)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.site = _site()
        self.frames = powder_frame_grid(3, 4, 2)
        self.cfg = MultiAxisSLSEConfig(
            site=self.site, frames=self.frames, n_coils=2, n_fine=12,
            num_echoes=4, rx_scheme="quadrature")
        rng = np.random.default_rng(2)
        x = np.zeros((2 * self.cfg.n_coils, 4))
        x[:, 0] = rng.uniform(0.5, 2.0, size=2 * self.cfg.n_coils)
        x[:, 1] = rng.uniform(-np.pi, np.pi, size=2 * self.cfg.n_coils)
        x[:, 2] = rng.uniform(0.0, 0.3, size=2 * self.cfg.n_coils)
        x[:, 3] = rng.uniform(0.4, 0.9, size=2 * self.cfg.n_coils)
        self.x = x.reshape(-1)

    def test_value_matches_numpy_reference(self):
        vg = make_multi_axis_slse_objective(self.cfg)
        value, _grad = vg(self.x)
        energy_np = float(np.sum(np.abs(slse_train_amplitudes(self.cfg, self.x))))
        self.assertAlmostEqual(value, energy_np, places=8)

    def test_gradient_finite_and_matches_fd(self):
        vg = make_multi_axis_slse_objective(self.cfg)
        _value, grad = vg(self.x)
        self.assertTrue(np.all(np.isfinite(grad)))
        energy = _make_energy_fn(self.cfg)

        def f(z):
            return float(energy(jnp.asarray(z)))

        eps = 1e-6
        for i in (0, 3, 5):
            xp = self.x.copy()
            xp[i] += eps
            xm = self.x.copy()
            xm[i] -= eps
            fd = (f(xp) - f(xm)) / (2 * eps)
            self.assertLess(abs(grad[i] - fd) / (abs(fd) + 1e-9), 1e-4)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class OptimizeTests(unittest.TestCase):
    def test_optimization_improves_on_seed(self):
        cfg = MultiAxisSLSEConfig(
            site=_site(), frames=powder_frame_grid(2, 4, 1), n_coils=2, n_fine=12,
            num_echoes=4, rx_scheme="quadrature")
        res = optimize_multi_axis_slse(cfg, n_starts=2, seed=0, amp_max=1.0)
        seed_energy = res.seed_energies[0]
        self.assertGreater(res.best_energy, seed_energy)
        self.assertEqual(res.excite_params().shape, (2, 4))

    def test_simultaneous_seed_shape(self):
        cfg = MultiAxisSLSEConfig(site=_site(), frames=powder_frame_grid(2, 2, 1),
                                  n_coils=3)
        seed = simultaneous_seed(cfg)
        self.assertEqual(seed.size, 2 * 3 * 4)


if __name__ == "__main__":
    unittest.main()
