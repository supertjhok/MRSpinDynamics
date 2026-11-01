"""Regression tests for code-review items M-38, M-40, M-42 and M-43.

* M-38: ``MultiAxisSLSEConfig``'s operator cache was shared through
  ``dataclasses.replace``, so a replaced config reused stale operators.
* M-40: with a gradient channel GRAPE forced the ``eigh`` propagator, whose
  gradient is NaN for a vanishing segment Hamiltonian; ``expm`` was rejected.
* M-42: ``GrapeOptimizationResult`` had no penalty-free fidelity, and
  ``initial_fidelity`` was evaluated at the unclipped start.
* M-43: non-uniform ``dt`` broke ``gradient_moments`` (``cumsum(g) * dt``) and
  the ``per_rf_power`` detected-SNR objective (non-scalar RF energy).
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.operators import total_operator  # noqa: E402
from spin_dynamics.coupling.systems import coupled_spin_system  # noqa: E402
from spin_dynamics.nqr.isotopes import quadrupolar_site  # noqa: E402
from spin_dynamics.nqr.orientations import powder_frame_grid  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import (  # noqa: E402
    JAX_AVAILABLE,
    propagate_state_numpy,
    propagate_state_numpy_grad,
)
from spin_dynamics.optimal_control.hamiltonians import (  # noqa: E402
    coupled_spin_control_model,
    gradient_control_operator,
    position_gradient_batch,
)
from spin_dynamics.optimal_control.multi_axis import (  # noqa: E402
    MultiAxisSLSEConfig,
    _build_model,
)

if JAX_AVAILABLE:
    import jax.numpy as jnp  # noqa: E402

    from spin_dynamics.optimal_control.control_response import (  # noqa: E402
        GradientDriverResponse,
    )
    from spin_dynamics.optimal_control.detection_objective import (  # noqa: E402
        make_detected_snr_objective,
    )
    from spin_dynamics.optimal_control.diffusion import gradient_moments  # noqa: E402
    from spin_dynamics.optimal_control.objectives import make_grape_objective  # noqa: E402
    from spin_dynamics.optimal_control.solvers import grape_optimize  # noqa: E402

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_PSI_DOWN = np.array([0.0, 1.0], dtype=np.complex128)


def _central_difference(fn, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (fn(xp)[0] - fn(xm)[0]) / (2.0 * eps)
    return grad


class MultiAxisCacheTests(unittest.TestCase):
    def test_replace_rebuilds_operators(self) -> None:
        site = quadrupolar_site("14N", nu_q_hz=1e6, eta=0.0)
        frames = powder_frame_grid(2, 2, 1)
        single = MultiAxisSLSEConfig(site=site, frames=frames, n_coils=1, rx_scheme="single")
        _build_model(single)
        replaced = dataclasses.replace(single, n_coils=3, rx_scheme="quadrature")
        # Independent reference: a freshly constructed 3-coil config.
        fresh = MultiAxisSLSEConfig(
            site=site, frames=frames, n_coils=3, rx_scheme="quadrature"
        )
        built = _build_model(replaced)
        expected = _build_model(fresh)
        self.assertIsNot(replaced._built, single._built)
        self.assertEqual(built["hx_batch"].shape, expected["hx_batch"].shape)
        self.assertEqual(built["hx_batch"].shape[1], 3)
        np.testing.assert_allclose(built["hx_batch"], expected["hx_batch"])
        np.testing.assert_allclose(built["det_batch"], expected["det_batch"])
        # The original config keeps its own single-coil operators.
        self.assertEqual(_build_model(single)["hx_batch"].shape[1], 1)


def _slice_case():
    system = coupled_spin_system([0.0], [[0.0]])
    model = coupled_spin_control_model(system)
    positions = np.linspace(-1.0, 1.0, 5)  # includes the on-resonance position 0
    batch = position_gradient_batch(gradient_control_operator(system), positions)
    targets = np.stack([_PSI_DOWN if abs(p) < 0.6 else _PSI_UP for p in positions])
    return model, batch, targets


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class GradientChannelExpmTests(unittest.TestCase):
    def test_zero_amplitude_segment_has_finite_expm_gradient(self) -> None:
        model, batch, targets = _slice_case()
        n = 8
        dt = np.full(n, 2e-5)
        common = dict(
            n_segments=n, dt=dt, target=targets, psi0=_PSI_UP,
            optimize_amplitude=True, optimize_gradient=True,
            gradient_operator_batch=batch,
        )
        # The last segment sits at the amplitude lower bound (0).
        amplitude = np.r_[np.full(n - 1, 3000.0), 0.0]
        x = np.concatenate([amplitude, np.linspace(0.0, 1.0, n), np.full(n, 100.0)])

        _value, eigh_grad = make_grape_objective(model, **common)(x)
        self.assertTrue(np.any(np.isnan(eigh_grad)))  # documents why expm is needed

        vg = make_grape_objective(model, propagator="expm", **common)
        value, grad = vg(x)
        # Independent references: the NumPy propagator and finite differences.
        phase = x[n : 2 * n]
        gradient = x[2 * n :]
        expected = np.mean([
            abs(np.vdot(t, propagate_state_numpy_grad(
                model.h_drift, model.h_x, model.h_y, h, amplitude, phase,
                gradient, dt, _PSI_UP,
            ))) ** 2
            for h, t in zip(batch, targets)
        ])
        np.testing.assert_allclose(value, expected, rtol=1e-10)
        self.assertTrue(np.all(np.isfinite(grad)))
        np.testing.assert_allclose(grad, _central_difference(vg, x), rtol=1e-4, atol=1e-8)

    def test_gradient_response_tail_has_finite_expm_gradient(self) -> None:
        # Only a gradient driver is modelled, so the appended RF tail is exactly
        # zero and the position-0 member's H_seg vanishes there.
        model, batch, targets = _slice_case()
        n = 8
        common = dict(
            n_segments=n, dt=2e-5, target=targets, psi0=_PSI_UP,
            fixed_amplitude=3000.0, optimize_gradient=True,
            gradient_operator_batch=batch,
            gradient_response=GradientDriverResponse(tau_rl=2e-6, oversample=2),
        )
        x = np.concatenate([np.linspace(0.0, 1.0, n), np.full(n, 100.0)])
        _value, eigh_grad = make_grape_objective(model, **common)(x)
        self.assertTrue(np.any(np.isnan(eigh_grad)))
        vg = make_grape_objective(model, propagator="expm", **common)
        _value, grad = vg(x)
        self.assertTrue(np.all(np.isfinite(grad)))
        np.testing.assert_allclose(grad, _central_difference(vg, x), rtol=1e-4, atol=1e-8)

    def test_expm_gate_mode_matches_eigh_value(self) -> None:
        model, batch, _targets = _slice_case()
        n = 6
        x = np.concatenate([np.linspace(0.0, 1.0, n), np.full(n, 100.0)])
        target = np.array([[0.0, -1j], [-1j, 0.0]])
        common = dict(
            n_segments=n, dt=np.full(n, 3e-5), target=target, mode="gate",
            fixed_amplitude=3000.0, optimize_gradient=True,
            gradient_operator_batch=batch,
        )
        eigh_value, _ = make_grape_objective(model, **common)(x)
        expm_value, _ = make_grape_objective(model, propagator="expm", **common)(x)
        np.testing.assert_allclose(expm_value, eigh_value, rtol=1e-10)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class UnpenalizedFidelityTests(unittest.TestCase):
    def test_result_reports_penalty_free_fidelity_at_clipped_start(self) -> None:
        model = coupled_spin_control_model(coupled_spin_system([300.0], [[0.0]]))
        n = 6
        dt = np.full(n, 2e-5)
        weight = 0.05
        bound = np.pi
        # The last phase lies outside the +-pi box: the optimizer starts at pi.
        initial = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 7.0])
        result = grape_optimize(
            model, initial, dt=dt, target=_PSI_DOWN, psi0=_PSI_UP,
            fixed_amplitude=3000.0, phase_bound_rad=bound,
            phase_smoothness_weight=weight,
            scipy_options={"maxiter": 5},
        )

        def fidelity(phase):
            psi = propagate_state_numpy(
                model.h_drift, model.h_x, model.h_y, np.full(n, 3000.0), phase, dt, _PSI_UP
            )
            return abs(np.vdot(_PSI_DOWN, psi)) ** 2

        def penalty(phase):
            return weight * np.sum(np.diff(phase) ** 2)

        start = np.clip(initial, -bound, bound)
        np.testing.assert_allclose(result.initial_controls, start)
        np.testing.assert_allclose(
            result.initial_unpenalized_fidelity, fidelity(start), rtol=1e-10
        )
        np.testing.assert_allclose(
            result.initial_fidelity, fidelity(start) - penalty(start), rtol=1e-10
        )
        best = result.best_phase
        np.testing.assert_allclose(
            result.best_unpenalized_fidelity, fidelity(best), rtol=1e-10
        )
        np.testing.assert_allclose(
            result.best_fidelity, fidelity(best) - penalty(best), rtol=1e-10
        )
        self.assertGreater(result.best_unpenalized_fidelity, result.best_fidelity)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class NonUniformDtTests(unittest.TestCase):
    def test_gradient_moments_integrate_non_uniform_dt(self) -> None:
        g = np.array([1.0, 1.0, -1.0, -1.0, 0.5])
        dt = np.array([1.0, 3.0, 1.0, 3.0, 2.0])
        sign = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
        q, q_encode, q_residual, b_value = gradient_moments(
            jnp.asarray(g), jnp.asarray(dt), encode_end=2, coherence_sign=jnp.asarray(sign)
        )
        # Independent reference: left-to-right sum of g*s*dt (before the fix,
        # cumsum(g*s)*dt gave q = [1, 6, 1, 6, 3]).
        expected_q = np.array([1.0, 4.0, 3.0, 6.0, 5.0])
        np.testing.assert_allclose(np.asarray(q), expected_q)
        np.testing.assert_allclose(float(q_encode), 4.0)
        np.testing.assert_allclose(float(q_residual), 5.0)
        np.testing.assert_allclose(float(b_value), np.sum(expected_q**2 * dt))
        # Uniform dt is unchanged by the fix.
        q_u, *_ = gradient_moments(jnp.asarray(g), 2.0, encode_end=2)
        np.testing.assert_allclose(np.asarray(q_u), np.cumsum(g) * 2.0)

    def test_per_rf_power_objective_with_non_uniform_dt(self) -> None:
        model = coupled_spin_control_model(coupled_spin_system([100.0], [[0.0]]))
        detect = total_operator(1, "x") + 1j * total_operator(1, "y")
        dt = np.array([1e-5, 2e-5, 1e-5, 3e-5])
        amplitude = np.array([5000.0, 4000.0, 6000.0, 3000.0])
        common = dict(
            drift_batch=[model.h_drift], hx_batch=[model.h_x], hy_batch=[model.h_y],
            psi0=_PSI_UP, detection_operator=detect, weights=np.ones(1),
            offsets_hz=np.array([100.0]), dt=dt, n_segments=4,
            amplitude_template=amplitude, acquisition_points=8, acquisition_dt=1e-5,
        )
        x = np.array([0.0, 0.4, 1.0, 0.2])
        raw_snr, _ = make_detected_snr_objective(**common)(x)
        # Before the fix this raised (non-scalar RF energy for array dt).
        value, grad = make_detected_snr_objective(per_rf_power=True, **common)(x)
        rf_norm = np.sqrt(np.sum(amplitude**2 * dt))
        np.testing.assert_allclose(value, raw_snr / rf_norm, rtol=1e-10)
        self.assertEqual(grad.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(grad)))


if __name__ == "__main__":
    unittest.main()
