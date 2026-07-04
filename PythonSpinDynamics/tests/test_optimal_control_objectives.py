"""Unit tests for GRAPE fidelity objectives."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.systems import coupled_spin_system  # noqa: E402
from spin_dynamics.core.rotations import calc_rot_axis_arba4  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE  # noqa: E402
from spin_dynamics.optimal_control.hamiltonians import (  # noqa: E402
    coupled_spin_control_model,
    position_gradient_batch,
)

if JAX_AVAILABLE:
    import jax.numpy as jnp  # noqa: E402

    from spin_dynamics.optimal_control._jax_propagation import (  # noqa: E402
        propagate_unitary_scan,
    )
    from spin_dynamics.optimal_control.objectives import (  # noqa: E402
        average_gate_fidelity,
        bloch_vector_to_ket,
        make_grape_objective,
        robust_ensemble_fidelity,
        state_transfer_fidelity,
        su2_effective_axis,
    )

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_PSI_DOWN = np.array([0.0, 1.0], dtype=np.complex128)

TAU = 2.0 * np.pi


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class StateTransferFidelityTests(unittest.TestCase):
    def test_identical_states_have_unit_fidelity(self) -> None:
        psi = jnp.asarray([0.6, 0.8j], dtype=jnp.complex128)
        fidelity = state_transfer_fidelity(psi, psi)
        np.testing.assert_allclose(float(fidelity), 1.0, atol=1e-12)

    def test_orthogonal_states_have_zero_fidelity(self) -> None:
        psi_up = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)
        psi_down = jnp.asarray([0.0, 1.0], dtype=jnp.complex128)
        fidelity = state_transfer_fidelity(psi_up, psi_down)
        np.testing.assert_allclose(float(fidelity), 0.0, atol=1e-12)

    def test_global_phase_invariant(self) -> None:
        psi = jnp.asarray([0.6, 0.8j], dtype=jnp.complex128)
        phased = psi * jnp.exp(1j * 0.37)
        fidelity = state_transfer_fidelity(phased, psi)
        np.testing.assert_allclose(float(fidelity), 1.0, atol=1e-12)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class AverageGateFidelityTests(unittest.TestCase):
    def test_identity_vs_identity_is_unity(self) -> None:
        identity = jnp.eye(2, dtype=jnp.complex128)
        fidelity = average_gate_fidelity(identity, identity)
        np.testing.assert_allclose(float(fidelity), 1.0, atol=1e-12)

    def test_x_gate_vs_identity_is_below_unity(self) -> None:
        identity = jnp.eye(2, dtype=jnp.complex128)
        x_gate = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex128)
        fidelity = average_gate_fidelity(x_gate, identity)
        self.assertLess(float(fidelity), 1.0)
        self.assertGreaterEqual(float(fidelity), 0.0)

    def test_global_phase_invariant(self) -> None:
        target = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex128)
        phased = target * jnp.exp(1j * 1.1)
        fidelity = average_gate_fidelity(phased, target)
        np.testing.assert_allclose(float(fidelity), 1.0, atol=1e-10)

    def test_scales_with_dimension(self) -> None:
        # Sanity-check the formula is dimension-aware, not hard-coded to d=2.
        for dimension in (2, 3, 4):
            identity = jnp.eye(dimension, dtype=jnp.complex128)
            # A pi-phase-per-basis-state diagonal unitary, maximally "wrong".
            diag = np.exp(1j * np.pi * np.arange(dimension))
            u = jnp.asarray(np.diag(diag), dtype=jnp.complex128)
            fidelity = float(average_gate_fidelity(u, identity))
            self.assertGreaterEqual(fidelity, 0.0)
            self.assertLessEqual(fidelity, 1.0 + 1e-12)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class RobustEnsembleFidelityTests(unittest.TestCase):
    def test_mean_reduction(self) -> None:
        fids = jnp.asarray([0.9, 0.7, 0.5])
        np.testing.assert_allclose(
            float(robust_ensemble_fidelity(fids, reduction="mean")), 0.7, atol=1e-9
        )

    def test_worst_case_reduction(self) -> None:
        fids = jnp.asarray([0.9, 0.7, 0.5])
        np.testing.assert_allclose(
            float(robust_ensemble_fidelity(fids, reduction="worst_case")), 0.5, atol=1e-9
        )

    def test_rejects_unknown_reduction(self) -> None:
        fids = jnp.asarray([0.9, 0.7])
        with self.assertRaises(ValueError):
            robust_ensemble_fidelity(fids, reduction="best_case")


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class Su2EffectiveAxisTests(unittest.TestCase):
    def _cycle_unitary(self, offset_hz: float, w1_hz: float, tp, phi):
        model = coupled_spin_control_model(coupled_spin_system([offset_hz], [[0.0]]))
        return propagate_unitary_scan(
            jnp.asarray(model.h_drift), jnp.asarray(model.h_x), jnp.asarray(model.h_y),
            jnp.asarray(np.full(len(tp), w1_hz)), jnp.asarray(phi), jnp.asarray(tp),
        )

    def test_matches_calc_rot_axis_arba4(self) -> None:
        offset_hz, w1_hz, n_seg = 350.0, 4000.0, 6
        rng = np.random.default_rng(5)
        tp = rng.uniform(1e-5, 4e-5, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)

        u = self._cycle_unitary(offset_hz, w1_hz, tp, phi)
        axis, angle = su2_effective_axis(u)

        n_gt, alpha_gt = calc_rot_axis_arba4(
            tp, phi, np.full(n_seg, TAU * w1_hz), np.array([TAU * offset_hz])
        )
        np.testing.assert_allclose(np.asarray(axis), n_gt[:, 0], atol=1e-9)
        np.testing.assert_allclose(float(angle), float(alpha_gt[0]), atol=1e-9)

    def test_axis_is_unit_norm(self) -> None:
        rng = np.random.default_rng(6)
        n_seg = 5
        tp = rng.uniform(1e-5, 4e-5, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        u = self._cycle_unitary(200.0, 3000.0, tp, phi)
        axis, _angle = su2_effective_axis(u)
        np.testing.assert_allclose(float(jnp.sum(axis**2)), 1.0, atol=1e-9)

    def test_identity_cycle_returns_zero_angle(self) -> None:
        # Zero amplitude, on resonance: the cycle unitary is exactly identity.
        u = self._cycle_unitary(0.0, 0.0, [1e-5, 1e-5], [0.0, 0.0])
        _axis, angle = su2_effective_axis(u)
        np.testing.assert_allclose(float(angle), 0.0, atol=1e-9)

    def test_bloch_vector_to_ket_is_fixed_point_of_its_own_cycle(self) -> None:
        # |n> for axis n is, by construction, an eigenvector of U -- applying
        # U changes only its global phase, so the Bloch vector it represents
        # (via su2_effective_axis-style projection) is unchanged.
        offset_hz, w1_hz, n_seg = 275.0, 3500.0, 5
        rng = np.random.default_rng(7)
        tp = rng.uniform(1e-5, 4e-5, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        u = self._cycle_unitary(offset_hz, w1_hz, tp, phi)
        axis, _angle = su2_effective_axis(u)
        ket = bloch_vector_to_ket(axis)

        ket_after = u @ ket
        # Same ray (equal up to a global phase): |<ket|ket_after>|^2 == 1.
        overlap = jnp.vdot(ket, ket_after)
        np.testing.assert_allclose(float(jnp.real(overlap * jnp.conj(overlap))), 1.0, atol=1e-9)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class GradientChannelObjectiveTests(unittest.TestCase):
    """The gradient control channel: autodiff, per-case targets, layout."""

    def _setup(self, n_seg=6, npos=7):
        system = coupled_spin_system([0.0], [[0.0]])
        model = coupled_spin_control_model(system, include_gradient=True)
        positions = np.linspace(-1.5, 1.5, npos)
        hgb = position_gradient_batch(model.h_grad, positions)
        dt = np.full(n_seg, 1.5e-5)
        # per-case target: invert inside the slice, leave alone outside.
        targets = np.stack([_PSI_DOWN if abs(p) < 0.5 else _PSI_UP for p in positions])
        return model, positions, hgb, dt, targets

    def test_gradient_and_amplitude_grad_matches_central_difference(self) -> None:
        n_seg = 6
        model, _positions, hgb, dt, targets = self._setup(n_seg=n_seg)
        vg = make_grape_objective(
            model, n_segments=n_seg, dt=dt, target=targets, psi0=_PSI_UP,
            optimize_amplitude=True, optimize_gradient=True,
            gradient_operator_batch=hgb,
        )
        rng = np.random.default_rng(11)
        x = np.concatenate([
            rng.uniform(1000.0, 4000.0, n_seg),   # amplitude
            rng.uniform(-np.pi, np.pi, n_seg),    # phase
            rng.uniform(-2000.0, 2000.0, n_seg),  # gradient
        ])
        _val, grad = vg(x)
        self.assertEqual(grad.shape, (3 * n_seg,))
        eps = 1e-4
        fd = np.zeros_like(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            fd[i] = (vg(xp)[0] - vg(xm)[0]) / (2 * eps)
        np.testing.assert_allclose(grad, fd, atol=5e-3)

    def test_phase_only_gradient_channel_grad_matches_fd(self) -> None:
        # gradient channel on, RF amplitude fixed (phase-only + gradient).
        n_seg = 5
        model, _positions, hgb, dt, targets = self._setup(n_seg=n_seg, npos=5)
        vg = make_grape_objective(
            model, n_segments=n_seg, dt=dt, target=targets, psi0=_PSI_UP,
            optimize_amplitude=False, fixed_amplitude=3000.0,
            optimize_gradient=True, gradient_operator_batch=hgb,
        )
        rng = np.random.default_rng(12)
        x = np.concatenate([
            rng.uniform(-np.pi, np.pi, n_seg),    # phase (amplitude fixed)
            rng.uniform(-2000.0, 2000.0, n_seg),  # gradient
        ])
        _val, grad = vg(x)
        self.assertEqual(grad.shape, (2 * n_seg,))
        eps = 1e-4
        fd = np.zeros_like(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            fd[i] = (vg(xp)[0] - vg(xm)[0]) / (2 * eps)
        np.testing.assert_allclose(grad, fd, atol=5e-3)

    def test_fixed_gradient_matches_manual_evaluation(self) -> None:
        # A held (non-optimized) constant gradient reproduces a direct per-case
        # NumPy evaluation of the same waveform.
        from spin_dynamics.optimal_control._jax_propagation import propagate_state_numpy_grad

        n_seg = 5
        model, _positions, hgb, dt, targets = self._setup(n_seg=n_seg, npos=5)
        g_const = 1200.0
        vg = make_grape_objective(
            model, n_segments=n_seg, dt=dt, target=targets, psi0=_PSI_UP,
            optimize_amplitude=False, fixed_amplitude=3000.0,
            optimize_gradient=False, fixed_gradient=g_const,
            gradient_operator_batch=hgb,
        )
        phase = np.linspace(0.0, 1.0, n_seg)
        val, _grad = vg(phase)
        amp = np.full(n_seg, 3000.0)
        g = np.full(n_seg, g_const)
        fids = []
        for k in range(len(hgb)):
            psi = propagate_state_numpy_grad(
                model.h_drift, model.h_x, model.h_y, hgb[k], amp, phase, g, dt, _PSI_UP
            )
            overlap = np.vdot(targets[k], psi)
            fids.append(np.real(overlap * np.conj(overlap)))
        np.testing.assert_allclose(val, float(np.mean(fids)), atol=1e-9)

    def test_optimize_gradient_requires_operator_batch(self) -> None:
        system = coupled_spin_system([0.0], [[0.0]])
        model = coupled_spin_control_model(system, include_gradient=True)
        with self.assertRaises(ValueError):
            make_grape_objective(
                model, n_segments=4, dt=np.full(4, 1e-5), target=_PSI_DOWN, psi0=_PSI_UP,
                optimize_gradient=True,
            )


if __name__ == "__main__":
    unittest.main()
