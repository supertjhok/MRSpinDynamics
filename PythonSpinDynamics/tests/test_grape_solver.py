"""Solver-level tests for GRAPE: gradients, optimization, bounds, multistart."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.systems import coupled_spin_system  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE  # noqa: E402
from spin_dynamics.optimal_control.hamiltonians import coupled_spin_control_model  # noqa: E402
from spin_dynamics.optimal_control.parameterization import rectangular_seed_phase  # noqa: E402

if JAX_AVAILABLE:
    from spin_dynamics.optimal_control.drivers import run_grape_multistart  # noqa: E402
    from spin_dynamics.optimal_control.objectives import make_grape_objective  # noqa: E402
    from spin_dynamics.optimal_control.solvers import (  # noqa: E402
        grape_optimize,
        grape_optimize_phase_only,
    )

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_PSI_DOWN = np.array([0.0, 1.0], dtype=np.complex128)


def _inversion_model(offset_hz: float = 300.0):
    system = coupled_spin_system([offset_hz], [[0.0]])
    return coupled_spin_control_model(system)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class GradientAccuracyTests(unittest.TestCase):
    def test_phase_only_grad_matches_finite_difference(self) -> None:
        model = _inversion_model()
        n_seg = 6
        dt = np.full(n_seg, 2e-5)
        vg = make_grape_objective(
            model, n_segments=n_seg, dt=dt, target=_PSI_DOWN, psi0=_PSI_UP,
            optimize_amplitude=False, fixed_amplitude=3000.0,
        )
        rng = np.random.default_rng(1)
        phases = rng.uniform(-np.pi, np.pi, size=n_seg)
        _value, grad = vg(phases)

        h = 1e-6
        fd = np.zeros_like(phases)
        for i in range(phases.size):
            pp, pm = phases.copy(), phases.copy()
            pp[i] += h
            pm[i] -= h
            fd[i] = (vg(pp)[0] - vg(pm)[0]) / (2 * h)
        rel = np.linalg.norm(grad - fd) / max(np.linalg.norm(fd), 1e-12)
        self.assertLess(rel, 5e-3, msg=f"grad rel error {rel:.2e}")

    def test_joint_amplitude_phase_grad_matches_finite_difference(self) -> None:
        model = _inversion_model()
        n_seg = 5
        dt = np.full(n_seg, 2e-5)
        vg = make_grape_objective(
            model, n_segments=n_seg, dt=dt, target=_PSI_DOWN, psi0=_PSI_UP,
            optimize_amplitude=True,
        )
        rng = np.random.default_rng(2)
        amp = rng.uniform(500.0, 4000.0, size=n_seg)
        phase = rng.uniform(-np.pi, np.pi, size=n_seg)
        x = np.concatenate([amp, phase])
        _value, grad = vg(x)

        h = 1e-4  # amplitude scale (Hz) needs a larger step than phase (rad)
        fd = np.zeros_like(x)
        for i in range(x.size):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            fd[i] = (vg(xp)[0] - vg(xm)[0]) / (2 * h)
        rel = np.linalg.norm(grad - fd) / max(np.linalg.norm(fd), 1e-12)
        self.assertLess(rel, 5e-3, msg=f"grad rel error {rel:.2e}")


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class GrapeOptimizeTests(unittest.TestCase):
    def test_phase_only_beats_rectangular_baseline(self) -> None:
        model = _inversion_model()
        n_seg = 8
        dt = np.full(n_seg, 2e-5)
        rect_phase = rectangular_seed_phase(n_seg)
        result = grape_optimize_phase_only(
            model, rect_phase, fixed_amplitude=3000.0, dt=dt,
            target=_PSI_DOWN, psi0=_PSI_UP,
        )
        self.assertTrue(result.improved)
        self.assertGreater(result.best_fidelity, result.initial_fidelity)
        self.assertGreater(result.best_fidelity, 0.99)
        self.assertTrue(result.optimizer_success)

    def test_phase_only_respects_loose_phase_bounds(self) -> None:
        model = _inversion_model()
        n_seg = 6
        dt = np.full(n_seg, 2e-5)
        result = grape_optimize_phase_only(
            model, rectangular_seed_phase(n_seg), fixed_amplitude=3000.0, dt=dt,
            target=_PSI_DOWN, psi0=_PSI_UP, phase_bound_rad=np.pi,
        )
        self.assertTrue(np.all(result.best_phase >= -np.pi - 1e-9))
        self.assertTrue(np.all(result.best_phase <= np.pi + 1e-9))

    def test_joint_amplitude_respects_peak_b1_bound(self) -> None:
        model = _inversion_model()
        n_seg = 6
        dt = np.full(n_seg, 1e-5)
        rng = np.random.default_rng(4)
        result = grape_optimize(
            model, rng.uniform(-np.pi, np.pi, size=n_seg),
            dt=dt, target=_PSI_DOWN, psi0=_PSI_UP,
            optimize_amplitude=True,
            initial_amplitude=np.full(n_seg, 1000.0),
            amplitude_max_hz=4000.0,
        )
        self.assertTrue(np.all(result.best_amplitude >= -1e-9))
        self.assertTrue(np.all(result.best_amplitude <= 4000.0 + 1e-6))

    def test_requires_fixed_amplitude_when_not_optimizing_amplitude(self) -> None:
        model = _inversion_model()
        n_seg = 4
        with self.assertRaises(ValueError):
            grape_optimize(
                model, np.zeros(n_seg), dt=np.full(n_seg, 1e-5),
                target=_PSI_DOWN, psi0=_PSI_UP, optimize_amplitude=False,
            )

    def test_requires_amplitude_max_when_optimizing_amplitude(self) -> None:
        model = _inversion_model()
        n_seg = 4
        with self.assertRaises(ValueError):
            grape_optimize(
                model, np.zeros(n_seg), dt=np.full(n_seg, 1e-5),
                target=_PSI_DOWN, psi0=_PSI_UP, optimize_amplitude=True,
                initial_amplitude=np.full(n_seg, 1000.0),
            )

    def test_ensemble_objective_beats_rectangular_worst_case(self) -> None:
        n_seg = 10
        dt = np.full(n_seg, 2e-5)
        # Deliberately asymmetric offsets: a perfectly offset-symmetric
        # ensemble combined with an exact-zero-phase rectangular seed sits at
        # a genuine saddle point (mirrored offsets' gradient contributions
        # cancel in the mean), which is a real symmetry, not a bug -- but a
        # poor choice for a "does the optimizer improve on the baseline" test.
        offsets = [-420.0, -140.0, 30.0, 180.0, 410.0]
        models = [_inversion_model(o) for o in offsets]
        batch = [m.h_drift for m in models]
        model0 = models[2]

        rect_phase = rectangular_seed_phase(n_seg)
        vg_ensemble = make_grape_objective(
            model0, n_segments=n_seg, dt=dt, target=_PSI_DOWN, psi0=_PSI_UP,
            fixed_amplitude=3000.0, hamiltonian_batch=batch, ensemble_reduction="mean",
        )
        baseline_mean_fidelity, _ = vg_ensemble(rect_phase)

        result = grape_optimize_phase_only(
            model0, rect_phase, fixed_amplitude=3000.0, dt=dt,
            target=_PSI_DOWN, psi0=_PSI_UP,
            hamiltonian_batch=batch, ensemble_reduction="mean",
        )
        self.assertGreater(result.best_fidelity, float(baseline_mean_fidelity))
        self.assertGreater(result.best_fidelity, 0.9)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class MultiStartTests(unittest.TestCase):
    def test_multistart_ranks_and_returns_best(self) -> None:
        model = _inversion_model()
        n_seg = 6
        dt = np.full(n_seg, 2e-5)
        result = run_grape_multistart(
            model, n_seg, num_starts=5, seed=11,
            dt=dt, target=_PSI_DOWN, psi0=_PSI_UP, fixed_amplitude=3000.0,
        )
        self.assertEqual(len(result.results), 5)
        scores = [r.best_fidelity for r in result.results]
        self.assertEqual(result.best_fidelity, max(scores))
        self.assertIs(result.best_result, result.results[result.best_index])

    def test_multistart_rejects_mismatched_initial_phase_shape(self) -> None:
        model = _inversion_model()
        with self.assertRaises(ValueError):
            run_grape_multistart(
                model, 6, initial_phases=np.zeros((3, 5)),
                dt=np.full(6, 1e-5), target=_PSI_DOWN, psi0=_PSI_UP,
                fixed_amplitude=3000.0,
            )


if __name__ == "__main__":
    unittest.main()
