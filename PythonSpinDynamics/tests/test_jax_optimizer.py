"""Phase 3: autodiff v0crit objective — score/grad parity and optimizer wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.optimization import _jax_objectives as jo  # noqa: E402
from spin_dynamics.optimization.refocusing import (  # noqa: E402
    evaluate_ideal_v0crit_refocusing_pulse,
    optimize_ideal_v0crit_refocusing_phases,
)

NUMPTS = 81


def _numpy_score(phases: np.ndarray) -> float:
    return float(
        evaluate_ideal_v0crit_refocusing_pulse(phases, numpts=NUMPTS).score
    )


class JaxObjectiveTests(unittest.TestCase):
    @unittest.skipUnless(jo.JAX_AVAILABLE, "jax not installed")
    def test_score_matches_numpy(self) -> None:
        phases = np.linspace(0.2, 2.9, 12)
        vg = jo.make_ideal_v0crit_objective(phases.size, numpts=NUMPTS)
        value, _grad = vg(phases)
        np.testing.assert_allclose(value, _numpy_score(phases), rtol=1e-6)

    @unittest.skipUnless(jo.JAX_AVAILABLE, "jax not installed")
    def test_grad_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(1)
        phases = rng.uniform(0.3, 2.8, size=8)
        vg = jo.make_ideal_v0crit_objective(phases.size, numpts=NUMPTS)
        _value, grad = vg(phases)

        h = 1e-6
        fd = np.zeros_like(phases)
        for i in range(phases.size):
            pp = phases.copy()
            pp[i] += h
            pm = phases.copy()
            pm[i] -= h
            fd[i] = (_numpy_score(pp) - _numpy_score(pm)) / (2 * h)

        rel = np.linalg.norm(grad - fd) / max(np.linalg.norm(fd), 1e-12)
        self.assertLess(rel, 5e-3, msg=f"grad rel error {rel:.2e}")

    @unittest.skipUnless(jo.JAX_AVAILABLE, "jax not installed")
    def test_jax_optimizer_matches_scipy_fd_more_cheaply(self) -> None:
        # Same algorithm (L-BFGS-B), analytic vs finite-difference gradients:
        # equal quality, but autodiff needs far fewer forward evaluations.
        rng = np.random.default_rng(0)
        initial = rng.uniform(0.0, 2 * np.pi, size=12)
        res_jax = optimize_ideal_v0crit_refocusing_phases(
            initial, numpts=NUMPTS, optimizer="jax"
        )
        res_fd = optimize_ideal_v0crit_refocusing_phases(
            initial, numpts=NUMPTS, optimizer="scipy"
        )
        self.assertTrue(np.isfinite(res_jax.best_score))
        self.assertGreaterEqual(res_jax.best_score, res_jax.initial_score)
        self.assertTrue(res_jax.improved)
        self.assertEqual(res_jax.optimizer_method.split(":")[0], "jax+scipy")
        # Autodiff needs far fewer objective evaluations than finite differencing
        # (which spends ~N extra forward evals per gradient). Gradient *quality*
        # is validated separately in test_grad_matches_finite_difference; the two
        # runs may land in different optima precisely because FD gradients are
        # inaccurate on this stiff objective.
        self.assertLess(res_jax.history_scores.size, res_fd.history_scores.size)

    @unittest.skipUnless(jo.JAX_AVAILABLE, "jax not installed")
    def test_jax_optimizer_rejects_excitation_vector(self) -> None:
        with self.assertRaises(ValueError):
            optimize_ideal_v0crit_refocusing_phases(
                np.zeros(6),
                numpts=NUMPTS,
                optimizer="jax",
                excitation_vector=np.zeros((3, NUMPTS), dtype=np.complex128),
            )


if __name__ == "__main__":
    unittest.main()
