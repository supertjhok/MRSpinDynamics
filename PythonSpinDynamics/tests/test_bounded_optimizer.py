"""Regression tests for the bounds-shape widening in optimization._bounded."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.optimization._bounded import (  # noqa: E402
    scipy_maximize_with_grad,
)


def _quadratic_bowl_value_and_grad(center: np.ndarray):
    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        diff = x - center
        score = -float(np.sum(diff**2))
        grad = -2.0 * diff
        return score, grad

    return value_and_grad


class ScalarBoundsUnchangedTests(unittest.TestCase):
    """Confirms the pre-existing scalar-(lower, upper) call path is untouched."""

    def test_scalar_bounds_reaches_unconstrained_optimum(self) -> None:
        vg = _quadratic_bowl_value_and_grad(np.array([0.3, -0.2, 0.1]))
        result = scipy_maximize_with_grad(
            vg, np.zeros(3), bounds=(-1.0, 1.0)
        )
        np.testing.assert_allclose(result.best_x, [0.3, -0.2, 0.1], atol=1e-6)
        self.assertTrue(result.success)
        self.assertEqual(result.method, "jax+scipy:L-BFGS-B")

    def test_scalar_bounds_clip_initial_guess(self) -> None:
        vg = _quadratic_bowl_value_and_grad(np.zeros(2))
        # Initial guess outside the bounds must be clipped, exactly as before.
        result = scipy_maximize_with_grad(vg, np.array([5.0, -5.0]), bounds=(-1.0, 1.0))
        self.assertTrue(np.all(np.abs(result.best_x) <= 1.0 + 1e-9))

    def test_scalar_bounds_active_constraint(self) -> None:
        vg = _quadratic_bowl_value_and_grad(np.array([5.0]))
        result = scipy_maximize_with_grad(vg, np.array([0.0]), bounds=(-1.0, 1.0))
        np.testing.assert_allclose(result.best_x, [1.0], atol=1e-6)


class PerParameterBoundsTests(unittest.TestCase):
    """New behavior: a sequence of per-parameter (lower, upper) pairs."""

    def test_per_parameter_bounds_each_clamp_independently(self) -> None:
        vg = _quadratic_bowl_value_and_grad(np.array([5.0, 5.0, 0.2]))
        result = scipy_maximize_with_grad(
            vg,
            np.zeros(3),
            bounds=[(-1.0, 1.0), (-2.0, 0.5), (-1.0, 1.0)],
        )
        np.testing.assert_allclose(result.best_x, [1.0, 0.5, 0.2], atol=1e-6)

    def test_per_parameter_bounds_wrong_length_raises(self) -> None:
        vg = _quadratic_bowl_value_and_grad(np.zeros(3))
        with self.assertRaises(ValueError):
            scipy_maximize_with_grad(vg, np.zeros(3), bounds=[(-1.0, 1.0), (-1.0, 1.0)])

    def test_per_parameter_bounds_reject_inverted_pair(self) -> None:
        vg = _quadratic_bowl_value_and_grad(np.zeros(2))
        with self.assertRaises(ValueError):
            scipy_maximize_with_grad(vg, np.zeros(2), bounds=[(1.0, -1.0), (-1.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
