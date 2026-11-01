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


class BoundsEnforcementTests(unittest.TestCase):
    """Review M-39: results must never escape the box bounds.

    Before the fix, ``scipy_method="BFGS"`` silently ignored the bounds (SciPy
    only warns) and returned the unconstrained optimum ``[5, 5]`` for a box of
    ``[0, 1]^2`` -- in GRAPE that exceeds ``amplitude_max_hz``.
    """

    def test_unbounded_named_methods_are_rejected(self) -> None:
        from spin_dynamics.optimization._bounded import scipy_maximize

        vg = _quadratic_bowl_value_and_grad(np.array([5.0, 5.0]))
        for method in ("BFGS", "CG", "Newton-CG"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "ignores bounds"):
                    scipy_maximize_with_grad(
                        vg, np.full(2, 0.5), bounds=(0.0, 1.0), scipy_method=method
                    )
                with self.assertRaisesRegex(ValueError, "ignores bounds"):
                    scipy_maximize(
                        lambda x: vg(x)[0], np.full(2, 0.5), bounds=(0.0, 1.0),
                        scipy_method=method,
                    )

    def test_bound_capable_methods_reach_the_box_corner(self) -> None:
        # Independent reference: the maximizer of -(x-5)^2 on [0, 1]^2 is (1, 1).
        vg = _quadratic_bowl_value_and_grad(np.array([5.0, 5.0]))
        for method in ("L-BFGS-B", "TNC", "SLSQP"):
            with self.subTest(method=method):
                run = scipy_maximize_with_grad(
                    vg, np.full(2, 0.5), bounds=(0.0, 1.0), scipy_method=method
                )
                np.testing.assert_allclose(run.best_x, [1.0, 1.0], atol=1e-6)
                self.assertTrue(np.all((run.best_x >= 0.0) & (run.best_x <= 1.0)))

    def test_custom_method_result_is_projected_and_rescored(self) -> None:
        from scipy.optimize import OptimizeResult

        def escaping_method(fun, x0, args=(), **_options):
            # A user-supplied method that ignores the bounds entirely.
            return OptimizeResult(
                x=np.array([5.0, -3.0]), success=True, message="escaped", nfev=1
            )

        vg = _quadratic_bowl_value_and_grad(np.array([5.0, 5.0]))
        run = scipy_maximize_with_grad(
            vg,
            np.full(2, 0.5),
            bounds=[(0.0, 1.0), (0.0, 2.0)],
            scipy_method=escaping_method,
        )
        np.testing.assert_allclose(run.best_x, [1.0, 0.0])
        np.testing.assert_allclose(run.best_score, -(16.0 + 25.0))


if __name__ == "__main__":
    unittest.main()
