"""Hybrid EPM array geometry, field-basis, and synthesis tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.electropermanent_array import (  # noqa: E402
    ElectropermanentArray,
    ElectropermanentArrayFieldBasis,
    affine_transport_target,
    illustrative_hybrid_epm_array,
    synthesize_epm_array_state,
    synthesize_field_off_state,
    synthesize_transport_state,
)


class HybridArrayGeometryTests(unittest.TestCase):
    def test_reference_hierarchy_has_two_panels_and_72_controls(self) -> None:
        array = illustrative_hybrid_epm_array()
        self.assertEqual(len(array.subunits), 18)
        self.assertEqual(len(array.fixed_magnets), 18)
        self.assertEqual(len(array.programmable_elements), 72)
        self.assertEqual({subunit.panel_index for subunit in array.subunits}, {0, 1})
        self.assertTrue(np.allclose(array.remanence_limits_t, 0.33))
        self.assertEqual(array.evidence[0].classification, "specified")
        self.assertEqual(array.evidence[1].classification, "inferred")

    def test_with_remanence_preserves_geometry_and_enforces_limits(self) -> None:
        array = illustrative_hybrid_epm_array()
        values = np.linspace(-0.30, 0.30, 72)
        updated = array.with_remanence(values)
        np.testing.assert_allclose(updated.retained_remanence_t, values)
        self.assertEqual(updated.fixed_magnets, array.fixed_magnets)
        with self.assertRaises(ValueError):
            array.with_remanence(np.full(72, 0.34))

    def test_cached_basis_matches_direct_superposition(self) -> None:
        full = illustrative_hybrid_epm_array()
        array = ElectropermanentArray(full.subunits[:2], label="two-subunit test")
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.005, -0.004, 0.002],
                [-0.008, 0.003, -0.004],
            ]
        )
        basis = array.build_field_basis(points, n_cross=3, n_length=7)
        values = np.linspace(-0.2, 0.2, len(array.programmable_elements))
        cached = basis.field_vectors(values)
        direct = array.field_vectors(
            points,
            remanence_t=values,
            n_cross=3,
            n_length=7,
        )
        np.testing.assert_allclose(cached, direct, rtol=2e-13, atol=2e-15)


class ArraySynthesisTests(unittest.TestCase):
    def _synthetic_basis(self) -> ElectropermanentArrayFieldBasis:
        full = illustrative_hybrid_epm_array()
        array = ElectropermanentArray((full.subunits[0],), label="synthetic basis")
        points = np.asarray(
            [
                [-0.01, 0.0, 0.0],
                [0.0, -0.01, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
            ]
        )
        programmable = np.zeros((4, 3, 4))
        programmable[:, 2, :] = np.eye(4)
        return ElectropermanentArrayFieldBasis(
            array=array,
            points_m=points,
            fixed_field_t=np.zeros((4, 3)),
            programmable_field_t_per_t=programmable,
            n_cross=1,
            n_length=1,
        )

    def test_numpy_bounded_solver_recovers_representable_target(self) -> None:
        basis = self._synthetic_basis()
        target = np.asarray([-0.20, -0.05, 0.10, 0.25])
        result = synthesize_epm_array_state(
            basis,
            target,
            field_direction=(0.0, 0.0, 1.0),
            regularization=0.0,
            backend="numpy",
            tolerance_t=1e-12,
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.remanence_t, target, atol=1e-12)
        np.testing.assert_allclose(result.predicted_projected_field_t, target, atol=1e-12)
        self.assertAlmostEqual(result.condition_number, 1.0)

    def test_field_off_synthesis_reduces_reference_array_field(self) -> None:
        array = illustrative_hybrid_epm_array()
        axis = np.linspace(-0.015, 0.015, 3)
        points = np.asarray(
            [
                (x, y, z)
                for x in axis
                for y in axis
                for z in axis
                if x * x + y * y + z * z <= 0.015**2 + 1e-15
            ]
        )
        basis = array.build_field_basis(points, n_cross=3, n_length=7)
        initial_rms = float(np.sqrt(np.mean(basis.fixed_field_t[:, 2] ** 2)))
        result = synthesize_field_off_state(
            basis,
            field_direction=(0.0, 0.0, 1.0),
            regularization=1e-8,
            backend="numpy",
            tolerance_t=1e-9,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.rms_error_t, 0.4 * initial_rms)
        self.assertTrue(np.all(np.abs(result.remanence_t) <= 0.33 + 1e-12))

    def test_transport_target_and_synthesis_preserve_gradient_direction(self) -> None:
        basis = self._synthetic_basis()
        target = affine_transport_target(
            basis.points_m,
            bias_field_t=0.10,
            gradient_t_per_m=(5.0, 0.0, 0.0),
            center_m=(0.0, 0.0, 0.0),
        )
        result = synthesize_transport_state(
            basis,
            bias_field_t=0.10,
            gradient_t_per_m=(5.0, 0.0, 0.0),
            center_m=(0.0, 0.0, 0.0),
            field_direction=(0.0, 0.0, 1.0),
            regularization=0.0,
            backend="numpy",
            tolerance_t=1e-12,
        )
        np.testing.assert_allclose(result.target_projected_field_t, target)
        np.testing.assert_allclose(result.predicted_projected_field_t, target, atol=1e-12)
        positive_x = np.argmax(basis.points_m[:, 0])
        negative_x = np.argmin(basis.points_m[:, 0])
        self.assertGreater(
            result.predicted_projected_field_t[positive_x],
            result.predicted_projected_field_t[negative_x],
        )


if __name__ == "__main__":
    unittest.main()
