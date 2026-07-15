"""Static geometry and field validation for electropermanent magnets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.electropermanent import (  # noqa: E402
    ALNICO5_AC500,
    ALNICO5_FEMM_2019,
    ElectropermanentRod,
    EvidenceRecord,
    RemanenceState,
    close_packed_rod_bundle,
    electropermanent_field,
    finite_cylinder_on_axis_field,
    hexagonal_rod_offsets,
    sample_electropermanent_field,
    variable_field_nmr_rod,
    weinberg_37_rod_bundle,
)
from spin_dynamics.fields.magnetostatics import GAMMA_PROTON, MU0  # noqa: E402


class ElectropermanentMaterialTests(unittest.TestCase):
    def test_local_material_presets_retain_provenance_and_femm_curve(self) -> None:
        self.assertAlmostEqual(ALNICO5_AC500.remanence_t, 1.27)
        self.assertAlmostEqual(ALNICO5_AC500.coercivity_a_per_m, 50929.6, delta=1.0)
        self.assertEqual(ALNICO5_AC500.evidence[0].classification, "specified")
        self.assertEqual(len(ALNICO5_FEMM_2019.bh_curve), 26)
        self.assertEqual(
            ALNICO5_FEMM_2019.bh_curve[-1],
            (1.2544, 50988.028),
        )
        self.assertEqual(ALNICO5_FEMM_2019.evidence[0].classification, "simulated")

    def test_evidence_and_state_validation_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceRecord("", "measured")
        with self.assertRaises(ValueError):
            RemanenceState(np.nan)
        with self.assertRaises(ValueError):
            RemanenceState(0.1, uncertainty_t=-1.0)


class ElectropermanentRodFieldTests(unittest.TestCase):
    def test_published_rod_matches_reported_surface_field_scale(self) -> None:
        rod = variable_field_nmr_rod()
        position = 0.5 * rod.length_m + 1.0e-3
        field = finite_cylinder_on_axis_field(position, rod)
        self.assertAlmostEqual(field, 0.15, delta=0.005)
        self.assertEqual(rod.coil.turns, 50)
        self.assertEqual(rod.coil.wire_gauge_awg, 14)
        self.assertAlmostEqual(rod.coil.inductance_h, 25.0e-6)
        self.assertEqual(rod.state.evidence[0].classification, "inferred")

    def test_cubature_converges_to_exact_on_axis_cylinder_field(self) -> None:
        rod = variable_field_nmr_rod()
        axial_position = 0.5 * rod.length_m + 0.05
        point = np.asarray([[0.0, 0.0, axial_position]])
        numerical = rod.field_vectors(point, n_cross=11, n_length=61)[0, 2]
        exact = finite_cylinder_on_axis_field(axial_position, rod)
        self.assertAlmostEqual(numerical, exact, delta=0.015 * abs(exact))

    def test_arbitrary_axis_rotates_geometry_and_field(self) -> None:
        rod_z = variable_field_nmr_rod(axis=(0.0, 0.0, 1.0))
        rod_x = variable_field_nmr_rod(axis=(1.0, 0.0, 0.0))
        distance = 0.5 * rod_z.length_m + 0.04
        field_z = rod_z.field_vectors(
            np.asarray([[0.0, 0.0, distance]]),
            n_cross=7,
            n_length=31,
        )[0]
        field_x = rod_x.field_vectors(
            np.asarray([[distance, 0.0, 0.0]]),
            n_cross=7,
            n_length=31,
        )[0]
        np.testing.assert_allclose(field_x, field_z[[2, 1, 0]], rtol=1e-12, atol=1e-15)
        self.assertGreater(field_x[0], 0.0)

    def test_far_field_matches_magnetic_dipole_limit(self) -> None:
        rod = variable_field_nmr_rod()
        distance = 2.0
        field = rod.field_vectors(
            np.asarray([[0.0, 0.0, distance]]),
            n_cross=3,
            n_length=9,
        )[0, 2]
        moment = rod.magnetic_moment_am2[2]
        expected = MU0 * moment / (2.0 * np.pi * distance**3)
        self.assertAlmostEqual(field, expected, delta=0.005 * expected)

    def test_reversing_retained_state_reverses_field(self) -> None:
        positive = variable_field_nmr_rod()
        negative = positive.with_state(
            RemanenceState(-positive.state.remanence_t, branch="partial")
        )
        point = np.asarray([[0.01, 0.0, 0.12]])
        b_positive = electropermanent_field(
            point,
            (positive,),
            n_cross=5,
            n_length=21,
        )
        b_negative = electropermanent_field(
            point,
            (negative,),
            n_cross=5,
            n_length=21,
        )
        np.testing.assert_allclose(b_negative, -b_positive, rtol=1e-13, atol=1e-15)


class ElectropermanentBundleTests(unittest.TestCase):
    def test_hexagonal_37_rod_geometry_has_three_complete_rings(self) -> None:
        radius = 0.5 * 0.125 * 0.0254
        offsets = hexagonal_rod_offsets(37, 2.0 * radius)
        self.assertEqual(offsets.shape, (37, 3))
        np.testing.assert_allclose(offsets[0], 0.0, atol=1e-15)
        radial = np.linalg.norm(offsets[:, :2], axis=1)
        self.assertAlmostEqual(float(radial.max()), 6.0 * radius, delta=1e-12)
        unique = np.unique(np.round(offsets, decimals=12), axis=0)
        self.assertEqual(unique.shape[0], 37)

    def test_local_bundle_preset_preserves_documented_geometry(self) -> None:
        bundle = weinberg_37_rod_bundle()
        self.assertEqual(len(bundle.rods), 37)
        self.assertAlmostEqual(bundle.rods[0].radius_m, 0.0015875)
        self.assertAlmostEqual(bundle.rods[0].length_m, 0.1016)
        self.assertAlmostEqual(bundle.outer_radius_m, 7.0 * 0.0015875)
        self.assertEqual(bundle.coil.turns, 60)
        self.assertEqual(bundle.coil.wire_gauge_awg, 16)
        self.assertAlmostEqual(bundle.coil.inductance_h, 50.0e-6)
        self.assertEqual(bundle.evidence[0].classification, "specified")
        self.assertEqual(bundle.rods[0].state.remanence_t, 0.0)

    def test_equivalent_cylinder_preserves_volume_and_moment(self) -> None:
        state = RemanenceState(0.42, branch="partial")
        bundle = weinberg_37_rod_bundle(state=state)
        equivalent = bundle.equivalent_cylinder()
        self.assertAlmostEqual(equivalent.volume_m3, bundle.volume_m3)
        np.testing.assert_allclose(
            equivalent.magnetic_moment_am2,
            bundle.magnetic_moment_am2,
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertAlmostEqual(
            equivalent.radius_m,
            np.sqrt(37.0) * bundle.rods[0].radius_m,
        )

    def test_generic_bundle_builder_supports_rotated_axis(self) -> None:
        state = RemanenceState(0.2, branch="partial")
        bundle = close_packed_rod_bundle(
            7,
            rod_radius_m=1.0e-3,
            rod_length_m=0.02,
            axis=(1.0, 0.0, 0.0),
            state=state,
        )
        centers = np.asarray([rod.center_m for rod in bundle.rods])
        np.testing.assert_allclose(centers[:, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(bundle.axis, (1.0, 0.0, 0.0), atol=1e-15)


class ElectropermanentWorkflowTests(unittest.TestCase):
    def test_sampled_maps_adapt_to_motion_and_spatial_workflows(self) -> None:
        rod = ElectropermanentRod(
            center_m=(0.0, 0.0, -0.06),
            axis=(0.0, 0.0, 1.0),
            length_m=0.04,
            radius_m=0.006,
            material=ALNICO5_AC500,
            state=RemanenceState(0.3, branch="partial"),
        )
        x_axis = np.linspace(-0.01, 0.01, 5)
        z_axis = np.linspace(0.0, 0.02, 7)
        maps = sample_electropermanent_field(
            (x_axis, z_axis),
            (rod,),
            n_cross=5,
            n_length=15,
        )
        self.assertEqual(maps.b0_vector.shape, (5, 7, 3))
        self.assertEqual(maps.b0_projected.shape, (5, 7))
        self.assertEqual(maps.b0_gradient.shape, (5, 7))
        self.assertTrue(np.all(np.isfinite(maps.larmor_hz)))
        center = maps.center_field_t
        center_index = (
            int(np.argmin(np.abs(x_axis))),
            int(np.argmin(np.abs(z_axis))),
        )
        self.assertAlmostEqual(
            maps.larmor_hz[center_index],
            GAMMA_PROTON * abs(center) / (2.0 * np.pi),
        )

        motion = maps.to_motion_field_maps()
        spatial = maps.to_spatial_field_maps(t1_map=0.8, t2_map=0.05)
        self.assertEqual(motion.domain.shape, (5, 7))
        self.assertEqual(spatial.domain.shape, (5, 7))
        self.assertAlmostEqual(motion.b0_map[center_index], 0.0)
        self.assertAlmostEqual(spatial.b0_map[center_index], 0.0)
        np.testing.assert_allclose(motion.b0_map, spatial.b0_map)


if __name__ == "__main__":
    unittest.main()
