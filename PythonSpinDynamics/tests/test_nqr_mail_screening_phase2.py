"""Phase 2 parcel conservation and field-map invariants."""

import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

PATH = (
    Path(__file__).resolve().parents[1]
    / "studies/nqr_mail_screening/phase2/aperture.py"
)
SPEC = importlib.util.spec_from_file_location("nqr_phase2", PATH)
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


class ParcelTests(unittest.TestCase):
    def test_voxel_mass_pose_and_refinement(self):
        region = MODEL.Region(
            "test", (0.1, 0.002, 0.1), (0.0, 0.0, 0.0), 0.001, 0.5, 293.15, 3.0, 1e-8
        )
        rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        for shape in ((3, 3, 3), (7, 5, 9)):
            points, mass = region.voxels(
                shape, translation=(0.0, 0.0, 0.12), rotation=rotation
            )
            self.assertAlmostEqual(mass.sum(), 0.001)
            np.testing.assert_allclose(
                points.mean(axis=0), (0.0, 0.0, 0.12), atol=1e-15
            )
        with self.assertRaises(ValueError):
            region.voxels(rotation=2 * np.eye(3))

    def test_fields_are_finite_and_refine(self):
        points = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.1]])
        for name, coil in MODEL.candidates(16).items():
            coarse = MODEL.field(coil, points)
            fine = MODEL.field(MODEL.candidates(32)[name], points)
            self.assertTrue(np.all(np.isfinite(coarse)))
            self.assertLess(np.linalg.norm(coarse - fine) / np.linalg.norm(fine), 0.05)

    def test_added_loading_increases_loss_and_reduces_current_at_fixed_carrier(self):
        rows = MODEL.loading_sweep(10e-6, 1.0, 1e-12, 3.3e6)
        self.assertGreater(rows[1]["johnson_psd_v2_hz"], rows[0]["johnson_psd_v2_hz"])
        self.assertLess(
            rows[1]["current_per_drive_volt_a"], rows[0]["current_per_drive_volt_a"]
        )
        self.assertLess(
            rows[3]["resonant_frequency_hz"], rows[0]["resonant_frequency_hz"]
        )
