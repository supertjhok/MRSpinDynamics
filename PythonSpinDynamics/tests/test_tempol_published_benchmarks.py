"""Frozen published-data provenance, conventions and reduced-model comparison."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from validate_tempol_water import (  # noqa: E402
    compare,
    digitized_points,
    error_metrics,
    load_data,
    profile_model,
    saturation_product_curves,
)


class TempolPublishedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.data = load_data()

    def test_pixel_axis_calibration_and_evidence_classification(self):
        profile = self.data["sezer_profile"]
        e = digitized_points(profile["figure5"])
        s = digitized_points(profile["figure6c"])
        self.assertEqual(e.shape, (31, 2))
        self.assertEqual(s.shape, (19, 2))
        self.assertTrue(np.all(np.diff(e[:, 0]) >= 0))
        self.assertTrue(np.all(e[:, 1] < 0))
        self.assertTrue(np.all((s[:, 1] >= 0) & (s[:, 1] <= 1)))
        calibration = copy.deepcopy(profile["figure5"])
        calibration["points_pixels"] = [[87, 8], [340, 244]]
        assert_allclose(digitized_points(calibration), [[-80, 0], [80, -5]])
        self.assertIn("experimental", profile["figure5"]["evidence_kind"])
        self.assertIn("not_experiment", profile["figure6c"]["evidence_kind"])
        self.assertIsNone(profile["figure5"]["experimental_uncertainty"])
        self.assertEqual(len(self.data["sources"]["sezer2009"]["pdf_sha256"]), 64)

    def test_published_parameters_are_unfitted_and_baseline_is_explicit(self):
        result = compare(self.data)
        self.assertEqual(result["fit_parameters"], [])
        self.assertTrue(result["within_reduction_envelopes"])
        self.assertAlmostEqual(
            result["enhancement_profile"]["rmse"], 0.4884946790981705, delta=1e-10
        )
        self.assertAlmostEqual(
            result["saturation_published_theory"]["rmse"],
            0.016572517088831236,
            delta=1e-11,
        )
        s, epsilon = profile_model(self.data, np.array([-42, 0, 42]))
        profile = self.data["sezer_profile"]
        ratio = 28.0249514242e9 / 42.57747892e6
        assert_allclose(
            epsilon,
            -ratio * profile["coupling_factor"] * profile["leakage_factor"] * s,
            atol=1e-13,
        )
        # No irradiation means epsilon=0, while the baseline-inclusive E is 1.
        _, off = profile_model(self.data, [0], b1_scale=0)
        assert_allclose(off, 0, atol=0)

    def test_full_operator_preserves_legacy_comparison_and_source_linewidths(self):
        result = compare(self.data)
        self.assertAlmostEqual(
            result["model_comparison"]["reduced"]["enhancement"]["rmse"],
            0.4828455827862031,
            delta=1e-10,
        )
        self.assertLess(
            result["saturation_published_theory"]["rmse"],
            result["model_comparison"]["reduced"]["saturation_theory"]["rmse"],
        )
        # Better theory agreement does not imply a better fit to measurements.
        self.assertGreater(
            result["enhancement_profile"]["rmse"],
            result["model_comparison"]["reduced"]["enhancement"]["rmse"],
        )
        diagnostic = result["full_relaxation_diagnostics"]
        assert_allclose(
            diagnostic["t2_diagonal_seconds_ascending_m"],
            self.data["sezer_profile"]["t2e_seconds_by_nitrogen_m_ascending"],
            rtol=0.002,
        )
        self.assertEqual(diagnostic["nuclear_population_r0_rank"], 2)
        self.assertGreater(diagnostic["nuclear_feedback_u_frobenius_per_second"], 1000)

    def test_exchange_improves_profile_without_claiming_measurement_error_agreement(
        self,
    ):
        profile = self.data["sezer_profile"]
        observed = digitized_points(profile["figure5"])
        off = error_metrics(
            observed[:, 1],
            profile_model(self.data, observed[:, 0], exchange_scale=0)[1],
        )
        on = compare(self.data)["enhancement_profile"]
        self.assertLess(on["rmse"], off["rmse"])
        self.assertGreater(
            on["rmse"], profile["figure5"]["digitization_half_width_epsilon"]
        )

    def test_wrong_sign_or_amplitude_is_detected_by_envelope(self):
        broken = copy.deepcopy(self.data)
        broken["sezer_profile"]["coupling_factor"] *= -1
        self.assertFalse(compare(broken)["within_reduction_envelopes"])
        broken = copy.deepcopy(self.data)
        broken["sezer_profile"]["b1_corotating_gauss"] *= 2
        self.assertFalse(compare(broken)["within_reduction_envelopes"])

    def test_cross_study_product_difference_and_reconstructed_curve_labels(self):
        result = compare(self.data)["epr_product_comparison"]
        self.assertAlmostEqual(result["ratio"], 4.322857142857143)
        a, b = saturation_product_curves(self.data, [0, 0.5, 1, 2])
        assert_allclose([a[0], b[0]], 0, atol=0)
        self.assertTrue(np.all(a[1:] < b[1:]))
        self.assertIn(
            "not_raw_curve", self.data["epr_saturation_product"]["evidence_kind"]
        )

    def test_lower_field_closure_is_never_independent_validation(self):
        anchors = compare(self.data)["hofer_factor_closure"]
        assert_allclose(
            [p["epsilon_factor_closure"] for p in anchors],
            [-98.5887200492097, -24.785774854375955],
            atol=1e-10,
        )
        self.assertTrue(all(not p["independent_validation"] for p in anchors))
        self.assertTrue(all(0 < p["leakage_from_nmrd"] < 1 for p in anchors))

    def test_invalid_digitization_axes_rejected(self):
        record = copy.deepcopy(self.data["sezer_profile"]["figure5"])
        record["axis_x_pixels_values"] = [[0, 0], [0, 1]]
        with self.assertRaises(ValueError):
            digitized_points(record)

    def test_committed_report_matches_frozen_data_and_current_model(self):
        recorded = json.loads(
            (
                ROOT / "validation/experimental/tempol_water_benchmark_report.json"
            ).read_text()
        )
        current = compare(self.data)
        self.assertEqual(recorded["data_sha256"], current["data_sha256"])
        for name in ("enhancement_profile", "saturation_published_theory"):
            for key in current[name]:
                self.assertAlmostEqual(
                    recorded[name][key], current[name][key], delta=1e-10
                )


if __name__ == "__main__":
    unittest.main()
