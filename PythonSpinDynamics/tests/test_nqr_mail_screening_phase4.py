"""Calibration, correlated Bayes updates and physical-time invariants."""

from pathlib import Path  # noqa: E402
import sys
import unittest
import numpy as np
from scipy.special import logsumexp  # noqa: E402

STUDY = Path(__file__).resolve().parents[1] / "studies/nqr_mail_screening/phase4"
sys.path.insert(0, str(STUDY))
from inference import Detector  # noqa: E402
from evaluate import auc, threshold  # noqa: E402

sys.path.remove(str(STUDY))


class Phase4Tests(unittest.TestCase):
    def detector(self):
        nodes = [
            {"id": 0, "path": [0], "cost_s": 0.2},
            {"id": 1, "path": [0, 1], "cost_s": 0.3},
            {"id": 2, "path": [0, 1, 2], "cost_s": 0.3},
        ]
        info = {"nodes": nodes, "sunk_and_exit_cost_s": 2.0}
        means = np.ones((3, 3, 2), complex) * 0.3
        covariance = 0.6 * np.ones((12, 12)) + 0.4 * np.eye(12)
        return Detector(
            info, {"means_v": means}, np.zeros((3, 4)), np.ones((3, 4)), covariance
        )

    def test_sequential_likelihood_equals_joint_correlated_likelihood(self):
        detector = self.detector()
        x = np.arange(12).reshape(3, 4) / 10
        result = detector.run(x, 3.0, adaptive=False)
        mu = detector.templates.reshape(len(detector.templates), 12)
        delta = x.ravel() - mu
        lp = -0.5 * np.einsum(
            "ni,ij,nj->n", delta, np.linalg.inv(detector.covariance), delta
        )
        expected = logsumexp(lp[1:]) - np.log(len(lp) - 1) - lp[0]
        self.assertAlmostEqual(result["score"], expected, places=11)
        self.assertAlmostEqual(result["seconds"], 2.8)

    def test_time_cap_charges_overhead_and_blocks_once(self):
        result = self.detector().run(np.zeros((3, 4)), 2.51, adaptive=False)
        self.assertEqual(result["path"], [0, 1])
        self.assertAlmostEqual(result["seconds"], 2.5)
        empty = self.detector().run(np.zeros((3, 4)), 2.1, adaptive=False)
        self.assertEqual(empty["path"], [])
        self.assertEqual(empty["seconds"], 2.0)

    def test_auc_ties_and_perfect_order(self):
        self.assertEqual(auc(np.zeros(10), np.zeros(10)), 0.5)
        self.assertEqual(auc(np.arange(10), np.arange(10) + 20), 1)
        self.assertEqual(auc(np.arange(10) + 20, np.arange(10)), 0)

    def test_finite_calibration_quantile_is_conservative(self):
        self.assertEqual(threshold(np.arange(19), 0.05), 18)
        self.assertTrue(np.isinf(threshold(np.arange(5), 0.05)))
        rng = np.random.default_rng(1)
        alarms = [rng.normal() > threshold(rng.normal(size=199)) for _ in range(3000)]
        self.assertLess(abs(np.mean(alarms) - 0.05), 0.015)
