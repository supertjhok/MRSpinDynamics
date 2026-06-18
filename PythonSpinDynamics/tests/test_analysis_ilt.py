from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.analysis import (
    Regularization,
    default_regularization_strengths,
    diffusion_kernel,
    estimate_noise_rms_from_snr,
    expected_residual_norm_from_snr,
    invert_d_t2,
    invert_laplace_1d,
    invert_t1,
    invert_t1_t2,
    invert_t2,
    laplace_kernel,
    select_regularization_1d,
    select_regularization_2d,
    t1_kernel,
    t2_kernel,
)


HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class InverseLaplaceTests(unittest.TestCase):
    def test_named_kernels_have_expected_values(self) -> None:
        times = np.array([0.0, 0.002], dtype=np.float64)
        t_axis = np.array([0.001, 0.004], dtype=np.float64)
        b_values = np.array([0.0, 1.0e9], dtype=np.float64)
        d_axis = np.array([0.0, 2.0e-9], dtype=np.float64)

        np.testing.assert_allclose(t2_kernel(times, t_axis)[0], [1.0, 1.0])
        np.testing.assert_allclose(
            t1_kernel(times, t_axis, mode="saturation")[0],
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            t1_kernel(times, t_axis, mode="inversion")[0],
            [-1.0, -1.0],
        )
        np.testing.assert_allclose(diffusion_kernel(b_values, d_axis)[0], [1.0, 1.0])
        np.testing.assert_allclose(
            laplace_kernel(times, t_axis, kind="t2"),
            t2_kernel(times, t_axis),
        )

    def test_unconstrained_t2_inverse_runs_without_scipy(self) -> None:
        echo_times = np.linspace(0.001, 0.02, 12)
        t2_axis = np.logspace(-3, -1, 18)
        true_distribution = np.zeros_like(t2_axis)
        true_distribution[8] = 1.0
        signal = t2_kernel(echo_times, t2_axis) @ true_distribution

        result = invert_laplace_1d(
            signal,
            echo_times,
            t2_axis,
            kernel="t2",
            regularization=Regularization(strength=1e-3, order=0),
            nonnegative=False,
        )

        self.assertEqual(result.distribution.shape, t2_axis.shape)
        np.testing.assert_allclose(result.prediction, signal, rtol=0.05, atol=0.05)

    def test_snr_noise_target_helpers(self) -> None:
        data = np.array([3.0, 4.0], dtype=np.float64)
        snr = 4.0
        observed_rms = np.sqrt(np.mean(data**2))
        expected_noise = observed_rms / np.sqrt(snr**2 + 1.0)

        self.assertAlmostEqual(estimate_noise_rms_from_snr(data, snr), expected_noise)
        self.assertAlmostEqual(
            expected_residual_norm_from_snr(data, snr),
            expected_noise * np.sqrt(data.size),
        )
        strengths = default_regularization_strengths(1e-6, 1e-2, 5)
        self.assertEqual(strengths.size, 5)
        self.assertAlmostEqual(float(strengths[0]), 1e-6)
        self.assertAlmostEqual(float(strengths[-1]), 1e-2)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for non-negative ILT")
    def test_t2_inverse_recovers_dominant_peak(self) -> None:
        echo_times = np.linspace(0.0005, 0.09, 36)
        t2_axis = np.logspace(-4, -1, 50)
        true_distribution = np.zeros_like(t2_axis)
        dominant_index = int(np.argmin(np.abs(t2_axis - 0.008)))
        shoulder_index = int(np.argmin(np.abs(t2_axis - 0.035)))
        true_distribution[dominant_index] = 1.0
        true_distribution[shoulder_index] = 0.35
        signal = t2_kernel(echo_times, t2_axis) @ true_distribution

        result = invert_t2(
            signal,
            echo_times,
            t2_axis,
            regularization=Regularization(strength=1e-4, order=2),
        )
        recovered_t2 = float(t2_axis[int(np.argmax(result.distribution))])

        self.assertLess(abs(np.log(recovered_t2 / t2_axis[dominant_index])), 0.35)
        self.assertLess(result.residual_norm / np.linalg.norm(signal), 0.05)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for non-negative ILT")
    def test_t1_inversion_recovery_inverse_recovers_peak(self) -> None:
        recovery_times = np.linspace(0.0003, 0.03, 32)
        t1_axis = np.logspace(-4, -1, 44)
        true_index = int(np.argmin(np.abs(t1_axis - 0.006)))
        true_distribution = np.zeros_like(t1_axis)
        true_distribution[true_index] = 1.0
        signal = (
            t1_kernel(recovery_times, t1_axis, mode="inversion")
            @ true_distribution
        )

        result = invert_t1(
            signal,
            recovery_times,
            t1_axis,
            mode="inversion",
            regularization=Regularization(strength=1e-5, order=2),
        )
        recovered_t1 = float(t1_axis[int(np.argmax(result.distribution))])

        self.assertLess(abs(np.log(recovered_t1 / t1_axis[true_index])), 0.35)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for non-negative ILT")
    def test_snr_based_regularization_selection_matches_noise_target(self) -> None:
        rng = np.random.default_rng(123)
        echo_times = np.linspace(0.0005, 0.08, 28)
        t2_axis = np.logspace(-4, -1, 36)
        true_distribution = np.zeros_like(t2_axis)
        true_distribution[int(np.argmin(np.abs(t2_axis - 0.009)))] = 1.0
        clean = t2_kernel(echo_times, t2_axis) @ true_distribution
        snr = 30.0
        sigma = np.sqrt(np.mean(clean**2)) / snr
        noisy = clean + rng.normal(scale=sigma, size=clean.shape)

        selection = select_regularization_1d(
            noisy,
            echo_times,
            t2_axis,
            snr=snr,
            kernel="t2",
            strengths=np.logspace(-7, -1, 13),
        )

        self.assertEqual(len(selection.candidates), 13)
        self.assertIn(selection.selected_strength, np.logspace(-7, -1, 13))
        self.assertLess(
            selection.result.residual_norm,
            2.5 * selection.target_residual_norm,
        )
        self.assertGreater(
            selection.result.residual_norm,
            0.25 * selection.target_residual_norm,
        )

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for non-negative ILT")
    def test_t1_t2_inverse_recovers_2d_peak(self) -> None:
        recovery_times = np.linspace(0.0005, 0.04, 18)
        echo_times = np.linspace(0.0005, 0.05, 16)
        t1_axis = np.logspace(-4, -1, 18)
        t2_axis = np.logspace(-4, -1, 16)
        t1_index = int(np.argmin(np.abs(t1_axis - 0.007)))
        t2_index = int(np.argmin(np.abs(t2_axis - 0.012)))
        true_distribution = np.zeros((t1_axis.size, t2_axis.size), dtype=np.float64)
        true_distribution[t1_index, t2_index] = 1.0
        data = (
            t1_kernel(recovery_times, t1_axis, mode="saturation")
            @ true_distribution
            @ t2_kernel(echo_times, t2_axis).T
        )

        result = invert_t1_t2(
            data,
            recovery_times,
            echo_times,
            t1_axis,
            t2_axis,
            regularization=(5e-4, 5e-4),
        )
        recovered = np.unravel_index(
            np.argmax(result.distribution),
            result.distribution.shape,
        )

        self.assertLess(abs(np.log(t1_axis[recovered[0]] / t1_axis[t1_index])), 0.55)
        self.assertLess(abs(np.log(t2_axis[recovered[1]] / t2_axis[t2_index])), 0.55)
        self.assertLess(result.residual_norm / np.linalg.norm(data), 0.08)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for non-negative ILT")
    def test_2d_regularization_selection_returns_axis_strengths(self) -> None:
        rng = np.random.default_rng(456)
        recovery_times = np.linspace(0.0005, 0.035, 12)
        echo_times = np.linspace(0.0005, 0.04, 11)
        t1_axis = np.logspace(-4, -1, 12)
        t2_axis = np.logspace(-4, -1, 10)
        true_distribution = np.zeros((t1_axis.size, t2_axis.size), dtype=np.float64)
        true_distribution[5, 4] = 1.0
        clean = (
            t1_kernel(recovery_times, t1_axis, mode="saturation")
            @ true_distribution
            @ t2_kernel(echo_times, t2_axis).T
        )
        snr = 25.0
        sigma = np.sqrt(np.mean(clean**2)) / snr
        noisy = clean + rng.normal(scale=sigma, size=clean.shape)

        selection = select_regularization_2d(
            noisy,
            recovery_times,
            echo_times,
            t1_axis,
            t2_axis,
            snr=snr,
            kernel1="t1",
            kernel2="t2",
            strengths=np.logspace(-7, -3, 5),
            axis_strength_ratio=(1.0, 2.0),
        )

        self.assertEqual(len(selection.candidates), 5)
        self.assertAlmostEqual(
            selection.selected_regularization[1].strength,
            2.0 * selection.selected_regularization[0].strength,
        )
        self.assertEqual(selection.result.distribution.shape, true_distribution.shape)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for non-negative ILT")
    def test_d_t2_inverse_recovers_2d_peak(self) -> None:
        b_values = np.linspace(0.0, 3.0e9, 14)
        echo_times = np.linspace(0.0005, 0.05, 14)
        d_axis = np.linspace(0.2e-9, 2.5e-9, 15)
        t2_axis = np.logspace(-4, -1, 15)
        d_index = int(np.argmin(np.abs(d_axis - 1.2e-9)))
        t2_index = int(np.argmin(np.abs(t2_axis - 0.01)))
        true_distribution = np.zeros((d_axis.size, t2_axis.size), dtype=np.float64)
        true_distribution[d_index, t2_index] = 1.0
        data = (
            diffusion_kernel(b_values, d_axis)
            @ true_distribution
            @ t2_kernel(echo_times, t2_axis).T
        )

        result = invert_d_t2(
            data,
            b_values,
            echo_times,
            d_axis,
            t2_axis,
            regularization=(1e-4, 5e-4),
        )
        recovered = np.unravel_index(
            np.argmax(result.distribution),
            result.distribution.shape,
        )

        self.assertLess(abs(d_axis[recovered[0]] - d_axis[d_index]), 0.35e-9)
        self.assertLess(abs(np.log(t2_axis[recovered[1]] / t2_axis[t2_index])), 0.55)


if __name__ == "__main__":
    unittest.main()
