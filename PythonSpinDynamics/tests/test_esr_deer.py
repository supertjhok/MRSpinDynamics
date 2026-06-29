from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.esr import (  # noqa: E402
    deer_dipolar_spectrum,
    deer_pair_trace,
    deer_pair_trace_quantum,
    deer_powder_kernel,
    dipolar_angular_frequency_hz,
    dipolar_constant_hz_nm3,
    dipolar_frequency_hz,
    distance_from_dipolar_frequency_nm,
    extract_distance_distribution,
    gaussian_distance_distribution,
    secular_dipolar_hamiltonian,
    simulate_deer,
)

try:
    import scipy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without SciPy
    HAVE_SCIPY = False

MAGIC_ANGLE_RAD = np.arccos(np.sqrt(1.0 / 3.0))


class DipolarCouplingTests(unittest.TestCase):
    def test_dipolar_constant_matches_canonical_value(self) -> None:
        # Two free-electron g values give the textbook 52.04 MHz nm^3.
        self.assertAlmostEqual(dipolar_constant_hz_nm3() / 1e6, 52.04, places=1)

    def test_dipolar_frequency_scales_as_inverse_cube(self) -> None:
        nu_1 = dipolar_frequency_hz(2.0)
        nu_2 = dipolar_frequency_hz(4.0)

        self.assertAlmostEqual(nu_1 / nu_2, 8.0, places=9)
        self.assertAlmostEqual(dipolar_frequency_hz(1.0), dipolar_constant_hz_nm3())

    def test_distance_frequency_round_trip(self) -> None:
        distance = 3.2
        frequency = dipolar_frequency_hz(distance)

        self.assertAlmostEqual(
            distance_from_dipolar_frequency_nm(frequency), distance, places=9
        )

    def test_angular_dependence_perpendicular_parallel_and_magic_angle(self) -> None:
        nu_perp = dipolar_frequency_hz(3.0)

        self.assertAlmostEqual(
            dipolar_angular_frequency_hz(3.0, np.pi / 2.0), nu_perp, places=3
        )
        self.assertAlmostEqual(
            dipolar_angular_frequency_hz(3.0, 0.0), -2.0 * nu_perp, places=3
        )
        self.assertAlmostEqual(
            dipolar_angular_frequency_hz(3.0, MAGIC_ANGLE_RAD), 0.0, places=3
        )

    def test_secular_hamiltonian_is_hermitian_diagonal_szsz(self) -> None:
        hamiltonian = secular_dipolar_hamiltonian(2.5, 0.7)

        np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12)
        # Pure S_zA S_zB coupling is diagonal in the product basis.
        off_diagonal = hamiltonian - np.diag(np.diag(hamiltonian))
        np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-12)

    def test_dipolar_frequency_rejects_nonpositive_distance(self) -> None:
        with self.assertRaises(ValueError):
            dipolar_frequency_hz(0.0)
        with self.assertRaises(ValueError):
            dipolar_frequency_hz(-1.0)


class DeerKernelTests(unittest.TestCase):
    def test_single_pair_trace_oscillates_at_dipolar_frequency(self) -> None:
        distance, theta = 2.5, 0.6
        nu_dd = dipolar_angular_frequency_hz(distance, theta)
        period = 1.0 / abs(nu_dd)
        times = np.array([0.0, 0.25 * period, 0.5 * period, period])

        trace = deer_pair_trace(times, distance, theta, lambda_depth=1.0)

        self.assertAlmostEqual(trace[0], 1.0, places=12)  # F(0) = 1
        self.assertAlmostEqual(trace[1], 0.0, places=9)  # cos = 0 -> 1 - lambda
        self.assertAlmostEqual(trace[2], -1.0, places=9)  # cos = -1 -> 1 - 2 lambda
        self.assertAlmostEqual(trace[3], 1.0, places=9)  # full period

    def test_lambda_sets_modulation_depth(self) -> None:
        distance, theta = 2.5, 0.6
        nu_dd = dipolar_angular_frequency_hz(distance, theta)
        half_period_time = 0.5 / abs(nu_dd)

        trace = deer_pair_trace([half_period_time], distance, theta, lambda_depth=0.3)

        # cos = -1 -> F = 1 - 2 lambda
        self.assertAlmostEqual(trace[0], 1.0 - 2.0 * 0.3, places=9)

    def test_magic_angle_pair_has_no_modulation(self) -> None:
        times = np.linspace(0, 3.0e-6, 50)

        trace = deer_pair_trace(times, 2.5, MAGIC_ANGLE_RAD, lambda_depth=1.0)

        np.testing.assert_allclose(trace, 1.0, atol=1e-9)

    def test_powder_kernel_starts_at_unity_and_is_bounded(self) -> None:
        times = np.linspace(0, 3.0e-6, 120)
        distances = np.array([2.0, 3.0, 4.0])
        lam = 0.4

        kernel = deer_powder_kernel(times, distances, lambda_depth=lam)

        np.testing.assert_allclose(kernel.matrix[0, :], 1.0, atol=1e-9)
        self.assertLessEqual(kernel.matrix.max(), 1.0 + 1e-9)
        self.assertGreaterEqual(kernel.matrix.min(), 1.0 - 2.0 * lam - 1e-9)

    def test_powder_spectrum_singularity_near_perpendicular_frequency(self) -> None:
        times = np.linspace(0, 4.0e-6, 1200)
        distance = 3.0

        kernel = deer_powder_kernel(times, [distance], lambda_depth=1.0)
        frequencies, spectrum = deer_dipolar_spectrum(times, kernel.matrix[:, 0])
        # The Pake singularity sits at nu_perp; mask the near-DC envelope band
        # (the finite-window zero-frequency integral) before locating it.
        band = frequencies > 0.6e6
        peak = frequencies[band][int(np.argmax(spectrum[band]))]

        self.assertAlmostEqual(peak, dipolar_frequency_hz(distance), delta=0.1e6)


class DeerQuantumValidationTests(unittest.TestCase):
    """The density-matrix simulation must reproduce the analytic kernel."""

    def test_quantum_matches_analytic_single_pair_full_pump(self) -> None:
        times = np.linspace(0, 2.0e-6, 41)
        for theta in (0.0, np.pi / 2.0, 0.6, MAGIC_ANGLE_RAD):
            analytic = deer_pair_trace(times, 2.5, theta, lambda_depth=1.0)
            quantum = deer_pair_trace_quantum(
                times,
                2.5,
                theta,
                pump_flip_rad=np.pi,
                observer_offset_hz=7.3e6,
            )
            np.testing.assert_allclose(quantum, analytic, atol=1e-9)

    def test_quantum_partial_pump_gives_sin_squared_modulation_depth(self) -> None:
        times = np.linspace(0, 2.0e-6, 41)
        beta = np.pi / 3.0
        lam = np.sin(beta / 2.0) ** 2

        analytic = deer_pair_trace(times, 2.5, 0.6, lambda_depth=lam)
        quantum = deer_pair_trace_quantum(times, 2.5, 0.6, pump_flip_rad=beta)

        np.testing.assert_allclose(quantum, analytic, atol=1e-9)

    def test_quantum_echo_refocuses_observer_offset(self) -> None:
        times = np.linspace(0, 1.5e-6, 31)

        low = deer_pair_trace_quantum(times, 2.5, 0.6, observer_offset_hz=1.0e6)
        high = deer_pair_trace_quantum(times, 2.5, 0.6, observer_offset_hz=40.0e6)

        np.testing.assert_allclose(low, high, atol=1e-9)


class DeerForwardModelTests(unittest.TestCase):
    def test_shorter_distance_modulates_faster(self) -> None:
        times = np.linspace(0, 2.0e-6, 200)
        distances = np.linspace(1.5, 5.0, 60)

        short = simulate_deer(
            times, distances, gaussian_distance_distribution(distances, 2.0, 0.1)
        )
        long = simulate_deer(
            times, distances, gaussian_distance_distribution(distances, 4.0, 0.1)
        )

        # The shorter distance reaches its first minimum sooner.
        self.assertLess(times[np.argmin(short)], times[np.argmin(long)])

    def test_distribution_is_normalized_internally(self) -> None:
        times = np.linspace(0, 1.0e-6, 20)
        distances = np.linspace(1.5, 5.0, 40)
        weights = gaussian_distance_distribution(distances, 3.0, 0.3)

        scaled = simulate_deer(times, distances, 7.5 * weights)
        unit = simulate_deer(times, distances, weights)

        np.testing.assert_allclose(scaled, unit, atol=1e-12)


class DeerInverseTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_SCIPY, "distance recovery needs SciPy NNLS")
    def test_recovers_gaussian_distance_distribution(self) -> None:
        times = np.linspace(0, 2.5e-6, 100)
        distances = np.linspace(1.5, 4.0, 60)
        truth = gaussian_distance_distribution(distances, 2.5, 0.2)
        form_factor = simulate_deer(times, distances, truth, lambda_depth=0.35)

        result = extract_distance_distribution(
            times,
            form_factor,
            distances,
            lambda_depth=0.35,
            snr=500.0,
        )
        recovered = result.distribution / np.sum(result.distribution)
        recovered_mean = float(np.sum(distances * recovered))

        # The recovered distribution should be concentrated near the true mean.
        self.assertAlmostEqual(recovered_mean, 2.5, delta=0.15)
        recovered_sigma = float(
            np.sqrt(np.sum(recovered * (distances - recovered_mean) ** 2))
        )
        self.assertLess(recovered_sigma, 0.6)
        # Fit explains the data to within the discrepancy-principle target.
        self.assertLess(result.residual_norm, 5e-2)


class DeerValidationErrorTests(unittest.TestCase):
    def test_lambda_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            deer_pair_trace([0.0, 1e-6], 2.5, 0.6, lambda_depth=1.5)

    def test_empty_distances_raises(self) -> None:
        with self.assertRaises(ValueError):
            deer_powder_kernel([0.0, 1e-6], [])

    def test_distribution_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            simulate_deer([0.0, 1e-6], [2.0, 3.0], [1.0, 1.0, 1.0])

    def test_spectrum_requires_uniform_time_axis(self) -> None:
        times = np.array([0.0, 1e-7, 3e-7])  # non-uniform
        with self.assertRaises(ValueError):
            deer_dipolar_spectrum(times, np.ones_like(times))

    def test_pump_time_outside_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            deer_pair_trace_quantum([3.0e-6], 2.5, 0.6, tau2_seconds=2.0e-6)


if __name__ == "__main__":
    unittest.main()
