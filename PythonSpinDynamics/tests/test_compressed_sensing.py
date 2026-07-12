import unittest

import numpy as np

from spin_dynamics.analysis.compressed_sensing import (
    adaptive_cs_reconstruction,
    centered_fft2,
    centered_ifft2,
    normalized_root_mean_square_error,
    reconstruct_wavelet_fista,
    variable_density_order,
)
from spin_dynamics.workflows.portable_halbach import (
    PortableHalbachMRIConfig,
    portable_phantom,
    simulate_portable_halbach_mri,
)


class TestCompressedSensing(unittest.TestCase):
    def test_centered_fourier_round_trip(self):
        rng = np.random.default_rng(2)
        image = rng.normal(size=(16, 24)) + 1j * rng.normal(size=(16, 24))
        np.testing.assert_allclose(centered_ifft2(centered_fft2(image)), image)

    def test_wavelet_reconstruction_improves_zero_fill(self):
        image = portable_phantom(32)
        kspace = centered_fft2(image)
        order = variable_density_order(image.shape, seed=3)
        count = int(round(0.32 * image.size))
        flat = np.ravel_multi_index((order[:count, 0], order[:count, 1]), image.shape)
        mask = np.zeros(image.size, dtype=bool)
        mask[flat] = True
        mask = mask.reshape(image.shape)
        zero_fill = centered_ifft2(np.where(mask, kspace, 0.0))
        reconstructed = reconstruct_wavelet_fista(
            kspace, mask, regularization=5e-4, iterations=60
        )
        self.assertLess(
            normalized_root_mean_square_error(image, reconstructed),
            normalized_root_mean_square_error(image, zero_fill),
        )

    def test_held_out_quality_stops_incomplete_acquisition(self):
        image = portable_phantom(32)
        result = adaptive_cs_reconstruction(
            centered_fft2(image),
            batch_size=41,
            min_sampling_fraction=0.32,
            patience=2,
            min_quality_improvement=0.035,
            regularization=5e-4,
            iterations=25,
            seed=4,
        )
        self.assertTrue(result.stopped_early)
        self.assertLess(result.sampling_fractions[-1], 1.0)
        self.assertFalse(np.any(result.reconstruction_mask & result.validation_mask))
        self.assertIn("quality plateaued", result.stop_reason)

    def test_portable_workflow_includes_noise_and_thermal_drift(self):
        config = PortableHalbachMRIConfig(matrix_size=32, reconstruction_iterations=20)
        result = simulate_portable_halbach_mri(config)
        self.assertGreater(result.noise_standard_deviation, 0.0)
        self.assertLess(result.larmor_drift_hz[-1], 0.0)
        self.assertGreater(result.magnet_temperature_k[-1], config.ambient_temperature_k)
        self.assertTrue(result.adaptive.stopped_early)
        self.assertLess(result.reference_nrmse, 0.50)
        self.assertGreater(result.predicted_single_scan_snr, 0.0)
        self.assertGreater(result.receive_coil_q_factor, 0.0)
        self.assertGreater(
            result.receive_coil_copper_q_factor, result.receive_coil_q_factor
        )
        self.assertAlmostEqual(result.receive_coil_q_factor, 128.0, delta=0.5)
        self.assertGreater(result.ferrite_rf_loss_resistance_ohm, 0.0)
        self.assertLess(result.gradient_coil_average_power_w, 0.1)
        axis = np.linspace(
            -0.5 * config.field_of_view_m,
            0.5 * config.field_of_view_m,
            config.matrix_size,
        )
        center = config.matrix_size // 2
        gx = np.gradient(result.gx_field_map_t_per_a, axis, axis)[0]
        gz = np.gradient(result.gz_field_map_t_per_a, axis, axis)[1]
        self.assertAlmostEqual(gx[center, center], 15.0e-3, delta=0.5e-3)
        self.assertAlmostEqual(gz[center, center], 15.0e-3, delta=0.5e-3)
        self.assertGreater(
            np.linalg.norm(result.measured_kspace - result.ideal_kspace), 0.0
        )


if __name__ == "__main__":
    unittest.main()
