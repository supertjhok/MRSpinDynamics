from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.fields import illustrative_hybrid_epm_array
from spin_dynamics.workflows import (
    build_epm_nonlinear_encoding,
    random_epm_encoding_states,
    run_epm_nonlinear_imaging,
    simple_tissue_phantom,
)


class TissuePhantomTests(unittest.TestCase):
    def test_simple_phantom_contains_off_center_target_and_relaxation_maps(self) -> None:
        phantom = simple_tissue_phantom(12, field_of_view_m=0.040)

        self.assertEqual(phantom.shape, (12, 12))
        self.assertTrue(np.any(phantom.target_mask))
        self.assertTrue(np.all(phantom.proton_density[phantom.target_mask] == 1.0))
        target_points = phantom.points_m[phantom.target_mask.ravel()]
        self.assertGreater(float(np.mean(target_points[:, 0])), 0.0)
        self.assertLess(float(np.mean(target_points[:, 1])), 0.0)

        contrast = phantom.spin_echo_image(
            repetition_time_s=1.2,
            echo_time_s=0.040,
        )
        tissue = phantom.tissue_labels == 1
        self.assertGreater(float(np.mean(contrast[phantom.target_mask])), float(np.mean(contrast[tissue])))
        self.assertTrue(np.all(contrast[phantom.tissue_labels == 0] == 0.0))


class NonlinearEPMImagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phantom = simple_tissue_phantom(10, field_of_view_m=0.040)
        cls.basis = illustrative_hybrid_epm_array().build_field_basis(
            cls.phantom.points_m,
            n_cross=1,
            n_length=5,
        )

    def test_random_states_are_seeded_and_respect_remanence_limits(self) -> None:
        first = random_epm_encoding_states(self.basis, 24, seed=7)
        second = random_epm_encoding_states(self.basis, 24, seed=7)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], 0.0)
        self.assertTrue(
            np.all(np.abs(first) <= self.basis.array.remanence_limits_t[None, :])
        )

    def test_encoding_matches_cached_fields_and_reference_demodulation(self) -> None:
        states = random_epm_encoding_states(self.basis, 16, seed=2)
        encoding = build_epm_nonlinear_encoding(
            self.basis,
            states,
            image_shape=self.phantom.shape,
            phase_encoding_s=250e-6,
        )

        expected = np.stack(
            [self.basis.field_vectors(state)[:, 2] for state in states],
            axis=0,
        )
        np.testing.assert_allclose(encoding.projected_fields_t, expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(
            encoding.phase_rad[:, encoding.reference_point_index],
            0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.abs(encoding.matrix),
            1.0 / np.sqrt(states.shape[0]),
            atol=1e-14,
        )

    def test_noisy_tissue_reconstruction_is_reproducible_and_accurate(self) -> None:
        states = random_epm_encoding_states(self.basis, 160, seed=4)
        encoding = build_epm_nonlinear_encoding(
            self.basis,
            states,
            image_shape=self.phantom.shape,
            phase_encoding_s=300e-6,
        )
        expected = self.phantom.spin_echo_image(
            repetition_time_s=1.2,
            echo_time_s=0.040,
        )

        first = run_epm_nonlinear_imaging(
            encoding,
            expected,
            snr_db=35.0,
            seed=8,
        )
        second = run_epm_nonlinear_imaging(
            encoding,
            expected,
            snr_db=35.0,
            seed=8,
        )

        self.assertTrue(first.converged)
        self.assertLess(first.nrmse, 0.03)
        self.assertLess(encoding.condition_number, 10.0)
        np.testing.assert_array_equal(first.signal, second.signal)
        np.testing.assert_array_equal(first.reconstructed_image, second.reconstructed_image)

    def test_state_limit_violation_is_rejected(self) -> None:
        states = random_epm_encoding_states(self.basis, 8, seed=1)
        states[1, 0] = 1.01 * self.basis.array.remanence_limits_t[0]
        with self.assertRaisesRegex(ValueError, "element limit"):
            build_epm_nonlinear_encoding(
                self.basis,
                states,
                image_shape=self.phantom.shape,
            )


if __name__ == "__main__":
    unittest.main()
