from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.absolute_phase import TAU
from spin_dynamics.workflows import run_matched_diffusion_cpmg, run_tuned_diffusion_cpmg


class DiffusionAbsolutePhaseTests(unittest.TestCase):
    def test_matched_diffusion_cpmg_records_absolute_phase_metadata(self) -> None:
        result = run_matched_diffusion_cpmg(
            num_echoes=4,
            echo_spacing_seconds=800e-6,
            diffusion_time=800e-6,
            numpts=9,
            q_value=20,
            rephase_action="ignore",
            q_stability_action="ignore",
            absolute_phase={
                "rf_frequency_hz": 0.25 / 800e-6,
                "phase_bins": 4,
            },
        )

        metadata = result.absolute_phase
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.phase_bins, 4)
        self.assertEqual(
            metadata.pulse_kind,
            (
                "excitation",
                "excitation",
                "diffusion_refocusing",
                "refocusing",
                "refocusing",
                "refocusing",
                "refocusing",
            ),
        )
        self.assertIsNotNone(metadata.encoding_absolute_phase_rad)
        self.assertIsNotNone(metadata.encoding_matrix_indices)
        self.assertIsNotNone(metadata.refocus_matrix_indices)
        self.assertIsNotNone(metadata.pulse_matrix_phase_rad)
        self.assertIsNotNone(metadata.refocus_pulse_library)
        self.assertEqual(metadata.refocus_absolute_phase_rad.size, 4)
        self.assertEqual(metadata.encoding_absolute_phase_rad.size, 1)
        self.assertEqual(metadata.pulse_matrix_phase_rad.size, 7)
        self.assertLessEqual(metadata.pulse_matrix_count, 2 + 4)
        self.assertGreater(result.echo_integrals.size, 0)

    def test_absolute_phase_perturbation_changes_diffusion_echo_train(self) -> None:
        common = {
            "num_echoes": 4,
            "echo_spacing_seconds": 800e-6,
            "diffusion_time": 800e-6,
            "numpts": 9,
            "q_value": 20,
            "rephase_action": "ignore",
            "q_stability_action": "ignore",
        }
        baseline = run_matched_diffusion_cpmg(**common)
        perturbed = run_matched_diffusion_cpmg(
            **common,
            absolute_phase={
                "rf_frequency_hz": 0.25 / 800e-6,
                "transient_model": {
                    "kind": "sinusoidal",
                    "phase_amplitude_rad": 0.15,
                    "periodicity": "full",
                },
            },
        )

        self.assertIsNotNone(perturbed.absolute_phase)
        self.assertAlmostEqual(
            perturbed.absolute_phase.delta_refocus_phase_rad,
            TAU * 0.25,
        )
        self.assertFalse(np.allclose(perturbed.echo_integrals, baseline.echo_integrals))

    def test_tuned_diffusion_cpmg_probe_shapes_show_absolute_phase_contrast(self) -> None:
        common = {
            "num_echoes": 4,
            "echo_spacing_seconds": 1000e-6,
            "diffusion_time": 1000e-6,
            "numpts": 17,
            "dz": 50e-6,
            "diffusion_coefficient": 0.0,
            "rephase_action": "ignore",
            "num_workers": 1,
        }
        baseline = run_tuned_diffusion_cpmg(**common)
        phase_advanced = run_tuned_diffusion_cpmg(
            **common,
            absolute_phase={
                "rf_frequency_hz": 0.25 / 1000e-6,
                "phase_bins": 4,
            },
        )

        self.assertIsNotNone(phase_advanced.absolute_phase)
        self.assertAlmostEqual(
            phase_advanced.absolute_phase.delta_refocus_phase_cycles,
            0.25,
        )
        rel_delta = np.max(
            np.abs(phase_advanced.echo_integrals - baseline.echo_integrals)
            / np.maximum(np.abs(baseline.echo_integrals), np.finfo(float).tiny)
        )
        self.assertGreater(float(rel_delta), 0.05)


if __name__ == "__main__":
    unittest.main()
