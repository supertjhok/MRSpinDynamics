from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    sample_linear_rf_pulse,
    simulate_lab_frame_rf,
)


class NQRLabFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(
            spin=1.0,
            quadrupole_frequency_hz=100.0e3,
            eta=0.2,
            gamma_hz_per_t=3.0e6,
        )

    def test_zero_rf_keeps_full_hamiltonian_equilibrium_stationary(self) -> None:
        durations = np.full(20, 1.0e-6)
        fields = np.zeros((20, 3))

        result = simulate_lab_frame_rf(
            self.site,
            [0.0, 0.0, 0.02],
            durations,
            fields,
        )

        for density in result.density_matrices:
            np.testing.assert_allclose(density, result.density_matrices[0], atol=1e-14)

    def test_arbitrary_rf_waveform_preserves_trace_and_hermiticity(self) -> None:
        durations, fields = sample_linear_rf_pulse(
            40.0e-6,
            0.1e-6,
            1.0e-3,
            100.0e3,
            direction_pas=(1.0, 1.0, 0.0),
        )

        result = simulate_lab_frame_rf(
            self.site,
            [0.0, 0.0, 0.0],
            durations,
            fields,
        )

        traces = np.trace(result.density_matrices, axis1=1, axis2=2)
        np.testing.assert_allclose(traces, 1.0, atol=1e-12)
        np.testing.assert_allclose(
            result.density_matrices,
            result.density_matrices.conj().transpose(0, 2, 1),
            atol=1e-13,
        )
        self.assertGreater(
            np.linalg.norm(result.density_matrices[-1] - result.density_matrices[0]),
            1e-9,
        )

    def test_resonant_pulse_converges_when_carrier_sampling_is_refined(self) -> None:
        common = dict(
            duration_seconds=40.0e-6,
            amplitude_tesla=1.0e-3,
            carrier_hz=100.0e3,
        )
        coarse_duration, coarse_fields = sample_linear_rf_pulse(
            time_step_seconds=0.1e-6,
            **common,
        )
        fine_duration, fine_fields = sample_linear_rf_pulse(
            time_step_seconds=0.05e-6,
            **common,
        )

        coarse = simulate_lab_frame_rf(
            self.site,
            [0.0, 0.0, 0.0],
            coarse_duration,
            coarse_fields,
        )
        fine = simulate_lab_frame_rf(
            self.site,
            [0.0, 0.0, 0.0],
            fine_duration,
            fine_fields,
        )

        np.testing.assert_allclose(
            coarse.density_matrices[-1],
            fine.density_matrices[-1],
            atol=2e-7,
        )


if __name__ == "__main__":
    unittest.main()
