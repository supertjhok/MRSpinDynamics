from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.absolute_phase import PulseShapeLibrary
from spin_dynamics.pulse_diagnostics import (
    build_probe_pulse_shape_library,
    solve_probe_pulse_shape,
    solve_probe_pulse_shape_sweep,
)


class PulseDiagnosticsTests(unittest.TestCase):
    def test_solve_probe_pulse_shape_returns_drive_metrics(self) -> None:
        shape = solve_probe_pulse_shape(
            probe="tuned",
            absolute_phase_rad=np.pi / 4,
            numpts=9,
        )

        self.assertEqual(shape.probe, "tuned")
        self.assertEqual(shape.pulse_kind, "refocusing")
        self.assertGreater(shape.time_seconds.size, 1)
        self.assertEqual(shape.drive.shape, shape.amplitude.shape)
        self.assertGreater(shape.peak_amplitude, 0.0)
        self.assertGreater(shape.rms_amplitude, 0.0)
        self.assertGreaterEqual(shape.quadrature_energy_fraction, 0.0)
        self.assertLessEqual(shape.quadrature_energy_fraction, 1.0)
        np.testing.assert_allclose(
            shape.to_pulse_shape().amplitude,
            shape.amplitude,
        )

    def test_tuned_probe_refocusing_shape_is_pi_periodic(self) -> None:
        first = solve_probe_pulse_shape(
            probe="tuned",
            absolute_phase_rad=0.25,
            numpts=9,
        )
        second = solve_probe_pulse_shape(
            probe="tuned",
            absolute_phase_rad=0.25 + np.pi,
            numpts=9,
        )

        np.testing.assert_allclose(first.amplitude, second.amplitude)
        np.testing.assert_allclose(
            np.exp(1j * first.phase),
            np.exp(1j * second.phase),
        )

    def test_probe_pulse_shape_sweep_exports_library(self) -> None:
        phases = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)

        sweep = solve_probe_pulse_shape_sweep(
            probe="tuned",
            absolute_phase_rad=phases,
            numpts=9,
        )

        self.assertEqual(len(sweep.shapes), 4)
        self.assertIsInstance(sweep.pulse_shape_library, PulseShapeLibrary)
        np.testing.assert_allclose(sweep.absolute_phase_rad, phases)
        self.assertEqual(
            len(sweep.pulse_shape_library.shapes["refocusing"]),
            4,
        )
        library_shape = sweep.pulse_shape_library.shape("refocusing", phases[1])
        np.testing.assert_allclose(library_shape.amplitude, sweep.shapes[1].amplitude)

    def test_build_probe_pulse_shape_library_returns_library_only(self) -> None:
        library = build_probe_pulse_shape_library(
            probe="untuned",
            absolute_phase_rad=np.array([0.0, np.pi]),
            numpts=9,
        )

        self.assertIsInstance(library, PulseShapeLibrary)
        self.assertEqual(len(library.shapes["refocusing"]), 2)


if __name__ == "__main__":
    unittest.main()
