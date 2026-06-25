from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.sequences.cpmg import (  # noqa: E402
    cpmg_pulse_times,
    dephasing_filter_function,
    interval_durations,
    toggling_frame_integral,
    udd_pulse_times,
)


class UDDSequenceTests(unittest.TestCase):
    def test_udd_pulse_times_follow_uhrig_formula(self) -> None:
        duration = 2.0
        count = 4
        pulse_index = np.arange(1, count + 1, dtype=np.float64)
        expected = duration * np.sin(np.pi * pulse_index / (2 * count + 2)) ** 2

        np.testing.assert_allclose(udd_pulse_times(count, duration), expected)

    def test_cpmg_pulse_times_have_half_edge_intervals(self) -> None:
        times = cpmg_pulse_times(4, 1.0)

        np.testing.assert_allclose(times, [0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(
            interval_durations(times, 1.0),
            [0.125, 0.25, 0.25, 0.25, 0.125],
        )

    def test_zero_pulses_return_empty_schedule(self) -> None:
        self.assertEqual(udd_pulse_times(0, 1.0).shape, (0,))
        self.assertEqual(cpmg_pulse_times(0, 1.0).shape, (0,))
        np.testing.assert_allclose(interval_durations([], 1.0), [1.0])

    def test_toggling_integral_handles_zero_frequency(self) -> None:
        no_pulses = toggling_frame_integral(0.0, [], 1.5)
        balanced = toggling_frame_integral(0.0, cpmg_pulse_times(3, 1.5), 1.5)

        self.assertEqual(no_pulses, 1.5 + 0j)
        self.assertAlmostEqual(abs(balanced), 0.0, places=14)

    def test_udd_suppresses_low_frequency_more_than_cpmg(self) -> None:
        duration = 1.0
        count = 4
        low_omega = 0.1 / duration
        udd_filter = dephasing_filter_function(
            low_omega,
            udd_pulse_times(count, duration),
            duration,
        )
        cpmg_filter = dephasing_filter_function(
            low_omega,
            cpmg_pulse_times(count, duration),
            duration,
        )

        self.assertLess(udd_filter, 1e-4 * cpmg_filter)

    def test_invalid_inputs_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            udd_pulse_times(-1, 1.0)
        with self.assertRaises(ValueError):
            cpmg_pulse_times(2, 0.0)
        with self.assertRaises(ValueError):
            interval_durations([0.4, 0.4], 1.0)
        with self.assertRaises(ValueError):
            interval_durations([1.0], 1.0)


if __name__ == "__main__":
    unittest.main()
