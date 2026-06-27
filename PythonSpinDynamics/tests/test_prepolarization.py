from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.prepolarization import (
    apply_prepolarization_to_parameters,
    field_ratio_equilibrium,
    longitudinal_recovery,
    prepolarized_flow_state,
    prepolarized_magnetization,
    prepolarized_state,
    residence_time_seconds,
)


class PrepolarizationTests(unittest.TestCase):
    def test_longitudinal_recovery_matches_t1_build_up(self) -> None:
        mz = longitudinal_recovery(
            initial_magnetization=0.0,
            equilibrium_magnetization=4.0,
            duration_seconds=2.0,
            t1_seconds=2.0,
        )

        self.assertAlmostEqual(float(mz), 4.0 * (1.0 - np.exp(-1.0)))

    def test_field_ratio_equilibrium_is_signed_detection_normalized(self) -> None:
        equilibrium = field_ratio_equilibrium(
            np.array([2.0, -1.0]),
            detection_field_tesla=0.5,
            detection_equilibrium_magnetization=3.0,
        )

        np.testing.assert_allclose(equilibrium, [12.0, -6.0])

    def test_prepolarized_magnetization_uses_polarizing_field_equilibrium(self) -> None:
        m0 = prepolarized_magnetization(
            polarizing_field_tesla=0.1,
            detection_field_tesla=50e-6,
            prepolarization_time_seconds=np.inf,
            t1_seconds=1.5,
        )

        self.assertAlmostEqual(float(m0), 2000.0)

    def test_flow_state_handles_velocity_dependent_residence_time(self) -> None:
        speeds = np.array([0.0, 0.05, 0.5])
        times = residence_time_seconds(path_length_meters=0.1, speed_meters_per_second=speeds)
        state = prepolarized_flow_state(
            polarizing_field_tesla=0.2,
            detection_field_tesla=0.1,
            path_length_meters=0.1,
            speed_meters_per_second=speeds,
            t1_seconds=2.0,
        )

        np.testing.assert_allclose(times, [np.inf, 2.0, 0.2])
        np.testing.assert_allclose(state.mth, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            state.m0,
            2.0 * (1.0 - np.exp(-times / 2.0)),
        )
        self.assertGreater(state.m0[1], state.m0[2])

    def test_prepolarized_state_returns_kernel_parameter_fields(self) -> None:
        state = prepolarized_state(
            polarizing_field_tesla=[1.0, 2.0],
            detection_field_tesla=1.0,
            prepolarization_time_seconds=np.inf,
            t1_seconds=1.0,
            detection_equilibrium_magnetization=[0.5, 0.25],
        )

        params = state.as_parameters()
        np.testing.assert_allclose(params["m0"], [0.5, 0.5])
        np.testing.assert_allclose(params["mth"], [0.5, 0.25])
        np.testing.assert_allclose(state.enhancement, [1.0, 2.0])

    def test_apply_prepolarization_to_mapping_or_dataclass(self) -> None:
        state = prepolarized_state(
            polarizing_field_tesla=[3.0, 4.0],
            detection_field_tesla=1.0,
            prepolarization_time_seconds=np.inf,
            t1_seconds=1.0,
        )

        mapped = apply_prepolarization_to_parameters({"m0": 1.0, "other": 7}, state)
        self.assertEqual(mapped["other"], 7)
        np.testing.assert_allclose(mapped["m0"], [3.0, 4.0])

        @dataclass(frozen=True)
        class Params:
            del_w: np.ndarray
            m0: float
            mth: float

        copied = apply_prepolarization_to_parameters(
            Params(del_w=np.array([0.0, 1.0]), m0=1.0, mth=1.0),
            state,
        )
        np.testing.assert_allclose(copied["del_w"], [0.0, 1.0])
        np.testing.assert_allclose(copied["mth"], [1.0, 1.0])

    def test_invalid_inputs_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "detection_field_tesla"):
            field_ratio_equilibrium(1.0, detection_field_tesla=0.0)
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            longitudinal_recovery(0.0, 1.0, -1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "speed_meters_per_second"):
            residence_time_seconds(1.0, -0.1)


if __name__ == "__main__":
    unittest.main()
