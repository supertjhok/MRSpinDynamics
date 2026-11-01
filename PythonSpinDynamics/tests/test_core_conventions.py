"""Regression tests for core numerical-convention fixes (review items M-03, M-04, M-07)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.core.rotations import calc_rot_axis_arba4, calc_rotation_matrix  # noqa: E402
from spin_dynamics.noise import add_received_noise  # noqa: E402
from spin_dynamics.pulses import quantize_phase  # noqa: E402

# Cartesian (Mx, My, Mz) <- coherence (Mz, M-, M+) and back.
_COH_TO_CART = np.array([[0.0, 0.5, 0.5], [0.0, 0.5j, -0.5j], [1.0, 0.0, 0.0]])
_CART_TO_COH = np.array([[0.0, 0.0, 1.0], [1.0, -1.0j, 0.0], [1.0, 1.0j, 0.0]])


class QuantizePhaseTests(unittest.TestCase):
    """M-03: phases are quantized around the circle."""

    def test_matches_rounding_of_wrapped_phase(self) -> None:
        rng = np.random.default_rng(0)
        phases = np.concatenate(
            [rng.uniform(-4.0 * np.pi, 4.0 * np.pi, 400), [2.0 * np.pi, -np.pi / 2]]
        )
        for count in (2, 4, 8):
            step = 2.0 * np.pi / count
            expected = (np.round(np.mod(phases, 2.0 * np.pi) / step) % count) * step
            np.testing.assert_allclose(quantize_phase(phases, count), expected, atol=1e-12)

    def test_previous_failures_and_legacy_mode(self) -> None:
        phases = [2.0 * np.pi, 2.0 * np.pi - 0.05, -np.pi / 2]
        np.testing.assert_allclose(quantize_phase(phases, 4), [0.0, 0.0, 1.5 * np.pi])
        # circular=False keeps the MATLAB linear-distance behaviour.
        np.testing.assert_allclose(
            quantize_phase(phases, 4, circular=False), [1.5 * np.pi, 1.5 * np.pi, 0.0]
        )


class RotationAxisTests(unittest.TestCase):
    """M-04: effective angle and axis stay defined for near-identity cycles."""

    def test_near_identity_cycles_match_rotation_matrix_reference(self) -> None:
        rng = np.random.default_rng(0)
        del_w = np.linspace(-1e-3, 1e-3, 2001)
        for _ in range(200):
            angle = rng.uniform(0.1, 3.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            tp = np.array([angle, angle])
            phi = np.array([phase, phase + np.pi])
            amp = np.ones(2)
            n, alpha = calc_rot_axis_arba4(tp, phi, amp, del_w)
            self.assertTrue(np.all(np.isfinite(alpha)))
            # Independent reference: compose the pulses as coherence matrices.
            mats = calc_rotation_matrix(del_w, 1.0, tp, phi, amp)
            full = np.empty((del_w.size, 3, 3), dtype=np.complex128)
            for i, row in enumerate("0mp"):
                for k, col in enumerate("0mp"):
                    full[:, i, k] = getattr(mats, f"R_{row}{col}")
            cart = np.real(_COH_TO_CART @ full @ _CART_TO_COH)
            cos_ref = 0.5 * (np.trace(cart, axis1=1, axis2=2) - 1.0)
            np.testing.assert_allclose(np.cos(alpha), cos_ref, atol=1e-12)
            norms = np.linalg.norm(n, axis=0)
            nonzero = norms > 0
            np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-12)

    def test_exact_identity_keeps_zero_axis(self) -> None:
        n, alpha = calc_rot_axis_arba4(
            np.array([np.pi / 2, np.pi / 2]),
            np.array([0.0, np.pi]),
            np.ones(2),
            np.array([0.0]),
        )
        self.assertTrue(np.isfinite(alpha[0]))
        self.assertAlmostEqual(float(alpha[0]), 0.0, places=7)
        self.assertLessEqual(float(np.linalg.norm(n[:, 0])), 1.0 + 1e-12)


class NoiseTargetSnrTests(unittest.TestCase):
    """M-07: target_snr means rms(signal) / rms(noise) for every model."""

    def test_white_and_probe_noise_realize_the_target(self) -> None:
        signal = np.exp(1j * np.linspace(0.0, 40.0, 100_000))
        for complex_noise in (True, False):
            noisy, metadata = add_received_noise(
                signal,
                {"model": "white", "target_snr": 10.0, "seed": 1, "complex_noise": complex_noise},
            )
            measured = np.sqrt(np.mean(np.abs(signal) ** 2)) / np.sqrt(
                np.mean(np.abs(noisy - signal) ** 2)
            )
            self.assertAlmostEqual(measured, 10.0, delta=0.1)
            self.assertAlmostEqual(metadata.realized_snr, measured)
        _noisy, probe_meta = add_received_noise(
            signal,
            {"model": "probe", "target_snr": 10.0, "seed": 1},
            pnoise=np.ones(signal.size),
            frequencies=np.arange(signal.size, dtype=np.float64),
        )
        self.assertAlmostEqual(probe_meta.realized_snr, 10.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
