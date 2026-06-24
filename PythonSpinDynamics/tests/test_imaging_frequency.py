from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows import (
    run_rare_imaging,
    run_spin_warp_imaging,
)


def _phantom(n: int = 16):
    rho = np.zeros((n, n), dtype=np.float64)
    rho[4:12, 5:9] = 1.0
    rho[6:10, 10:13] = 0.6
    return rho


def _relative_error(image: np.ndarray, rho: np.ndarray) -> float:
    image = np.abs(image)
    if image.max() > 0:
        image = image / image.max() * rho.max()
    return float(np.linalg.norm(image - rho) / np.linalg.norm(rho))


class FrequencyEncodedImagingTests(unittest.TestCase):
    def test_spin_warp_localizes_a_point(self) -> None:
        n = 8
        rho = np.zeros((n, n))
        rho[5, 2] = 1.0
        result = run_spin_warp_imaging(rho, fov=(0.02, 0.02))
        magnitude = result.magnitude[:, :, 0]
        peak = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        self.assertEqual(peak, (5, 2))
        # A single point is a clean delta: the next-brightest pixel is negligible.
        second = np.sort(magnitude.ravel())[-2]
        self.assertLess(second, 1e-6 * magnitude.max())

    def test_spin_warp_recovers_phantom(self) -> None:
        rho = _phantom()
        result = run_spin_warp_imaging(rho, fov=(0.02, 0.02))
        self.assertEqual(result.num_shots, rho.shape[1])
        self.assertEqual(result.echo_train_length, 1)
        self.assertLess(_relative_error(result.image[:, :, 0], rho), 0.05)

    def test_rare_matches_spin_warp_without_relaxation(self) -> None:
        rho = _phantom()
        spin_warp = run_spin_warp_imaging(rho, fov=(0.02, 0.02))
        rare = run_rare_imaging(rho, fov=(0.02, 0.02), echo_train_length=rho.shape[1])
        self.assertEqual(rare.num_shots, 1)
        # With T2 = inf every line is unweighted, so the k-space is identical.
        np.testing.assert_allclose(rare.kspace, spin_warp.kspace, atol=1e-9)

    def test_rare_blurring_increases_as_t2_shortens(self) -> None:
        rho = _phantom()
        n = rho.shape[1]
        long_t2 = run_rare_imaging(
            rho, fov=(0.02, 0.02), echo_train_length=n,
            t2_map=np.full(rho.shape, 120e-3),
        )
        short_t2 = run_rare_imaging(
            rho, fov=(0.02, 0.02), echo_train_length=n,
            t2_map=np.full(rho.shape, 20e-3),
        )
        self.assertGreater(
            _relative_error(short_t2.image[:, :, 0], rho),
            _relative_error(long_t2.image[:, :, 0], rho) + 0.05,
        )

    def test_rare_shot_count_follows_echo_train_length(self) -> None:
        rho = _phantom(16)
        self.assertEqual(run_rare_imaging(rho, echo_train_length=8).num_shots, 2)
        self.assertEqual(run_rare_imaging(rho, echo_train_length=5).num_shots, 4)
        self.assertEqual(run_rare_imaging(rho, echo_train_length=16).num_shots, 1)

    def test_rejects_non_2d_density(self) -> None:
        with self.assertRaises(ValueError):
            run_spin_warp_imaging(np.ones(8))


if __name__ == "__main__":
    unittest.main()
