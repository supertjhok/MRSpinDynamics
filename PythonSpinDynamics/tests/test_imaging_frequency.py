from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows import (
    imaging_slice_sensitivity,
    make_imaging_field_maps,
    run_rare_imaging,
    run_spin_warp_imaging,
)


def _readout_phase_widths(image: np.ndarray) -> tuple[float, float]:
    m = np.abs(image)
    m = m / m.max()
    n0, n1 = m.shape
    gx, gz = np.mgrid[0:n0, 0:n1]
    total = m.sum()
    mx = (gx * m).sum() / total
    mz = (gz * m).sum() / total
    sx = float(np.sqrt(((gx - mx) ** 2 * m).sum() / total))
    sz = float(np.sqrt(((gz - mz) ** 2 * m).sum() / total))
    return sx, sz


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

    def test_accepts_imaging_field_maps_like_array_inputs(self) -> None:
        rho = _phantom()
        t1 = np.full(rho.shape, 1.0)
        t2 = np.full(rho.shape, 1.0)
        fields = make_imaging_field_maps(
            rho, t1_map=t1, t2_map=t2, b0_map=np.zeros(rho.shape),
            b1_tx_map=np.ones(rho.shape), b1_rx_map=np.ones(rho.shape),
        )
        from_fields = run_spin_warp_imaging(fields, fov=(0.02, 0.02))
        from_arrays = run_spin_warp_imaging(rho, fov=(0.02, 0.02), t1_map=t1, t2_map=t2)
        np.testing.assert_allclose(from_fields.kspace, from_arrays.kspace, atol=1e-9)

    def test_rejects_map_kwargs_with_field_maps(self) -> None:
        rho = _phantom()
        fields = make_imaging_field_maps(rho)
        with self.assertRaises(ValueError):
            run_spin_warp_imaging(fields, b0_map=np.zeros(rho.shape))

    def test_subvoxel_b0_spread_blurs_readout_not_phase_encode(self) -> None:
        rho = np.zeros((24, 24))
        rho[8:16, 9:15] = 1.0
        spread = 2.0 * np.pi * 1500.0  # rad/s
        base = run_spin_warp_imaging(rho, fov=(0.02, 0.02))
        blurred = run_spin_warp_imaging(
            rho, fov=(0.02, 0.02), num_offsets=9, offset_spread=spread
        )
        sx0, sz0 = _readout_phase_widths(base.image[:, :, 0])
        sx1, sz1 = _readout_phase_widths(blurred.image[:, :, 0])
        self.assertEqual(blurred.num_offsets, 9)
        # The spread broadens the readout (x) axis but leaves phase encode (z).
        self.assertGreater(sx1, sx0 + 0.3)
        self.assertAlmostEqual(sz1, sz0, delta=0.1)

    def test_slice_sensitivity_peaks_on_resonance(self) -> None:
        n = 21
        rho = np.ones((n, n))
        ramp = np.linspace(-80000.0, 80000.0, n)  # rad/s along x
        b0 = ramp[:, np.newaxis] * np.ones((1, n))
        res = imaging_slice_sensitivity(
            rho, b0_map=b0, center_frequency=0.0,
            excitation_flip=np.pi / 2, excitation_duration=100e-6,
        )
        # On-resonance (centre row) a 90 gives full transverse magnetization.
        self.assertAlmostEqual(res.excitation[n // 2, 0], 1.0, delta=0.02)
        # The excited band is centred on resonance and falls off well off it (a
        # hard pulse has a broad, sinc-like profile, not an ideal box).
        self.assertEqual(int(np.argmax(res.excitation[:, 0])), n // 2)
        self.assertLess(res.excitation[0, 0], 0.4)

    def test_slice_sensitivity_scales_with_receive_b1(self) -> None:
        n = 16
        rho = np.ones((n, n))
        b0 = np.linspace(-2e4, 2e4, n)[:, np.newaxis] * np.ones((1, n))
        base = imaging_slice_sensitivity(rho, b0_map=b0, b1_rx_map=np.ones((n, n)))
        doubled = imaging_slice_sensitivity(
            rho, b0_map=b0, b1_rx_map=2.0 * np.ones((n, n))
        )
        np.testing.assert_allclose(doubled.sensitivity, 2.0 * base.sensitivity, atol=1e-9)

    def test_slice_follows_curved_b0_contour(self) -> None:
        n = 31
        rho = np.ones((n, n))
        axis = np.linspace(-1.0, 1.0, n)
        xx = axis[:, np.newaxis] * np.ones((1, n))
        zz = np.ones((n, 1)) * axis[np.newaxis, :]
        b0 = 2 * np.pi * (30000.0 * zz + 18000.0 * xx**2)  # parabolic contours
        res = imaging_slice_sensitivity(rho, b0_map=b0, excitation_duration=120e-6)
        centers = axis[np.argmin(np.abs(res.off_resonance), axis=1)]
        # The on-resonance depth varies with position (curved, not flat) and is
        # deeper at the edges than on axis.
        self.assertGreater(centers.std(), 0.1)
        self.assertLess(centers[0], centers[n // 2] - 0.2)
        self.assertLess(centers[-1], centers[n // 2] - 0.2)

    def test_slice_sensitivity_accepts_field_maps(self) -> None:
        n = 12
        rho = np.ones((n, n))
        b0 = np.linspace(-2e4, 2e4, n)[:, np.newaxis] * np.ones((1, n))
        fields = make_imaging_field_maps(rho, b0_map=b0, b1_tx_map=np.ones((n, n)),
                                         b1_rx_map=np.ones((n, n)))
        from_fields = imaging_slice_sensitivity(fields)
        from_arrays = imaging_slice_sensitivity(rho, b0_map=b0)
        np.testing.assert_allclose(from_fields.sensitivity, from_arrays.sensitivity)

    def test_static_spread_does_not_decay_the_echo_train(self) -> None:
        rho = np.zeros((20, 20))
        rho[6:14, 7:13] = 1.0
        spread = 2.0 * np.pi * 1200.0
        kw = dict(fov=(0.02, 0.02), echo_train_length=20,
                  t2_map=np.full(rho.shape, np.inf))
        plain = run_rare_imaging(rho, **kw)
        spread_run = run_rare_imaging(rho, num_offsets=9, offset_spread=spread, **kw)
        # The last echo (longest TE) keeps most of its amplitude: a static spread
        # is refocused each echo, so it does not act like a T2' train decay.
        last = 19
        ratio = abs(spread_run.kspace[10, last, 0]) / abs(plain.kspace[10, last, 0])
        self.assertGreater(ratio, 0.8)


if __name__ == "__main__":
    unittest.main()
