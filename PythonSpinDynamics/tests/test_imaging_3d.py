from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows.imaging_3d import (
    run_multislice_imaging,
    run_multislice_imaging_separable,
)


def _point_phantom(shape, index):
    rho = np.zeros(shape, dtype=np.float64)
    rho[index] = 1.0
    return rho


class EngineMultiSliceTests(unittest.TestCase):
    def test_localizes_a_point_in_three_dimensions(self):
        rho = _point_phantom((4, 5, 4), (1, 2, 3))
        res = run_multislice_imaging(
            rho, slice_gradient=1.5e7, fov=(0.02, 0.02, 0.02), num_substeps=40,
        )
        self.assertEqual(res.method, "engine_3d")
        self.assertEqual(res.magnitude.shape, (4, 5, 4))
        peak = np.unravel_index(np.argmax(res.magnitude), res.magnitude.shape)
        self.assertEqual(peak, (1, 2, 3))

    def test_slice_selectivity(self):
        rho = _point_phantom((4, 5, 4), (1, 2, 3))
        res = run_multislice_imaging(rho, slice_gradient=1.5e7, num_substeps=40)
        per_slice = res.magnitude.sum(axis=(0, 2))
        self.assertEqual(int(np.argmax(per_slice)), 2)
        self.assertGreater(per_slice[2], 20.0 * per_slice[1])
        self.assertGreater(per_slice[2], 20.0 * per_slice[3])

    def test_mild_b0_inhomogeneity_still_localizes(self):
        # A mild through-plane B0 ramp shifts the slice slightly but the point
        # still reconstructs at the right voxel.
        shape = (4, 5, 4)
        rho = _point_phantom(shape, (1, 2, 3))
        y = (np.arange(shape[1]) - shape[1] // 2)
        b0 = (300.0 * y)[None, :, None] * np.ones(shape)  # ~hundreds of rad/s
        res = run_multislice_imaging(
            rho, slice_gradient=1.5e7, b0_map=b0, num_substeps=40,
        )
        peak = np.unravel_index(np.argmax(res.magnitude), res.magnitude.shape)
        self.assertEqual(peak, (1, 2, 3))

    def test_spin_echo_refocuses_on_the_k0_sample(self):
        # Regression (M-11): the 180 was placed for an echo at readout/2 with the
        # sinc centre taken as exc_total/2, so the echo fell one dwell early
        # (rephase=False) or one dwell late (default rephase=True, 1 ms pulse)
        # relative to the k=0 sample. Reference: the spin-echo condition. Voxels
        # on the in-plane isocentre line (no in-plane encoding) with random B0
        # inside a thick slice trace the echo envelope during the readout.
        n, ny = 16, 120
        rng = np.random.default_rng(1)
        rho = np.zeros((n, ny, n))
        rho[n // 2, :, n // 2] = 1.0
        b0 = np.zeros_like(rho)
        b0[n // 2, :, n // 2] = rng.uniform(-2 * np.pi * 1500, 2 * np.pi * 1500, ny)
        for rephase in (True, False):
            res = run_multislice_imaging(
                rho, slice_gradient=1.0, slice_positions=[0.0], b0_map=b0,
                rephase=rephase,
            )
            envelope = np.abs(res.kspace[:, 0, n // 2])
            self.assertEqual(int(np.argmax(envelope)), n // 2)

    def test_relaxation_maps_must_match_rho_shape(self):
        # Regression (L-11): a same-size but transposed T1/T2 volume used to be
        # accepted and silently assigned to the wrong voxels.
        rho = _point_phantom((4, 6, 4), (1, 2, 3))
        with self.assertRaises(ValueError):
            run_multislice_imaging(rho, slice_gradient=1.5e7, t1_map=np.ones((6, 4, 4)))
        with self.assertRaises(ValueError):
            run_multislice_imaging(rho, slice_gradient=1.5e7, t2_map=np.zeros(rho.shape))

    def test_slice_axis_zero(self):
        rho = _point_phantom((5, 4, 4), (2, 1, 3))
        res = run_multislice_imaging(rho, slice_axis=0, slice_gradient=1.5e7, num_substeps=32)
        peak = np.unravel_index(np.argmax(res.magnitude), res.magnitude.shape)
        self.assertEqual(peak, (1, 2, 3))

    def test_validation(self):
        with self.assertRaises(ValueError):
            run_multislice_imaging(np.zeros((4, 4)), slice_gradient=1e7)
        with self.assertRaises(ValueError):
            run_multislice_imaging(np.zeros((4, 4, 4)), slice_gradient=0.0)
        with self.assertRaises(ValueError):
            run_multislice_imaging(np.zeros((4, 4, 4)), slice_gradient=1e7, slice_axis=3)
        with self.assertRaises(ValueError):
            run_multislice_imaging(
                np.zeros((4, 4, 4)), slice_gradient=1e7, b0_map=np.zeros((4, 4))
            )


class SeparableMultiSliceTests(unittest.TestCase):
    def test_localizes_a_point(self):
        rho = _point_phantom((4, 5, 4), (1, 2, 3))
        res = run_multislice_imaging_separable(
            rho, slice_gradient=1.5e7, num_substeps=40, readout_time=2.0e-3,
        )
        self.assertEqual(res.method, "separable")
        peak = np.unravel_index(np.argmax(res.magnitude), res.magnitude.shape)
        self.assertEqual(peak, (1, 2, 3))

    def test_selectivity_and_explicit_positions(self):
        rho = _point_phantom((4, 5, 4), (1, 2, 3))
        res = run_multislice_imaging_separable(
            rho, slice_gradient=1.5e7, num_substeps=32,
            slice_positions=[0.0, 0.004],
        )
        self.assertEqual(res.magnitude.shape, (4, 2, 4))
        np.testing.assert_allclose(res.slice_positions, [0.0, 0.004])

    def test_engine_and_separable_agree_in_uniform_field(self):
        rho = _point_phantom((4, 5, 4), (2, 1, 0))
        common = dict(slice_gradient=1.5e7, num_substeps=40)
        a = run_multislice_imaging(rho, **common)
        b = run_multislice_imaging_separable(rho, **common)
        pa = np.unravel_index(np.argmax(a.magnitude), a.magnitude.shape)
        pb = np.unravel_index(np.argmax(b.magnitude), b.magnitude.shape)
        self.assertEqual(pa, pb)


if __name__ == "__main__":
    unittest.main()
