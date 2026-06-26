from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import SpatialDomain, SpatialFieldMaps
from spin_dynamics.motion import (
    _bilinear_sample,
    initialize_ensemble_from_density,
    initialize_ensemble_from_domain,
)
from spin_dynamics.workflows.imaging import _default_gradient_maps
from spin_dynamics.workflows.imaging_types import ImagingFieldMaps


def _random_maps(shape, rng):
    rho = rng.random(shape)
    return dict(
        rho=rho,
        t1_map=rng.random(shape) + 1.0,
        t2_map=rng.random(shape) + 1.0,
        b0_map=rng.random(shape) - 0.5,
        b1_tx_map=rng.random(shape),
        b1_rx_map=rng.random(shape),
    )


class SpatialDomainTests(unittest.TestCase):
    def test_rejects_non_increasing_axis(self):
        with self.assertRaises(ValueError):
            SpatialDomain((np.array([0.0, 0.0, 1.0]),))

    def test_rejects_too_many_axes(self):
        ax = np.linspace(0, 1, 2)
        with self.assertRaises(ValueError):
            SpatialDomain((ax, ax, ax, ax))

    def test_shape_bounds_ndim(self):
        dom = SpatialDomain((np.linspace(0, 1, 3), np.linspace(-2, 2, 5)))
        self.assertEqual(dom.ndim, 2)
        self.assertEqual(dom.shape, (3, 5))
        self.assertEqual(dom.bounds, ((0.0, 1.0), (-2.0, 2.0)))

    def test_normalized_grids_match_default_gradient_maps(self):
        shape = (4, 6)
        dom = SpatialDomain.normalized(shape)
        g0, g1 = dom.normalized_coordinate_grids()
        d0, d1 = _default_gradient_maps(shape)
        np.testing.assert_array_equal(g0, d0)
        np.testing.assert_array_equal(g1, d1)


class FlattenParityTests(unittest.TestCase):
    def _reference(self, m, dwx, dwz, ny, maxoffs, scale):
        reps = ny
        offsets = np.linspace(-maxoffs, maxoffs, reps)
        b0 = m["b0_map"].reshape(-1)
        dens = scale * m["rho"].reshape(-1)
        return {
            "del_w": np.concatenate([o + b0 for o in offsets]),
            "del_wx": np.tile(dwx.reshape(-1), reps),
            "del_wz": np.tile(dwz.reshape(-1), reps),
            "w_1": np.tile(m["b1_tx_map"].reshape(-1), reps),
            "w_1r": np.tile(m["b1_rx_map"].reshape(-1), reps),
            "m0": np.tile(dens, reps),
            "mth": np.tile(dens, reps),
            "T1": np.tile(m["t1_map"].reshape(-1), reps),
            "T2": np.tile(m["t2_map"].reshape(-1), reps),
        }

    def test_flatten_matches_inline_formula(self):
        rng = np.random.default_rng(1)
        shape = (3, 4)
        m = _random_maps(shape, rng)
        dwx, dwz = _default_gradient_maps(shape)
        dom = SpatialDomain.normalized(shape)
        sm = SpatialFieldMaps(domain=dom, gradient_sensitivity=(dwx, dwz), **m)
        for scale, norm in ((1.0, "legacy"), (1.0 / 5, "preserve")):
            out = sm.flatten(5, 2.0, norm, axis_names=("del_wx", "del_wz"))
            ref = self._reference(m, dwx, dwz, 5, 2.0, scale)
            self.assertEqual(set(out), set(ref))
            for key in ref:
                np.testing.assert_array_equal(out[key], ref[key], err_msg=key)

    def test_imaging_field_maps_adapter_unchanged(self):
        rng = np.random.default_rng(2)
        shape = (2, 3)
        m = _random_maps(shape, rng)
        dwx, dwz = _default_gradient_maps(shape)
        maps = ImagingFieldMaps(del_wx=dwx, del_wz=dwz, **m)
        out = maps.kernel_maps(4, 1.5)
        ref = self._reference(m, dwx, dwz, 4, 1.5, 1.0)
        for key in ref:
            np.testing.assert_array_equal(out[key], ref[key], err_msg=key)


class SampleTests(unittest.TestCase):
    def test_sample_matches_legacy_bilinear(self):
        rng = np.random.default_rng(3)
        xa = np.linspace(0, 1, 4)
        za = np.linspace(-1, 2, 5)
        m = _random_maps((4, 5), rng)
        dom = SpatialDomain((xa, za))
        sm = SpatialFieldMaps(domain=dom, **m)
        pos = np.column_stack(
            [rng.uniform(-0.3, 1.3, 50), rng.uniform(-1.3, 2.3, 50)]
        )
        got = sm.sample(pos)
        np.testing.assert_array_equal(
            got["b0"], _bilinear_sample(m["b0_map"], xa, za, pos)
        )
        np.testing.assert_array_equal(
            got["b1_tx"], _bilinear_sample(m["b1_tx_map"], xa, za, pos)
        )

    def test_affine_field_is_interpolated_exactly_in_1d_2d_3d(self):
        rng = np.random.default_rng(4)
        coeffs = [2.0, -1.5, 0.7, 3.1]
        for axes in (
            (np.linspace(0, 1, 5),),
            (np.linspace(0, 1, 4), np.linspace(-1, 1, 6)),
            (np.linspace(0, 1, 3), np.linspace(-1, 1, 4), np.linspace(2, 5, 5)),
        ):
            dom = SpatialDomain(axes)
            grids = dom.meshgrid("ij")
            values = np.full(dom.shape, coeffs[0])
            for k, grid in enumerate(grids):
                values = values + coeffs[k + 1] * grid
            sm = SpatialFieldMaps(
                domain=dom,
                rho=values,
                t1_map=np.ones(dom.shape),
                t2_map=np.ones(dom.shape),
                b0_map=values,
                b1_tx_map=values,
                b1_rx_map=values,
            )
            pos = np.column_stack(
                [rng.uniform(a[0], a[-1], 40) for a in axes]
            )
            expected = coeffs[0] + sum(
                coeffs[k + 1] * pos[:, k] for k in range(dom.ndim)
            )
            np.testing.assert_allclose(sm.sample(pos)["b0"], expected)

    def test_voxel_center_sampling_returns_voxel_values(self):
        rng = np.random.default_rng(5)
        axes = (np.linspace(0, 2, 3), np.linspace(-1, 1, 4), np.linspace(0, 1, 2))
        dom = SpatialDomain(axes)
        vol = rng.random(dom.shape)
        sm = SpatialFieldMaps(
            domain=dom,
            rho=vol,
            t1_map=np.ones(dom.shape),
            t2_map=np.ones(dom.shape),
            b0_map=vol,
            b1_tx_map=vol,
            b1_rx_map=vol,
        )
        grids = dom.meshgrid("ij")
        centers = np.column_stack([g.ravel() for g in grids])
        np.testing.assert_allclose(sm.sample(centers)["b0"], vol.ravel())


class GradientCouplingTests(unittest.TestCase):
    def test_lagrangian_and_eulerian_couplings_agree(self):
        for axes, grad in (
            ((np.linspace(0, 1, 5),), [0.8]),
            ((np.linspace(0, 1, 4), np.linspace(-1, 1, 3)), [0.5, -1.2]),
            (
                (np.linspace(0, 1, 3), np.linspace(-1, 1, 2), np.linspace(0, 2, 4)),
                [0.3, 0.9, -0.4],
            ),
        ):
            dom = SpatialDomain(axes)
            sm = SpatialFieldMaps(
                domain=dom,
                rho=np.ones(dom.shape),
                t1_map=np.ones(dom.shape),
                t2_map=np.ones(dom.shape),
                b0_map=np.zeros(dom.shape),
                b1_tx_map=np.ones(dom.shape),
                b1_rx_map=np.ones(dom.shape),
            )
            grids = dom.meshgrid("ij")
            centers = np.column_stack([g.ravel() for g in grids])
            lagrangian = sm.gradient_coupling(grad, positions=centers)
            eulerian = sm.gradient_coupling(grad, grids=grids).ravel()
            np.testing.assert_allclose(lagrangian, eulerian)


class EnsembleInitTests(unittest.TestCase):
    def test_domain_path_matches_axis_path_in_2d(self):
        rng = np.random.default_rng(7)
        xa = np.linspace(0, 1, 3)
        za = np.linspace(-1, 1, 4)
        rho = rng.random((3, 4))
        a = initialize_ensemble_from_density(
            rho, xa, za, walkers_per_cell=2, seed=11, jitter=True
        )
        b = initialize_ensemble_from_domain(
            SpatialDomain((xa, za)), rho, walkers_per_cell=2, seed=11, jitter=True
        )
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_domain_path_supports_3d(self):
        rng = np.random.default_rng(8)
        dom = SpatialDomain(
            (np.linspace(0, 1, 2), np.linspace(0, 1, 3), np.linspace(0, 1, 2))
        )
        rho = rng.random(dom.shape)
        ens = initialize_ensemble_from_domain(dom, rho)
        self.assertEqual(ens.positions.shape, (rho.size, 3))
        np.testing.assert_allclose(np.sum(ens.weights), np.sum(rho))


if __name__ == "__main__":
    unittest.main()
