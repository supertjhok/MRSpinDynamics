from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.experiment import (
    Hardware,
    Phantom,
    SampledB0,
    sampled_b0_from_solution,
    solve_imaging_field_maps,
)
from spin_dynamics.experiment.hardware import ImagingPlane, PlanarSpiralCoil, TxCoil, UniformB0
from spin_dynamics.experiment.serialization import decode, encode
from spin_dynamics.fields.magnetostatics import GAMMA_PROTON
from spin_dynamics.fields.nonlinear_magnetostatics import MU0
from spin_dynamics.fields.scalar_potential_3d import ReducedScalarPotential3D


def _ramp_b0(shape=(4, 5), b0=0.40, delta=0.05):
    """A B0 vector map along z whose magnitude ramps across the grid."""

    field = np.zeros((*shape, 3), dtype=np.float64)
    field[..., 2] = b0 + delta * np.linspace(0.0, 1.0, shape[1])[np.newaxis, :]
    return field


class SampledB0SpecTests(unittest.TestCase):
    def test_off_resonance_direction_and_nutation_scale(self) -> None:
        field = _ramp_b0()
        spec = SampledB0(field, carrier_hz=17e6)
        expected = GAMMA_PROTON * np.linalg.norm(field, axis=-1) - 2.0 * np.pi * 17e6
        np.testing.assert_allclose(spec.off_resonance(GAMMA_PROTON), expected)
        np.testing.assert_allclose(np.linalg.norm(spec.direction(), axis=-1), 1.0)
        # Normalizing by the nutation frequency divides the offset by omega_1.
        w1 = 2.0 * np.pi * 5e4
        spec_n = SampledB0(field, carrier_hz=17e6, nutation_rad_s=w1)
        np.testing.assert_allclose(spec_n.off_resonance(GAMMA_PROTON), expected / w1)

    def test_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            SampledB0(np.zeros((4, 5)), 17e6)
        with self.assertRaisesRegex(ValueError, "carrier"):
            SampledB0(_ramp_b0(), -1.0)
        with self.assertRaisesRegex(ValueError, "nutation"):
            SampledB0(_ramp_b0(), 17e6, nutation_rad_s=0.0)

    def test_serialization_roundtrip(self) -> None:
        spec = SampledB0(_ramp_b0(), carrier_hz=17e6, nutation_rad_s=3.1e5)
        self.assertEqual(decode(encode(spec)), spec)


class SolvedB0ImagingTests(unittest.TestCase):
    def test_builder_samples_the_solution(self) -> None:
        g = np.linspace(-0.05, 0.05, 15)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_uniform_source_field((0.0, 0.0, 1.0e5))  # uniform Bz = mu0 * 1e5
        sol = prob.solve()
        plane = ImagingPlane(extent_m=(0.02, 0.02), plane="xz", center_m=(0.0, 0.0, 0.0))
        spec = sampled_b0_from_solution(sol, plane, (5, 6), carrier_hz=4e6)
        self.assertEqual(spec.b0_tesla.shape, (5, 6, 3))
        np.testing.assert_allclose(spec.magnitude_tesla(), MU0 * 1.0e5, atol=1e-9)

    def test_wiring_sets_spatially_varying_b0_map(self) -> None:
        rho = np.ones((4, 5))
        spec = SampledB0(_ramp_b0((4, 5)), carrier_hz=17e6, nutation_rad_s=2.0 * np.pi * 5e4)
        maps = solve_imaging_field_maps(Phantom(rho=rho), Hardware(b0=spec))
        np.testing.assert_allclose(maps.b0_map, spec.off_resonance(GAMMA_PROTON))
        self.assertGreater(float(maps.b0_map.max() - maps.b0_map.min()), 0.0)

    def test_uniform_b0_leaves_zero_off_resonance(self) -> None:
        maps = solve_imaging_field_maps(Phantom(rho=np.ones((4, 5))), Hardware(b0=UniformB0()))
        np.testing.assert_allclose(maps.b0_map, 0.0)

    def test_sampled_b0_with_coil_uses_per_voxel_direction(self) -> None:
        # B0 along z, a y-axis surface coil (B1 along y) is transverse to it, so the
        # per-voxel-direction projection path produces a valid transmit map.
        rho = np.ones((5, 5))
        spec = SampledB0(_ramp_b0((5, 5)), carrier_hz=17e6, nutation_rad_s=2.0 * np.pi * 5e4)
        coil = TxCoil(geometry=PlanarSpiralCoil(
            r_inner_m=0.002, r_outer_m=0.01, turns=5, axis="y"))
        plane = ImagingPlane(extent_m=(0.02, 0.02), plane="xz", center_m=(0.0, 0.005, 0.0))
        maps = solve_imaging_field_maps(
            Phantom(rho=rho), Hardware(b0=spec, tx_coil=coil, plane=plane)
        )
        self.assertIsNotNone(maps.b1_tx_map)
        self.assertTrue(np.all(np.isfinite(maps.b1_tx_map)))
        self.assertGreater(float(np.mean(maps.b1_tx_map)), 0.0)
        np.testing.assert_allclose(maps.b0_map, spec.off_resonance(GAMMA_PROTON))

    def test_rejects_unknown_b0_spec(self) -> None:
        with self.assertRaisesRegex(ValueError, "UniformB0 or SampledB0"):
            solve_imaging_field_maps(Phantom(rho=np.ones((3, 3))), Hardware(b0="bogus"))


if __name__ == "__main__":
    unittest.main()
