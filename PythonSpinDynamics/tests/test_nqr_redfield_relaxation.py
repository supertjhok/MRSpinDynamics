from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    OrientationSample,
    QuadrupolarSite,
    diagonalize_site,
    simulate_slse,
    slse_sequence,
)
from spin_dynamics.relaxation import (  # noqa: E402
    DipolarRelaxationSource,
    IsotropicLiquidMotionalAveraging,
    PhenomenologicalRelaxationModel,
    RedfieldDipolarRelaxationModel,
    RigidSolidMotionalAveraging,
    dipolar_coupling_tensor,
    propagate_density_liouville,
)


class RedfieldDipolarRelaxationTests(unittest.TestCase):
    def test_shared_liouville_relaxation_supports_spin_half(self) -> None:
        density = np.array([[0.5, 1.0], [0.25, -0.5]], dtype=np.complex128)
        hamiltonian = np.zeros((2, 2), dtype=np.complex128)

        final = propagate_density_liouville(
            density,
            hamiltonian,
            0.2,
            relaxation=PhenomenologicalRelaxationModel(t1_seconds=0.4, t2_seconds=0.5),
        )

        self.assertAlmostEqual(np.trace(final).real, np.trace(density).real)
        np.testing.assert_allclose(final[0, 1], density[0, 1] * np.exp(-0.2 / 0.5))
        self.assertLess(np.linalg.norm(np.diag(final)), np.linalg.norm(np.diag(density)))

    def test_dipolar_source_covariance_scales_as_distance_to_minus_six(self) -> None:
        near = DipolarRelaxationSource((1.0, 0.0, 0.0))
        far = DipolarRelaxationSource((2.0, 0.0, 0.0))

        near_norm = np.linalg.norm(near.covariance_rad2_per_s2)
        far_norm = np.linalg.norm(far.covariance_rad2_per_s2)

        self.assertAlmostEqual(near_norm / far_norm, 64.0, delta=1.0e-10)

    def test_coupling_tensor_has_expected_point_dipole_anisotropy(self) -> None:
        tensor = dipolar_coupling_tensor((0.0, 0.0, 1.0), coupling_hz=2.0)

        np.testing.assert_allclose(
            tensor / (2.0 * np.pi * 2.0),
            np.diag([1.0, 1.0, -2.0]),
            atol=1.0e-14,
        )

    def test_motion_regimes_distinguish_solid_and_isotropic_liquid(self) -> None:
        source_x = DipolarRelaxationSource((1.0, 0.0, 0.0), coupling_hz=1.0e3)
        source_z = DipolarRelaxationSource((0.0, 0.0, 1.0), coupling_hz=1.0e3)
        solid = RigidSolidMotionalAveraging(correlation_time_seconds=1.0e-6)
        liquid = IsotropicLiquidMotionalAveraging(correlation_time_seconds=1.0e-6)

        solid_x = solid.covariance_from_source(source_x)
        solid_z = solid.covariance_from_source(source_z)
        liquid_x = liquid.covariance_from_source(source_x)
        liquid_z = liquid.covariance_from_source(source_z)

        self.assertFalse(np.allclose(solid_x, solid_z))
        np.testing.assert_allclose(liquid_x, liquid_z, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(
            liquid_x,
            np.trace(solid_x) / 3.0 * np.eye(3),
            rtol=1.0e-12,
        )

    def test_redfield_superoperator_preserves_trace_and_damps_state(self) -> None:
        model = RedfieldDipolarRelaxationModel.from_dipolar_sources(
            1.0,
            (DipolarRelaxationSource((1.0, 0.5, 0.25), coupling_hz=2.0e5),),
            correlation_time_seconds=1.0e-6,
        )
        hamiltonian = np.diag([0.0, 2.0 * np.pi * 0.9e6, 2.0 * np.pi * 1.1e6])
        density = np.array(
            [
                [1.0, 0.4, 0.0],
                [0.2, -0.2, 0.1j],
                [0.0, -0.1j, -0.8],
            ],
            dtype=np.complex128,
        )

        derivative = (model.superoperator(hamiltonian) @ density.reshape(-1, order="F"))
        derivative = derivative.reshape(density.shape, order="F")
        final = propagate_density_liouville(
            density,
            hamiltonian,
            20.0e-6,
            relaxation=model,
        )

        self.assertAlmostEqual(np.trace(derivative).real, 0.0, places=9)
        self.assertAlmostEqual(np.trace(final).real, np.trace(density).real, places=9)
        self.assertLess(np.linalg.norm(final), np.linalg.norm(density))

    def test_redfield_model_records_explicit_motion_regime(self) -> None:
        model = RedfieldDipolarRelaxationModel.from_dipolar_sources(
            1.0,
            (DipolarRelaxationSource((1.0, 0.0, 0.0), coupling_hz=1.0e4),),
            motion=IsotropicLiquidMotionalAveraging(5.0e-9),
        )

        self.assertEqual(model.regime, "isotropic_liquid")
        self.assertAlmostEqual(model.correlation_time_seconds, 5.0e-9)

    def test_nqr_namespace_reexports_shared_redfield_model(self) -> None:
        from spin_dynamics.nqr import RedfieldDipolarRelaxationModel as nqr_model

        self.assertIs(nqr_model, RedfieldDipolarRelaxationModel)

    def test_existing_slse_path_accepts_redfield_relaxation_as_opt_in(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0e3, eta=0.3)
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=0.0,
            nutation_hz=0.0,
            echo_spacing_seconds=100.0e-6,
            num_echoes=4,
        )
        transition = diagonalize_site(site).transition("x")
        density = np.zeros((3, 3), dtype=np.complex128)
        density[transition.upper, transition.lower] = 1.0
        model = RedfieldDipolarRelaxationModel.from_dipolar_sources(
            site.spin,
            (DipolarRelaxationSource((1.0, 0.0, 0.0), coupling_hz=3.0e5),),
            correlation_time_seconds=2.0e-6,
        )

        bare = simulate_slse(
            site,
            sequence,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            initial_density=density,
        )
        damped = simulate_slse(
            site,
            sequence,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            initial_density=density,
            relaxation=model,
        )

        self.assertLess(
            abs(damped.echo_amplitudes[-1]),
            abs(bare.echo_amplitudes[-1]),
        )
        self.assertIsNotNone(damped.local_effective_t2eff_seconds)


if __name__ == "__main__":
    unittest.main()
