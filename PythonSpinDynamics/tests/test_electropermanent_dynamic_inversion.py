from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.workflows.electropermanent_dynamic_inversion import (
    DynamicInversionHardwareConfig,
    FerromagneticParticle,
    assess_dynamic_inversion_hardware,
    nacev_2015_sequence,
    simulate_dynamic_inversion,
)


def _rod(
    length_m: float = 20e-6,
    diameter_m: float = 500e-9,
    *,
    internal_relaxation_time_s: float | None = None,
    remanent_magnetization_a_m: float | None = None,
) -> FerromagneticParticle:
    return FerromagneticParticle(
        shape="rod",
        length_m=length_m,
        diameter_m=diameter_m,
        volume_susceptibility=0.65,
        saturation_magnetization_a_m=1.4e6,
        remanent_magnetization_a_m=remanent_magnetization_a_m,
        fluid_viscosity_pa_s=0.7e-3,
        temperature_k=298.0,
        internal_relaxation_time_s=internal_relaxation_time_s,
    )


class FerromagneticParticleTests(unittest.TestCase):
    def test_sphere_drag_recovers_stokes_and_rotational_limits(self) -> None:
        diameter = 2e-6
        viscosity = 1.2e-3
        particle = FerromagneticParticle(
            shape="sphere",
            length_m=diameter,
            diameter_m=diameter,
            volume_susceptibility=0.5,
            saturation_magnetization_a_m=8e5,
            fluid_viscosity_pa_s=viscosity,
        )

        self.assertAlmostEqual(
            particle.translational_drag_parallel_n_s_m,
            3.0 * np.pi * viscosity * diameter,
        )
        self.assertAlmostEqual(
            particle.translational_drag_perpendicular_n_s_m,
            3.0 * np.pi * viscosity * diameter,
        )
        self.assertAlmostEqual(
            particle.rotational_drag_n_m_s,
            np.pi * viscosity * diameter**3,
        )

    def test_rod_diffusion_matches_tirado_finite_cylinder_expressions(self) -> None:
        particle = _rod(length_m=8e-6, diameter_m=1e-6)
        p = particle.aspect_ratio
        expected_rotation = (
            3.0
            * 1.380649e-23
            * particle.temperature_k
            * (np.log(p) - 0.662 + 0.917 / p - 0.050 / p**2)
            / (np.pi * particle.fluid_viscosity_pa_s * particle.length_m**3)
        )

        self.assertAlmostEqual(
            particle.rotational_diffusion_rad2_s,
            expected_rotation,
        )
        self.assertLess(
            particle.translational_drag_parallel_n_s_m,
            particle.translational_drag_perpendicular_n_s_m,
        )

    def test_fast_internal_relaxation_shortens_orientation_memory(self) -> None:
        locked = _rod()
        relaxing = _rod(internal_relaxation_time_s=100e-9)

        self.assertLess(
            relaxing.orientation_memory_time_s(10e-3, polarizing_field_t=50e-3),
            locked.orientation_memory_time_s(10e-3, polarizing_field_t=50e-3),
        )


class DynamicInversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = nacev_2015_sequence(
            polarizing_field_t=50e-3,
            gradient_field_at_center_t=15e-3,
            actuator_radius_m=12e-3,
        )
        angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        self.initial = 3.0e-3 * np.column_stack((np.cos(angle), np.sin(angle)))

    def test_seeded_dynamic_inversion_is_reproducible_and_nonsticky(self) -> None:
        kwargs = dict(
            duration_s=2.0,
            target_radius_m=2.0e-3,
            seed=4,
        )
        first = simulate_dynamic_inversion(
            self.sequence,
            _rod(),
            self.initial,
            **kwargs,
        )
        second = simulate_dynamic_inversion(
            self.sequence,
            _rod(),
            self.initial,
            **kwargs,
        )

        np.testing.assert_array_equal(first.positions_m, second.positions_m)
        np.testing.assert_array_equal(first.body_angles_rad, second.body_angles_rad)
        self.assertGreater(first.stability.repulsive_gradient_fraction, 0.5)
        self.assertGreater(first.element_count, 4)

    def test_rapid_internal_relaxation_destroys_repulsive_phase(self) -> None:
        locked = simulate_dynamic_inversion(
            self.sequence,
            _rod(),
            self.initial,
            duration_s=1.0,
            target_radius_m=2e-3,
            brownian=True,
            seed=8,
        )
        relaxing = simulate_dynamic_inversion(
            self.sequence,
            _rod(internal_relaxation_time_s=100e-9),
            self.initial,
            duration_s=1.0,
            target_radius_m=2e-3,
            brownian=True,
            seed=8,
        )

        self.assertGreater(
            locked.stability.repulsive_gradient_fraction,
            relaxing.stability.repulsive_gradient_fraction,
        )

    def test_rigid_long_rod_contracts_while_fast_moment_relaxation_expands(self) -> None:
        sequence = nacev_2015_sequence(
            polarizing_field_t=0.5,
            gradient_field_at_center_t=0.2,
            actuator_radius_m=8e-3,
        )
        rigid = _rod(
            length_m=200e-6,
            diameter_m=200e-9,
            remanent_magnetization_a_m=1.0e6,
        )
        relaxing = _rod(
            length_m=200e-6,
            diameter_m=200e-9,
            internal_relaxation_time_s=100e-9,
            remanent_magnetization_a_m=1.0e6,
        )
        angle = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        initial = 3e-3 * np.column_stack((np.cos(angle), np.sin(angle)))

        trapped = simulate_dynamic_inversion(
            sequence,
            rigid,
            initial,
            duration_s=10.0,
            target_radius_m=2e-3,
            seed=8,
        )
        escaped = simulate_dynamic_inversion(
            sequence,
            relaxing,
            initial,
            duration_s=10.0,
            target_radius_m=2e-3,
            seed=8,
        )

        self.assertGreater(trapped.stability.concentration_gain, 1.0)
        self.assertLess(escaped.stability.concentration_gain, 1.0)

    def test_scaled_up_rod_retains_antialignment_at_least_as_well(self) -> None:
        short = simulate_dynamic_inversion(
            self.sequence,
            _rod(length_m=2e-6, diameter_m=200e-9),
            self.initial,
            duration_s=1.0,
            target_radius_m=2e-3,
            seed=11,
        )
        long = simulate_dynamic_inversion(
            self.sequence,
            _rod(length_m=20e-6, diameter_m=2e-6),
            self.initial,
            duration_s=1.0,
            target_radius_m=2e-3,
            seed=11,
        )

        self.assertGreaterEqual(
            long.stability.repulsive_gradient_fraction,
            short.stability.repulsive_gradient_fraction,
        )


class DynamicInversionHardwareTests(unittest.TestCase):
    def test_architecture_counts_and_waveform_consequences_are_explicit(self) -> None:
        sequence = nacev_2015_sequence()
        particle = _rod(length_m=200e-6, diameter_m=200e-9)
        duration = 9.1 * 60.0
        coil = assess_dynamic_inversion_hardware(
            sequence,
            particle,
            duration_s=duration,
            config=DynamicInversionHardwareConfig(architecture="coils"),
        )
        epm = assess_dynamic_inversion_hardware(
            sequence,
            particle,
            duration_s=duration,
            config=DynamicInversionHardwareConfig(architecture="epm"),
        )
        hybrid = assess_dynamic_inversion_hardware(
            sequence,
            particle,
            duration_s=duration,
            config=DynamicInversionHardwareConfig(architecture="hybrid"),
        )

        self.assertEqual(coil.coil_pulse_count, 2 * coil.element_count)
        self.assertEqual(epm.epm_retained_state_changes, 3 * epm.element_count)
        self.assertGreater(epm.epm_channel_pulse_count, 1_000_000)
        self.assertEqual(hybrid.epm_retained_state_changes, 2)
        self.assertFalse(epm.waveform_fidelity_feasible)
        self.assertTrue(coil.waveform_fidelity_feasible)
        self.assertTrue(hybrid.waveform_fidelity_feasible)


if __name__ == "__main__":
    unittest.main()
