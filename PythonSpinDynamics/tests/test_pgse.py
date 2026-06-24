from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.motion import (
    make_circular_reflector,
    make_elliptical_reflector,
    make_motion_field_maps_2d,
)
from spin_dynamics.workflows import (
    gradient_moment_b_value,
    pgse_b_value,
    run_dde_walkers,
    run_ogse_walkers,
    run_pgse,
    run_pgse_moment,
    run_pgse_walkers,
    run_pgste_walkers,
)


def _wide_slab(num_cells: int = 48):
    # A wide slab lets the unwanted stimulated anti-echo dephase spatially.
    x_axis = np.linspace(-1.0e-3, 1.0e-3, num_cells)
    z_axis = np.array([-1.0e-6, 1.0e-6])
    rho = np.ones((x_axis.size, z_axis.size), dtype=np.float64)
    return rho, x_axis, z_axis


def _pore(semi_axes, grid: int = 13):
    ax, az = semi_axes
    x = np.linspace(-ax, ax, grid)
    z = np.linspace(-az, az, grid)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    rho = ((xx / ax) ** 2 + (zz / az) ** 2 <= 1.0).astype(np.float64)
    return rho, x, z, make_motion_field_maps_2d(x, z)


def _dde_cos2psi(reflector, pore) -> float:
    # Estimate the cos(2 psi) amplitude from psi = 0, 90, 180 degrees:
    # E2 = ((E(0) + E(180)) / 2 - E(90)) / 2.
    rho, x, z, fields = pore
    echoes = {}
    for psi in (0.0, np.pi / 2, np.pi):
        result = run_dde_walkers(
            rho=rho, x_axis=x, z_axis=z, fields=fields,
            gradient_amplitude=1.0, gradient_duration=1.0e-3, diffusion_time=12.0e-3,
            mixing_time=1.0e-3, angle1=0.0, angle2=psi,
            diffusion_coefficient=2.0e-9, walkers_per_cell=48, seed=11, jitter=True,
            excitation_duration=40.0e-6, refocusing_duration=80.0e-6,
            boundary=reflector, substeps_per_interval=8,
        )
        echoes[psi] = abs(result.signal[0]) / float(rho.sum())
    return 0.5 * (0.5 * (echoes[0.0] + echoes[np.pi]) - echoes[np.pi / 2])


class PGSETests(unittest.TestCase):
    def test_moment_b_value_matches_stejskal_tanner_formula(self) -> None:
        gamma = 2.675e8
        gradient = 0.08
        delta = 3.0e-3
        delta_big = 24.0e-3

        direct = pgse_b_value(gradient, delta, delta_big, gamma=gamma)
        moment = gradient_moment_b_value(
            [
                (delta, gradient),
                (delta_big - delta, 0.0),
                (delta, -gradient),
            ],
            gamma=gamma,
        )

        self.assertAlmostEqual(direct, moment, places=6)

    def test_moment_signal_has_expected_diffusion_slope(self) -> None:
        diffusion = 2.2e-9
        result = run_pgse_moment(
            num_echoes=3,
            gradient_amplitude=0.07,
            gradient_duration=2.5e-3,
            diffusion_time=18.0e-3,
            diffusion_coefficient=diffusion,
            t2_seconds=0.150,
            first_echo_time_seconds=40.0e-3,
            echo_spacing_seconds=20.0e-3,
        )

        expected_diffusion = np.exp(-result.b_value * diffusion)
        expected_t2 = np.exp(-result.echo_times / 0.150)
        np.testing.assert_allclose(result.diffusion_attenuation, expected_diffusion)
        np.testing.assert_allclose(np.abs(result.signal), expected_diffusion * expected_t2)

    def test_zero_gradient_or_diffusion_has_no_pgse_loss(self) -> None:
        no_gradient = run_pgse_moment(
            gradient_amplitude=0.0,
            gradient_duration=2.0e-3,
            diffusion_time=20.0e-3,
            diffusion_coefficient=2.3e-9,
        )
        no_diffusion = run_pgse_moment(
            gradient_amplitude=0.08,
            gradient_duration=2.0e-3,
            diffusion_time=20.0e-3,
            diffusion_coefficient=0.0,
        )

        self.assertAlmostEqual(no_gradient.diffusion_attenuation, 1.0)
        self.assertAlmostEqual(no_diffusion.diffusion_attenuation, 1.0)

    def test_dispatches_to_requested_backend(self) -> None:
        result = run_pgse(
            backend="moment",
            gradient_amplitude=0.03,
            gradient_duration=1.0e-3,
            diffusion_time=10.0e-3,
        )

        self.assertEqual(result.backend, "moment")

    def test_walker_pgse_converges_toward_moment_attenuation(self) -> None:
        common = {
            "gradient_amplitude": 0.05,
            "gradient_duration": 2.0e-3,
            "diffusion_time": 16.0e-3,
            "walkers_per_cell": 12000,
            "seed": 123,
            "substeps_per_interval": 6,
            "excitation_duration": 60.0e-6,
            "refocusing_duration": 120.0e-6,
            "boundary": "reflect",
        }
        diffusion = 2.5e-9
        reference = run_pgse_walkers(
            **common,
            diffusion_coefficient=0.0,
        )
        diffusing = run_pgse_walkers(
            **common,
            diffusion_coefficient=diffusion,
        )
        expected = np.exp(-diffusing.b_value * diffusion)
        measured = abs(diffusing.signal[0]) / abs(reference.signal[0])

        self.assertLess(measured, 1.0)
        self.assertAlmostEqual(measured, expected, delta=0.08)

    def test_walker_spin_echo_is_num_echoes_one_special_case(self) -> None:
        result = run_pgse_walkers(
            num_echoes=1,
            gradient_amplitude=0.02,
            gradient_duration=1.0e-3,
            diffusion_time=8.0e-3,
            diffusion_coefficient=0.0,
            walkers_per_cell=128,
            seed=9,
        )

        self.assertEqual(result.signal.shape, (1,))
        self.assertEqual(result.sequence.sample_labels, ("echo_1",))

    def test_stimulated_echo_is_half_amplitude_with_stejskal_tanner_slope(self) -> None:
        rho, x_axis, z_axis = _wide_slab()
        common = {
            "rho": rho,
            "x_axis": x_axis,
            "z_axis": z_axis,
            "gradient_amplitude": 0.1,
            "gradient_duration": 1.0e-3,
            "diffusion_time": 20.0e-3,
            "walkers_per_cell": 128,
            "seed": 123,
            "jitter": True,
            "excitation_duration": 50.0e-6,
            "substeps_per_interval": 6,
        }
        diffusion = 2.5e-9
        norm = float(rho.sum())

        ste_reference = run_pgste_walkers(**common, diffusion_coefficient=0.0)
        ste_diffusing = run_pgste_walkers(**common, diffusion_coefficient=diffusion)
        spin_echo = run_pgse_walkers(
            **common, diffusion_coefficient=0.0, refocusing_duration=100.0e-6
        )

        # The stimulated echo carries half of the spin-echo amplitude.
        self.assertAlmostEqual(abs(ste_reference.signal[0]) / norm, 0.5, delta=0.05)
        self.assertAlmostEqual(abs(spin_echo.signal[0]) / norm, 1.0, delta=0.05)

        # ... and still follows the Stejskal-Tanner diffusion attenuation.
        measured = abs(ste_diffusing.signal[0]) / abs(ste_reference.signal[0])
        expected = float(np.exp(-ste_diffusing.b_value * diffusion))
        self.assertAlmostEqual(measured, expected, delta=0.08)

    def test_stimulated_echo_storage_decays_with_t1(self) -> None:
        rho, x_axis, z_axis = _wide_slab()
        common = {
            "rho": rho,
            "x_axis": x_axis,
            "z_axis": z_axis,
            "gradient_amplitude": 0.05,
            "gradient_duration": 1.0e-3,
            "diffusion_coefficient": 0.0,
            "walkers_per_cell": 96,
            "seed": 123,
            "jitter": True,
            "excitation_duration": 50.0e-6,
            # Strong spoiler + enough substeps so the residual transverse
            # coherence is fully crushed even at the shorter storage time.
            "spoiler_gradient": 0.3,
            "substeps_per_interval": 8,
            "t1_seconds": 50.0e-3,
        }
        short = run_pgste_walkers(**common, diffusion_time=20.0e-3)
        long = run_pgste_walkers(**common, diffusion_time=60.0e-3)

        ratio = abs(long.signal[0]) / abs(short.signal[0])
        expected = float(
            np.exp(-(long.storage_time - short.storage_time) / 50.0e-3)
        )
        self.assertAlmostEqual(ratio, expected, delta=0.05)

    def test_stimulated_echo_requires_room_for_storage(self) -> None:
        with self.assertRaises(ValueError):
            run_pgste_walkers(
                gradient_duration=1.0e-3,
                diffusion_time=0.5e-3,  # shorter than the encode + pulse overhead
                excitation_duration=50.0e-6,
            )

    def test_dde_refocuses_stationary_spins(self) -> None:
        rho, x, z, fields = _pore((8.0e-6, 3.0e-6))
        reflector = make_elliptical_reflector((0.0, 0.0), (8.0e-6, 3.0e-6))
        result = run_dde_walkers(
            rho=rho, x_axis=x, z_axis=z, fields=fields,
            gradient_amplitude=1.0, gradient_duration=1.0e-3, diffusion_time=12.0e-3,
            mixing_time=1.0e-3, angle1=0.0, angle2=np.pi / 2,
            diffusion_coefficient=0.0, walkers_per_cell=32, seed=5, jitter=True,
            excitation_duration=40.0e-6, refocusing_duration=80.0e-6,
            boundary=reflector, substeps_per_interval=8,
        )
        self.assertEqual(result.signal.shape, (1,))
        self.assertEqual(result.sequence.sample_labels, ("dde_echo",))
        # With no diffusion the two refocused blocks recover the full echo.
        self.assertAlmostEqual(abs(result.signal[0]) / float(rho.sum()), 1.0, delta=1e-6)

    def test_dde_cos2psi_modulation_reveals_pore_anisotropy(self) -> None:
        ellipse = _pore((8.0e-6, 3.0e-6))
        radius = float(np.sqrt(8.0e-6 * 3.0e-6))  # equal area
        circle = _pore((radius, radius))
        ellipse_e2 = _dde_cos2psi(
            make_elliptical_reflector((0.0, 0.0), (8.0e-6, 3.0e-6)), ellipse
        )
        circle_e2 = _dde_cos2psi(make_circular_reflector((0.0, 0.0), radius), circle)

        # The anisotropic pore carries a clear cos(2 psi) term; the isotropic
        # pore does not.
        self.assertLess(ellipse_e2, -0.01)
        self.assertLess(abs(circle_e2), 0.01)

    def test_dde_requires_room_for_refocusing(self) -> None:
        with self.assertRaises(ValueError):
            run_dde_walkers(
                gradient_duration=2.0e-3,
                diffusion_time=2.0e-3,  # not greater than delta + refocusing_duration
                refocusing_duration=200.0e-6,
            )

    def test_ogse_b_value_matches_cosine_spectrum_and_free_attenuation(self) -> None:
        gradient = 0.3
        frequency = 150.0
        periods = 2
        diffusion = 2.0e-9
        result = run_ogse_walkers(
            gradient_amplitude=gradient, oscillation_frequency=frequency,
            num_periods=periods, samples_per_period=16,
            diffusion_coefficient=diffusion, walkers_per_cell=4000, seed=3,
            excitation_duration=40.0e-6, refocusing_duration=80.0e-6,
            substeps_per_interval=4,
        )
        # Cosine-OGSE b-value: b = (gamma G / omega)^2 * N / f.
        omega = 2.0 * np.pi * frequency
        analytic_b = (2.675e8 * gradient / omega) ** 2 * periods / frequency
        self.assertAlmostEqual(result.b_value / analytic_b, 1.0, delta=0.02)

        # Free diffusion follows the Stejskal-Tanner exponential.
        measured = abs(result.signal[0]) / float(result.initial_ensemble.weights.sum())
        expected = float(np.exp(-result.b_value * diffusion))
        self.assertAlmostEqual(measured, expected, delta=0.05)

    def test_ogse_apparent_diffusion_rises_with_frequency(self) -> None:
        # A reflecting slab: restrict along x, thin along z (gradient is along x).
        x = np.linspace(-2.5e-6, 2.5e-6, 15)
        z = np.array([-0.5e-6, 0.5e-6])
        rho = np.ones((x.size, z.size), dtype=np.float64)
        fields = make_motion_field_maps_2d(x, z)

        def d_app(frequency):
            omega = 2.0 * np.pi * frequency
            gradient = float(omega / 2.675e8 * np.sqrt(3.0e8 * frequency / 2))
            result = run_ogse_walkers(
                rho=rho, x_axis=x, z_axis=z, fields=fields,
                gradient_amplitude=gradient, oscillation_frequency=frequency,
                num_periods=2, samples_per_period=12, diffusion_coefficient=2.0e-9,
                walkers_per_cell=160, seed=7, jitter=True,
                excitation_duration=40.0e-6, refocusing_duration=80.0e-6,
                boundary="reflect", substeps_per_interval=6,
            )
            echo = abs(result.signal[0]) / float(rho.sum())
            return -np.log(max(echo, 1e-9)) / result.b_value

        low = d_app(40.0)
        high = d_app(400.0)
        # Restriction is strong at low frequency and lifts at high frequency.
        self.assertLess(low, 0.3 * 2.0e-9)
        self.assertGreater(high, 0.4 * 2.0e-9)
        self.assertGreater(high, low)

    def test_ogse_validates_frequency_and_periods(self) -> None:
        with self.assertRaises(ValueError):
            run_ogse_walkers(oscillation_frequency=0.0)
        with self.assertRaises(ValueError):
            run_ogse_walkers(num_periods=0)


if __name__ == "__main__":
    unittest.main()
