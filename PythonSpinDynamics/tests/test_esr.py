from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.esr import (  # noqa: E402
    BOHR_MAGNETON_HZ_PER_T,
    ESRDistributionSample,
    ESROrientationSample,
    ESRRelaxationModel,
    ESRSpinSystem,
    NuclearSite,
    diagonalize_system,
    diagonalize_hyperfine_system,
    effective_g_value,
    electron_nuclear_system,
    flip_angle_duration,
    gaussian_lineshape,
    lorentzian_lineshape,
    powder_average_grid,
    propagate_density_liouville,
    resonance_field_tesla,
    resonance_frequency_hz,
    simulate_field_sweep,
    simulate_field_sweep_distribution,
    simulate_fid,
    simulate_frequency_spectrum,
    simulate_hahn_echo,
    simulate_hyperfine_field_sweep,
    spectrum_from_lines,
    static_disorder_grid,
    zeeman_hamiltonian,
)


class ESRTests(unittest.TestCase):
    def test_isotropic_resonance_frequency_uses_bohr_magneton_scale(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)

        frequency = resonance_frequency_hz(system, [0.0, 0.0, 0.35])

        self.assertAlmostEqual(frequency, BOHR_MAGNETON_HZ_PER_T * 2.0 * 0.35)

    def test_anisotropic_effective_g_depends_on_field_direction(self) -> None:
        system = ESRSpinSystem(g_tensor=[2.0, 2.1, 2.2])

        self.assertAlmostEqual(effective_g_value(system, [1.0, 0.0, 0.0]), 2.0)
        self.assertAlmostEqual(effective_g_value(system, [0.0, 0.0, 1.0]), 2.2)

    def test_zeeman_hamiltonian_is_hermitian_and_linear_in_field(self) -> None:
        system = ESRSpinSystem(g_tensor=[2.0, 2.1, 2.2])

        h1 = zeeman_hamiltonian(system, [0.0, 0.0, 0.1])
        h2 = zeeman_hamiltonian(system, [0.0, 0.0, 0.2])

        np.testing.assert_allclose(h1, h1.conj().T, atol=1e-14)
        np.testing.assert_allclose(h2, 2.0 * h1, atol=1e-6)

    def test_diagonalized_transition_matches_resonance_frequency(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)

        eigensystem = diagonalize_system(system, [0.0, 0.0, 0.35])

        self.assertEqual(len(eigensystem.transitions), 1)
        self.assertEqual(eigensystem.transitions[0].label, "e")
        self.assertAlmostEqual(
            eigensystem.transitions[0].frequency_hz,
            resonance_frequency_hz(system, [0.0, 0.0, 0.35]),
        )
        self.assertAlmostEqual(eigensystem.transitions[0].strength, np.sqrt(0.5))

    def test_field_sweep_places_isotropic_line_at_resonance_field(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        microwave_frequency_hz = 9.5e9

        result = simulate_field_sweep(
            system,
            microwave_frequency_hz,
            broadening_tesla=1e-4,
            points=257,
        )

        expected = resonance_field_tesla(system, microwave_frequency_hz)
        self.assertEqual(len(result.lines), 1)
        self.assertAlmostEqual(result.lines[0].field_tesla, expected)
        self.assertAlmostEqual(result.lines[0].intensity, 0.25)
        self.assertAlmostEqual(
            result.fields_tesla[np.argmax(result.spectrum)],
            expected,
            delta=result.fields_tesla[1] - result.fields_tesla[0],
        )

    def test_frequency_spectrum_respects_rf_dark_parallel_geometry(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        orientations = [
            ESROrientationSample(
                b0_direction_g=(0.0, 0.0, 1.0),
                b1_direction_g=(0.0, 0.0, 1.0),
            )
        ]

        result = simulate_frequency_spectrum(
            system,
            0.35,
            orientations=orientations,
            broadening_hz=1e6,
        )

        self.assertEqual(len(result.lines), 0)
        np.testing.assert_allclose(result.spectrum, 0.0)

    def test_powder_grid_weights_are_normalized(self) -> None:
        grid = powder_average_grid(n_theta=3, n_phi=4, n_chi=5)

        self.assertEqual(len(grid), 60)
        self.assertAlmostEqual(sum(sample.weight for sample in grid), 1.0)

    def test_lorentzian_has_heavier_tails_than_gaussian(self) -> None:
        axis = np.array([0.0, 3.0])

        gaussian = gaussian_lineshape(axis, 0.0, 1.0)
        lorentzian = lorentzian_lineshape(axis, 0.0, 1.0)

        self.assertAlmostEqual(gaussian[0], 1.0)
        self.assertAlmostEqual(lorentzian[0], 1.0)
        self.assertGreater(lorentzian[1], gaussian[1])

    def test_derivative_lineshape_crosses_zero_at_line_center(self) -> None:
        axis = np.array([-1.0, 0.0, 1.0])

        spectrum = spectrum_from_lines(
            axis,
            [0.0],
            [1.0],
            width=1.0,
            detection_mode="derivative",
        )

        self.assertGreater(spectrum[0], 0.0)
        self.assertAlmostEqual(spectrum[1], 0.0)
        self.assertLess(spectrum[2], 0.0)

    def test_field_sweep_supports_derivative_detection_mode(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        microwave_frequency_hz = 9.5e9
        center = resonance_field_tesla(system, microwave_frequency_hz)
        fields = np.linspace(center - 5.0e-4, center + 5.0e-4, 1001)

        result = simulate_field_sweep(
            system,
            microwave_frequency_hz,
            broadening_tesla=1.0e-4,
            fields_tesla=fields,
            detection_mode="derivative",
        )

        self.assertEqual(result.detection_mode, "derivative")
        self.assertGreater(np.max(result.spectrum), 0.0)
        self.assertLess(np.min(result.spectrum), 0.0)
        self.assertAlmostEqual(result.spectrum[fields.size // 2], 0.0, places=12)

    def test_static_disorder_grid_weights_are_normalized(self) -> None:
        system = ESRSpinSystem(g_tensor=[2.0, 2.1, 2.2])

        samples = static_disorder_grid(
            system,
            g_std=[0.0, 0.0, 0.01],
            field_std_tesla=1.0e-4,
            g_points=3,
            field_points=3,
        )

        self.assertEqual(len(samples), 9)
        self.assertAlmostEqual(sum(sample.weight for sample in samples), 1.0)

    def test_g_strain_broadens_field_sweep_line_positions(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        microwave_frequency_hz = 9.5e9
        samples = static_disorder_grid(
            system,
            g_std=[0.0, 0.0, 0.02],
            g_points=5,
            field_points=1,
        )

        result = simulate_field_sweep_distribution(
            samples,
            microwave_frequency_hz,
            broadening_tesla=1.0e-4,
            points=257,
        )
        centers = np.array([line.field_tesla for line in result.lines])

        self.assertEqual(len(result.lines), 5)
        self.assertGreater(np.ptp(centers), 5.0e-3)

    def test_field_offset_distribution_shifts_applied_resonance_field(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        microwave_frequency_hz = 9.5e9
        center = resonance_field_tesla(system, microwave_frequency_hz)
        samples = [
            ESRDistributionSample(system, weight=0.5, field_offset_tesla=-1.0e-3),
            ESRDistributionSample(system, weight=0.5, field_offset_tesla=1.0e-3),
        ]

        result = simulate_field_sweep_distribution(
            samples,
            microwave_frequency_hz,
            broadening_tesla=1.0e-4,
            points=257,
        )
        centers = sorted(line.field_tesla for line in result.lines)

        self.assertAlmostEqual(centers[0], center - 1.0e-3)
        self.assertAlmostEqual(centers[1], center + 1.0e-3)

    def test_flip_angle_duration_uses_spin_half_nutation_rate(self) -> None:
        self.assertAlmostEqual(flip_angle_duration(np.pi / 2.0, 1.0e6), 0.25e-6)
        self.assertAlmostEqual(flip_angle_duration(np.pi, 1.0e6), 0.5e-6)

    def test_resonant_90_degree_pulse_creates_transverse_fid(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        b0 = [0.0, 0.0, 0.35]
        carrier = resonance_frequency_hz(system, b0)
        nutation_hz = 1.0e6

        result = simulate_fid(
            system,
            b0,
            nutation_hz=nutation_hz,
            pulse_duration_seconds=flip_angle_duration(np.pi / 2.0, nutation_hz),
            times_seconds=np.array([0.0, 1.0e-6]),
            rf_frequency_hz=carrier,
        )

        np.testing.assert_allclose(np.abs(result.signal), 0.5, atol=1e-12)
        self.assertAlmostEqual(result.rf_frequency_hz, carrier)

    def test_esr_t1_relaxation_damps_population_difference(self) -> None:
        density = np.diag([1.0, -1.0]).astype(np.complex128)
        hamiltonian = np.zeros((2, 2), dtype=np.complex128)
        t1_seconds = 2.0e-6

        relaxed = propagate_density_liouville(
            density,
            hamiltonian,
            t1_seconds,
            relaxation=ESRRelaxationModel(t1_seconds=t1_seconds),
        )

        expected = np.exp(-1.0) * density
        np.testing.assert_allclose(relaxed, expected, atol=1e-12)

    def test_esr_relaxation_model_controls_fid_t2_decay(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        b0 = [0.0, 0.0, 0.35]
        carrier = resonance_frequency_hz(system, b0)
        nutation_hz = 1.0e6
        t2_seconds = 2.0e-6

        result = simulate_fid(
            system,
            b0,
            nutation_hz=nutation_hz,
            pulse_duration_seconds=flip_angle_duration(np.pi / 2.0, nutation_hz),
            times_seconds=np.array([0.0, t2_seconds]),
            rf_frequency_hz=carrier,
            relaxation=ESRRelaxationModel(t2_seconds=t2_seconds),
        )

        ratio = abs(result.signal[1] / result.signal[0])
        self.assertAlmostEqual(ratio, np.exp(-1.0), places=12)

    def test_finite_t2_envelope_warns_when_relaxation_model_is_supplied(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        b0 = [0.0, 0.0, 0.35]
        carrier = resonance_frequency_hz(system, b0)

        with self.assertWarns(RuntimeWarning):
            simulate_fid(
                system,
                b0,
                nutation_hz=1.0e6,
                pulse_duration_seconds=flip_angle_duration(np.pi / 2.0, 1.0e6),
                times_seconds=np.array([0.0, 1.0e-6]),
                rf_frequency_hz=carrier,
                t2_seconds=10.0e-6,
                relaxation=ESRRelaxationModel(t2_seconds=10.0e-6),
            )

    def test_single_nucleus_hyperfine_doublet_splitting_matches_coupling(self) -> None:
        system = electron_nuclear_system(
            [20.0e6],
            nuclei=[NuclearSite("H1", gamma_hz_per_t=0.0)],
            g_tensor=2.0,
        )

        eigensystem = diagonalize_hyperfine_system(system, [0.0, 0.0, 0.35])
        strong = sorted(
            eigensystem.transitions,
            key=lambda transition: abs(transition.dipole_vector[0]),
            reverse=True,
        )[:2]
        frequencies = sorted(transition.frequency_hz for transition in strong)

        self.assertEqual(system.dimension, 4)
        self.assertAlmostEqual(frequencies[1] - frequencies[0], 20.0e6, delta=1e3)

    def test_hyperfine_field_sweep_resolves_two_strong_lines(self) -> None:
        system = electron_nuclear_system(
            [20.0e6],
            nuclei=[NuclearSite("H1", gamma_hz_per_t=0.0)],
            g_tensor=2.0,
        )
        microwave_frequency_hz = 9.5e9

        result = simulate_hyperfine_field_sweep(
            system,
            microwave_frequency_hz,
            broadening_hz=0.5e6,
            points=801,
        )
        spectrum = result.spectrum
        local = np.where(
            (spectrum[1:-1] > spectrum[:-2])
            & (spectrum[1:-1] > spectrum[2:])
        )[0] + 1
        peak_indices = local[np.argsort(spectrum[local])[-2:]]
        peak_fields = np.sort(result.fields_tesla[peak_indices])
        expected_split = 20.0e6 / (BOHR_MAGNETON_HZ_PER_T * 2.0)

        self.assertEqual(len(peak_fields), 2)
        self.assertAlmostEqual(
            peak_fields[1] - peak_fields[0],
            expected_split,
            delta=2e-5,
        )

    def test_detuned_isochromat_ensemble_rephases_at_hahn_echo_time(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        b0 = 0.35
        carrier = resonance_frequency_hz(system, [0.0, 0.0, b0])
        gamma_eff_hz_per_t = BOHR_MAGNETON_HZ_PER_T * 2.0
        nutation_hz = 1.0e6
        tau = 10.0e-6
        times = np.linspace(0.0, 20.0e-6, 101)
        summed = np.zeros(times.size, dtype=np.complex128)

        for offset_hz in np.linspace(-2.0e5, 2.0e5, 21):
            shifted_b0 = (carrier + offset_hz) / gamma_eff_hz_per_t
            result = simulate_hahn_echo(
                system,
                [0.0, 0.0, shifted_b0],
                nutation_hz=nutation_hz,
                excitation_duration_seconds=flip_angle_duration(
                    np.pi / 2.0,
                    nutation_hz,
                ),
                refocus_duration_seconds=flip_angle_duration(np.pi, nutation_hz),
                tau_seconds=tau,
                times_seconds=times,
                rf_frequency_hz=carrier,
            )
            summed += result.signal

        peak_time = times[int(np.argmax(np.abs(summed)))]
        self.assertAlmostEqual(peak_time, tau, delta=0.5e-6)
        self.assertGreater(np.max(np.abs(summed)), 10.0 * abs(summed[0]))


if __name__ == "__main__":
    unittest.main()
