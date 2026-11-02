from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.constants import FREE_ELECTRON_G  # noqa: E402
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
    effective_nutation_hz,
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
    simulate_frequency_spectrum_distribution,
    simulate_hahn_echo,
    simulate_hyperfine_field_sweep,
    spectrum_from_lines,
    spherical_direction,
    static_disorder_grid,
    width_from_fwhm,
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
        # The transition moment is (g / g_e) S, relative to a free electron.
        self.assertAlmostEqual(
            eigensystem.transitions[0].strength, np.sqrt(0.5) * 2.0 / FREE_ELECTRON_G
        )

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
        # |b1 . (g / g_e) S|^2 = 0.25 (g / g_e)^2, times the field-swept g_e / g.
        self.assertAlmostEqual(result.lines[0].intensity, 0.25 * 2.0 / FREE_ELECTRON_G)
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

    def test_powder_grid_order_and_b1_geometry(self) -> None:
        # The grid is built with array operations (L-64); samples still run
        # over theta, then phi, then chi, with B1 at the requested angle.
        n_theta, n_phi, n_chi, angle = 3, 5, 4, 1.1
        grid = powder_average_grid(n_theta, n_phi, n_chi, b1_b0_angle=angle)
        mu, _ = np.polynomial.legendre.leggauss(n_theta)
        index = 0
        for beta in np.arccos(mu):
            for phi_idx in range(n_phi):
                expected = spherical_direction(2.0 * np.pi * phi_idx / n_phi, beta)
                b1_set = []
                for _ in range(n_chi):
                    sample = grid[index]
                    np.testing.assert_allclose(sample.b0_direction_g, expected, atol=1e-15)
                    self.assertAlmostEqual(np.linalg.norm(sample.b1_direction_g), 1.0, places=14)
                    self.assertAlmostEqual(
                        float(sample.b0_direction_g @ sample.b1_direction_g), np.cos(angle), places=14
                    )
                    b1_set.append(sample.b1_direction_g)
                    index += 1
                # The n_chi B1 directions are equally spaced about B0.
                np.testing.assert_allclose(np.mean(b1_set, axis=0), np.cos(angle) * expected, atol=1e-14)

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
        system = ESRSpinSystem(g_tensor=FREE_ELECTRON_G)  # nominal rate exact (M-47)
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

    def test_esr_t1_recovers_toward_supplied_equilibrium(self) -> None:
        # Bloch inversion recovery: p(t) = p_eq * (1 - 2 exp(-t/T1)).
        equilibrium = np.diag([1.0, -1.0]).astype(np.complex128)
        hamiltonian = np.diag([-1.0e6, 1.0e6]).astype(np.complex128)
        t1_seconds = 2.0e-6
        for duration in (0.5e-6, 2.0e-6, 10.0e-6):
            relaxed = propagate_density_liouville(
                -equilibrium,
                hamiltonian,
                duration,
                relaxation=ESRRelaxationModel(t1_seconds=t1_seconds),
                equilibrium_density=equilibrium,
            )
            expected = (1.0 - 2.0 * np.exp(-duration / t1_seconds)) * equilibrium
            np.testing.assert_allclose(relaxed, expected, atol=1e-12)

    def test_continuous_drive_reaches_bloch_steady_state_with_t1(self) -> None:
        # Regression: T1 relaxed the pulse helpers toward zero polarization,
        # so a long resonant drive decayed to no signal. The Bloch steady
        # state on resonance is M_perp = M0 * w1*T2 / (1 + w1^2*T1*T2).
        system = ESRSpinSystem(g_tensor=FREE_ELECTRON_G)  # nominal rate exact (M-47)
        b0 = [0.0, 0.0, 0.35]
        carrier = resonance_frequency_hz(system, b0)
        nutation_hz = 0.2e6
        t1_seconds, t2_seconds = 2.0e-6, 1.0e-6
        omega1 = 2.0 * np.pi * nutation_hz
        common = dict(rf_frequency_hz=carrier, times_seconds=np.array([0.0]))

        ideal = simulate_fid(
            system,
            b0,
            nutation_hz=nutation_hz,
            pulse_duration_seconds=flip_angle_duration(np.pi / 2.0, nutation_hz),
            **common,
        )
        driven = simulate_fid(
            system,
            b0,
            nutation_hz=nutation_hz,
            pulse_duration_seconds=40.0 * t1_seconds,
            relaxation=ESRRelaxationModel(t1_seconds=t1_seconds, t2_seconds=t2_seconds),
            **common,
        )

        expected = omega1 * t2_seconds / (1.0 + omega1**2 * t1_seconds * t2_seconds)
        ratio = abs(driven.signal[0]) / abs(ideal.signal[0])
        self.assertAlmostEqual(ratio, expected, places=9)

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

    def test_hyperfine_auto_field_axis_contains_all_nitroxide_14n_lines(self) -> None:
        # Regression: the automatic span used 0.6*sum|A| and ignored I, so the
        # outer m_I = +/-1 lines of a 14N nitroxide fell outside the axis.
        hyperfine_hz = 45.0e6
        microwave_frequency_hz = 9.5e9
        system = electron_nuclear_system(
            [hyperfine_hz],
            nuclei=[NuclearSite("N", isotope="14N", spin=1.0, gamma_hz_per_t=3.077e6)],
        )

        result = simulate_hyperfine_field_sweep(
            system,
            microwave_frequency_hz,
            broadening_hz=0.5e6,
            points=2001,
        )
        spectrum = result.spectrum
        local = np.where(
            (spectrum[1:-1] > spectrum[:-2])
            & (spectrum[1:-1] > spectrum[2:])
            & (spectrum[1:-1] > 0.1 * spectrum.max())
        )[0] + 1
        peak_fields = np.sort(result.fields_tesla[local])
        # Independent first-order reference: B = (nu_mw - A*m_I) / (g mu_B/h).
        hz_per_t = BOHR_MAGNETON_HZ_PER_T * 2.00231930436256
        expected = np.sort(
            (microwave_frequency_hz - hyperfine_hz * np.array([-1.0, 0.0, 1.0]))
            / hz_per_t
        )

        self.assertEqual(len(peak_fields), 3)
        np.testing.assert_allclose(peak_fields, expected, rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(
            expected, [0.33738, 0.33898, 0.34059], rtol=0.0, atol=1.0e-5
        )

    def test_hyperfine_derivative_matches_field_sweep_derivative_convention(
        self,
    ) -> None:
        # Regression: derivative mode returned d/d(nu_mw) in per-hertz units,
        # with the opposite sign to simulate_field_sweep's d/dB.
        g = 2.0
        hz_per_t = BOHR_MAGNETON_HZ_PER_T * g
        microwave_frequency_hz = 9.5e9
        broadening_hz = 2.0e6
        center = microwave_frequency_hz / hz_per_t
        fields = np.linspace(center - 4.0e-4, center + 4.0e-4, 401)
        system = electron_nuclear_system(
            [1.0],
            nuclei=[NuclearSite("H1", gamma_hz_per_t=0.0)],
            g_tensor=g,
        )

        hyperfine = simulate_hyperfine_field_sweep(
            system,
            microwave_frequency_hz,
            broadening_hz=broadening_hz,
            fields_tesla=fields,
            detection_mode="derivative",
        )
        reference = simulate_field_sweep(
            ESRSpinSystem(g_tensor=g),
            microwave_frequency_hz,
            broadening_tesla=broadening_hz / hz_per_t,
            fields_tesla=fields,
            detection_mode="derivative",
        )
        # Intensities are per electron (L-16), so no multiplicity factor. The
        # field sweep applies the Aasa-Vanngard factor g_e / g explicitly; the
        # frequency-broadened hyperfine sweep gets 1/g through its width, so
        # at matched widths the two differ by g / g_e (1 for a free electron).
        scale = float(np.max(np.abs(reference.spectrum)))
        np.testing.assert_allclose(
            hyperfine.spectrum,
            reference.spectrum * g / FREE_ELECTRON_G,
            rtol=0.0,
            atol=1.0e-6 * scale,
        )

    def test_hyperfine_absorption_intensity_is_per_electron(self) -> None:
        # Regression (L-16): each nuclear state carried the full electron
        # intensity, so the A -> 0 limit was (2I+1)^N times the bare-electron
        # spectrum (2x for one proton, 6x for 14N plus a proton).
        g = 2.0
        hz_per_t = BOHR_MAGNETON_HZ_PER_T * g
        microwave_frequency_hz = 9.5e9
        broadening_hz = 2.0e6
        center = microwave_frequency_hz / hz_per_t
        fields = np.linspace(center - 4.0e-4, center + 4.0e-4, 201)
        reference = simulate_field_sweep(
            ESRSpinSystem(g_tensor=g),
            microwave_frequency_hz,
            broadening_tesla=broadening_hz / hz_per_t,
            fields_tesla=fields,
        )
        for nuclei in (
            [NuclearSite("H1", gamma_hz_per_t=0.0)],
            [
                NuclearSite("N", isotope="14N", spin=1.0, gamma_hz_per_t=0.0),
                NuclearSite("H1", gamma_hz_per_t=0.0),
            ],
        ):
            system = electron_nuclear_system(
                [1.0] * len(nuclei), nuclei=nuclei, g_tensor=g
            )
            hyperfine = simulate_hyperfine_field_sweep(
                system,
                microwave_frequency_hz,
                broadening_hz=broadening_hz,
                fields_tesla=fields,
            )
            # g / g_e: see test_hyperfine_derivative_matches_field_sweep_derivative_convention.
            np.testing.assert_allclose(
                hyperfine.spectrum,
                reference.spectrum * g / FREE_ELECTRON_G,
                rtol=0.0,
                atol=1.0e-9,
            )

    def test_microwave_field_couples_through_the_g_tensor(self) -> None:
        # Found by the EasySpin comparison: the transition moment was <S>, so
        # anisotropic powder patterns weighted orientations wrongly. With B0
        # along z and B1 along x the weight is (g_x / g_e)^2 / 4, times the
        # field-swept g_e / g_z.
        g = (1.95, 2.05, 2.20)
        result = simulate_field_sweep(ESRSpinSystem(g_tensor=g), 9.5e9, points=65)
        expected = (g[0] / FREE_ELECTRON_G) ** 2 / 4.0 * FREE_ELECTRON_G / g[2]
        self.assertAlmostEqual(result.lines[0].intensity, expected, places=12)
        eigensystem = diagonalize_system(ESRSpinSystem(g_tensor=g), [0.0, 0.0, 0.3])
        moment = eigensystem.transitions[0].dipole_vector
        np.testing.assert_allclose(
            np.abs(moment[:2]), np.array(g[:2]) / FREE_ELECTRON_G / 2.0, rtol=1e-12
        )

    def test_pulsed_drive_couples_through_the_g_tensor(self) -> None:
        # M-47: the drive was e1 . S, so an anisotropic g did not change the
        # nutation rate with orientation. For spin 1/2 the Rabi rate is
        # nu1 |d x m| with drive d = g^T e1 / g_e and quantization axis
        # m = g^T b0 / |g^T b0|.
        nutation = 5.0e6
        free = ESRSpinSystem(g_tensor=FREE_ELECTRON_G)
        self.assertAlmostEqual(
            effective_nutation_hz(free, [0.0, 0.0, 0.34], nutation) / nutation, 1.0, places=12
        )
        g = np.diag([1.95, 2.05, 2.20])
        system = ESRSpinSystem(g_tensor=g)
        rng = np.random.default_rng(4)
        for _ in range(6):
            b0 = rng.normal(size=3)
            b0 *= 0.33 / np.linalg.norm(b0)
            e1 = np.cross(b0, rng.normal(size=3))
            e1 /= np.linalg.norm(e1)
            axis = g.T @ b0 / np.linalg.norm(g.T @ b0)
            drive = g.T @ e1 / FREE_ELECTRON_G
            expected = nutation * np.linalg.norm(np.cross(drive, axis))
            self.assertAlmostEqual(
                effective_nutation_hz(system, b0, nutation, b1_direction_g=e1) / expected,
                1.0,
                places=10,
            )
        # A pi pulse timed with the effective rate inverts the spin.
        b0 = [0.0, 0.0, 0.31]
        rate = effective_nutation_hz(system, b0, nutation, b1_direction_g=[1.0, 0.0, 0.0])
        self.assertAlmostEqual(rate / nutation, 1.95 / FREE_ELECTRON_G, places=12)
        inverted = simulate_fid(
            system,
            b0,
            nutation_hz=nutation,
            pulse_duration_seconds=flip_angle_duration(np.pi, rate),
            times_seconds=[0.0],
        )
        quarter = simulate_fid(
            system,
            b0,
            nutation_hz=nutation,
            pulse_duration_seconds=flip_angle_duration(np.pi / 2, rate),
            times_seconds=[0.0],
        )
        self.assertLess(abs(inverted.signal[0]), 1e-9 * abs(quarter.signal[0]))

    def test_vectorized_spin_half_lines_match_diagonalization(self) -> None:
        # L-64: spin-1/2 powder lines are computed in closed form; they must
        # equal per-orientation diagonalization, including a non-diagonal g,
        # tilted B1 directions and the b1_direction_g=None default.
        from spin_dynamics.esr.orientations import ESROrientationSample
        from spin_dynamics.esr.spectra import (
            _field_lines_by_diagonalization,
            _frequency_lines_by_diagonalization,
            _spin_half_lines,
        )

        system = ESRSpinSystem(
            g_tensor=[[1.95, 0.02, 0.0], [0.01, 2.05, 0.03], [0.0, 0.02, 2.20]]
        )
        samples = powder_average_grid(6, 12, 3, b1_b0_angle=1.2) + (
            ESROrientationSample(b0_direction_g=(0.3, -0.2, 0.9), weight=0.5),
        )
        for fast, slow in (
            (
                _spin_half_lines(system, samples, microwave_frequency_hz=9.5e9),
                _field_lines_by_diagonalization(system, 9.5e9, samples),
            ),
            (
                _spin_half_lines(system, samples, field_tesla=0.34),
                _frequency_lines_by_diagonalization(system, 0.34, samples),
            ),
        ):
            self.assertEqual(len(fast), len(slow))
            for a, b in zip(fast, slow):
                self.assertEqual(a.orientation_index, b.orientation_index)
                self.assertAlmostEqual(a.field_tesla, b.field_tesla, places=14)
                self.assertAlmostEqual(a.frequency_hz / b.frequency_hz, 1.0, places=13)
                self.assertAlmostEqual(a.intensity / b.intensity, 1.0, places=12)
                self.assertAlmostEqual(a.transition_strength, b.transition_strength, places=13)
                np.testing.assert_allclose(a.b1_direction_g, b.b1_direction_g, atol=1e-15)

    def test_default_powder_grid_follows_the_anisotropy(self) -> None:
        # L-64/L-65: 2.1 polar knots per linewidth of resonance spread, 16 to 256.
        isotropic = simulate_field_sweep(
            ESRSpinSystem(g_tensor=2.0036), 9.5e9, orientations="powder", points=33
        )
        self.assertEqual(len(isotropic.lines), 16 * 32 * 3)
        rhombic = ESRSpinSystem(g_tensor=[1.95, 2.05, 2.20])
        wide = simulate_field_sweep(
            rhombic, 9.5e9, orientations="powder", broadening_tesla=3e-3, points=33
        )
        spread = 9.5e9 / BOHR_MAGNETON_HZ_PER_T * (1 / 1.95 - 1 / 2.20)
        knots = int(np.ceil(2.1 * spread / 3e-3))
        self.assertEqual(len(wide.lines), knots * 2 * knots * 3)
        narrow = simulate_field_sweep(
            rhombic, 9.5e9, orientations="powder", broadening_tesla=0.05e-3, points=33
        )
        self.assertEqual(len(narrow.lines), 256 * 512 * 3)

    def test_default_powder_grid_converges_derivative_spectra(self) -> None:
        # L-65: differentiation amplifies grid ripple; at 1.4 knots per
        # linewidth this case was 2.7% off a converged grid.
        system = ESRSpinSystem(g_tensor=[2.0, 2.1, 2.2])
        fields = np.linspace(0.29, 0.35, 1201)
        kwargs = dict(broadening_tesla=1e-3, fields_tesla=fields, detection_mode="derivative")
        default = simulate_field_sweep(system, 9.5e9, orientations="powder", **kwargs)
        converged = simulate_field_sweep(
            system, 9.5e9, orientations=powder_average_grid(200, 400, 3), **kwargs
        )
        error = np.max(np.abs(default.spectrum - converged.spectrum))
        self.assertLess(error, 1e-3 * np.max(np.abs(converged.spectrum)))

    def test_distribution_samples_share_the_powder_budget(self) -> None:
        # L-64: 25 disorder samples cap the shared grid at 256 / 5 knots.
        system = ESRSpinSystem(g_tensor=[2.0, 2.1, 2.2])
        disorder = static_disorder_grid(
            system, g_std=[0.0, 0.0, 0.005], field_std_tesla=0.1e-3, g_points=5, field_points=5
        )
        field = simulate_field_sweep_distribution(
            disorder, 9.5e9, orientations="powder", broadening_tesla=0.05e-3, points=33
        )
        self.assertEqual(len(field.lines), 25 * (51 * 102 * 3))
        frequency = simulate_frequency_spectrum_distribution(
            disorder, 0.34, orientations="powder", broadening_hz=1e6, points=33
        )
        self.assertEqual(len(frequency.lines), 25 * (51 * 102 * 3))

    def test_width_from_fwhm_matches_measured_half_maximum(self) -> None:
        # L-15: ``width`` is sigma for Gaussian but HWHM for Lorentzian; the
        # helper must give both lineshapes the requested full width.
        axis = np.linspace(-5.0, 5.0, 200001)
        fwhm = 1.7
        for name, profile in (
            ("gaussian", gaussian_lineshape),
            ("lorentzian", lorentzian_lineshape),
        ):
            values = profile(axis, 0.0, width_from_fwhm(fwhm, name))
            above = axis[values >= 0.5]
            self.assertAlmostEqual(above[-1] - above[0], fwhm, delta=1e-4)
        self.assertAlmostEqual(
            width_from_fwhm(fwhm, "gaussian"),
            fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))),
        )
        self.assertAlmostEqual(width_from_fwhm(fwhm, "lorentzian"), fwhm / 2.0)
        with self.assertRaises(ValueError):
            width_from_fwhm(fwhm, "voigt")

    def test_hyperfine_derivative_matches_numerical_field_gradient(self) -> None:
        # A resolved doublet with nuclear Zeeman: derivative mode must equal the
        # finite-difference field gradient of the absorption trace.
        system = electron_nuclear_system(
            [20.0e6],
            nuclei=[NuclearSite("H1", gamma_hz_per_t=42.57747892e6)],
            g_tensor=2.0,
        )
        microwave_frequency_hz = 9.5e9
        center = microwave_frequency_hz / (BOHR_MAGNETON_HZ_PER_T * 2.0)
        fields = np.linspace(center - 8.0e-4, center + 8.0e-4, 1601)
        kwargs = dict(broadening_hz=0.5e6, fields_tesla=fields)

        absorption = simulate_hyperfine_field_sweep(
            system, microwave_frequency_hz, **kwargs
        )
        derivative = simulate_hyperfine_field_sweep(
            system, microwave_frequency_hz, detection_mode="derivative", **kwargs
        )
        gradient = np.gradient(absorption.spectrum, fields)
        scale = float(np.max(np.abs(gradient)))

        np.testing.assert_allclose(
            derivative.spectrum[1:-1],
            gradient[1:-1],
            rtol=0.0,
            atol=2.0e-3 * scale,
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
