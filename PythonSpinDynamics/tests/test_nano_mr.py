from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.esr import (  # noqa: E402
    BOHR_MAGNETON_HZ_PER_T,
    ESRSpinSystem,
    diagonalize_system,
)
from spin_dynamics.nano_mr import (  # noqa: E402
    DEFAULT_MAX_HILBERT_DIMENSION,
    DIAMOND_NV_ZFS_HZ,
    ISOTOPE_GAMMA_HZ_PER_T,
    MU0_OVER_4PI,
    PLANCK_CONSTANT_J_S,
    SIC_PL6_ZFS_HZ,
    ClockModel,
    CoherentNMRSite,
    CoordinateFrame,
    CorrelatedFieldNoiseModel,
    DNPModel,
    DefectSpinSensor,
    FieldNoiseComponent,
    HighResolutionBudget,
    NuclearSpin,
    NuclearScalarCoupling,
    NuclearBathSpecies,
    OpticalCycleModel,
    OpticalReadoutModel,
    QdyneProtocol,
    ResolvedNucleus,
    ResolvedSpinCluster,
    SensingSequence,
    SPADDetectorModel,
    SurfaceGeometry,
    TimedControlPulse,
    UniformBathComponent,
    UniformNuclearLayer,
    VoxelBathComponent,
    VoxelNuclearSample,
    arbitrary_scan,
    build_depth_profile_operator,
    build_dipolar_forward_operator,
    build_voxel_density_forward_operator,
    compile_qubit_sequence,
    correlation_spectrum_2d,
    cpmg_sequence,
    dephasing_filter_function,
    defect_sensor_from_esr,
    diamond_nv_minus,
    diagonalize_sensor,
    dipolar_field_from_moments,
    dipole_spatial_tensor_inverse_m3,
    dnp_polarization,
    field_autocorrelation,
    field_power_spectral_density,
    effective_sample_size,
    free_diffusion_return_density,
    hahn_echo_sequence,
    ideal_nuclear_rotation,
    kdd_sequence,
    gaussian_filter_coherence,
    localize_point_sources,
    linear_field_covariance,
    modulation_function,
    nuclear_voxel_variance_amplitudes,
    planar_voxel_grid,
    point_dipolar_hyperfine_tensor_hz,
    propagate_controlled_qubit,
    ramsey_sequence,
    raster_scan,
    reconstruct_nonnegative_density,
    rotating_lorentzian_psd,
    sample_optical_readout,
    sample_correlated_field_noise,
    sample_time_resolved_optical_readout,
    sensor_memory_correlation,
    sensor_nuclear_hyperfine_hamiltonian,
    sensor_array,
    sensor_hamiltonian,
    simulate_statistical_spectrum,
    simulate_resolved_cw_spectrum,
    simulate_diffusing_dipolar_field,
    simulate_coherent_nmr_spectrum,
    simulate_qdyne,
    simulate_synchronized_readout,
    simulate_two_block_correlation,
    sic_pl6,
    toggling_integral,
    trajectory_displacement_statistics,
    with_nuclear_rf_pulses,
    xy_sequence,
    zeeman_hamiltonian,
    zfs_hamiltonian,
    zfs_tensor_from_d_e,
    esr_system_from_defect,
    nuclear_dipolar_hamiltonian,
    nuclear_rf_hamiltonian,
    nuclear_scalar_coupling_hamiltonian,
    nuclear_zeeman_hamiltonian,
    resolved_cluster_hamiltonian,
)


class NanoMRFrameTests(unittest.TestCase):
    def test_frame_round_trip_and_axes(self) -> None:
        frame = CoordinateFrame.from_z_axis([1.0, 0.0, 0.0], name="tilted")
        local = np.array([0.2, -0.4, 0.7])

        np.testing.assert_allclose(frame.z_axis_lab, [1.0, 0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(
            frame.vector_to_local(frame.vector_to_lab(local)),
            local,
            atol=1e-14,
        )
        self.assertAlmostEqual(np.linalg.det(frame.rotation_lab_from_local), 1.0)

    def test_frame_rejects_reflection(self) -> None:
        with self.assertRaisesRegex(ValueError, "determinant"):
            CoordinateFrame(np.diag([1.0, 1.0, -1.0]))


class NanoMRSensorTests(unittest.TestCase):
    def test_diamond_nv_zero_field_transition_matches_preset_zfs(self) -> None:
        sensor = diamond_nv_minus()

        result = diagonalize_sensor(sensor)

        self.assertEqual(sensor.spin, 1.0)
        self.assertEqual(sensor.dimension, 3)
        self.assertEqual(len(result.transitions), 2)
        for transition in result.transitions:
            self.assertAlmostEqual(transition.frequency_hz, DIAMOND_NV_ZFS_HZ)

    def test_axial_field_produces_d_plus_minus_zeeman_transitions(self) -> None:
        sensor = diamond_nv_minus()
        field_tesla = 5.0e-3

        result = diagonalize_sensor(sensor, [0.0, 0.0, field_tesla])

        shift = BOHR_MAGNETON_HZ_PER_T * 2.0028 * field_tesla
        expected = np.array(
            [DIAMOND_NV_ZFS_HZ - shift, DIAMOND_NV_ZFS_HZ + shift]
        )
        observed = np.sort(
            [transition.frequency_hz for transition in result.transitions]
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-5)

    def test_tilted_sensor_is_covariant_when_field_and_drive_rotate(self) -> None:
        axial = diamond_nv_minus(axis_lab=[0.0, 0.0, 1.0])
        tilted = diamond_nv_minus(axis_lab=[1.0, 0.0, 0.0])
        field_tesla = 8.0e-3

        axial_result = diagonalize_sensor(
            axial,
            [0.0, 0.0, field_tesla],
            drive_direction_lab=[1.0, 0.0, 0.0],
        )
        tilted_result = diagonalize_sensor(
            tilted,
            [field_tesla, 0.0, 0.0],
            drive_direction_lab=[0.0, 1.0, 0.0],
        )

        np.testing.assert_allclose(
            axial_result.levels_hz,
            tilted_result.levels_hz,
            rtol=1e-12,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            sorted(item.frequency_hz for item in axial_result.transitions),
            sorted(item.frequency_hz for item in tilted_result.transitions),
            rtol=1e-12,
            atol=1e-5,
        )

    def test_pl6_preset_has_spin_one_and_expected_zero_field_splitting(self) -> None:
        sensor = sic_pl6()

        result = diagonalize_sensor(sensor)

        self.assertEqual(sensor.material, "4H-SiC")
        self.assertEqual(sensor.defect, "PL6")
        self.assertEqual(len(result.transitions), 2)
        self.assertTrue(
            all(
                np.isclose(item.frequency_hz, SIC_PL6_ZFS_HZ)
                for item in result.transitions
            )
        )

    def test_generic_spin_three_halves_uses_arbitrary_spin_operators(self) -> None:
        d_hz = 35.0e6
        sensor = DefectSpinSensor(
            spin=1.5,
            g_tensor=2.0,
            zfs_tensor_hz=zfs_tensor_from_d_e(d_hz),
            frame=CoordinateFrame.from_z_axis([0.0, 0.0, 1.0]),
            depth_nm=3.0,
            material="4H-SiC",
            defect="V2 test model",
        )

        result = diagonalize_sensor(sensor)

        self.assertEqual(sensor.dimension, 4)
        self.assertEqual(len(result.transitions), 2)
        self.assertTrue(
            all(np.isclose(item.frequency_hz, 2.0 * d_hz) for item in result.transitions)
        )

    def test_hamiltonian_terms_are_hermitian_and_zeeman_is_linear(self) -> None:
        sensor = diamond_nv_minus(e_hz=2.0e6)
        zfs = zfs_hamiltonian(sensor)
        zeeman_1 = zeeman_hamiltonian(sensor, [1.0e-4, 2.0e-4, 3.0e-4])
        zeeman_2 = zeeman_hamiltonian(sensor, [2.0e-4, 4.0e-4, 6.0e-4])
        total = sensor_hamiltonian(sensor, [1.0e-4, 2.0e-4, 3.0e-4])

        np.testing.assert_allclose(zfs, zfs.conj().T, atol=1e-8)
        np.testing.assert_allclose(zeeman_2, 2.0 * zeeman_1, atol=1e-8)
        np.testing.assert_allclose(total, total.conj().T, atol=1e-8)


class NanoMRGeometryTests(unittest.TestCase):
    def test_surface_places_sensor_below_sample_and_reports_signed_distance(self) -> None:
        surface = SurfaceGeometry([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        sensor = diamond_nv_minus(depth_nm=4.5)

        position = surface.sensor_position_lab_nm(sensor)

        np.testing.assert_allclose(position, [0.0, 0.0, -3.5])
        distances = surface.signed_distance_nm(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], position]
        )
        np.testing.assert_allclose(distances, [0.0, 2.0, -4.5])

    def test_dipole_spatial_tensor_is_traceless_and_inverse_cube(self) -> None:
        near = dipole_spatial_tensor_inverse_m3([0.0, 0.0, 2.0])
        far = dipole_spatial_tensor_inverse_m3([0.0, 0.0, 4.0])

        self.assertAlmostEqual(float(np.trace(near) / np.linalg.norm(near)), 0.0)
        np.testing.assert_allclose(near, 8.0 * far, rtol=1e-14, atol=1e-4)

    def test_point_hyperfine_tensor_scales_as_inverse_cube(self) -> None:
        sensor = diamond_nv_minus(depth_nm=1.0)
        proton_near = NuclearSpin.from_isotope("1H", [0.0, 0.0, 1.0])
        proton_far = NuclearSpin.from_isotope("1H", [0.0, 0.0, 2.0])

        near = point_dipolar_hyperfine_tensor_hz(
            sensor, proton_near, sensor_position_lab_nm=[0.0, 0.0, 0.0]
        )
        far = point_dipolar_hyperfine_tensor_hz(
            sensor, proton_far, sensor_position_lab_nm=[0.0, 0.0, 0.0]
        )

        np.testing.assert_allclose(near, 8.0 * far, rtol=1e-14, atol=1e-8)
        self.assertGreater(np.linalg.norm(near), 1.0e4)

    def test_unknown_isotope_requires_explicit_gamma(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown isotope"):
            NuclearSpin.from_isotope("2H", [0.0, 0.0, 1.0])


class NanoMRSequenceTests(unittest.TestCase):
    def test_standard_sequence_timing_and_xy_phases(self) -> None:
        ramsey = ramsey_sequence(10.0e-6)
        hahn = hahn_echo_sequence(5.0e-6)
        cpmg = cpmg_sequence(4, 16.0e-6)
        xy8 = xy_sequence(8, 1, 16.0e-6)

        self.assertEqual(ramsey.electron_pulses, ())
        self.assertAlmostEqual(hahn.electron_pulses[0].center_seconds, 5.0e-6)
        np.testing.assert_allclose(
            [pulse.center_seconds for pulse in cpmg.electron_pulses],
            [2.0e-6, 6.0e-6, 10.0e-6, 14.0e-6],
        )
        np.testing.assert_allclose(
            xy8.cycle_pulse_phases_rad,
            [0.0, np.pi / 2, 0.0, np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0],
        )

    def test_kdd_phase_cycle_and_pulse_error_robustness(self) -> None:
        duration = 400.0e-6
        kdd = kdd_sequence(1, duration)
        knill = np.array([np.pi / 6.0, 0.0, np.pi / 2.0, 0.0, np.pi / 6.0])
        expected_phases = np.concatenate((knill, knill + np.pi / 2.0) * 2)

        self.assertEqual(kdd.name, "KDD-20")
        np.testing.assert_allclose(kdd.cycle_pulse_phases_rad, expected_phases)
        np.testing.assert_allclose(
            [pulse.center_seconds for pulse in kdd.electron_pulses],
            [(index + 0.5) * duration / 20.0 for index in range(20)],
        )
        probe_frequencies = 2.0 * np.pi * np.array([100.0, 10.0e3, 25.0e3])
        np.testing.assert_allclose(
            dephasing_filter_function(kdd, probe_frequencies),
            dephasing_filter_function(
                cpmg_sequence(20, duration),
                probe_frequencies,
            ),
        )

        flip_error = 0.02

        def with_flip_error(sequence: SensingSequence) -> SensingSequence:
            pulses = tuple(
                replace(
                    pulse,
                    flip_angle_rad=pulse.flip_angle_rad * (1.0 + flip_error),
                )
                for pulse in sequence.electron_pulses
            )
            return SensingSequence(sequence.total_duration_seconds, pulses)

        kdd_result = propagate_controlled_qubit(with_flip_error(kdd))
        cpmg_result = propagate_controlled_qubit(
            with_flip_error(cpmg_sequence(20, duration))
        )
        kdd_state_infidelity = 0.5 * (1.0 - kdd_result.bloch_vector[0])
        cpmg_state_infidelity = 0.5 * (1.0 - cpmg_result.bloch_vector[0])

        self.assertLess(kdd_state_infidelity, 1.0e-10)
        self.assertGreater(cpmg_state_infidelity, 0.3)

    def test_simultaneous_microwave_and_nuclear_rf_are_distinct_channels(self) -> None:
        microwave = hahn_echo_sequence(
            5.0e-6,
            pulse_duration_seconds=1.0e-6,
        )
        rf = TimedControlPulse(
            center_seconds=5.0e-6,
            duration_seconds=2.0e-6,
            flip_angle_rad=np.pi,
            channel="nuclear_rf",
        )

        combined = with_nuclear_rf_pulses(microwave, [rf])

        self.assertEqual(len(combined.electron_pulses), 1)
        self.assertEqual(len(combined.nuclear_rf_pulses), 1)
        np.testing.assert_allclose(
            modulation_function(combined, [0.0, 6.0e-6]),
            [1.0, -1.0],
        )

    def test_overlapping_pulses_on_one_channel_are_rejected(self) -> None:
        pulses = (
            TimedControlPulse(4.0e-6, np.pi, duration_seconds=4.0e-6),
            TimedControlPulse(5.0e-6, np.pi, duration_seconds=4.0e-6),
        )

        with self.assertRaisesRegex(ValueError, "same channel"):
            SensingSequence(10.0e-6, pulses)


class NanoMRFilterFunctionTests(unittest.TestCase):
    def test_ideal_and_finite_width_modulation(self) -> None:
        ideal = hahn_echo_sequence(5.0e-6)
        finite = hahn_echo_sequence(
            5.0e-6,
            pulse_duration_seconds=2.0e-6,
        )
        times = [0.0, 4.0e-6, 5.0e-6, 6.0e-6, 10.0e-6]

        np.testing.assert_allclose(
            modulation_function(ideal, times),
            [1.0, 1.0, -1.0, -1.0, -1.0],
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            modulation_function(finite, times, pulse_model="finite"),
            [1.0, 1.0, 0.0, -1.0, -1.0],
            atol=1.0e-14,
        )

    def test_finite_model_supports_mixed_instantaneous_and_finite_pulses(self) -> None:
        sequence = SensingSequence(
            10.0e-6,
            (
                TimedControlPulse(3.0e-6, np.pi),
                TimedControlPulse(
                    7.0e-6,
                    np.pi,
                    duration_seconds=2.0e-6,
                ),
            ),
        )

        values = modulation_function(
            sequence,
            [2.0e-6, 4.0e-6, 7.0e-6, 8.0e-6, 9.0e-6],
            pulse_model="finite",
        )

        np.testing.assert_allclose(values, [1.0, -1.0, 0.0, 1.0, 1.0], atol=1e-14)

    def test_cpmg_filter_peaks_near_sequence_passband(self) -> None:
        sequence = cpmg_sequence(8, 80.0e-6)
        resonance = np.pi * 8 / sequence.total_duration_seconds
        frequencies = resonance * np.array([0.5, 1.0, 1.5])

        values = dephasing_filter_function(sequence, frequencies)

        self.assertGreater(values[1], values[0])
        self.assertGreater(values[1], values[2])

    def test_weak_ac_propagation_matches_analytic_toggling_integral(self) -> None:
        sequence = cpmg_sequence(
            4,
            40.0e-6,
            phase_rad=0.0,
        )
        omega = np.pi * 4 / sequence.total_duration_seconds
        amplitude = 2.0 * np.pi * 300.0
        phase = 0.37
        analytic_phase = np.real(
            amplitude
            * np.exp(1.0j * phase)
            * toggling_integral(sequence, omega)
        )

        result = propagate_controlled_qubit(
            sequence,
            lambda time: amplitude * np.cos(omega * time + phase),
            max_step_seconds=25.0e-9,
        )

        self.assertAlmostEqual(
            np.angle(result.coherence),
            analytic_phase,
            places=5,
        )

    def test_short_finite_pulse_filter_approaches_ideal_result(self) -> None:
        duration = 40.0e-6
        frequency = np.pi * 4 / duration
        ideal = cpmg_sequence(4, duration)
        finite = cpmg_sequence(
            4,
            duration,
            pulse_duration_seconds=10.0e-9,
        )

        ideal_value = dephasing_filter_function(ideal, frequency)
        finite_value = dephasing_filter_function(
            finite,
            frequency,
            pulse_model="finite",
            samples_per_pulse=128,
        )

        self.assertAlmostEqual(finite_value / ideal_value, 1.0, places=3)


class NanoMRCompilerTests(unittest.TestCase):
    def test_compiler_separates_free_and_pulse_steps(self) -> None:
        ideal = compile_qubit_sequence(hahn_echo_sequence(5.0e-6))
        finite = compile_qubit_sequence(
            hahn_echo_sequence(5.0e-6, pulse_duration_seconds=1.0e-6)
        )

        self.assertEqual([step.kind for step in ideal.steps], ["free", "instantaneous", "free"])
        self.assertEqual([step.kind for step in finite.steps], ["free", "pulse", "free"])

    def test_ramsey_phase_and_hahn_static_detuning_cancellation(self) -> None:
        duration = 17.0e-6
        detuning = 2.0 * np.pi * 25.0e3

        ramsey = propagate_controlled_qubit(
            ramsey_sequence(duration),
            detuning,
        )
        hahn = propagate_controlled_qubit(
            hahn_echo_sequence(duration / 2.0),
            detuning,
        )

        self.assertAlmostEqual(
            np.angle(ramsey.coherence),
            detuning * duration,
            places=12,
        )
        np.testing.assert_allclose(hahn.bloch_vector, [1.0, 0.0, 0.0], atol=1e-12)


class NanoMROpticalReadoutTests(unittest.TestCase):
    def test_expected_counts_include_contrast_background_and_repetitions(self) -> None:
        model = OpticalReadoutModel(
            initialization_fidelity=0.95,
            initialization_seconds=2.0e-6,
            bright_count_rate_hz=200.0e3,
            readout_contrast=0.25,
            readout_seconds=1.0e-6,
            background_count_rate_hz=20.0e3,
            dead_time_seconds=3.0e-6,
        )

        counts = model.expected_counts([1.0, 0.0], repetitions=100)

        np.testing.assert_allclose(counts, [22.0, 17.0])
        self.assertAlmostEqual(model.initialized_bright_probability, 0.95)
        self.assertAlmostEqual(model.cycle_seconds(4.0e-6), 10.0e-6)

    def test_initialization_fidelity_reduces_spin_contrast(self) -> None:
        model = OpticalReadoutModel(
            initialization_fidelity=0.75,
            bright_count_rate_hz=1.0,
            readout_contrast=1.0,
            readout_seconds=1.0,
        )

        result = sample_optical_readout(model, [0.0, 1.0], seed=4)

        np.testing.assert_allclose(
            result.effective_bright_probability,
            [0.25, 0.75],
        )
        np.testing.assert_allclose(result.expected_counts, [0.25, 0.75])

    def test_seeded_poisson_readout_is_reproducible(self) -> None:
        model = OpticalReadoutModel(
            bright_count_rate_hz=1.0e6,
            readout_contrast=0.3,
            readout_seconds=1.0e-6,
        )

        first = sample_optical_readout(
            model,
            [0.0, 0.5, 1.0],
            repetitions=10_000,
            sensing_seconds=20.0e-6,
            seed=1234,
        )
        second = sample_optical_readout(
            model,
            [0.0, 0.5, 1.0],
            repetitions=10_000,
            sensing_seconds=20.0e-6,
            seed=1234,
        )

        np.testing.assert_array_equal(first.sampled_counts, second.sampled_counts)
        self.assertAlmostEqual(first.acquisition_seconds, 0.21)


class NanoMRBathTests(unittest.TestCase):
    def test_statistical_thermal_and_fixed_spin_half_populations(self) -> None:
        statistical = NuclearBathSpecies.from_isotope("1H")
        thermal = NuclearBathSpecies.from_isotope(
            "1H",
            polarization_mode="thermal",
            temperature_kelvin=0.1,
        )
        fixed = NuclearBathSpecies.from_isotope(
            "1H",
            polarization_mode="fixed",
            polarization_fraction=0.6,
        )
        field_tesla = 1.0
        thermal_exponent = (
            PLANCK_CONSTANT_J_S
            * thermal.gamma_hz_per_t
            * field_tesla
            / (1.380649e-23 * thermal.temperature_kelvin)
        )

        np.testing.assert_allclose(
            statistical.level_probabilities(field_tesla),
            [0.5, 0.5],
        )
        self.assertAlmostEqual(statistical.mean_spin_projection(field_tesla), 0.0)
        self.assertAlmostEqual(
            statistical.transverse_spin_second_moment(field_tesla),
            0.25,
        )
        self.assertAlmostEqual(
            thermal.mean_spin_projection(field_tesla),
            0.5 * np.tanh(0.5 * thermal_exponent),
        )
        self.assertAlmostEqual(fixed.mean_spin_projection(field_tesla), 0.3)

    def test_fixed_spin_one_polarization_changes_transverse_second_moment(self) -> None:
        unpolarized = NuclearBathSpecies(
            isotope="test",
            gamma_hz_per_t=10.0e6,
            spin=1.0,
        )
        polarized = NuclearBathSpecies(
            isotope="test",
            gamma_hz_per_t=10.0e6,
            spin=1.0,
            polarization_mode="fixed",
            polarization_fraction=1.0,
        )

        self.assertAlmostEqual(unpolarized.transverse_spin_second_moment(1.0), 2 / 3)
        self.assertAlmostEqual(polarized.mean_spin_projection(1.0), 1.0)
        self.assertAlmostEqual(polarized.transverse_spin_second_moment(1.0), 0.5)

    def test_statistical_and_coherent_polarization_have_sqrt_n_and_n_scaling(
        self,
    ) -> None:
        statistical = NuclearBathSpecies.from_isotope("1H")
        fixed = NuclearBathSpecies.from_isotope(
            "1H",
            polarization_mode="fixed",
            polarization_fraction=0.2,
        )
        statistical_small = statistical.polarization_scaling(0.1, 100.0)
        statistical_large = statistical.polarization_scaling(0.1, 400.0)
        fixed_small = fixed.polarization_scaling(0.1, 100.0)
        fixed_large = fixed.polarization_scaling(0.1, 400.0)

        self.assertEqual(statistical_small.coherent_mean_projection, 0.0)
        self.assertAlmostEqual(
            statistical_large.statistical_rms_projection
            / statistical_small.statistical_rms_projection,
            2.0,
        )
        self.assertAlmostEqual(
            fixed_large.coherent_mean_projection
            / fixed_small.coherent_mean_projection,
            4.0,
        )

    def test_uniform_half_space_matches_axial_analytic_variance(self) -> None:
        density_m3 = 6.7e28
        depth_nm = 5.0
        sensor = diamond_nv_minus(depth_nm=depth_nm)
        surface = SurfaceGeometry([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        proton = NuclearBathSpecies.from_isotope("1H")
        sample = UniformNuclearLayer(
            surface,
            (UniformBathComponent(proton, density_m3),),
        )
        result = simulate_statistical_spectrum(
            sensor,
            sample,
            [0.0, 0.0, 0.05],
            np.linspace(-3.0e7, 3.0e7, 101),
        )
        magnetic_moment_scale = (
            MU0_OVER_4PI
            * PLANCK_CONSTANT_J_S
            * proton.gamma_hz_per_t
        )
        expected = (
            density_m3
            * 0.25
            * magnetic_moment_scale**2
            * np.pi
            / 4.0
            * (depth_nm * 1.0e-9) ** -3
        )

        self.assertAlmostEqual(
            result.total_field_variance_t2 / expected,
            1.0,
            places=12,
        )

    def test_half_space_variance_has_inverse_cube_depth_scaling(self) -> None:
        surface = SurfaceGeometry([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        sample = UniformNuclearLayer(
            surface,
            (
                UniformBathComponent(
                    NuclearBathSpecies.from_isotope("1H"),
                    6.7e28,
                ),
            ),
        )
        frequency_axis = np.linspace(-1.0e7, 1.0e7, 11)
        near = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=2.0),
            sample,
            [0.0, 0.0, 0.02],
            frequency_axis,
        )
        far = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=4.0),
            sample,
            [0.0, 0.0, 0.02],
            frequency_axis,
        )

        self.assertAlmostEqual(
            near.total_field_variance_t2 / far.total_field_variance_t2,
            8.0,
            places=12,
        )

    def test_uniform_variance_is_covariant_under_joint_rotation(self) -> None:
        proton = NuclearBathSpecies.from_isotope("1H")
        axial_sample = UniformNuclearLayer(
            SurfaceGeometry([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            (UniformBathComponent(proton, 6.7e28),),
        )
        rotated_sample = UniformNuclearLayer(
            SurfaceGeometry([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            (UniformBathComponent(proton, 6.7e28),),
        )
        frequencies = np.array([-1.0, 1.0])
        axial = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=5.0, axis_lab=[0.0, 0.0, 1.0]),
            axial_sample,
            [0.0, 0.0, 0.02],
            frequencies,
        )
        rotated = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=5.0, axis_lab=[1.0, 0.0, 0.0]),
            rotated_sample,
            [0.02, 0.0, 0.0],
            frequencies,
        )

        self.assertAlmostEqual(
            axial.total_field_variance_t2 / rotated.total_field_variance_t2,
            1.0,
            places=12,
        )

    def test_voxel_density_is_linear_and_requires_sensor_position(self) -> None:
        proton = NuclearBathSpecies.from_isotope("1H")
        positions = np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 6.0]])
        volumes = np.array([8.0, 8.0])
        first = VoxelNuclearSample(
            positions,
            volumes,
            (VoxelBathComponent(proton, 2.0e28),),
        )
        second = VoxelNuclearSample(
            positions,
            volumes,
            (VoxelBathComponent(proton, 4.0e28),),
        )
        arguments = (
            diamond_nv_minus(),
            [0.0, 0.0, 0.05],
            np.linspace(-2.0e7, 2.0e7, 21),
        )

        with self.assertRaisesRegex(ValueError, "sensor_position"):
            simulate_statistical_spectrum(arguments[0], first, *arguments[1:])
        low = simulate_statistical_spectrum(
            arguments[0],
            first,
            *arguments[1:],
            sensor_position_lab_nm=[0.0, 0.0, 0.0],
        )
        high = simulate_statistical_spectrum(
            arguments[0],
            second,
            *arguments[1:],
            sensor_position_lab_nm=[0.0, 0.0, 0.0],
        )

        self.assertAlmostEqual(
            high.total_field_variance_t2 / low.total_field_variance_t2,
            2.0,
        )


class NanoMRStatisticalSpectrumTests(unittest.TestCase):
    def test_rotating_lorentzian_integrates_to_field_variance(self) -> None:
        variance = 3.0e-12
        larmor = 2.0e6
        correlation = 10.0e-6
        frequencies = np.linspace(-20.0e6, 20.0e6, 200_001)
        psd = rotating_lorentzian_psd(
            frequencies,
            variance,
            larmor,
            correlation,
        )
        integrated = np.sum(
            0.5 * (psd[1:] + psd[:-1]) * np.diff(frequencies)
        ) / (2.0 * np.pi)

        np.testing.assert_allclose(integrated, variance, rtol=5.0e-3)

    def test_multi_isotope_spectrum_resolves_proton_and_fluorine(self) -> None:
        field_tesla = 0.02
        proton = NuclearBathSpecies.from_isotope(
            "1H",
            correlation_time_seconds=200.0e-6,
        )
        fluorine = NuclearBathSpecies.from_isotope(
            "19F",
            polarization_mode="fixed",
            polarization_fraction=0.1,
            correlation_time_seconds=200.0e-6,
        )
        sample = UniformNuclearLayer(
            SurfaceGeometry([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            (
                UniformBathComponent(proton, 5.0e28),
                UniformBathComponent(fluorine, 2.0e28),
            ),
            thickness_nm=10.0,
        )
        frequencies_hz = np.linspace(760.0e3, 900.0e3, 4001)
        result = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=5.0),
            sample,
            [0.0, 0.0, field_tesla],
            2.0 * np.pi * frequencies_hz,
        )
        peaks_hz = frequencies_hz[
            np.argmax(result.component_psd_t2_s, axis=1)
        ]

        np.testing.assert_allclose(
            peaks_hz,
            [proton.gamma_hz_per_t * field_tesla, fluorine.gamma_hz_per_t * field_tesla],
            atol=40.0,
        )
        self.assertEqual(result.polarization_modes, ("statistical", "fixed"))

    def test_filter_overlap_returns_physical_gaussian_coherence(self) -> None:
        field_tesla = 0.01
        sequence = cpmg_sequence(8, 80.0e-6)
        proton = NuclearBathSpecies.from_isotope(
            "1H",
            correlation_time_seconds=50.0e-6,
        )
        sample = UniformNuclearLayer(
            SurfaceGeometry([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            (UniformBathComponent(proton, 6.7e28),),
        )
        larmor = 2.0 * np.pi * proton.gamma_hz_per_t * field_tesla
        spectrum = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=5.0),
            sample,
            [0.0, 0.0, field_tesla],
            np.linspace(-1.5 * larmor, 1.5 * larmor, 20_001),
        )

        coherence = gaussian_filter_coherence(
            diamond_nv_minus(depth_nm=5.0),
            sequence,
            spectrum,
        )

        self.assertGreater(coherence.chi, 0.0)
        self.assertGreater(coherence.coherence, 0.0)
        self.assertLess(coherence.coherence, 1.0)
        self.assertAlmostEqual(
            coherence.phase_variance_rad2,
            2.0 * coherence.chi,
        )


    def test_filter_overlap_rejects_one_sided_frequency_grid(self) -> None:
        field_tesla = 0.01
        proton = NuclearBathSpecies.from_isotope("1H")
        sample = UniformNuclearLayer(
            SurfaceGeometry([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            (UniformBathComponent(proton, 6.7e28),),
        )
        spectrum = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=5.0),
            sample,
            [0.0, 0.0, field_tesla],
            np.linspace(0.0, 2.0 * np.pi * 1.0e6, 1001),
        )

        with self.assertRaisesRegex(ValueError, "two-sided"):
            gaussian_filter_coherence(
                diamond_nv_minus(depth_nm=5.0),
                ramsey_sequence(2.0e-6),
                spectrum,
            )


class NanoMRResolvedClusterTests(unittest.TestCase):
    @staticmethod
    def _cluster(*, chemical_shift_ppm=0.0) -> ResolvedSpinCluster:
        return ResolvedSpinCluster(
            diamond_nv_minus(depth_nm=3.0),
            (
                ResolvedNucleus.from_isotope(
                    "1H",
                    [0.0, 0.0, 3.0],
                    chemical_shift_ppm=chemical_shift_ppm,
                ),
            ),
            sensor_position_lab_nm=[0.0, 0.0, 0.0],
        )

    def test_secular_interactions_are_covariant_with_field_direction(self) -> None:
        nuclei_z = (
            ResolvedNucleus.from_isotope("1H", [0.0, 0.0, 2.0]),
            ResolvedNucleus.from_isotope("1H", [0.0, 0.0, 3.0]),
        )
        nuclei_x = (
            ResolvedNucleus.from_isotope("1H", [2.0, 0.0, 0.0]),
            ResolvedNucleus.from_isotope("1H", [3.0, 0.0, 0.0]),
        )
        coupling = (NuclearScalarCoupling(0, 1, 1.0e6, "secular"),)
        cluster_z = ResolvedSpinCluster(
            diamond_nv_minus(axis_lab=[0.0, 0.0, 1.0]),
            nuclei_z,
            [0.0, 0.0, 0.0],
            scalar_couplings=coupling,
        )
        cluster_x = ResolvedSpinCluster(
            diamond_nv_minus(axis_lab=[1.0, 0.0, 0.0]),
            nuclei_x,
            [0.0, 0.0, 0.0],
            scalar_couplings=coupling,
        )
        scalar_options = {
            "include_sensor_nuclear": False,
            "include_nuclear_dipolar": False,
        }
        scalar_z = np.linalg.eigvalsh(
            resolved_cluster_hamiltonian(
                cluster_z,
                [0.0, 0.0, 0.01],
                **scalar_options,
            )
        )
        scalar_x = np.linalg.eigvalsh(
            resolved_cluster_hamiltonian(
                cluster_x,
                [0.01, 0.0, 0.0],
                **scalar_options,
            )
        )
        dipolar_options = {
            "include_sensor_nuclear": False,
            "include_scalar": False,
            "nuclear_dipolar_model": "secular",
        }
        dipolar_z = np.linalg.eigvalsh(
            resolved_cluster_hamiltonian(
                cluster_z,
                [0.0, 0.0, 0.01],
                **dipolar_options,
            )
        )
        dipolar_x = np.linalg.eigvalsh(
            resolved_cluster_hamiltonian(
                cluster_x,
                [0.01, 0.0, 0.0],
                **dipolar_options,
            )
        )

        np.testing.assert_allclose(scalar_z, scalar_x, atol=1.0e-4)
        np.testing.assert_allclose(dipolar_z, dipolar_x, atol=1.0e-4)

    def test_default_dense_limit_accepts_four_nuclei_and_rejects_five(self) -> None:
        sensor = diamond_nv_minus()
        four = tuple(
            ResolvedNucleus.from_isotope("1H", [index + 1.0, 0.0, 4.0])
            for index in range(4)
        )
        cluster = ResolvedSpinCluster(sensor, four, [0.0, 0.0, 0.0])

        self.assertEqual(DEFAULT_MAX_HILBERT_DIMENSION, 64)
        self.assertEqual(cluster.dimension, 48)
        with self.assertRaisesRegex(ValueError, "dimension 96"):
            ResolvedSpinCluster(
                sensor,
                four
                + (ResolvedNucleus.from_isotope("1H", [5.0, 0.0, 4.0]),),
                [0.0, 0.0, 0.0],
            )

    def test_chemical_shift_scales_nuclear_zeeman_term(self) -> None:
        base = self._cluster()
        shifted = self._cluster(chemical_shift_ppm=1_000.0)
        field = [0.0, 0.0, 0.1]
        norm_base = np.linalg.norm(nuclear_zeeman_hamiltonian(base, field))
        norm_shifted = np.linalg.norm(nuclear_zeeman_hamiltonian(shifted, field))

        self.assertAlmostEqual(norm_shifted / norm_base, 1.001, places=12)

    def test_all_resolved_interaction_terms_are_hermitian(self) -> None:
        nuclei = (
            ResolvedNucleus.from_isotope("1H", [0.0, 0.0, 3.0]),
            ResolvedNucleus.from_isotope("13C", [1.0, 0.0, 3.0]),
        )
        cluster = ResolvedSpinCluster(
            diamond_nv_minus(),
            nuclei,
            [0.0, 0.0, 0.0],
            scalar_couplings=(NuclearScalarCoupling(0, 1, 140.0),),
        )
        terms = (
            sensor_nuclear_hyperfine_hamiltonian(cluster),
            nuclear_scalar_coupling_hamiltonian(cluster),
            nuclear_dipolar_hamiltonian(cluster),
            nuclear_rf_hamiltonian(cluster, 10.0e3, phase_rad=0.3),
            resolved_cluster_hamiltonian(cluster, [0.0, 0.0, 0.02]),
        )

        for term in terms:
            np.testing.assert_allclose(term, term.conj().T, atol=1.0e-7)

    def test_nuclear_dipolar_interaction_scales_as_inverse_cube(self) -> None:
        def norm_at_separation(separation_nm: float) -> float:
            cluster = ResolvedSpinCluster(
                diamond_nv_minus(),
                (
                    ResolvedNucleus.from_isotope("1H", [0.0, 0.0, 4.0]),
                    ResolvedNucleus.from_isotope(
                        "1H", [separation_nm, 0.0, 4.0]
                    ),
                ),
                [0.0, 0.0, 0.0],
            )
            return float(np.linalg.norm(nuclear_dipolar_hamiltonian(cluster)))

        self.assertAlmostEqual(norm_at_separation(1.0) / norm_at_separation(2.0), 8.0)

    def test_resolved_cw_spectrum_has_hyperfine_split_lines(self) -> None:
        result = simulate_resolved_cw_spectrum(
            self._cluster(),
            [0.0, 0.0, 0.02],
            broadening_hz=20.0e3,
            points=501,
        )

        self.assertEqual(result.spectrum.shape, (501,))
        self.assertTrue(np.all(np.isfinite(result.spectrum)))
        self.assertGreater(len(result.eigensystem.transitions), 2)

    def test_ideal_nuclear_rotation_and_two_block_spectrum(self) -> None:
        cluster = self._cluster()
        rotation = ideal_nuclear_rotation(cluster, np.pi, phase_rad=np.pi / 4)
        np.testing.assert_allclose(
            rotation.conj().T @ rotation,
            np.eye(cluster.dimension),
            atol=1.0e-12,
        )
        times = np.linspace(0.2e-6, 2.0e-6, 5)
        result = simulate_two_block_correlation(
            cluster,
            [0.0, 0.0, 0.02],
            times,
            times,
            mixing_seconds=0.5e-6,
            nuclear_rf_flip_angle_rad=np.pi,
        )
        spectrum = correlation_spectrum_2d(result)

        self.assertEqual(result.bright_probability.shape, (5, 5))
        self.assertTrue(np.all((result.bright_probability >= 0.0)))
        self.assertTrue(np.all((result.bright_probability <= 1.0)))
        self.assertEqual(spectrum.amplitude.shape, (5, 5))
        self.assertTrue(np.all(np.isfinite(spectrum.amplitude)))


class NanoMRESRBridgeTests(unittest.TestCase):
    def test_spin_half_zero_zfs_round_trip_matches_pure_esr_levels(self) -> None:
        system = ESRSpinSystem(g_tensor=[2.01, 2.02, 2.03], label="radical")
        sensor = defect_sensor_from_esr(system, depth_nm=2.0)
        field = np.array([0.002, -0.003, 0.15])
        pure = diagonalize_system(system, field)
        promoted = diagonalize_sensor(sensor, field)

        np.testing.assert_allclose(
            pure.levels_hz - pure.levels_hz[0],
            promoted.levels_hz - promoted.levels_hz[0],
            rtol=1.0e-13,
            atol=1.0e-5,
        )
        self.assertEqual(esr_system_from_defect(sensor), system)

    def test_higher_spin_defect_is_not_silently_demoted_to_pure_esr(self) -> None:
        with self.assertRaisesRegex(ValueError, "spin-1/2"):
            esr_system_from_defect(diamond_nv_minus())


class NanoMRMotionTests(unittest.TestCase):
    @staticmethod
    def _proton() -> NuclearBathSpecies:
        return NuclearBathSpecies.from_isotope("1H")

    def test_point_dipole_field_has_inverse_cube_scaling(self) -> None:
        moment = np.array([[0.0, 0.0, 1.0e-26]])

        near = dipolar_field_from_moments([[0.0, 0.0, 1.0e-8]], moment)
        far = dipolar_field_from_moments([[0.0, 0.0, 2.0e-8]], moment)

        np.testing.assert_allclose(near[:2], 0.0, atol=1.0e-30)
        self.assertAlmostEqual(near[2] / far[2], 8.0)

    def test_seeded_diffusing_field_records_are_reproducible(self) -> None:
        positions = np.array(
            [[-4.0, 1.0, 5.0], [3.0, -2.0, 7.0], [1.0, 4.0, 9.0]]
        )
        kwargs = dict(
            sensor_position_lab_nm=[0.0, 0.0, -5.0],
            static_field_lab_tesla=[0.0, 0.0, 0.02],
            sample_interval_seconds=2.0e-9,
            sample_count=12,
            diffusion_coefficient_m2_s=1.5e-9,
            seed=73,
        )

        first = simulate_diffusing_dipolar_field(
            positions,
            self._proton(),
            **kwargs,
        )
        second = simulate_diffusing_dipolar_field(
            positions,
            self._proton(),
            **kwargs,
        )

        np.testing.assert_array_equal(
            first.positions_lab_nm,
            second.positions_lab_nm,
        )
        np.testing.assert_array_equal(first.field_lab_tesla, second.field_lab_tesla)

    def test_free_diffusion_msd_matches_six_d_t(self) -> None:
        particle_count = 5000
        diffusion = 1.2e-9
        dt = 1.0e-8
        record = simulate_diffusing_dipolar_field(
            np.zeros((particle_count, 3)),
            self._proton(),
            sensor_position_lab_nm=[0.0, 0.0, -1000.0],
            sample_interval_seconds=dt,
            sample_count=21,
            diffusion_coefficient_m2_s=diffusion,
            seed=5,
        )

        statistics = trajectory_displacement_statistics(record)
        expected_nm2 = (
            6.0 * diffusion * record.times_seconds[-1] * 1.0e18
        )

        self.assertAlmostEqual(
            statistics.mean_squared_displacement_nm2[-1] / expected_nm2,
            1.0,
            delta=0.04,
        )

    def test_drift_and_reflecting_confinement(self) -> None:
        drift = simulate_diffusing_dipolar_field(
            np.zeros((4, 3)),
            self._proton(),
            sensor_position_lab_nm=[0.0, 0.0, -1000.0],
            sample_interval_seconds=1.0e-7,
            sample_count=11,
            motion_substeps_per_sample=5,
            drift_velocity_m_s=[0.01, -0.02, 0.0],
            initial_phases_rad=0.0,
        )
        mean = trajectory_displacement_statistics(drift).mean_displacement_nm[-1]
        np.testing.assert_allclose(mean, [10.0, -20.0, 0.0], atol=1.0e-12)

        confined = simulate_diffusing_dipolar_field(
            [[0.0, 0.0, 5.0], [2.0, -1.0, 7.0]],
            self._proton(),
            sensor_position_lab_nm=[0.0, 0.0, -10.0],
            sample_interval_seconds=2.0e-8,
            sample_count=30,
            diffusion_coefficient_m2_s=8.0e-9,
            bounds_lab_nm=((-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0)),
            boundary="reflect",
            seed=19,
        )
        lower = np.array([-5.0, -5.0, 0.0])
        upper = np.array([5.0, 5.0, 10.0])
        self.assertTrue(np.all(confined.positions_lab_nm >= lower))
        self.assertTrue(np.all(confined.positions_lab_nm <= upper))

    def test_autocorrelation_and_psd_recover_a_stationary_tone(self) -> None:
        dt = 1.0e-4
        times = np.arange(4096) * dt
        frequency = 1000.0
        values = 2.0e-9 * np.cos(2.0 * np.pi * frequency * times)

        correlation = field_autocorrelation(
            values,
            sample_interval_seconds=dt,
            max_lag=20,
            normalize=True,
        )
        spectrum = field_power_spectral_density(
            values,
            sample_interval_seconds=dt,
            segment_length=512,
        )

        self.assertAlmostEqual(correlation.autocorrelation_tesla2[0], 1.0)
        self.assertGreater(spectrum.segment_count, 1)
        peak = spectrum.frequency_hz[
            np.argmax(spectrum.power_spectral_density_tesla2_per_hz)
        ]
        self.assertAlmostEqual(peak, frequency, delta=1.0 / (512 * dt))

    def test_stationary_record_has_constant_raw_autocorrelation(self) -> None:
        values = np.full(32, 3.0e-9)

        correlation = field_autocorrelation(
            values,
            sample_interval_seconds=1.0e-6,
            max_lag=12,
            demean=False,
        )

        np.testing.assert_allclose(correlation.autocorrelation_tesla2, 9.0e-18)

    def test_free_diffusion_return_density_has_three_dimensional_tail(self) -> None:
        times = np.geomspace(1.0e-9, 1.0e-3, 50)
        density = free_diffusion_return_density(times, 2.0e-9)
        slope = np.polyfit(np.log(times), np.log(density), 1)[0]

        self.assertAlmostEqual(slope, -1.5, places=12)


class NanoMRNoiseAndDetectorTests(unittest.TestCase):
    def test_correlated_field_covariance_and_seeded_sampling(self) -> None:
        times = np.arange(6) * 10.0e-6
        positions = np.column_stack(
            (np.arange(6, dtype=float), np.zeros((6, 2)))
        )
        model = CorrelatedFieldNoiseModel(
            (
                FieldNoiseComponent(
                    "surface bath",
                    rms_field_tesla=2.0e-9,
                    correlation_time_seconds=20.0e-6,
                    spatial_correlation_length_nm=2.0,
                ),
                FieldNoiseComponent(
                    "diffusing target",
                    rms_field_tesla=1.0e-9,
                    correlation_time_seconds=5.0e-6,
                    larmor_frequency_hz=50.0e3,
                    envelope="power_law",
                ),
            )
        )

        covariance = model.covariance(
            times,
            positions_lab_nm=positions,
        )
        first = sample_correlated_field_noise(
            model,
            times,
            positions_lab_nm=positions,
            seed=17,
        )
        second = sample_correlated_field_noise(
            model,
            times,
            positions_lab_nm=positions,
            seed=17,
        )

        np.testing.assert_allclose(np.diag(covariance), 5.0e-18)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) >= -1.0e-30))
        self.assertLess(effective_sample_size(covariance), times.size)
        np.testing.assert_array_equal(first.field_tesla, second.field_tesla)

    def test_linear_source_variance_maps_to_measurement_covariance(self) -> None:
        matrix = np.array([[1.0, 2.0], [-1.0, 0.5]])
        variances = np.array([4.0, 9.0])

        covariance = linear_field_covariance(matrix, variances)

        np.testing.assert_allclose(
            covariance,
            matrix @ np.diag(variances) @ matrix.T,
        )

    def test_optical_cycle_conserves_population_and_has_transient_contrast(
        self,
    ) -> None:
        model = OpticalCycleModel()
        times = np.linspace(0.0, 500.0e-9, 101)
        bright = model.population_trace(1.0, times)
        dark = model.population_trace(0.0, times)

        np.testing.assert_allclose(np.sum(bright, axis=1), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(np.sum(dark, axis=1), 1.0, atol=1.0e-12)
        self.assertGreater(
            model.expected_emitted_photons(
                1.0,
                readout_seconds=300.0e-9,
            ),
            model.expected_emitted_photons(
                0.0,
                readout_seconds=300.0e-9,
            ),
        )

    def test_time_resolved_readout_matches_emission_mean(self) -> None:
        optical = OpticalCycleModel()
        detector = SPADDetectorModel(
            detection_efficiency=1.0,
            dark_count_rate_hz=0.0,
        )
        expected = optical.expected_emitted_photons(
            0.35,
            readout_seconds=300.0e-9,
        )

        result = sample_time_resolved_optical_readout(
            optical,
            detector,
            0.35,
            repetitions=3000,
            readout_seconds=300.0e-9,
            seed=29,
        )

        self.assertAlmostEqual(
            float(np.mean(result.detected_counts)),
            expected,
            delta=0.12,
        )
        self.assertEqual(result.detected_counts.shape, (3000,))
        self.assertGreater(result.fano_factor, 0.0)

    def test_spad_nonparalyzable_dead_time_and_afterpulse(self) -> None:
        emitted = np.arange(0.0, 100.0e-9, 2.0e-9)
        detector = SPADDetectorModel(
            detection_efficiency=1.0,
            dark_count_rate_hz=0.0,
            dead_time_seconds=10.0e-9,
        )
        detected = detector.detect(
            emitted,
            readout_seconds=100.0e-9,
            seed=3,
        )
        afterpulse_detector = SPADDetectorModel(
            detection_efficiency=1.0,
            dark_count_rate_hz=0.0,
            afterpulse_probability=1.0,
            afterpulse_time_seconds=5.0e-9,
        )
        afterpulsed = afterpulse_detector.detect(
            [10.0e-9],
            readout_seconds=100.0e-9,
            seed=4,
        )

        self.assertEqual(detected.size, 10)
        self.assertEqual(afterpulsed.size, 2)
        self.assertGreater(afterpulsed[1], afterpulsed[0])

    def test_covariance_weighted_density_fit_rejects_noisy_channel(self) -> None:
        matrix = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        measurements = np.array([2.0, 4.0, 3.0])
        covariance = np.diag([0.01, 100.0, 0.01])
        ordinary = reconstruct_nonnegative_density(
            matrix,
            measurements,
            regularization=0.0,
            regularization_order=0,
        )
        weighted = reconstruct_nonnegative_density(
            matrix,
            measurements,
            regularization=0.0,
            regularization_order=0,
            noise_covariance=covariance,
        )

        truth = np.array([2.0, 1.0])
        self.assertLess(
            np.linalg.norm(weighted.density - truth),
            np.linalg.norm(ordinary.density - truth),
        )
        np.testing.assert_array_equal(
            weighted.measurement_covariance,
            covariance,
        )


class NanoMRImagingTests(unittest.TestCase):
    @staticmethod
    def _axis() -> np.ndarray:
        return np.array([np.sqrt(2.0 / 3.0), 0.0, np.sqrt(1.0 / 3.0)])

    @staticmethod
    def _proton() -> NuclearBathSpecies:
        return NuclearBathSpecies.from_isotope("1H")

    def test_raster_and_sensor_array_geometry(self) -> None:
        scan = raster_scan([0.0, 1.0, 2.0], [10.0, 11.0], z_nm=-5.0)

        np.testing.assert_allclose(
            scan.positions_lab_nm[:, 0],
            [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
        )
        np.testing.assert_array_equal(
            scan.reshape_raster(np.arange(6)),
            [[0, 1, 2], [5, 4, 3]],
        )

        array = sensor_array(
            [[0.0, 0.0, -5.0], [1.0, 0.0, -5.0]],
            sensor_axes_lab=[[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(np.linalg.norm(array.sensor_axes_lab, axis=1), 1.0)

    def test_projected_field_forward_operator_scales_as_inverse_cube(self) -> None:
        scan = arbitrary_scan([[0.0, 0.0, 0.0]])
        near = build_dipolar_forward_operator(
            scan,
            [[0.0, 0.0, 10.0]],
            response_kind="field",
        )
        far = build_dipolar_forward_operator(
            scan,
            [[0.0, 0.0, 20.0]],
            response_kind="field",
        )

        self.assertAlmostEqual(near.matrix[0, 0] / far.matrix[0, 0], 8.0)
        self.assertEqual(near.source_units, "J/T")
        self.assertEqual(near.measurement_units, "T")

    def test_voxel_variance_operator_matches_phase3_statistical_backend(self) -> None:
        axis = self._axis()
        sensor = diamond_nv_minus(depth_nm=5.0, axis_lab=axis)
        species = self._proton()
        density = 4.0e28
        volume_nm3 = 8.0
        position = np.array([[2.0, -1.0, 3.0]])
        scan = arbitrary_scan(
            [[0.0, 0.0, -5.0]],
            sensor_axes_lab=axis,
        )
        operator = build_dipolar_forward_operator(
            scan,
            position,
            response_kind="transverse_variance",
            field_axis_lab=axis,
        )
        amplitude = nuclear_voxel_variance_amplitudes(
            species,
            0.02,
            density,
            [volume_nm3],
        )
        expected = operator.predict(amplitude)[0]

        sample = VoxelNuclearSample(
            position,
            np.array([volume_nm3]),
            (VoxelBathComponent(species, density),),
        )
        spectrum = simulate_statistical_spectrum(
            sensor,
            sample,
            axis * 0.02,
            np.array([-1.0, 1.0]),
            sensor_position_lab_nm=[0.0, 0.0, -5.0],
        )

        self.assertAlmostEqual(
            expected / spectrum.total_field_variance_t2,
            1.0,
            places=12,
        )

    def test_depth_profile_operator_matches_uniform_layer(self) -> None:
        axis = self._axis()
        species = self._proton()
        density = 5.0e28
        depth_operator = build_depth_profile_operator(
            [5.0],
            [0.0, 10.0],
            species,
            field_tesla=0.02,
            sensor_axis_lab=axis,
            field_axis_lab=axis,
        )
        predicted = depth_operator.predict(np.array([density]))[0]

        sensor = diamond_nv_minus(depth_nm=5.0, axis_lab=axis)
        sample = UniformNuclearLayer(
            SurfaceGeometry([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            (UniformBathComponent(species, density),),
            thickness_nm=10.0,
        )
        spectrum = simulate_statistical_spectrum(
            sensor,
            sample,
            axis * 0.02,
            np.array([-1.0, 1.0]),
        )

        self.assertAlmostEqual(
            predicted / spectrum.total_field_variance_t2,
            1.0,
            places=12,
        )

    def test_nonnegative_density_reconstruction_recovers_peak_and_uncertainty(
        self,
    ) -> None:
        axis = self._axis()
        scan = raster_scan(
            np.linspace(-6.0, 6.0, 11),
            np.linspace(-6.0, 6.0, 11),
            z_nm=-8.0,
            sensor_axis_lab=axis,
        )
        grid = planar_voxel_grid(
            np.linspace(-4.0, 4.0, 5),
            np.linspace(-4.0, 4.0, 5),
            z_nm=0.0,
            thickness_nm=2.0,
        )
        operator = build_voxel_density_forward_operator(
            scan,
            grid,
            self._proton(),
            field_tesla=0.02,
            field_axis_lab=axis,
            minimum_distance_nm=8.0,
        )
        truth = np.zeros(grid.shape)
        truth[2, 2] = 5.0e28
        clean = operator.predict(truth)
        noise_std = 0.002 * float(np.max(clean))
        noisy = clean + np.random.default_rng(13).normal(
            scale=noise_std,
            size=clean.size,
        )

        result = reconstruct_nonnegative_density(
            operator,
            noisy,
            regularization=1.0e-5,
            regularization_order=0,
            noise_std=noise_std,
        )

        self.assertTrue(result.converged)
        self.assertEqual(np.unravel_index(np.argmax(result.density), grid.shape), (2, 2))
        self.assertLess(np.linalg.norm(result.residual), 0.08 * np.linalg.norm(noisy))
        self.assertTrue(np.all(result.density >= 0.0))
        self.assertTrue(np.all(result.standard_deviation > 0.0))

    def test_sparse_point_localization_recovers_position(self) -> None:
        axis = self._axis()
        scan = raster_scan(
            np.linspace(-7.0, 7.0, 13),
            np.linspace(-7.0, 7.0, 13),
            z_nm=-8.0,
            sensor_axis_lab=axis,
        )
        true_position = np.array([[1.2, -0.8, 0.5]])
        amplitude = np.array([1.3e-26])
        operator = build_dipolar_forward_operator(
            scan,
            true_position,
            response_kind="field",
            moment_direction_lab=axis,
            minimum_distance_nm=5.0,
        )
        measurements = operator.predict(amplitude)

        result = localize_point_sources(
            scan,
            measurements,
            [[0.2, 0.0, 0.0]],
            response_kind="field",
            moment_direction_lab=axis,
            initial_amplitudes=[1.0e-26],
            bounds_lab_nm=((-4.0, 4.0), (-4.0, 4.0), (-1.0, 3.0)),
            minimum_distance_nm=5.0,
            noise_std=1.0e-12,
        )

        self.assertTrue(result.success)
        self.assertLess(
            np.linalg.norm(result.positions_lab_nm[0] - true_position[0]),
            0.05,
        )
        self.assertAlmostEqual(result.amplitudes[0] / amplitude[0], 1.0, places=3)
        self.assertTrue(np.all(result.position_standard_deviation_nm > 0.0))

        correlated = localize_point_sources(
            scan,
            measurements,
            [[0.2, 0.0, 0.0]],
            response_kind="field",
            moment_direction_lab=axis,
            initial_amplitudes=[1.0e-26],
            bounds_lab_nm=((-4.0, 4.0), (-4.0, 4.0), (-1.0, 3.0)),
            minimum_distance_nm=5.0,
            noise_covariance=np.eye(scan.measurement_count) * 1.0e-24,
        )
        self.assertLess(
            np.linalg.norm(correlated.positions_lab_nm[0] - true_position[0]),
            0.05,
        )
        self.assertIsNotNone(correlated.measurement_covariance)


class NanoMRHighResolutionTests(unittest.TestCase):
    def test_clock_zero_error_limit_and_seeded_instability(self) -> None:
        ideal = ClockModel()
        nominal, actual = ideal.sample_times(5, 2.0e-3, seed=1)
        np.testing.assert_allclose(actual, nominal, atol=0.0)

        clock = ClockModel(
            fractional_frequency_offset=2.0e-6,
            interval_fractional_frequency_instability=1.0e-7,
            trigger_jitter_seconds=2.0e-9,
        )
        first = clock.sample_times(20, 1.0e-3, seed=9)[1]
        second = clock.sample_times(20, 1.0e-3, seed=9)[1]
        ideal_times = ideal.sample_times(20, 1.0e-3, seed=9)[1]
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, ideal_times))

    def test_qdyne_sensor_coherence_reduces_visibility_not_phase(self) -> None:
        sensing_seconds = 1.0e-3
        protocol = QdyneProtocol(
            ramsey_sequence(sensing_seconds),
            repetition_interval_seconds=sensing_seconds,
            budget=HighResolutionBudget(
                sensor_coherence_seconds=sensing_seconds / np.log(2.0),
            ),
        )
        result = simulate_qdyne(
            protocol,
            signal_frequency_hz=0.0,
            field_amplitude_tesla=np.pi / (2.0 * sensing_seconds),
            sensor_gamma_rad_s_t=1.0,
            shot_count=8,
        )

        self.assertAlmostEqual(result.sensor_contrast, 0.5)
        self.assertAlmostEqual(result.normalized_signal[0], 0.5)

    def test_qdyne_reports_raw_and_nyquist_aliased_beats(self) -> None:
        protocol = QdyneProtocol(
            ramsey_sequence(10.0e-6),
            repetition_interval_seconds=1.0e-3,
        )
        result = simulate_qdyne(
            protocol,
            signal_frequency_hz=1200.0,
            field_amplitude_tesla=1.0e-6,
            sensor_gamma_rad_s_t=1.0e9,
            shot_count=4096,
        )

        resolution = 1.0 / (
            result.nominal_times_seconds.size
            * protocol.repetition_interval_seconds
        )
        self.assertAlmostEqual(result.raw_beat_frequency_hz, 1200.0)
        self.assertAlmostEqual(result.expected_beat_frequency_hz, 200.0)
        self.assertEqual(result.alias_order, 1)
        self.assertLess(
            abs(result.estimated_beat_frequency_hz - 200.0),
            resolution,
        )

    def test_qdyne_recovers_clocked_beat_and_optical_record(self) -> None:
        protocol = QdyneProtocol(
            ramsey_sequence(8.0e-6),
            repetition_interval_seconds=1.0e-3,
            reference_frequency_hz=20_000.0,
            analysis_contrast=0.8,
            budget=HighResolutionBudget(sensor_coherence_seconds=80.0e-6),
        )
        result = simulate_qdyne(
            protocol,
            signal_frequency_hz=20_007.0,
            field_amplitude_tesla=8.0e-9,
            sensor_gamma_rad_s_t=2.0 * np.pi * 28.0e9,
            shot_count=2048,
            optical_model=OpticalReadoutModel(),
            seed=12,
        )

        resolution = 1.0 / (
            result.nominal_times_seconds.size
            * protocol.repetition_interval_seconds
        )
        self.assertAlmostEqual(result.expected_beat_frequency_hz, 7.0)
        self.assertLess(abs(result.estimated_beat_frequency_hz - 7.0), resolution)
        self.assertGreater(abs(result.filter_response_seconds), 0.0)
        self.assertIsNotNone(result.optical_readout)
        self.assertEqual(result.optical_readout.sampled_counts.shape, (2048,))

    def test_synchronized_readout_preserves_signed_beat(self) -> None:
        protocol = QdyneProtocol(
            ramsey_sequence(5.0e-6),
            repetition_interval_seconds=2.0e-3,
            reference_frequency_hz=50_000.0,
        )
        result = simulate_synchronized_readout(
            protocol,
            signal_frequency_hz=49_995.0,
            field_amplitude_tesla=5.0e-9,
            sensor_gamma_rad_s_t=2.0 * np.pi * 28.0e9,
            shot_count=1024,
        )

        resolution = 1.0 / (
            result.nominal_times_seconds.size
            * protocol.repetition_interval_seconds
        )
        self.assertLess(
            abs(result.estimated_beat_frequency_hz + 5.0),
            resolution,
        )
        self.assertTrue(np.iscomplexobj(result.complex_signal))

    def test_memory_correlation_reports_separate_envelopes(self) -> None:
        times = np.array([0.0, 1.0e-3])
        budget = HighResolutionBudget(
            sensor_coherence_seconds=100.0e-6,
            sample_coherence_seconds=4.0e-3,
            diffusion_correlation_seconds=5.0e-3,
            memory_coherence_seconds=10.0e-3,
        )
        result = sensor_memory_correlation(
            times,
            frequency_hz=0.0,
            sensing_seconds=20.0e-6,
            budget=budget,
            contrast=0.8,
            transfer_fidelity=0.9,
            retrieval_fidelity=0.95,
        )

        expected_zero = 0.8 * 0.9 * 0.95 * np.exp(-0.4)
        self.assertAlmostEqual(result.signal[0], expected_zero)
        self.assertAlmostEqual(result.sample_envelope[1], np.exp(-0.25))
        self.assertAlmostEqual(result.diffusion_envelope[1], np.exp(-0.2))
        self.assertAlmostEqual(result.memory_envelope[1], np.exp(-0.1))

    def test_coherent_nmr_doublet_of_doublets_positions(self) -> None:
        site = CoherentNMRSite(
            "1H",
            chemical_shift_ppm=2.0,
            scalar_couplings_hz=(10.0, 4.0),
            transverse_relaxation_seconds=0.5,
        )
        times = np.arange(4096) * 2.0e-4
        result = simulate_coherent_nmr_spectrum(
            [site],
            1.0,
            times,
            diffusion_correlation_seconds=1.0,
            window=False,
        )
        center = (
            abs(ISOTOPE_GAMMA_HZ_PER_T["1H"])
            * (1.0 + 2.0e-6)
        )

        np.testing.assert_allclose(
            result.component_frequencies_hz - center,
            [-7.0, -3.0, 3.0, 7.0],
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            np.abs(result.component_amplitudes),
            np.full(4, 0.25),
        )
        self.assertLess(abs(result.fid[-1]), abs(result.fid[0]))

        triplet = simulate_coherent_nmr_spectrum(
            [
                CoherentNMRSite(
                    "1H",
                    chemical_shift_ppm=0.0,
                    scalar_couplings_hz=(8.0, 8.0),
                )
            ],
            1.0,
            times,
        )
        triplet_center = abs(ISOTOPE_GAMMA_HZ_PER_T["1H"])
        np.testing.assert_allclose(
            triplet.component_frequencies_hz - triplet_center,
            [-8.0, 0.0, 8.0],
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            np.abs(triplet.component_amplitudes),
            [0.25, 0.5, 0.25],
        )

    def test_coherent_nmr_clock_scales_all_physical_evolution(self) -> None:
        times = np.arange(5, dtype=np.float64) * 0.1
        result = simulate_coherent_nmr_spectrum(
            [CoherentNMRSite("1H", 0.0, scalar_couplings_hz=(1.0,))],
            0.0,
            times,
            reference_frequency_hz=0.0,
            clock=ClockModel(fractional_frequency_offset=0.1),
            window=False,
        )
        elapsed_actual = result.actual_times_seconds - result.actual_times_seconds[0]

        np.testing.assert_allclose(
            result.fid.real,
            np.cos(np.pi * elapsed_actual),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(result.fid.imag, 0.0, atol=1.0e-12)

    def test_dnp_build_up_and_decay_limits(self) -> None:
        model = DNPModel(
            enhancement_factor=50.0,
            buildup_time_seconds=2.0,
            nuclear_t1_seconds=5.0,
        )
        times = np.array([0.0, 2.0, 20.0])
        buildup = dnp_polarization(model, times, 0.01)
        self.assertAlmostEqual(buildup.polarization[0], 0.01)
        self.assertAlmostEqual(buildup.steady_state_polarization, 0.5)
        self.assertAlmostEqual(
            buildup.polarization[1],
            0.5 + (0.01 - 0.5) * np.exp(-1.0),
        )
        decay = dnp_polarization(
            model,
            times,
            0.01,
            pumping=False,
            initial_polarization=0.5,
        )
        self.assertAlmostEqual(decay.polarization[0], 0.5)
        self.assertLess(decay.polarization[-1], 0.02)


if __name__ == "__main__":
    unittest.main()
