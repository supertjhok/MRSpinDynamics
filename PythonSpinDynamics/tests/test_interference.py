from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.interference import (
    AcquisitionMask,
    CompensationActuator,
    MagneticDipoleSource,
    ReferenceCoil,
    SampleLabel,
    UniformPlaneWaveSource,
    adaptive_lms_canceller,
    adaptive_rls_canceller,
    am_carrier_waveform,
    blank_mask,
    coil_voltage,
    colored_noise_waveform,
    coupling_matrix,
    feedforward_cancel,
    fit_gated_ridge_fir,
    fit_scalar_canceller,
    frequency_domain_canceller,
    gated_ridge_fir_canceller,
    joint_signal_reference_canceller,
    mask_from_intervals,
    matched_filter_snr_improvement,
    reference_design_diagnostics,
    reference_matrix,
    reference_noise_injection,
    residual_spectral_lines,
    rfi_suppression_db,
    robust_fir_canceller,
    scalar_canceller,
    saturation_diagnostics,
    signal_bias,
    tone_waveform,
    total_field,
    windowed_joint_signal_reference_canceller,
    windowed_ridge_fir_canceller,
)
from spin_dynamics.interference._signal import (
    sample_times,
    spectral_derivative,
    spectral_lowpass,
    spectral_phase_shift,
)
from spin_dynamics.nqr import slse_acquisition_mask, slse_sequence
from spin_dynamics.nqr import sorc_acquisition_mask, sorc_sequence
from spin_dynamics.interference.sources import MU0


class SignalHelperTests(unittest.TestCase):
    def test_spectral_derivative_of_sine_is_scaled_cosine(self):
        fs, f, n = 10_000.0, 50.0, 2000
        t = sample_times(n, fs)
        signal = np.sin(2 * np.pi * f * t)
        deriv = spectral_derivative(signal, fs)
        expected = 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        np.testing.assert_allclose(deriv, expected, atol=1e-6 * 2 * np.pi * f)

    def test_spectral_derivative_real_input_stays_real(self):
        deriv = spectral_derivative(np.cos(np.linspace(0, 6, 128)), 1.0)
        self.assertTrue(np.isrealobj(deriv))

    def test_phase_shift_by_ninety_degrees_maps_cos_to_sin(self):
        fs, f, n = 8_000.0, 40.0, 1600
        t = sample_times(n, fs)
        shifted = spectral_phase_shift(np.cos(2 * np.pi * f * t), np.pi / 2)
        # A +90 deg Hilbert phase advance turns cos into +sin.
        interior = slice(n // 8, -n // 8)
        np.testing.assert_allclose(
            shifted[interior], np.sin(2 * np.pi * f * t)[interior], atol=1e-3
        )

    def test_lowpass_removes_high_tone_keeps_low_tone(self):
        fs, n = 10_000.0, 4000
        t = sample_times(n, fs)
        low = np.cos(2 * np.pi * 100 * t)
        high = np.cos(2 * np.pi * 2000 * t)
        filtered = spectral_lowpass(low + high, fs, cutoff_hz=500.0)
        np.testing.assert_allclose(filtered, low, atol=1e-9)


class MaskTests(unittest.TestCase):
    def test_masks_partition_all_samples(self):
        mask = mask_from_intervals(
            1_000.0,
            1.0,
            transmit=[(0.0, 0.1)],
            ringdown=[(0.1, 0.15)],
            baseline=[(0.9, 1.0)],
        )
        total = (
            mask.transmit_mask.sum()
            + mask.ringdown_mask.sum()
            + mask.signal_mask.sum()
            + mask.baseline_mask.sum()
        )
        self.assertEqual(total, mask.num_samples)
        # receive = signal + baseline, disjoint from transmit/ringdown.
        self.assertTrue(np.all(mask.receive_mask == (mask.signal_mask | mask.baseline_mask)))
        self.assertFalse(np.any(mask.receive_mask & mask.transmit_mask))

    def test_transmit_priority_wins_over_baseline_overlap(self):
        mask = mask_from_intervals(
            1_000.0, 1.0, transmit=[(0.0, 0.5)], baseline=[(0.0, 0.5)]
        )
        self.assertEqual(int(mask.baseline_mask.sum()), 0)
        self.assertGreater(int(mask.transmit_mask.sum()), 0)

    def test_masked_fraction(self):
        mask = mask_from_intervals(1_000.0, 1.0, transmit=[(0.0, 0.25)])
        self.assertAlmostEqual(mask.masked_fraction, 0.25, places=6)

    def test_blank_mask_defaults_to_signal(self):
        mask = blank_mask(1_000.0, 10)
        self.assertTrue(np.all(mask.signal_mask))

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):
            AcquisitionMask(sample_rate_hz=1.0, labels=np.array([99]))

    def test_sample_label_enum(self):
        self.assertEqual(int(SampleLabel.SIGNAL), 2)


class UniformSourceTests(unittest.TestCase):
    def test_field_is_position_independent(self):
        wf = tone_waveform(256, 1e6, frequency_hz=1e5, amplitude=2.0)
        src = UniformPlaneWaveSource(polarization=[0.0, 0.0, 1.0], waveform=wf)
        positions = np.array([[0, 0, 0], [1.0, 2.0, 3.0], [-5.0, 0.0, 0.5]])
        field = src.field(positions)
        self.assertEqual(field.shape, (3, 3, 256))
        np.testing.assert_allclose(field[0], field[1])
        np.testing.assert_allclose(field[0], field[2])
        # z-polarized: x and y components vanish.
        np.testing.assert_allclose(field[0, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(field[0, 1], 0.0, atol=1e-15)


class DipoleSourceTests(unittest.TestCase):
    def test_on_axis_field_matches_analytic_dipole(self):
        wf = tone_waveform(4, 1e6, frequency_hz=1e5, amplitude=1.0)
        # Unit moment along z; observe on the z-axis at distance d.
        d = 0.1
        src = MagneticDipoleSource(
            position_m=[0.0, 0.0, 0.0],
            moment_direction=[0.0, 0.0, 1.0],
            waveform=wf,
            amplitude=1.0,
        )
        pattern = src.spatial_pattern(np.array([[0.0, 0.0, d]]))[0]
        # On-axis dipole field B_z = mu0/(2 pi) m / r^3 (m = 1 here).
        expected = MU0 / (2.0 * np.pi) / d**3
        self.assertAlmostEqual(pattern[2], expected, delta=1e-9 * expected)
        self.assertAlmostEqual(pattern[0], 0.0, places=12)

    def test_field_decays_as_inverse_cube(self):
        wf = tone_waveform(4, 1e6, frequency_hz=1e5)
        src = MagneticDipoleSource(
            position_m=[0.0, 0.0, 0.0],
            moment_direction=[0.0, 0.0, 1.0],
            waveform=wf,
        )
        near = np.linalg.norm(src.spatial_pattern(np.array([[0, 0, 0.1]]))[0])
        far = np.linalg.norm(src.spatial_pattern(np.array([[0, 0, 0.2]]))[0])
        self.assertAlmostEqual(near / far, 8.0, delta=1e-6)  # (0.2/0.1)^3

    def test_near_field_varies_across_positions(self):
        wf = tone_waveform(64, 1e6, frequency_hz=1e5)
        src = MagneticDipoleSource(
            position_m=[0.0, 0.0, 0.0],
            moment_direction=[1.0, 0.0, 0.0],
            waveform=wf,
        )
        field = src.field(np.array([[0.05, 0, 0.1], [0.0, 0.05, 0.1]]))
        self.assertFalse(np.allclose(field[0], field[1]))


class WaveformTests(unittest.TestCase):
    def test_colored_noise_is_reproducible_and_scaled(self):
        a = colored_noise_waveform(4096, 1e6, amplitude=3.0, seed=7)
        b = colored_noise_waveform(4096, 1e6, amplitude=3.0, seed=7)
        np.testing.assert_allclose(a.samples, b.samples)
        self.assertAlmostEqual(float(np.sqrt(np.mean(a.samples**2))), 3.0, places=6)
        self.assertAlmostEqual(float(np.mean(a.samples)), 0.0, places=6)

    def test_am_carrier_envelope_bounds(self):
        wf = am_carrier_waveform(
            8000, 1e6, carrier_hz=2e5, modulation_hz=1e3, modulation_depth=0.5
        )
        self.assertLessEqual(float(np.max(np.abs(wf.samples))), 1.5 + 1e-9)

    def test_total_field_requires_matching_clock(self):
        wf1 = tone_waveform(64, 1e6, frequency_hz=1e5)
        wf2 = tone_waveform(32, 1e6, frequency_hz=1e5)
        s1 = UniformPlaneWaveSource(polarization=[1, 0, 0], waveform=wf1)
        s2 = UniformPlaneWaveSource(polarization=[0, 1, 0], waveform=wf2)
        with self.assertRaises(ValueError):
            total_field([s1, s2], np.zeros((1, 3)))


class ReferenceCoilTests(unittest.TestCase):
    def test_faraday_voltage_of_tone_has_expected_amplitude(self):
        fs, f, n = 2e6, 1e5, 4000
        amp = 2.0
        wf = tone_waveform(n, fs, frequency_hz=f, amplitude=amp)
        src = UniformPlaneWaveSource(polarization=[0, 0, 1], waveform=wf)
        coil = ReferenceCoil(pickup_vector=[0, 0, 1])
        v = coil_voltage(coil, [src])
        # d/dt[amp cos(2 pi f t)] has amplitude amp * 2 pi f.
        interior = slice(n // 8, -n // 8)
        self.assertAlmostEqual(
            float(np.max(np.abs(v[interior]))), amp * 2 * np.pi * f, delta=1e-3 * amp * 2 * np.pi * f
        )

    def test_orthogonal_pickup_sees_nothing(self):
        wf = tone_waveform(1024, 2e6, frequency_hz=1e5, amplitude=1.0)
        src = UniformPlaneWaveSource(polarization=[0, 0, 1], waveform=wf)
        coil = ReferenceCoil(pickup_vector=[1, 0, 0])  # perpendicular to field
        v = coil_voltage(coil, [src])
        np.testing.assert_allclose(v, 0.0, atol=1e-6)

    def test_saturation_clips_voltage(self):
        wf = tone_waveform(1024, 2e6, frequency_hz=1e5, amplitude=1.0)
        src = UniformPlaneWaveSource(polarization=[0, 0, 1], waveform=wf)
        coil = ReferenceCoil(pickup_vector=[0, 0, 1], saturation=10.0)
        v = coil_voltage(coil, [src])
        self.assertLessEqual(float(np.max(np.abs(v))), 10.0 + 1e-12)

    def test_reference_matrix_is_reproducible_with_seed(self):
        wf = tone_waveform(512, 2e6, frequency_hz=1e5)
        src = UniformPlaneWaveSource(polarization=[0, 0, 1], waveform=wf)
        coils = [
            ReferenceCoil(pickup_vector=[0, 0, 1], noise_sigma=1.0),
            ReferenceCoil(pickup_vector=[0, 1, 0], noise_sigma=1.0),
        ]
        x1 = reference_matrix(coils, [src], seed=3)
        x2 = reference_matrix(coils, [src], seed=3)
        self.assertEqual(x1.shape, (2, 512))
        np.testing.assert_allclose(x1, x2)

    def test_nqr_leakage_adds_signal(self):
        wf = tone_waveform(256, 2e6, frequency_hz=1e5)
        src = UniformPlaneWaveSource(polarization=[0, 0, 1], waveform=wf)
        clean = ReferenceCoil(pickup_vector=[1, 0, 0])  # sees no RFI, no leak
        leaky = ReferenceCoil(pickup_vector=[1, 0, 0], nqr_leakage=0.5)
        nqr = np.ones(256)
        v_clean = coil_voltage(clean, [src], nqr_signal=nqr)
        v_leaky = coil_voltage(leaky, [src], nqr_signal=nqr)
        np.testing.assert_allclose(v_leaky - v_clean, 0.5 * nqr, atol=1e-9)

    def test_coupling_matrix_rank(self):
        three = [
            ReferenceCoil(pickup_vector=[1, 0, 0]),
            ReferenceCoil(pickup_vector=[0, 1, 0]),
            ReferenceCoil(pickup_vector=[0, 0, 1]),
        ]
        c3 = coupling_matrix(three)
        self.assertEqual(c3.shape, (3, 3))
        self.assertEqual(np.linalg.matrix_rank(c3), 3)
        one = coupling_matrix([ReferenceCoil(pickup_vector=[0, 0, 1])])
        self.assertEqual(np.linalg.matrix_rank(one), 1)


class CancellerTests(unittest.TestCase):
    def test_scalar_canceller_recovers_complex_coefficients(self):
        n = 256
        t = np.arange(n, dtype=np.float64) / 1_000.0
        x = np.vstack(
            [
                np.exp(1j * 2 * np.pi * 30.0 * t),
                np.exp(1j * 2 * np.pi * 75.0 * t),
            ]
        )
        coeff = np.array([0.7 - 0.2j, -0.3 + 0.5j])
        y = x.T @ coeff
        fit = np.ones(n, dtype=bool)

        model = fit_scalar_canceller(y, x, fit)
        result = scalar_canceller(y, x, fit)

        np.testing.assert_allclose(model.coefficients[:, 0], coeff, atol=1e-10)
        np.testing.assert_allclose(result.cleaned, 0.0, atol=1e-10)

    def test_gated_ridge_fir_ignores_signal_gap_when_fitting(self):
        rng = np.random.default_rng(4)
        n = 600
        x = rng.normal(size=n)
        delayed = np.zeros_like(x)
        delayed[1:] = x[:-1]
        rfi = 0.8 * x - 0.25 * delayed
        signal = np.zeros(n)
        signal[250:350] = 5.0 * np.hanning(100)
        y = rfi + signal
        fit = np.ones(n, dtype=bool)
        fit[240:360] = False

        result = gated_ridge_fir_canceller(y, x, fit, taps=2, ridge=1e-9)

        np.testing.assert_allclose(result.coefficients[0], [0.8, -0.25], atol=1e-6)
        np.testing.assert_allclose(result.predicted[360:], rfi[360:], atol=1e-6)
        np.testing.assert_allclose(result.cleaned[250:350], signal[250:350], atol=1e-6)

    def test_adaptive_nlms_tracks_randomly_modulated_am_coupling(self):
        rng = np.random.default_rng(8)
        n = 5000
        t = np.arange(n, dtype=np.float64) / 20_000.0
        carrier_i = np.cos(2 * np.pi * 900.0 * t)
        carrier_q = np.sin(2 * np.pi * 900.0 * t)
        drift = np.cumsum(rng.normal(scale=0.006, size=n))
        drift = np.convolve(drift, np.ones(41) / 41.0, mode="same")
        gain_i = 0.7 + drift
        gain_q = -0.25 + 0.5 * drift
        x = np.vstack([carrier_i, carrier_q])
        rfi = gain_i * carrier_i + gain_q * carrier_q
        y = rfi.copy()
        y[2200:2600] += 2.0 * np.hanning(400)
        update = np.ones(n, dtype=bool)
        update[2200:2600] = False

        result = adaptive_lms_canceller(y, x, update, step=0.35, normalized=True)
        fixed = scalar_canceller(y, x, update)
        test = np.r_[300:2200, 2600:n]
        adaptive_mse = float(np.mean(np.abs(result.cleaned[test]) ** 2))
        fixed_mse = float(np.mean(np.abs(fixed.cleaned[test]) ** 2))

        self.assertLess(adaptive_mse, 0.25 * fixed_mse)
        frozen = result.coefficient_history[2200:2600]
        np.testing.assert_allclose(
            frozen,
            np.repeat(frozen[0:1], frozen.shape[0], axis=0),
            atol=1e-12,
        )

    def test_rls_converges_quickly_for_correlated_references(self):
        n = 400
        t = np.arange(n, dtype=np.float64) / 2_000.0
        x0 = np.cos(2 * np.pi * 40.0 * t)
        x1 = 0.95 * x0 + 0.05 * np.sin(2 * np.pi * 90.0 * t)
        x = np.vstack([x0, x1])
        y = 0.4 * x0 - 0.6 * x1

        result = adaptive_rls_canceller(y, x, forgetting=0.99, initial_covariance=100.0)

        self.assertLess(float(np.mean(np.abs(result.cleaned[100:]) ** 2)), 1e-6)

    def test_windowed_ridge_tracks_offline_random_modulation(self):
        rng = np.random.default_rng(12)
        n = 4096
        t = np.arange(n, dtype=np.float64) / 20_000.0
        carrier_i = np.cos(2 * np.pi * 850.0 * t)
        carrier_q = np.sin(2 * np.pi * 850.0 * t)
        window_samples = 512
        num_windows = n // window_samples
        window_gain = 0.7 + np.cumsum(rng.normal(scale=0.12, size=num_windows))
        window_phase = -0.2 + np.cumsum(rng.normal(scale=0.08, size=num_windows))
        gain_i = np.repeat(window_gain, window_samples)
        gain_q = np.repeat(window_phase, window_samples)
        x = np.vstack([carrier_i, carrier_q])
        rfi = gain_i * carrier_i + gain_q * carrier_q
        signal = np.zeros(n)
        signal[1800:2200] = 2.5 * np.hanning(400)
        y = rfi + signal
        fit = np.ones(n, dtype=bool)
        fit[1760:2240] = False

        windowed = windowed_ridge_fir_canceller(
            y,
            x,
            fit,
            window_samples=window_samples,
            ridge=1e-6,
            smoothness=0.05,
        )
        fixed = scalar_canceller(y, x, fit)
        test = np.r_[0:1760, 2240:n]
        windowed_mse = float(np.mean(np.abs(windowed.cleaned[test]) ** 2))
        fixed_mse = float(np.mean(np.abs(fixed.cleaned[test]) ** 2))

        self.assertEqual(windowed.coefficients.shape, (num_windows, 2, 1))
        self.assertLess(windowed_mse, 0.2 * fixed_mse)
        np.testing.assert_allclose(
            windowed.cleaned[1800:2200],
            signal[1800:2200],
            atol=0.35,
        )

    def test_joint_signal_reference_canceller_preserves_echo_basis(self):
        n = 800
        t = np.arange(n, dtype=np.float64) / 10_000.0
        x0 = np.cos(2 * np.pi * 420.0 * t)
        x1 = np.sin(2 * np.pi * 420.0 * t)
        references = np.vstack([x0, x1])
        rfi = 0.55 * x0 - 0.35 * x1

        center = 0.042
        width = 0.004
        echo_basis = np.exp(-0.5 * ((t - center) / width) ** 2)
        echo_basis = echo_basis * np.exp(1j * 2 * np.pi * 55.0 * center)
        signal = (1.4 - 0.2j) * echo_basis
        primary = rfi + signal
        fit = np.ones(n, dtype=bool)

        result = joint_signal_reference_canceller(
            primary,
            references,
            fit,
            echo_basis,
            reference_ridge=1e-9,
            signal_ridge=1e-9,
        )

        np.testing.assert_allclose(result.predicted, rfi, atol=1e-6)
        np.testing.assert_allclose(result.cleaned, signal, atol=1e-6)
        np.testing.assert_allclose(result.signal_estimate, signal, atol=1e-6)

    def test_windowed_joint_signal_reference_canceller_tracks_drift_with_echo_basis(self):
        rng = np.random.default_rng(18)
        n = 1024
        t = np.arange(n, dtype=np.float64) / 10_000.0
        x0 = np.cos(2 * np.pi * 360.0 * t)
        x1 = np.sin(2 * np.pi * 360.0 * t)
        references = np.vstack([x0, x1])
        window_samples = 256
        gain_i = np.repeat([0.45, 0.60, 0.35, 0.55], window_samples)
        gain_q = np.repeat([-0.20, -0.35, -0.10, -0.25], window_samples)
        rfi = gain_i * x0 + gain_q * x1

        center = 0.055
        width = 0.004
        echo_basis = np.exp(-0.5 * ((t - center) / width) ** 2)
        signal = (1.1 + 0.3j) * echo_basis
        primary = rfi + signal + 1e-6 * rng.normal(size=n)
        fit = np.ones(n, dtype=bool)

        result = windowed_joint_signal_reference_canceller(
            primary,
            references,
            fit,
            echo_basis,
            window_samples=window_samples,
            reference_ridge=1e-8,
            signal_ridge=1e-8,
        )

        self.assertEqual(result.coefficients.shape, (4, 2, 1))
        np.testing.assert_allclose(result.predicted, rfi, atol=2e-5)
        np.testing.assert_allclose(result.cleaned, signal, atol=2e-5)

    def test_robust_fir_resists_impulsive_outliers_that_bias_least_squares(self):
        rng = np.random.default_rng(21)
        n = 2000
        x = rng.normal(size=n)
        delayed = np.zeros_like(x)
        delayed[1:] = x[:-1]
        true = np.array([0.8, -0.35])
        references = x.reshape(1, -1)
        rfi = true[0] * x + true[1] * delayed
        y = rfi + rng.normal(scale=0.05, size=n)
        spikes = rng.choice(np.arange(5, n), size=40, replace=False)
        y[spikes] += rng.normal(scale=40.0, size=spikes.size)
        fit = np.ones(n, dtype=bool)

        robust = robust_fir_canceller(y, references, fit, taps=2, huber_delta=1.345)
        l2 = gated_ridge_fir_canceller(y, references, fit, taps=2)
        robust_err = float(np.linalg.norm(robust.coefficients[0] - true))
        l2_err = float(np.linalg.norm(l2.coefficients[0] - true))

        self.assertGreater(l2_err, 0.02)
        self.assertLess(robust_err, 0.25 * l2_err)
        self.assertGreaterEqual(robust.iterations, 1)
        self.assertTrue(np.all(robust.fit_weights[spikes] < 1.0))
        self.assertEqual(robust.fit_weights.shape, (n,))

    def test_robust_fir_matches_least_squares_without_outliers(self):
        rng = np.random.default_rng(22)
        n = 800
        x = rng.normal(size=n)
        references = x.reshape(1, -1)
        y = 0.6 * x + rng.normal(scale=0.02, size=n)
        fit = np.ones(n, dtype=bool)

        robust = robust_fir_canceller(y, references, fit, taps=1, huber_delta=1.345)
        l2 = gated_ridge_fir_canceller(y, references, fit, taps=1)

        np.testing.assert_allclose(robust.coefficients, l2.coefficients, atol=5e-3)

    def test_frequency_domain_canceller_passthrough_is_exact(self):
        rng = np.random.default_rng(31)
        n = 2048
        length = 256
        x = rng.normal(size=n)

        result = frequency_domain_canceller(x, x.reshape(1, -1), segment_length=length)

        # Interior reconstruction is exact (edges taper to a zero Hann window).
        np.testing.assert_allclose(result.cleaned[length : n - length], 0.0, atol=1e-9)
        self.assertEqual(result.transfer_function.shape, (1, length))
        self.assertEqual(result.coherence.shape, (length,))
        interior_coherence = result.coherence[np.abs(result.frequencies) > 0.02]
        self.assertGreater(float(np.mean(interior_coherence)), 0.99)

    def test_frequency_domain_canceller_removes_frequency_dependent_coupling(self):
        rng = np.random.default_rng(33)
        n = 4096
        fs = 4_000.0
        t = np.arange(n, dtype=np.float64) / fs
        reference = rng.normal(size=n)
        impulse_response = np.array([1.0, -0.7, 0.35, -0.1])
        rfi = np.convolve(reference, impulse_response)[:n]
        signal = 0.5 * np.cos(2 * np.pi * 300.0 * t)
        primary = rfi + signal

        result = frequency_domain_canceller(
            primary, reference.reshape(1, -1), segment_length=256, sample_rate_hz=fs
        )

        # Suppression is scored on the interior; the first/last segment_length
        # samples are tapered by the overlap-add windows.
        interior_mask = np.zeros(n, dtype=bool)
        interior_mask[256 : n - 256] = True
        suppression = rfi_suppression_db(
            primary, result.cleaned, interior_mask, clean_signal=signal
        ).suppression_db
        self.assertGreater(suppression, 20.0)
        # The uncorrelated tone survives the cancellation.
        interior = slice(256, n - 256)
        signal_error = float(
            np.sqrt(np.mean(np.abs(result.cleaned[interior] - signal[interior]) ** 2))
        )
        self.assertLess(signal_error, 0.2 * float(np.sqrt(np.mean(signal[interior] ** 2))))
        self.assertTrue(np.all(result.coherence >= 0.0))
        self.assertTrue(np.all(result.coherence <= 1.0))

    def test_frequency_domain_canceller_validation(self):
        x = np.ones(100)
        with self.assertRaises(ValueError):
            frequency_domain_canceller(x, x.reshape(1, -1), segment_length=200)
        with self.assertRaises(ValueError):
            frequency_domain_canceller(x, x.reshape(1, -1), segment_length=32, sample_rate_hz=0.0)
        empty_fit = np.zeros(100, dtype=bool)
        with self.assertRaises(ValueError):
            frequency_domain_canceller(x, x.reshape(1, -1), empty_fit, segment_length=32)


class DiagnosticTests(unittest.TestCase):
    def test_rfi_suppression_uses_clean_residual_when_available(self):
        n = 256
        clean = np.ones(n)
        rfi = np.cos(2 * np.pi * np.arange(n) / 16)
        before = clean + rfi
        after = clean + 0.1 * rfi
        mask = np.zeros(n, dtype=bool)
        mask[64:192] = True

        result = rfi_suppression_db(before, after, mask, clean_signal=clean)

        self.assertAlmostEqual(result.suppression_db, 20.0, places=10)
        self.assertEqual(result.sample_count, 128)

    def test_matched_filter_snr_improvement_wraps_noise_helper(self):
        rng = np.random.default_rng(5)
        clean = np.exp(-0.5 * ((np.arange(64) - 32) / 8) ** 2)
        before = clean + rng.normal(scale=0.2, size=(12, clean.size))
        after = clean + rng.normal(scale=0.05, size=(12, clean.size))

        result = matched_filter_snr_improvement(clean, before, after)

        self.assertGreater(result.after.measured_snr, result.before.measured_snr)
        self.assertGreater(result.improvement_db, 6.0)

    def test_signal_bias_reports_complex_gain(self):
        n = 128
        clean = np.exp(1j * 2 * np.pi * np.arange(n) / n)
        gain = 1.2 * np.exp(0.3j)
        result = signal_bias(clean, gain * clean)

        self.assertAlmostEqual(abs(result.complex_gain), 1.2, places=12)
        self.assertAlmostEqual(result.amplitude_bias, 0.2, places=12)
        self.assertAlmostEqual(result.phase_bias_rad, 0.3, places=12)

    def test_residual_spectral_lines_finds_complex_tone(self):
        fs = 1000.0
        n = 1000
        freq = 123.0
        t = np.arange(n) / fs
        residual = np.exp(1j * 2 * np.pi * freq * t)

        result = residual_spectral_lines(residual, fs, top_n=1)

        self.assertAlmostEqual(result.top_lines[0].frequency_hz, freq, places=6)
        self.assertGreater(result.top_lines[0].amplitude, 0.9)

    def test_reference_design_diagnostics_reports_rank_deficiency(self):
        n = 128
        x = np.cos(2 * np.pi * np.arange(n) / 16)
        refs = np.vstack([x, 2.0 * x])

        result = reference_design_diagnostics(refs)

        self.assertEqual(result.rank, 1)
        self.assertEqual(result.column_count, 2)
        self.assertGreater(result.condition_number, 1e12)

    def test_saturation_diagnostics_flags_threshold_crossings(self):
        signal = np.array([0.0, 0.5, -1.2, 2.0])

        result = saturation_diagnostics(signal, threshold=1.0)

        self.assertEqual(result.saturated_count, 2)
        self.assertAlmostEqual(result.saturated_fraction, 0.5)
        self.assertTrue(np.array_equal(result.saturated_mask, [False, False, True, True]))

    def test_reference_noise_injection_matches_closed_form(self):
        coefficients = np.array([[0.6, 0.0], [0.0, 0.8]], dtype=np.complex128)
        sigma = np.array([1.0, 0.5])

        result = reference_noise_injection(coefficients, sigma)

        np.testing.assert_allclose(result.per_channel_rms, [0.6, 0.4])
        self.assertAlmostEqual(result.injected_rms, float(np.sqrt(0.52)), places=12)
        self.assertAlmostEqual(result.noise_gain, 1.0, places=12)

    def test_reference_noise_injection_predicts_measured_output_noise(self):
        rng = np.random.default_rng(31)
        n = 200_000
        h = np.array([0.6 - 0.2j, 0.3 + 0.1j])
        sigma = np.array([0.7, 1.3])
        noise = np.vstack(
            [rng.normal(scale=sigma[0], size=n), rng.normal(scale=sigma[1], size=n)]
        )
        injected = h @ noise  # what a scalar canceller subtracts from the primary
        measured_rms = float(np.sqrt(np.mean(np.abs(injected) ** 2)))

        result = reference_noise_injection(h, sigma)

        self.assertAlmostEqual(result.injected_rms, measured_rms, delta=0.01 * measured_rms)

    def test_reference_noise_injection_accepts_scalar_sigma(self):
        result = reference_noise_injection(np.array([3.0, 4.0]), 2.0)

        self.assertAlmostEqual(result.injected_rms, 2.0 * 5.0, places=12)
        np.testing.assert_allclose(result.reference_noise_sigma, [2.0, 2.0])


class NQRMaskAdapterTests(unittest.TestCase):
    def test_slse_mask_places_pulses_after_pre_baseline(self):
        seq = slse_sequence(
            "x",
            pulse_duration_seconds=10e-6,
            nutation_hz=10e3,
            echo_spacing_seconds=100e-6,
            num_echoes=2,
        )
        mask = slse_acquisition_mask(
            seq,
            1e6,
            ringdown_seconds=5e-6,
            pre_baseline_seconds=20e-6,
            post_baseline_seconds=30e-6,
        )

        self.assertEqual(mask.num_samples, 250)
        self.assertTrue(np.all(mask.baseline_mask[:20]))
        self.assertTrue(np.all(mask.transmit_mask[20:30]))
        self.assertTrue(np.all(mask.ringdown_mask[30:35]))
        self.assertTrue(np.all(mask.signal_mask[35:120]))
        self.assertTrue(np.all(mask.transmit_mask[120:130]))
        self.assertTrue(np.all(mask.baseline_mask[220:]))

    def test_slse_rejects_pulse_longer_than_spacing(self):
        seq = slse_sequence(
            "x",
            pulse_duration_seconds=20e-6,
            nutation_hz=10e3,
            echo_spacing_seconds=10e-6,
            num_echoes=1,
        )
        with self.assertRaises(ValueError):
            slse_acquisition_mask(seq, 1e6)

    def test_sorc_mask_mirrors_tau_pulse_tau_timing(self):
        seq = sorc_sequence(
            "x",
            pulse_duration_seconds=10e-6,
            nutation_hz=10e3,
            half_spacing_seconds=40e-6,
            num_pulses=2,
        )
        mask = sorc_acquisition_mask(seq, 1e6, ringdown_seconds=5e-6)

        self.assertEqual(mask.num_samples, 180)
        self.assertTrue(np.all(mask.baseline_mask[:40]))
        self.assertTrue(np.all(mask.transmit_mask[40:50]))
        self.assertTrue(np.all(mask.ringdown_mask[50:55]))
        self.assertTrue(np.all(mask.signal_mask[55:130]))
        self.assertTrue(np.all(mask.transmit_mask[130:140]))
        self.assertTrue(np.all(mask.ringdown_mask[140:145]))
        self.assertTrue(np.all(mask.signal_mask[145:]))

    def test_sorc_can_treat_initial_gap_as_signal(self):
        seq = sorc_sequence(
            "x",
            pulse_duration_seconds=10e-6,
            nutation_hz=10e3,
            half_spacing_seconds=40e-6,
            num_pulses=1,
        )
        mask = sorc_acquisition_mask(seq, 1e6, initial_gap_is_baseline=False)
        self.assertTrue(np.all(mask.signal_mask[:40]))


class ActiveFeedforwardTests(unittest.TestCase):
    @staticmethod
    def _rms(values):
        return float(np.sqrt(np.mean(np.abs(values) ** 2)))

    def _model_reproducing(self, rfi, signal):
        # Fit a unit transfer on baseline samples where only the RFI is present.
        primary = rfi + signal
        references = rfi.reshape(1, -1)
        baseline = np.abs(signal) <= 0.0
        return fit_gated_ridge_fir(primary, references, baseline, taps=1), primary, references

    def test_ideal_actuator_matches_digital_subtraction(self):
        n = 400
        t = np.arange(n, dtype=np.float64) / 1_000.0
        rfi = np.cos(2 * np.pi * 60.0 * t)
        signal = np.zeros(n)
        signal[150:250] = 0.5 * np.hanning(100)
        model, primary, references = self._model_reproducing(rfi, signal)

        result = feedforward_cancel(
            primary, references, model, CompensationActuator(), 1_000.0
        )

        np.testing.assert_allclose(result.digitized, signal, atol=1e-9)
        self.assertEqual(result.clipped_fraction, 0.0)

    def test_active_avoids_saturation_that_defeats_digital_cancellation(self):
        n = 500
        t = np.arange(n, dtype=np.float64) / 1_000.0
        rfi = 8.0 * np.cos(2 * np.pi * 50.0 * t)
        signal = np.zeros(n)
        signal[200:300] = 0.3 * np.hanning(100)
        model, primary, references = self._model_reproducing(rfi, signal)
        saturation = 4.0  # the raw RFI (amplitude 8) saturates the ADC

        digital = model.apply(np.clip(primary, -saturation, saturation), references)
        active = feedforward_cancel(
            primary, references, model, CompensationActuator(), 1_000.0,
            adc_saturation=saturation,
        )

        digital_err = self._rms(np.real(digital.cleaned) - signal)
        active_err = self._rms(active.digitized - signal)
        self.assertLess(active_err, 0.02 * digital_err)
        self.assertEqual(active.clipped_fraction, 0.0)

    def test_latency_limits_high_frequency_cancellation(self):
        n = 1024
        fs = 1_000.0
        t = np.arange(n, dtype=np.float64) / fs
        high = np.cos(2 * np.pi * 400.0 * t)
        low = np.cos(2 * np.pi * 5.0 * t)
        model_hi = fit_gated_ridge_fir(high, high.reshape(1, -1), np.ones(n, bool), taps=1)
        model_lo = fit_gated_ridge_fir(low, low.reshape(1, -1), np.ones(n, bool), taps=1)

        ideal = feedforward_cancel(high, high.reshape(1, -1), model_hi, CompensationActuator(), fs)
        delayed_hi = feedforward_cancel(
            high, high.reshape(1, -1), model_hi, CompensationActuator(latency_samples=1), fs
        )
        delayed_lo = feedforward_cancel(
            low, low.reshape(1, -1), model_lo, CompensationActuator(latency_samples=1), fs
        )

        self.assertLess(self._rms(ideal.analog_residual), 1e-9)
        self.assertGreater(self._rms(delayed_hi.analog_residual), 1.0)
        # A one-sample delay barely affects a low-frequency tone.
        self.assertLess(
            self._rms(delayed_lo.analog_residual),
            0.1 * self._rms(delayed_hi.analog_residual),
        )

    def test_finite_drive_range_leaves_partial_residual(self):
        n = 256
        t = np.arange(n, dtype=np.float64) / 1_000.0
        rfi = 5.0 * np.cos(2 * np.pi * 40.0 * t)
        references = rfi.reshape(1, -1)
        model = fit_gated_ridge_fir(rfi, references, np.ones(n, bool), taps=1)

        limited = feedforward_cancel(
            rfi, references, model, CompensationActuator(max_field=2.0), 1_000.0
        )
        full = feedforward_cancel(rfi, references, model, CompensationActuator(), 1_000.0)

        self.assertLess(self._rms(full.analog_residual), 1e-9)
        self.assertGreater(self._rms(limited.analog_residual), 0.5)

    def test_actuator_and_feedforward_validation(self):
        with self.assertRaises(ValueError):
            CompensationActuator(latency_samples=-1)
        with self.assertRaises(ValueError):
            CompensationActuator(max_field=-1.0)
        with self.assertRaises(ValueError):
            CompensationActuator().realize(np.ones(4), 1_000.0, rng=np.random.default_rng(0), seed=1)
        n = 32
        rfi = np.cos(np.arange(n) / 3.0)
        references = rfi.reshape(1, -1)
        model = fit_gated_ridge_fir(rfi, references, np.ones(n, bool), taps=1)
        with self.assertRaises(ValueError):
            feedforward_cancel(rfi, references, model, CompensationActuator(), 1_000.0, adc_saturation=-1.0)

    def test_nqr_mask_accepts_absolute_baseline_windows(self):
        seq = slse_sequence(
            "x",
            pulse_duration_seconds=10e-6,
            nutation_hz=10e3,
            echo_spacing_seconds=100e-6,
            num_echoes=1,
        )
        mask = slse_acquisition_mask(seq, 1e6, baseline_intervals=[(50e-6, 60e-6)])
        self.assertTrue(np.all(mask.baseline_mask[50:60]))


if __name__ == "__main__":
    unittest.main()
