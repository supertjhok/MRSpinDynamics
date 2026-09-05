"""Physical scaling and failure-boundary checks for the Phase 1 reference."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

PATH = (
    Path(__file__).resolve().parents[1]
    / "studies/nqr_mail_screening/phase1/reference.py"
)
SPEC = importlib.util.spec_from_file_location("nqr_phase1_reference", PATH)
REF = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REF
SPEC.loader.exec_module(REF)


class AbsoluteNQRReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = REF.ReferenceConfig(monte_carlo_trials=512)
        cls.report, cls.arrays = REF.build_reference(cls.cfg)

    def test_absolute_paths_and_scaling(self):
        for key, value in self.report["checks"].items():
            with self.subTest(check=key):
                self.assertLess(value, 1e-4)
        # Known analytic spin-1 factor, prevents normalized signal from passing.
        eig, tr, _ = REF.material_reference()
        expected = REF.H * tr.frequency_hz / (3 * REF.KB * self.cfg.temperature_k)
        actual = self.report["budget"]["equilibrium_population_difference"]
        self.assertAlmostEqual(actual / expected, 1, places=6)
        self.assertLess(actual, 1e-6)

    def test_gibbs_trace_positivity_and_library_equivalence(self):
        eig, _, _ = REF.material_reference()
        for temp in (1e-5, 273.15, 293.15, 323.15, 1e7):
            dev = REF.thermal_deviation(eig.levels_hz, temp)
            populations = np.diag(dev).real + 1 / 3
            np.testing.assert_allclose(
                populations, REF.boltzmann_populations(eig.levels_hz, temp), atol=2e-16
            )
            self.assertAlmostEqual(populations.sum(), 1)
            self.assertGreaterEqual(populations.min(), -1e-16)

    def test_real_rf_moment_is_full_operator_trace(self):
        cfg = self.cfg
        eig, tr, _ = REF.material_reference()
        axis = int(np.argmax(abs(tr.dipole_vector)))
        direction = np.eye(3)[axis]
        pulse = REF.SelectivePulse(
            tr.label,
            cfg.pulse_duration_s,
            cfg.flip_angle_rad / (2 * np.pi * cfg.pulse_duration_s),
        )
        rho = REF.apply_selective_pulse(
            REF.thermal_deviation(eig.levels_hz, cfg.temperature_k),
            tr,
            pulse,
            b1_direction_pas=direction,
        )
        ops = REF.spin_matrices(1)
        op = (
            eig.eigenvectors.conj().T
            @ (ops.ix, ops.iy, ops.iz)[axis]
            @ eig.eigenvectors
        )
        phasor = (
            2
            * REF.H
            * REF.GAMMA_HZ_T
            * op[tr.lower, tr.upper]
            * rho[tr.upper, tr.lower]
        )
        for time in np.arange(1, 8) / (13 * tr.frequency_hz):
            u = np.diag(np.exp(-2j * np.pi * eig.levels_hz * time))
            full = REF.H * REF.GAMMA_HZ_T * np.trace((u @ rho @ u.conj().T) @ op).real
            expected = np.real(phasor * np.exp(-2j * np.pi * tr.frequency_hz * time))
            self.assertAlmostEqual(
                full / abs(phasor), expected / abs(phasor), places=12
            )

    def test_adc_clipping_and_real_quantization(self):
        cfg = REF.ReferenceConfig(adc_bits=4, adc_full_scale_v=2)
        volts, codes, clipped = REF.adc_quantize([-2, -1, 0, 0.26, 2], cfg)
        np.testing.assert_array_equal(codes, [-8, -8, 0, 2, 7])
        np.testing.assert_array_equal(volts, codes * 0.125)
        self.assertEqual(clipped, 2)

    def test_adc_noise_and_averaging(self):
        result = REF.monte_carlo(self.cfg, self.report, dict(self.arrays))
        self.assertLess(result["noise_sd_relative_error"], 0.12)
        self.assertLess(result["four_average_ratio_relative_error"], 0.2)
        self.assertEqual(result["adc_clipped_samples"], 0)

    def test_prepolarization_state_relaxes_and_costs_are_explicit(self):
        eig, tr, line = REF.material_reference()
        eq = REF.thermal_deviation(eig.levels_hz, self.cfg.temperature_k)
        prepared = np.real(np.diag(eq)) + 1 / 3
        # Synthetic transfer-state fixture, not a material gain prediction.
        prepared[tr.lower] += 1e-7
        prepared[tr.upper] -= 1e-7
        state = REF.PreparedState(
            tuple(prepared), 0.5, 0.02, 0.039, 1.5, "synthetic test only", "predicted"
        )
        dev, cost = state.prepare(eq, line["t1_s"])
        self.assertAlmostEqual(cost, 0.559)

        actual = (dev[tr.lower, tr.lower] - eq[tr.lower, tr.lower]).real
        self.assertAlmostEqual(actual / (1e-7 / np.e), 1, places=7)
        moment = REF.powder_pulse(self.cfg, initial_deviation=dev)[0]
        self.assertGreater(abs(moment), self.report["budget"]["moment_peak_am2"])
        with self.assertRaises(ValueError):
            replace(state, source_reference="").prepare(eq, line["t1_s"])
        with self.assertRaises(ValueError):
            replace(state, populations=(1.0, 1.0, -1.0)).prepare(eq, line["t1_s"])

    def test_invalid_configuration_and_aliasing_rejected(self):
        for kwargs in (
            {"mass_kg": -1},
            {"crystalline_fraction": 1.1},
            {"temperature_k": 0},
            {"turns": 2.5},
            {"turns": 2.0},
            {"adc_bits": True},
            {"voltage_gain": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                REF.ReferenceConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            REF.build_reference(replace(self.cfg, sample_rate_hz=1e6))
        with self.assertRaisesRegex(ValueError, "dither"):
            REF.build_reference(replace(self.cfg, voltage_gain=1))


if __name__ == "__main__":
    unittest.main()
