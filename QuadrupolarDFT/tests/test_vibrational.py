import unittest

import numpy as np

from quadrupolar_dft import (
    EFGTensor,
    VibrationalMode,
    bayer_frequency,
    bayer_slope_hz_per_k,
    efg_curvature_central_difference,
    efg_temperature_sweep,
    fit_bayer_single_mode,
    thermally_averaged_efg,
)


def _principal_tensor(vzz, eta):
    vxx = -vzz * (1.0 - eta) / 2.0
    vyy = -vzz * (1.0 + eta) / 2.0
    return EFGTensor.from_components(np.diag([vxx, vyy, vzz]), unit="si")


class CentralDifferenceTests(unittest.TestCase):
    def test_recovers_known_curvature(self):
        curvature = np.array([[2.0, 0.5, 0.0], [0.5, -1.0, 0.0], [0.0, 0.0, -1.0]])
        delta = 0.01
        v0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -1.5]])
        v_plus = v0 + 0.5 * curvature * delta**2
        v_minus = v0 + 0.5 * curvature * delta**2  # symmetric in +/- for pure quadratic
        recovered = efg_curvature_central_difference(
            v_minus, v0, v_plus, delta_q=delta
        )
        np.testing.assert_allclose(recovered, curvature, atol=1e-9)

    def test_rejects_bad_delta(self):
        zero = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            efg_curvature_central_difference(zero, zero, zero, delta_q=0.0)


class VibrationalModeTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            VibrationalMode(wavenumber_cm_inv=-1.0, efg_curvature_si=np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            VibrationalMode(wavenumber_cm_inv=100.0, efg_curvature_si=np.zeros((2, 2)))


class ThermalAveragingTests(unittest.TestCase):
    def _mode(self, czz_sign=-1.0):
        # Traceless curvature that reduces |Vzz| and shifts eta with T.
        czz = czz_sign * 4.0e69
        curvature = np.diag([-0.6 * czz, -0.4 * czz, czz])
        return VibrationalMode(wavenumber_cm_inv=200.0, efg_curvature_si=curvature)

    def test_zero_curvature_returns_equilibrium(self):
        eq = _principal_tensor(1.0e22, 0.3)
        mode = VibrationalMode(
            wavenumber_cm_inv=200.0, efg_curvature_si=np.zeros((3, 3))
        )
        averaged = thermally_averaged_efg(eq, [mode], 300.0)
        np.testing.assert_allclose(averaged.matrix_si, eq.matrix_si, atol=1e6)

    def test_zero_point_shift_at_zero_temperature(self):
        eq = _principal_tensor(1.0e22, 0.3)
        averaged0 = thermally_averaged_efg(eq, [self._mode()], 0.0)
        # Zero-point motion already lowers |Vzz| below the static value.
        self.assertLess(abs(averaged0.vzz_si), abs(eq.vzz_si))

    def test_vzz_decreases_with_temperature(self):
        eq = _principal_tensor(1.0e22, 0.3)
        mode = self._mode()
        vzz = [abs(thermally_averaged_efg(eq, [mode], t).vzz_si) for t in (0, 100, 300, 500)]
        self.assertTrue(np.all(np.diff(vzz) < 0.0))

    def test_eta_changes_with_temperature(self):
        # The tensor is averaged in the crystal frame; eta is not constant,
        # which a scalar V_zz-only average could not reproduce.
        eq = _principal_tensor(1.0e22, 0.3)
        mode = self._mode()
        eta_low = thermally_averaged_efg(eq, [mode], 50.0).eta
        eta_high = thermally_averaged_efg(eq, [mode], 500.0).eta
        self.assertGreater(abs(eta_high - eta_low), 1e-3)


class TemperatureSweepTests(unittest.TestCase):
    def test_frequencies_decrease_with_temperature(self):
        eq = _principal_tensor(1.0e22, 0.3)
        czz = -4.0e69
        mode = VibrationalMode(
            wavenumber_cm_inv=200.0,
            efg_curvature_si=np.diag([-0.6 * czz, -0.4 * czz, czz]),
        )
        points = efg_temperature_sweep(
            eq, [mode], [0.0, 150.0, 300.0, 450.0],
            spin=1.0, quadrupole_moment_barns=0.02044,
        )
        nu_plus = [np.sort(p.frequencies_hz)[-1] for p in points]
        self.assertTrue(np.all(np.diff(nu_plus) < 0.0))


class BayerTests(unittest.TestCase):
    def test_zero_temperature_value(self):
        nu = bayer_frequency(5.0e6, [0.05], [200.0], 1e-6)
        self.assertAlmostEqual(nu, 5.0e6 * (1.0 - 0.05), delta=1.0)

    def test_slope_is_negative(self):
        slope = bayer_slope_hz_per_k(5.0e6, 0.05, 200.0, 300.0)
        self.assertLess(slope, 0.0)

    def test_fit_round_trip(self):
        nu0, amplitude, wavenumber = 5.0e6, 0.05, 200.0
        temps = np.array([50.0, 120.0, 200.0, 300.0, 420.0])
        freqs = np.array(
            [bayer_frequency(nu0, [amplitude], [wavenumber], t) for t in temps]
        )
        fit = fit_bayer_single_mode(temps, freqs)
        self.assertAlmostEqual(fit.wavenumber_cm_inv, wavenumber, delta=3.0)
        self.assertAlmostEqual(fit.nu0_hz, nu0, delta=2.0e3)
        self.assertAlmostEqual(fit.amplitude, amplitude, delta=2e-3)
        self.assertLess(fit.rms_hz, 1.0)

    def test_fit_requires_enough_points(self):
        with self.assertRaises(ValueError):
            fit_bayer_single_mode([100.0, 200.0], [4.0e6, 3.9e6])

    def test_fit_on_measured_nano2_is_physical(self):
        # Ordered-phase NaNO2 14N nu_+ (database).
        temps = np.array([77.0, 80.0, 293.0, 300.0])
        freqs = np.array([4929.0, 4929.0, 4647.0, 4637.0]) * 1e3
        fit = fit_bayer_single_mode(temps, freqs)
        self.assertTrue(50.0 < fit.wavenumber_cm_inv < 350.0)
        self.assertLess(fit.rms_hz, 5.0e3)
        self.assertLess(fit.slope_hz_per_k(296.0), 0.0)


if __name__ == "__main__":
    unittest.main()
