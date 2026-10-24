"""FFHS limits and independent numerical diffusion-spectrum validation."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from spin_dynamics.dnp import (
    TranslationalDiffusionModel,
    ffhs_spectral_density,
    rotational_spectral_density,
    translational_relaxation_rates,
)

try:
    from scipy.integrate import quad
except ImportError:
    quad = None


class DNPSpectralTests(unittest.TestCase):
    def test_zero_frequency_and_even_broadcast_spectrum(self):
        tau = np.array([1e-12, 1e-10, 1e-8])
        assert_allclose(ffhs_spectral_density(0, tau), 4 * tau / 9, rtol=1e-15)
        omega = np.array([-1e8, 0, 1e8])[:, None]
        spectrum = ffhs_spectral_density(omega, tau)
        self.assertEqual(spectrum.shape, (3, 3))
        assert_allclose(spectrum[0], spectrum[2], rtol=1e-15)
        self.assertTrue(np.all(spectrum > 0))

    def test_small_and_large_frequency_asymptotes(self):
        # Hwang-Freed Appendix A: normalized shape starts 1-3*sqrt(2*u)/8;
        # high-frequency one-sided integral tends 9*tau/u^2.
        u = 1e-12
        actual = ffhs_spectral_density(u, 1) / (4 / 9)
        assert_allclose((1 - actual) / np.sqrt(u), 3 * np.sqrt(2) / 8, rtol=2e-6)
        u = 1e12
        assert_allclose(ffhs_spectral_density(u, 1) * u**2, 9, rtol=5e-6)
        # Real rational evaluation also survives ranges where complex leading
        # terms cancel or a direct sixth-degree polynomial overflows.
        values = ffhs_spectral_density([1e100, 1e160, 1e300], 1)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0))
        assert_allclose(ffhs_spectral_density(1, 1e200), 9e-200, rtol=1e-12, atol=0)

    def test_diffusion_time_and_rotational_convention(self):
        model = TranslationalDiffusionModel(3e-10, 3e-9)
        self.assertAlmostEqual(model.diffusion_time_seconds / 1e-12, 30)
        self.assertEqual(rotational_spectral_density(0, 2), 2)
        self.assertNotEqual(
            model.spectral_density(1e10), rotational_spectral_density(1e10, 30e-12)
        )

    def test_invalid_spectral_inputs(self):
        for value in [0, -1, np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ffhs_spectral_density(1, value)
            with self.assertRaises(ValueError):
                TranslationalDiffusionModel(value, 1e-9)
        with self.assertRaises(ValueError):
            ffhs_spectral_density(np.inf, 1)
        with self.assertRaises(ValueError):
            TranslationalDiffusionModel([1, 2], 1)

    @unittest.skipIf(quad is None, "independent integral requires optional SciPy")
    def test_against_hwang_freed_diffusion_eigenmode_integral(self):
        # Eq. A6 integrated in time analytically, then integrated over the
        # diffusion eigenmode x=pd. Independent of the production rational form.
        # G(0)=n/(3*d^3); J_one_sided/(tau*G(0)) is the integral below.
        for u in [0, 1e-3, 0.1, 1, 10, 100, 1e3]:

            def integrand(x):
                polynomial = x**6 - 2 * x**4 + 9 * x**2 + 81
                frequency = 1 if u == 0 else x**4 / (x**4 + u**2)
                return frequency / polynomial

            integral, error = quad(
                integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-11, limit=300
            )
            reference = 54 / np.pi * integral
            with self.subTest(omega_tau=u):
                assert_allclose(ffhs_spectral_density(u, 1), reference, rtol=2e-10)
                self.assertLess(error, 1e-11)

    @unittest.skipIf(quad is None, "independent integral requires optional SciPy")
    def test_absolute_rate_prefactor_from_spatial_variance(self):
        # Angular integration gives 4*pi; integrate r^-6 over exterior volume.
        # S=1/2 spin factor is 1/10 for the one-sided normalized spectrum.
        d, diffusion, concentration = 3e-10, 3e-9, 0.001
        model = TranslationalDiffusionModel(d, diffusion)
        rates = translational_relaxation_rates(1e-12, model, concentration)
        radial, _ = quad(lambda x: x**-4, 1, np.inf)
        spatial_variance = 4 * np.pi * 6.02214076e26 * concentration * radial / d**3
        hbar = 6.62607015e-34 / (2 * np.pi)
        b = (
            1.00000000055e-7
            * hbar
            * (2 * np.pi * 42.57747892e6)
            * (2 * np.pi * 28.0249514242e9)
        )
        # Zero-frequency Poisson solution of -D laplacian u=r^-3 Y_2m:
        # u=r^-1/(6D)-d^2*r^-3/(18D), with du/dr(d)=0.
        time_integral, _ = quad(lambda x: x**-2 / 6 - x**-4 / 18, 1, np.inf)
        tau_integral = (time_integral / radial) * d**2 / diffusion
        expected = b**2 * spatial_variance * tau_integral
        assert_allclose(rates.self_relaxation_per_second, expected, rtol=2e-6)
        assert_allclose(rates.cross_relaxation_per_second, expected / 2, rtol=3e-6)

    def test_rates_from_orientation_averaged_two_spin_hamiltonian(self):
        # Independent spin-factor and sign check. Electron gamma is negative:
        # H_Z=+we*Sz-wn*Iz. DQ spin flips sample we-wn, not we+wn.
        sx = np.array([[0, 1], [1, 0]]) / 2
        sy = np.array([[0, -1j], [1j, 0]]) / 2
        sz = np.diag([0.5, -0.5])
        electron = [np.kron(op, np.eye(2)) for op in [sx, sy, sz]]
        nucleus = [np.kron(np.eye(2), op) for op in [sx, sy, sz]]
        z_axis, weights = np.polynomial.legendre.leggauss(6)
        squared_elements = np.zeros((4, 4))
        for z, weight in zip(z_axis, weights):
            for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                vector = [
                    np.sqrt(1 - z * z) * np.cos(phi),
                    np.sqrt(1 - z * z) * np.sin(phi),
                    z,
                ]
                sdotr = sum(v * op for v, op in zip(vector, electron))
                idotr = sum(v * op for v, op in zip(vector, nucleus))
                h = sum(s @ i for s, i in zip(electron, nucleus)) - 3 * sdotr @ idotr
                squared_elements += weight / 24 * np.abs(h) ** 2
        field = 0.35
        we, wn = 2 * np.pi * field * np.array([28.0249514242e9, 42.57747892e6])
        energies = np.diag(we * electron[2] - wn * nucleus[2]).real
        model = TranslationalDiffusionModel(3e-10, 3e-9)
        b0 = (
            1.00000000055e-7
            * (6.62607015e-34 / (2 * np.pi))
            * (2 * np.pi * 28.0249514242e9)
            * (2 * np.pi * 42.57747892e6)
        )
        variance = b0 * b0 * 4 * np.pi * (6.02214076e26 * 0.001) / (3 * (3e-10) ** 3)
        frequencies = energies[:, None] - energies[None, :]
        generator = (
            2 * variance * squared_elements * model.spectral_density(frequencies)
        )
        np.fill_diagonal(generator, 0)
        generator -= np.diag(generator.sum(axis=0))
        iz, ez = np.diag(nucleus[2]).real, np.diag(electron[2]).real
        expected_self = -iz @ generator @ iz / (iz @ iz)
        expected_cross = -iz @ generator @ ez / (iz @ iz)
        rates = translational_relaxation_rates(field, model, 0.001)
        assert_allclose(rates.self_relaxation_per_second, expected_self, rtol=2e-14)
        assert_allclose(rates.cross_relaxation_per_second, expected_cross, rtol=2e-14)


if __name__ == "__main__":
    unittest.main()
