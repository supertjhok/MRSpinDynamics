"""Analytic Appendix-A blocks and independent full-density CW validation."""

from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from spin_dynamics.dnp import (
    NitroxideRelaxationMatrix,
    NitroxideSaturationModel,
    bloch_saturation,
)
from spin_dynamics.dnp.overhauser import ELECTRON_GAMMA_HZ_PER_T as GE


class NitroxideRelaxationMatrixTests(unittest.TestCase):
    def setUp(self):
        self.g = np.array([2.0088, 2.006, 2.002])
        self.a = np.array([6.3, 6.3, 37.5]) * GE * 1e-4
        self.field = 9.21238
        self.tau = 12e-12
        self.options = dict(
            field_tesla=self.field,
            correlation_time_seconds=self.tau,
            g_tensor=self.g,
            hyperfine_tensor_hz=self.a,
            spin_rotation_j0_per_second=4.85e5,
            spin_rotation_jomega_per_second=4.89e5,
            additional_r1_per_second=3.5e5,
            additional_r2_per_second=1.4e7,
        )

    def test_line_dependent_bloch_limits_and_broadcasting(self):
        for spin in (0.5, 1):
            n = int(2 * spin + 1)
            t1 = np.linspace(0.3e-6, 0.8e-6, n)
            t2 = np.linspace(0.06e-6, 0.15e-6, n)
            model = NitroxideSaturationModel(
                nitrogen_spin=spin, electron_t1_seconds=t1, electron_t2_seconds=t2
            )
            result = model.steady_state(
                field_tesla=np.array([0.1, 0.35])[:, None],
                concentration_molar=np.array([0, 0.001, 0.01]),
                nutation_rad_per_s=1e7,
            )
            expected = bloch_saturation(
                1e7, t1, t2, detuning_rad_per_s=2 * np.pi * 45e6 * model.nitrogen_m
            )
            assert_allclose(
                result.line_saturation, np.broadcast_to(expected, (2, 3, n)), rtol=1e-13
            )
            self.assertEqual(result.t2e_seconds.shape, (2, 3, n))
            t1[:] = 100
            self.assertLess(np.max(model.electron_t1_seconds), 1e-6)

    def test_generalized_bloch_recovers_independent_lines(self):
        model = NitroxideSaturationModel(
            electron_t1_seconds=[0.3e-6, 0.5e-6, 0.8e-6],
            electron_t2_seconds=[0.08e-6, 0.1e-6, 0.15e-6],
        )
        params = dict(
            field_tesla=0.35,
            concentration_molar=0.01,
            nutation_rad_per_s=2e7,
            detuning_rad_per_s=np.array([-3e7, 0, 2e7]),
        )
        original = model.steady_state(**params)
        full = replace(model, coherence_transfer=True).steady_state(**params)
        assert_allclose(full.saturation_factor, original.saturation_factor, atol=2e-15)
        assert_allclose(full.populations, original.populations, atol=1e-16)
        self.assertEqual(full.electron_coherences.shape, (3, 3))

    def test_14n_appendix_a_analytic_blocks(self):
        omega = 2 * np.pi * GE * self.field
        g = omega * (self.g - self.g.mean()) / 2.00231930436256
        a = 2 * np.pi * (self.a - self.a.mean())
        g0, ga0, a0 = np.array([g @ g, g @ a, a @ a]) * self.tau / 5
        gw, gaw, aw = np.array([g0, ga0, a0]) / (1 + (omega * self.tau) ** 2)
        j0, jw, r1extra, r2extra = 4.85e5, 4.89e5, 3.5e5, 1.4e7
        matrix = NitroxideRelaxationMatrix.from_isotropic_tumbling(**self.options)
        r1, u, v, r0 = matrix.longitudinal_blocks()
        m = np.arange(-1, 2)
        off = np.diag(np.ones(2), 1) + np.diag(np.ones(2), -1)
        expected_r1 = (
            (gw + 2 * jw + (3 * a0 + 7 * aw) / 6 + r1extra) * np.eye(3)
            + 2 * gaw * np.diag(m)
            + (5 * aw - 3 * a0) / 12 * np.diag(m * m)
            + (7 * aw - 3 * a0) / 12 * off
        )
        expected_r2 = (
            ((4 * g0 + 3 * gw) / 6 + j0 + jw + (3 * a0 + 7 * aw) / 6 + r2extra)
            * np.eye(3)
            + (4 * ga0 + 3 * gaw) / 3 * np.diag(m)
            + (5 * a0 - aw) / 12 * np.diag(m * m)
            + a0 / 4 * off
        )
        expected_u = 5 * aw / 12 * np.array([[1, -1, 0], [1, 0, -1], [0, 1, -1]])
        assert_allclose(u, expected_u, atol=3e-10)
        assert_allclose(r1, expected_r1, rtol=3e-14, atol=3e-10)
        assert_allclose(
            matrix.transverse_per_second, expected_r2, rtol=3e-14, atol=3e-9
        )
        laplacian = np.diag(off.sum(axis=0)) - off
        assert_allclose(r0, (3 * a0 + 7 * aw) / 12 * laplacian, rtol=3e-14, atol=3e-10)
        assert_allclose(v, u.T, atol=1e-10)
        self.assertGreater(np.linalg.norm(u), 0)

    def test_15n_hyperfine_operator_against_independent_lindblad_jumps(self):
        # Project the general A1d operator through independent jump dissipators,
        # avoiding normalization ambiguities in compact isotope formulas.
        matrix = NitroxideRelaxationMatrix.from_isotropic_tumbling(
            field_tesla=self.field,
            correlation_time_seconds=self.tau,
            g_tensor=[2.00231930436256] * 3,
            hyperfine_tensor_hz=self.a,
            nitrogen_spin=0.5,
        )
        omega = 2 * np.pi * GE * self.field
        anisotropy = 2 * np.pi * (self.a - self.a.mean())
        a0 = (anisotropy @ anisotropy) * self.tau / 5
        aw = a0 / (1 + (omega * self.tau) ** 2)
        sz = np.kron(np.eye(2), np.diag([0.5, -0.5]))
        sp = np.kron(np.eye(2), [[0, 1], [0, 0]])
        ip = np.kron([[0, 0], [1, 0]], np.eye(2))
        iz = np.kron(np.diag([-0.5, 0.5]), np.eye(2))
        jumps = [np.sqrt(4 * a0 / 3) * iz @ sz]
        for coefficient, operator in (
            (a0 / 2, ip @ sz),
            (aw / 2, iz @ sp),
            (aw / 12, ip @ sp.T),
            (aw / 2, ip @ sp),
        ):
            jumps.extend(
                [np.sqrt(coefficient) * operator, np.sqrt(coefficient) * operator.T]
            )

        def relaxation(rho):
            result = np.zeros_like(rho)
            for jump in jumps:
                loss = jump.T @ jump
                result -= jump @ rho @ jump.T - (loss @ rho + rho @ loss) / 2
            return result

        p = np.zeros((4, 4))
        t = np.zeros((2, 2))
        for i in range(4):
            state = np.zeros((4, 4))
            state[i, i] = 1
            p[:, i] = np.diag(relaxation(state))
        for i in range(2):
            state = np.zeros((4, 4))
            state[2 * i, 2 * i + 1] = 1
            t[:, i] = relaxation(state)[[0, 2], [1, 3]]
        assert_allclose(p, matrix.population_per_second, atol=2e-10)
        assert_allclose(t, matrix.transverse_per_second, atol=2e-10)

    def test_full_nitrogen_loss_includes_nonadjacent_transitions(self):
        weights = np.array([[0, 3, 7], [3, 0, 5], [7, 5, 0]], float)
        loss = np.diag(weights.sum(axis=0)) - weights
        model = NitroxideSaturationModel(
            electron_t1_seconds=np.inf,
            electron_t2_seconds=0.001,
            nitrogen_relaxation_matrix_per_second=loss,
        )
        p = np.array([[0.04, 0.06], [0.15, 0.25], [0.20, 0.30]])
        dp = model.population_derivative(
            p, field_tesla=0.35, concentration_molar=0, nutation_rad_per_s=0
        )
        assert_allclose(dp, -loss @ p, atol=2e-15)
        assert_allclose(dp.sum(axis=0), 0, atol=2e-15)

    def test_independent_full_density_stationary_solution(self):
        blocks = NitroxideRelaxationMatrix.from_isotropic_tumbling(**self.options)
        model = NitroxideSaturationModel(
            hyperfine_hz=self.a.mean(),
            electron_t1_seconds=np.inf,
            electron_t2_seconds=np.inf,
            relaxation_matrix=blocks,
            exchange_coefficient_per_molar_second=2e9,
        )
        params = dict(
            field_tesla=self.field,
            concentration_molar=0.00315,
            nutation_rad_per_s=1.5e7,
            detuning_rad_per_s=2e7,
            temperature_kelvin=313.15,
        )
        result = model.steady_state(**params)
        pe = float(result.thermal_electron_polarization)
        reference = np.diag(np.tile([(1 - pe) / 6, (1 + pe) / 6], 3))
        offsets = 2e7 + 2 * np.pi * self.a.mean() * np.arange(-1, 2)
        h = np.zeros((6, 6))
        for m in range(3):
            h[2 * m : 2 * m + 2, 2 * m : 2 * m + 2] = [
                [offsets[m] / 2, 0.75e7],
                [0.75e7, -offsets[m] / 2],
            ]
        mixing = np.eye(3) - np.ones((3, 3)) / 3
        exchange = 2e9 * 0.00315

        def density_generator(rho):
            derivative = -1j * (h @ rho - rho @ h)
            diagonal = -blocks.population_per_second @ (
                rho.diagonal() - reference.diagonal() * np.trace(rho)
            )
            d = rho.diagonal()[1::2] - rho.diagonal()[::2]
            diagonal[::2] += exchange * mixing @ d / 2
            diagonal[1::2] -= exchange * mixing @ d / 2
            derivative[np.diag_indices(6)] += diagonal
            for indices in (
                (np.arange(0, 6, 2), np.arange(1, 6, 2)),
                (np.arange(1, 6, 2), np.arange(0, 6, 2)),
            ):
                derivative[indices] -= (
                    blocks.transverse_per_second + exchange * mixing
                ) @ rho[indices]
            for i in range(6):
                for j in range(6):
                    if i // 2 != j // 2:
                        derivative[i, j] -= 1e8 * rho[i, j]
            return derivative

        columns = []
        for i in range(36):
            unit = np.zeros((6, 6), complex)
            unit.flat[i] = 1
            columns.append(density_generator(unit).ravel())
        generator = np.column_stack(columns)
        system = generator.copy()
        system[0] = np.eye(6).ravel()
        rhs = np.zeros(36, complex)
        rhs[0] = 1
        rho = np.linalg.solve(system, rhs).reshape(6, 6)
        assert_allclose(
            rho.diagonal().real.reshape(3, 2), result.populations, atol=3e-12
        )
        assert_allclose(
            rho[np.arange(0, 6, 2), np.arange(1, 6, 2)],
            result.electron_coherences,
            atol=3e-12,
        )
        self.assertGreater(np.linalg.eigvalsh(rho).min(), 0)
        assert_allclose(np.trace(rho), 1, atol=2e-14)
        assert_allclose(
            model.population_derivative(result.populations, **params), 0, atol=2e-8
        )
        assert_allclose(
            result.saturation_factor, 1 - result.electron_polarization / pe, atol=3e-14
        )
        self.assertGreater(np.ptp(result.nitrogen_populations), 1e-7)

    def test_thermal_state_trace_and_disconnected_sectors(self):
        blocks = NitroxideRelaxationMatrix.from_isotropic_tumbling(**self.options)
        model = NitroxideSaturationModel(
            electron_t1_seconds=np.inf,
            electron_t2_seconds=np.inf,
            relaxation_matrix=blocks,
        )
        result = model.steady_state(
            field_tesla=self.field, concentration_molar=[0, 0.005], nutation_rad_per_s=0
        )
        assert_allclose(result.saturation_factor, 0, atol=0)
        assert_allclose(result.nitrogen_populations, 1 / 3, atol=1e-16)
        assert_allclose(result.populations.sum(axis=(-1, -2)), 1, atol=1e-16)
        assert_allclose(result.electron_coherences, 0, atol=0)
        zero = NitroxideRelaxationMatrix(np.zeros((6, 6)), np.eye(3))
        with self.assertRaises(ValueError):
            replace(model, relaxation_matrix=zero).steady_state(
                field_tesla=0.35, concentration_molar=0, nutation_rad_per_s=0
            )

    def test_invalid_matrices_and_parameter_vectors(self):
        for kwargs in (
            {"electron_t1_seconds": [1, 2]},
            {"electron_t2_seconds": [0.1e-6, 0.1e-6, np.nan]},
            {"nitrogen_relaxation_matrix_per_second": np.eye(3)},
            {"nitrogen_relaxation_matrix_per_second": np.ones((2, 2))},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NitroxideSaturationModel(**kwargs)
        for population, transverse in (
            (np.eye(6), np.eye(3)),
            (np.zeros((6, 6)), -np.eye(3)),
            (np.zeros((6, 6)), np.ones((2, 2))),
        ):
            with self.assertRaises(ValueError):
                NitroxideRelaxationMatrix(population, transverse)
        matrix = NitroxideRelaxationMatrix.from_isotropic_tumbling(**self.options)
        with self.assertRaises(ValueError):
            matrix.population_per_second[0, 0] = 0

    def test_inconsistent_population_and_coherence_baths_reject_unphysical_state(self):
        # A classical Markov population block and PSD R2 need not form a
        # completely positive combined quantum bath. Detect the resulting
        # excessive coherence instead of clipping the state into validity.
        population = np.kron(np.eye(3), 5e7 * np.array([[1, -1], [-1, 1]]))
        blocks = NitroxideRelaxationMatrix(population, np.eye(3) * 0.001)
        model = NitroxideSaturationModel(
            electron_t1_seconds=np.inf,
            electron_t2_seconds=np.inf,
            relaxation_matrix=blocks,
        )
        with self.assertRaisesRegex(ValueError, "nonphysical"):
            model.steady_state(
                field_tesla=0.35, concentration_molar=0, nutation_rad_per_s=np.sqrt(1e5)
            )


if __name__ == "__main__":
    unittest.main()
