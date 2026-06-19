from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling import (  # noqa: E402
    coupled_isochromat_ensemble,
    coupled_spin_system,
    fit_known_j_spectrum,
    free_precession_step,
    isotropic_j_hamiltonian,
    j_modulation_curve,
    product_operator,
    rf_step,
    simulate_coupled_isochromat_sequence,
    simulate_slic_spectrum,
    spin_operator,
    tango_b_filter,
    two_spin_slic_transfer_time,
)


class CoupledSpinTests(unittest.TestCase):
    def test_spin_operators_use_standard_commutator(self) -> None:
        ix = spin_operator(1, 0, "x")
        iy = spin_operator(1, 0, "y")
        iz = spin_operator(1, 0, "z")

        np.testing.assert_allclose(ix @ iy - iy @ ix, 1j * iz)

    def test_two_spin_isotropic_j_gap_matches_coupling(self) -> None:
        system = coupled_spin_system([0.0, 0.0], [[0.0, 7.0], [7.0, 0.0]])
        hamiltonian = isotropic_j_hamiltonian(system)

        eigenvalues = np.linalg.eigvalsh(hamiltonian) / (2.0 * np.pi)
        unique = np.unique(np.round(eigenvalues, decimals=12))
        np.testing.assert_allclose(unique, [-5.25, 1.75])

    def test_product_operator_embeds_two_spin_term(self) -> None:
        iziz = product_operator(2, [(0, "z"), (1, "z")])
        np.testing.assert_allclose(np.diag(iziz), [0.25, -0.25, -0.25, 0.25])

    def test_j_modulation_curve_matches_methanol_model(self) -> None:
        times = np.linspace(0.0, 8e-3, 9)
        curve = j_modulation_curve(
            times,
            [141.0],
            [0.75],
            background=0.25,
            cycles=1,
        )
        expected = 0.25 + 0.75 * np.cos(2.0 * np.pi * 141.0 * times)
        np.testing.assert_allclose(curve, expected)

    def test_known_j_fit_recovers_two_component_spectrum(self) -> None:
        times = np.linspace(0.0, 12.8e-3, 65)
        signal = j_modulation_curve(
            times,
            [125.0, 160.0],
            [0.85, 0.15],
            cycles=1,
        )

        fit = fit_known_j_spectrum(times, signal, [125.0, 160.0], include_background=False)

        np.testing.assert_allclose(fit.amplitudes, [0.85, 0.15], atol=1e-12)
        self.assertLess(fit.residual_norm, 1e-12)

    def test_tango_b_filter_selects_target_coupling(self) -> None:
        response = tango_b_filter([125.0, 160.0], target_coupling_hz=160.0, order=3)

        self.assertGreater(response[1], 0.99)
        self.assertLess(response[0], 0.5)

    def test_two_spin_slic_dip_occurs_at_j_coupling(self) -> None:
        coupling_hz = 7.0
        offset_difference_hz = 0.7
        system = coupled_spin_system(
            [-offset_difference_hz / 2.0, offset_difference_hz / 2.0],
            [[0.0, coupling_hz], [coupling_hz, 0.0]],
        )
        frequencies = np.linspace(4.0, 10.0, 121)

        result = simulate_slic_spectrum(
            system,
            frequencies,
            spin_lock_time=two_spin_slic_transfer_time(offset_difference_hz),
        )

        self.assertAlmostEqual(result.strongest_dip_frequency_hz, coupling_hz, delta=0.1)
        self.assertGreater(result.dip.max(), 0.45)
        self.assertLess(result.dip[0], 0.02)

    def test_coupled_isochromat_b1_scales_rf_angle(self) -> None:
        system = coupled_spin_system([0.0], [[0.0]])
        ensemble = coupled_isochromat_ensemble(
            system,
            [0.0, 0.0],
            b1_tx_scale=[1.0, 0.5],
            b1_rx_scale=1.0,
        )
        result = simulate_coupled_isochromat_sequence(
            ensemble,
            [rf_step(0.25, 1.0, phase=np.pi / 2.0)],
            initial_axis="x",
            detect_axis="x",
        )

        expected = [0.5 * np.cos(0.5 * np.pi), 0.5 * np.cos(0.25 * np.pi)]
        np.testing.assert_allclose(result.local_signals.real, expected, atol=1e-12)

    def test_coupled_isochromat_b0_offsets_dephase_signal(self) -> None:
        system = coupled_spin_system([0.0], [[0.0]])
        ensemble = coupled_isochromat_ensemble(
            system,
            [-1.0, 1.0],
            weights=[0.5, 0.5],
            b1_tx_scale=1.0,
            b1_rx_scale=1.0,
        )

        result = simulate_coupled_isochromat_sequence(
            ensemble,
            [free_precession_step(0.25)],
            initial_axis="x",
            detect_axis="x",
        )

        self.assertAlmostEqual(result.signal.real, 0.0, places=12)

    def test_coupled_isochromat_receive_weights_scale_signal(self) -> None:
        system = coupled_spin_system([0.0], [[0.0]])
        ensemble = coupled_isochromat_ensemble(
            system,
            [0.0, 0.0],
            weights=[1.0, 2.0],
            b1_tx_scale=1.0,
            b1_rx_scale=[1.0, 0.25],
        )

        result = simulate_coupled_isochromat_sequence(
            ensemble,
            [free_precession_step(0.0)],
            initial_axis="x",
            detect_axis="x",
        )

        self.assertAlmostEqual(result.signal.real, 0.75)

    def test_step_level_b0_override_supports_time_varying_fields(self) -> None:
        system = coupled_spin_system([0.0], [[0.0]])
        ensemble = coupled_isochromat_ensemble(system, [0.0])

        result = simulate_coupled_isochromat_sequence(
            ensemble,
            [
                free_precession_step(0.125, b0_offsets_hz=[1.0]),
                free_precession_step(0.125, b0_offsets_hz=[-1.0]),
            ],
            initial_axis="x",
            detect_axis="x",
        )

        self.assertAlmostEqual(result.signal.real, 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
