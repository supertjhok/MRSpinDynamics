from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from spin_dynamics.phase_cycling import (
    PhaseCycle,
    PhaseStep,
    cpmg_two_step_phase_cycle,
    diff_stebp_phase_cycle,
    eseem_stimulated_echo_phase_cycle,
    pgste_stimulated_echo_phase_cycle,
)
from spin_dynamics.workflows import run_ideal_cpmg_train, run_pgste_walkers


class PhaseCyclingTests(unittest.TestCase):
    def test_eseem_stimulated_echo_cycle_structure_and_receiver_phases(self) -> None:
        cycle = eseem_stimulated_echo_phase_cycle(n_phase=4)

        self.assertEqual(cycle.name, "eseem_stimulated_echo")
        self.assertEqual(cycle.num_steps, 4**3)
        self.assertEqual(
            tuple(cycle.pulse_names),
            ("excitation_90", "store_90", "read_90"),
        )
        # Normalized weights sum to unit modulus.
        self.assertAlmostEqual(float(np.sum(np.abs(cycle.branch_weights))), 1.0)
        # Each receiver phase selects dp = (+1, -1, -1): rec = -(phi1 - phi2 - phi3).
        phi1 = cycle.pulse_phases("excitation_90")
        phi2 = cycle.pulse_phases("store_90")
        phi3 = cycle.pulse_phases("read_90")
        expected = np.exp(1j * (phi1 - phi2 - phi3)) / cycle.num_steps
        np.testing.assert_allclose(cycle.branch_weights, expected, atol=1e-12)
        self.assertEqual(eseem_stimulated_echo_phase_cycle(5).num_steps, 5**3)

    def test_eseem_cycle_requires_enough_phase_steps(self) -> None:
        with self.assertRaises(ValueError):
            eseem_stimulated_echo_phase_cycle(n_phase=3)

    def test_cpmg_two_step_cycle_combines_like_existing_subtraction(self) -> None:
        cycle = cpmg_two_step_phase_cycle()
        first = np.array([1.0 + 2.0j, 3.0 - 1.0j])
        second = np.array([-1.0 + 1.0j, 5.0 + 0.0j])

        combined = cycle.combine([first, second])

        np.testing.assert_allclose(combined, (first - second) / 2)
        np.testing.assert_allclose(
            cycle.pulse_phases("excitation"),
            [np.pi / 2.0, 3.0 * np.pi / 2.0],
        )

    def test_receiver_phase_rotates_branch_before_accumulation(self) -> None:
        cycle = PhaseCycle(
            steps=(
                PhaseStep({"pulse": 0.0}, receiver_phase_rad=0.0, weight=1.0),
                PhaseStep({"pulse": np.pi}, receiver_phase_rad=np.pi / 2.0, weight=1.0),
            ),
            pulse_names=("pulse",),
            normalize=False,
        )

        combined = cycle.combine([1.0 + 0.0j, 1.0 + 0.0j])

        self.assertAlmostEqual(complex(combined).real, 1.0)
        self.assertAlmostEqual(complex(combined).imag, -1.0)

    def test_four_step_cyclops_cycle_rejects_mirror_coherence_artifact(self) -> None:
        phases = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
        cycle = PhaseCycle(
            steps=tuple(
                PhaseStep(
                    {"excitation": phase},
                    receiver_phase_rad=phase,
                    label=f"cyclops_{index}",
                )
                for index, phase in enumerate(phases)
            ),
            pulse_names=("excitation",),
            name="synthetic_cyclops",
        )
        two_step = PhaseCycle(
            steps=(
                PhaseStep({"excitation": phases[0]}, receiver_phase_rad=phases[0]),
                PhaseStep({"excitation": phases[2]}, receiver_phase_rad=phases[2]),
            ),
            pulse_names=("excitation",),
            name="synthetic_two_step",
        )
        desired = np.array([2.0 - 0.5j, -1.0 + 3.0j])
        dc_artifact = np.array([10.0 + 4.0j, -7.0 + 2.0j])
        mirror_artifact = np.array([0.25 + 1.5j, -2.0 + 0.75j])
        harmonic_artifact = np.array([4.0 - 3.0j, 0.5 + 0.5j])
        branches = [
            desired * np.exp(1j * phase)
            + dc_artifact
            + mirror_artifact * np.exp(-1j * phase)
            + harmonic_artifact * np.exp(2j * phase)
            for phase in phases
        ]

        combined = cycle.combine(branches)
        two_step_combined = two_step.combine([branches[0], branches[2]])

        np.testing.assert_allclose(combined, desired, atol=1e-14)
        np.testing.assert_allclose(two_step_combined, desired + mirror_artifact)

    def test_three_pulse_stimulated_echo_cycle_selects_pathway(self) -> None:
        phases = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
        branch_phases = tuple(product(phases, repeat=3))
        cycle = PhaseCycle(
            steps=tuple(
                PhaseStep(
                    {
                        "excitation_90": phi1,
                        "store_90": phi2,
                        "read_90": phi3,
                    },
                    receiver_phase_rad=phi1 - phi2 + phi3,
                    label=f"ste_{index}",
                )
                for index, (phi1, phi2, phi3) in enumerate(branch_phases)
            ),
            pulse_names=("excitation_90", "store_90", "read_90"),
            name="synthetic_stimulated_echo",
        )
        desired = np.array([0.75 + 1.25j, -2.0 + 0.5j])
        dc_artifact = np.array([5.0 - 1.0j, 2.0 + 3.0j])
        anti_echo = np.array([-0.25 + 0.5j, 1.5 - 0.25j])
        last_pulse_fid = np.array([0.1 + 2.0j, -0.75 - 0.5j])
        double_quantum_like = np.array([1.0 - 0.5j, 0.25 + 0.25j])
        branches = []
        for phi1, phi2, phi3 in branch_phases:
            branches.append(
                desired * np.exp(1j * (phi1 - phi2 + phi3))
                + dc_artifact
                + anti_echo * np.exp(1j * (-phi1 + phi2 + phi3))
                + last_pulse_fid * np.exp(1j * phi3)
                + double_quantum_like * np.exp(1j * (phi1 + phi2 + phi3))
            )

        combined = cycle.combine(branches)

        self.assertEqual(cycle.num_steps, 64)
        self.assertEqual(
            cycle.pulse_names,
            ("excitation_90", "store_90", "read_90"),
        )
        np.testing.assert_allclose(combined, desired, atol=1e-14)

    def test_diff_stebp_cycle_matches_bruker_phase_program(self) -> None:
        cycle = diff_stebp_phase_cycle()
        self.assertEqual(cycle.name, "diff_stebp")
        self.assertEqual(cycle.num_steps, 16)
        self.assertEqual(
            cycle.pulse_names,
            ("excitation_90", "refocus_180", "store_90", "read_90"),
        )
        quarter = np.pi / 2.0
        # the Bruker ph1..ph4 and ph31 columns, in quarter-cycle units
        ph1 = [0, 0, 2, 2, 1, 1, 3, 3, 2, 2, 0, 0, 3, 3, 1, 1]
        ph2 = [0, 2, 0, 2, 1, 3, 1, 3, 2, 0, 2, 0, 3, 1, 3, 1]
        ph3 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        ph31 = [0, 2, 2, 0, 1, 3, 3, 1, 2, 0, 0, 2, 3, 1, 1, 3]
        np.testing.assert_allclose(
            cycle.pulse_phases("excitation_90"), np.array(ph1) * quarter
        )
        np.testing.assert_allclose(
            cycle.pulse_phases("store_90"), np.array(ph2) * quarter
        )
        np.testing.assert_allclose(
            cycle.pulse_phases("read_90"), np.array(ph3) * quarter
        )
        np.testing.assert_allclose(
            cycle.pulse_phases("refocus_180"), np.zeros(16)
        )
        receiver = np.array(
            [step.receiver_phase_rad for step in cycle.steps]
        )
        np.testing.assert_allclose(receiver, np.array(ph31) * quarter)
        # ph31 == -(ph1 + ph2 + ph3) mod 4: the stimulated-echo pathway rule
        predicted = (-(np.array(ph1) + np.array(ph2) + np.array(ph3))) % 4
        np.testing.assert_array_equal(predicted, np.array(ph31))

    def test_diff_stebp_cycle_selects_stimulated_echo_pathway(self) -> None:
        cycle = diff_stebp_phase_cycle()
        ph1 = cycle.pulse_phases("excitation_90")
        ph2 = cycle.pulse_phases("store_90")
        ph3 = cycle.pulse_phases("read_90")
        desired = np.array([0.75 + 1.25j, -2.0 + 0.5j])
        dc_artifact = np.array([5.0 - 1.0j, 2.0 + 3.0j])
        anti_echo = np.array([-0.25 + 0.5j, 1.5 - 0.25j])
        read_fid = np.array([0.1 + 2.0j, -0.75 - 0.5j])
        branches = []
        for phi1, phi2, phi3 in zip(ph1, ph2, ph3):
            # selected pathway response is exp(-i(phi1+phi2+phi3)); add rejected ones
            branches.append(
                desired * np.exp(-1j * (phi1 + phi2 + phi3))
                + dc_artifact
                + anti_echo * np.exp(-1j * (phi1 - phi2 + phi3))
                + read_fid * np.exp(-1j * phi3)
            )
        combined = cycle.combine(branches)
        np.testing.assert_allclose(combined, desired, atol=1e-13)

    def test_phase_cycle_rejects_missing_pulse_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing pulse phases"):
            PhaseCycle(
                steps=(
                    PhaseStep({"excitation": 0.0, "refocus": 0.0}),
                    PhaseStep({"excitation": np.pi}),
                ),
                pulse_names=("excitation", "refocus"),
            )

    def test_phase_cycle_rejects_wrong_number_of_signals(self) -> None:
        cycle = cpmg_two_step_phase_cycle()

        with self.assertRaisesRegex(ValueError, "length must match"):
            cycle.combine([np.array([1.0])])

    def test_default_cpmg_train_exposes_phase_cycle_metadata(self) -> None:
        result = run_ideal_cpmg_train(
            numpts=9,
            num_echoes=2,
            rephase_action="ignore",
        )

        self.assertIsNotNone(result.phase_cycle)
        self.assertEqual(result.phase_cycle.name, "cpmg_two_step")
        self.assertEqual(result.phase_cycle.pulse_names, ("excitation",))

    def test_pgste_cycle_records_selected_pathway(self) -> None:
        cycle = pgste_stimulated_echo_phase_cycle()

        self.assertEqual(cycle.name, "pgste_stimulated_echo")
        self.assertEqual(
            cycle.pulse_names,
            ("excitation_90", "store_90", "read_90"),
        )
        self.assertEqual(cycle.num_steps, 1)
        np.testing.assert_allclose(
            [cycle.pulse_phases(name)[0] for name in cycle.pulse_names],
            [np.pi / 2.0, np.pi / 2.0, np.pi / 2.0],
        )

    def test_pgste_result_exposes_phase_cycle_metadata(self) -> None:
        result = run_pgste_walkers(
            gradient_amplitude=0.01,
            gradient_duration=1.0e-3,
            diffusion_time=6.0e-3,
            diffusion_coefficient=0.0,
            walkers_per_cell=2,
            excitation_duration=40.0e-6,
            substeps_per_interval=1,
            seed=7,
        )

        self.assertIsNotNone(result.phase_cycle)
        self.assertEqual(result.phase_cycle.name, "pgste_stimulated_echo")


if __name__ == "__main__":
    unittest.main()
