"""Shared imports and helper methods for MATLAB/Octave fixture tests.

The concrete test classes live in sibling modules so fixture failures are
easier to localize while preserving the original assertion bodies.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from dataclasses import replace
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.core.echo import calc_time_domain_echo, calc_time_domain_echo_arb
from spin_dynamics.core.isochromats import (
    analyze_rephasing,
    check_rephasing,
    recommended_numpts_for_rephasing,
)
from spin_dynamics.core.kernels import (
    sim_spin_dynamics_arb10,
    sim_spin_dynamics_arb10_chunked,
    sim_spin_dynamics_arb10_diffusion,
    sim_spin_dynamics_arb10_diffusion_chunked,
)
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import (
    calc_v0crit,
    calc_rotation_matrix,
    calc_rot_axis_arba3,
    calc_rot_axis_arba4,
    sim_spin_dynamics_asymp_mag3,
    sim_spin_dynamics_exc,
)
from spin_dynamics.noise import (
    NoiseSpec,
    add_received_noise,
    estimate_matched_filter_snr,
    tuned_probe_output_noise_density,
)
from spin_dynamics.parameters import (
    set_params_ideal,
    set_params_ideal_fid,
    set_params_matched_jmr,
    set_params_matched_orig,
    set_params_matched_spa,
    set_params_tuned_jmr,
    set_params_tuned_orig,
    set_params_tuned_spa,
    set_params_untuned_jmr,
    set_params_untuned_orig,
    set_params_untuned_spa,
)
from spin_dynamics.optimization import (
    analyze_matlab_optimization_results,
    analyze_optimization_result_file,
    analyze_tuned_inverse_result_files,
    analyze_tuned_inverse_result_pair,
    evaluate_ideal_time_varying_refocusing_pulse,
    evaluate_ideal_v0crit_excited_refocusing_pulse,
    evaluate_ideal_v0crit_refocusing_pulse,
    evaluate_tuned_excitation_pulse,
    evaluate_tuned_inverse_excitation_pulse,
    evaluate_matched_refocusing_pulse,
    evaluate_tuned_refocusing_pulse,
    evaluate_untuned_refocusing_pulse,
    evaluate_spa_metrics,
    ideal_time_varying_excitation_vector,
    optimize_ideal_time_varying_refocusing_phases,
    optimize_ideal_v0crit_excited_refocusing_phases,
    optimize_ideal_v0crit_refocusing_phases,
    optimize_matched_refocusing_phases,
    optimize_spa_phase_program,
    optimize_tuned_excitation_phases,
    optimize_tuned_inverse_excitation_phases,
    optimize_tuned_refocusing_phases,
    optimize_untuned_refocusing_phases,
    random_phase_starts,
    rectangular_refocusing_lengths,
    get_matlab_result_layout,
    load_optimization_results,
    load_matlab_results_mat,
    load_multistart_results_npz,
    matlab_result_layouts,
    multistart_summary_arrays,
    multistart_to_matlab_results,
    run_ideal_time_varying_refocusing_multistart,
    run_ideal_v0crit_excited_refocusing_multistart,
    run_ideal_v0crit_refocusing_multistart,
    run_tuned_excitation_inverse_pipeline,
    run_tuned_excitation_multistart,
    run_tuned_inverse_excitation_multistart,
    run_tuned_refocusing_multistart,
    save_multistart_results_mat,
    save_multistart_results_npz,
    select_matlab_result_program,
    spa_pulse_list,
    summarize_matlab_results,
    summarize_matched_spa_refocusing,
    summarize_tuned_spa_refocusing,
    summarize_untuned_spa_refocusing,
)
from spin_dynamics.pulses import (
    create_wurst_pulse,
    adjust_untuned_segment_lengths,
    matched_rectangular_pulse_response,
    matched_wurst_pulse_response,
    quantize_phase,
    tuned_rectangular_pulse_response,
    untuned_rectangular_pulse_response,
)
import spin_dynamics.optimization.drivers as driver_module
import spin_dynamics.optimization.excitation as excitation_module
import spin_dynamics.optimization.pipeline as pipeline_module
import spin_dynamics.optimization.refocusing as refocusing_module
import spin_dynamics.workflows.imaging as imaging_module
from spin_dynamics.probes.tuned import (
    calc_masy_tuned_probe_lp_orig,
    tuned_probe_lp_orig,
)
from spin_dynamics.probes.untuned import (
    calc_masy_untuned_probe_lp,
    untuned_probe_lp,
)
from spin_dynamics.probes.matched import (
    calc_masy_matched_probe_orig,
    find_coil_current,
    matching_network_design2,
)
from spin_dynamics.workflows.cpmg import calc_masy_ideal
from spin_dynamics.workflows import (
    calc_macq_ideal_probe_relax4,
    calc_macq_matched_probe_relax4,
    calc_macq_tuned_probe_relax4,
    calc_macq_untuned_probe_relax4,
    fit_imaging_echo_decay,
    form_imaging_image,
    load_imaging_field_maps_npz,
    make_imaging_field_maps,
    reconstruct_image_from_kspace,
    run_ideal_cpmg,
    run_ideal_cpmg_imaging,
    run_ideal_cpmg_ir_train,
    run_ideal_phase_encoded_cpmg_imaging,
    run_ideal_cpmg_train,
    run_ideal_time_varying_amplitude_sweep,
    run_ideal_time_varying_cpmg_final,
    run_ideal_wurst_inversion,
    run_t1_encoded_phase_encoded_cpmg_imaging,
    run_matched_cpmg,
    run_matched_cpmg_imaging,
    run_matched_cpmg_ir_train,
    run_matched_phase_encoded_cpmg_imaging,
    run_matched_cpmg_train,
    run_matched_diffusion_cpmg,
    run_matched_diffusion_q_sweep,
    run_matched_finite_mistuning_sweep,
    run_matched_finite_q_sweep,
    run_matched_mistuning_sweep,
    run_matched_q_sweep,
    run_matched_time_varying_amplitude_sweep,
    run_matched_time_varying_cpmg_final,
    run_matched_wurst_cpmg,
    run_matched_wurst_inversion,
    run_matched_z_magnetization_q_sweep,
    run_tuned_cpmg,
    run_tuned_cpmg_imaging,
    run_tuned_cpmg_ir_train,
    run_tuned_phase_encoded_cpmg_imaging,
    run_tuned_cpmg_train,
    run_tuned_finite_mistuning_sweep,
    run_tuned_finite_q_sweep,
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
    run_tuned_time_varying_amplitude_sweep,
    run_tuned_time_varying_cpmg_final,
    run_untuned_cpmg,
    run_untuned_cpmg_ir_train,
    run_untuned_cpmg_train,
    run_untuned_finite_mistuning_sweep,
    run_untuned_finite_q_sweep,
    run_untuned_time_varying_amplitude_sweep,
    run_untuned_time_varying_cpmg_final,
    VALIDATED_MATCHED_DIFFUSION_Q_MAX,
    check_matched_diffusion_q_stability,
    sinusoidal_field_waveform,
    summarize_imaging_noise_trials,
)
from spin_dynamics.workflows.fid import sim_fid_ideal


FIXTURES = ROOT / "validation" / "fixtures"




class OctaveFixtureBase(unittest.TestCase):
    def _probe_relax4_inputs(
        self,
        numpts: int,
    ) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray | float | list]]:
        del_w = np.linspace(-5, 5, numpts)
        sp = {
            "del_w": del_w,
            "del_wg": np.linspace(-0.75, 0.75, numpts),
            "w_1": 0.95 + 0.07 * np.cos(del_w / 2),
            "T1": 1.4 + 0.08 * np.cos(del_w / 3),
            "T2": 0.9 + 0.05 * np.sin(del_w / 4),
            "m0": 0.8 + 0.04 * np.cos(del_w),
            "mth": 1.05 + 0.02 * np.sin(del_w),
        }
        rtot = [
            calc_rotation_matrix(
                del_w,
                sp["w_1"],
                np.array([np.pi / 2]),
                np.array([np.pi / 2]),
                np.array([1.0]),
            ),
            calc_rotation_matrix(
                del_w,
                sp["w_1"],
                np.array([0.45 * np.pi, 0.45 * np.pi]),
                np.array([0, np.pi / 4]),
                np.array([0.8, 1.05]),
            ),
            calc_rotation_matrix(
                del_w,
                sp["w_1"],
                np.array([1.1 * np.pi]),
                np.array([-np.pi / 5]),
                np.array([0.9]),
            ),
        ]
        pp = {
            "T_90": 25e-6,
            "tp": np.array(
                [
                    np.pi / 2,
                    0.35 * np.pi,
                    0.9 * np.pi,
                    0.4 * np.pi,
                    0.55 * np.pi,
                    1.1 * np.pi,
                ]
            ),
            "amp": np.array([1, 0, 1, 0, 0, 1]),
            "acq": np.array([0, 1, 0, 1, 1, 0]),
            "grad": np.array([0, 0.25, 0, -0.2, 0.15, 0]),
            "pul": np.array([1, 0, 2, 0, 0, 3]),
            "Rtot": rtot,
        }
        return sp, pp
    def _load_probe_relax4_fixture(
        self,
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        table = np.loadtxt(FIXTURES / filename, delimiter=",")
        acq_count = int(np.max(table[:, 0]))
        numpts = int(np.max(table[:, 1]))
        macq_ref = np.zeros((acq_count, numpts), dtype=np.complex128)
        mrx_ref = np.zeros((acq_count, numpts), dtype=np.complex128)
        for row in table:
            idx = (int(row[0]) - 1, int(row[1]) - 1)
            macq_ref[idx] = row[2] + 1j * row[3]
            mrx_ref[idx] = row[4] + 1j * row[5]
        return macq_ref, mrx_ref
    def _assert_train_fixture(
        self,
        fixture_stem: str,
        runner,
        numpts_expected: int | None,
        maxoffs: float,
        rtol: float,
        atol: float,
    ) -> None:
        mrx_table = np.loadtxt(FIXTURES / f"{fixture_stem}_mrx.csv", delimiter=",")
        echo_table = np.loadtxt(FIXTURES / f"{fixture_stem}_echo.csv", delimiter=",")
        int_table = np.loadtxt(FIXTURES / f"{fixture_stem}_integrals.csv", delimiter=",")

        num_echoes = int(np.max(mrx_table[:, 0]))
        numpts = int(np.max(mrx_table[:, 1]))
        if numpts_expected is not None:
            self.assertEqual(numpts, numpts_expected)
        mrx_ref = np.zeros((num_echoes, numpts), dtype=np.complex128)
        for row in mrx_table:
            mrx_ref[int(row[0]) - 1, int(row[1]) - 1] = row[2] + 1j * row[3]

        nacq = int(np.max(echo_table[:, 1]))
        echo_ref = np.zeros((num_echoes, nacq), dtype=np.complex128)
        tvect_ref = np.zeros(nacq, dtype=np.float64)
        for row in echo_table:
            echo_ref[int(row[0]) - 1, int(row[1]) - 1] = row[2] + 1j * row[3]
            tvect_ref[int(row[1]) - 1] = row[4]

        echo_int_ref = int_table[:, 1] + 1j * int_table[:, 2]

        result = runner(
            numpts=numpts,
            maxoffs=maxoffs,
            num_echoes=num_echoes,
            t1_seconds=1.7,
            t2_seconds=1.1,
            rephase_action="ignore",
        )

        np.testing.assert_allclose(result.mrx, mrx_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(result.echo, echo_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(result.tvect, tvect_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            result.echo_integrals,
            echo_int_ref,
            rtol=rtol,
            atol=atol,
        )
        self.assertEqual(result.probe, fixture_stem.removeprefix("run_").removesuffix("_cpmg_train"))
    def _assert_imaging_fixture(
        self,
        fixture_stem: str,
        runner,
        rtol: float,
        atol: float,
    ) -> None:
        table = np.loadtxt(FIXTURES / f"{fixture_stem}_kspace.csv", delimiter=",")
        px = int(np.max(table[:, 0]))
        pz = int(np.max(table[:, 1]))
        num_echoes = int(np.max(table[:, 2]))
        kspace_ref = np.zeros((px, pz, num_echoes), dtype=np.complex128)
        for row in table:
            kspace_ref[int(row[0]) - 1, int(row[1]) - 1, int(row[2]) - 1] = row[3] + 1j * row[4]

        rho = np.array([[0.0, 1.0], [1.0, 0.35]], dtype=np.float64)
        relaxation = 5e-3 * np.ones_like(rho)
        result = runner(
            rho,
            t1_map=relaxation,
            t2_map=relaxation,
            num_echoes=num_echoes,
            echo_spacing_seconds=0.2e-3,
            gradient_duration_seconds=0.5e-3,
            fov=(20.0, 20.0),
            ny=400,
            num_workers=1,
            phase_workers=1,
        )

        np.testing.assert_allclose(result.kspace, kspace_ref, rtol=rtol, atol=atol)
    def _assert_pulse_response_fixture(
        self,
        fixture_stem: str,
        result,
        rtol: float,
        atol: float,
    ) -> None:
        table = np.loadtxt(FIXTURES / f"{fixture_stem}.csv", delimiter=",")
        rot_idx = self._fixture_sample_indices(result.rotating_time.size)
        raw_idx = self._fixture_sample_indices(result.raw_time.size)
        tf_idx = table[:, 6]
        tf_idx = tf_idx[np.isfinite(tf_idx)].astype(int) - 1

        np.testing.assert_allclose(result.rotating_time[rot_idx], table[: rot_idx.size, 0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            result.rotating_current[rot_idx],
            table[: rot_idx.size, 1] + 1j * table[: rot_idx.size, 2],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(result.raw_time[raw_idx], table[: raw_idx.size, 3], rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            result.raw_current[raw_idx],
            table[: raw_idx.size, 4] + 1j * table[: raw_idx.size, 5],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            result.receiver_tf[tf_idx],
            table[: tf_idx.size, 7] + 1j * table[: tf_idx.size, 8],
            rtol=rtol,
            atol=atol,
        )
    @staticmethod
    def _fixture_sample_indices(size: int) -> np.ndarray:
        return np.unique(np.rint(np.linspace(1, size, min(8, size))).astype(int)) - 1
