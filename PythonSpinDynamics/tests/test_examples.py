from __future__ import annotations

import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
LOCAL_TMP = ROOT / ".tmp"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _mandal2015_absolute_phase import (  # noqa: E402
    matched_filter_ratio,
    run_phase_resolved_probe_case,
)


def run_example(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


class ExampleSmokeTests(unittest.TestCase):
    def test_mandal2015_helper_modulates_echo_energy(self) -> None:
        baseline = run_phase_resolved_probe_case(
            probe="tuned",
            numpts=9,
            num_echoes=24,
            phase_step_cycles=0.0,
            rephase_action="ignore",
        )
        pi_periodic = run_phase_resolved_probe_case(
            probe="tuned",
            numpts=9,
            num_echoes=24,
            phase_step_cycles=0.5,
            rephase_action="ignore",
        )
        modulated = run_phase_resolved_probe_case(
            probe="tuned",
            numpts=9,
            num_echoes=24,
            phase_step_cycles=0.25,
            rephase_action="ignore",
        )

        static_ratio = np.abs(matched_filter_ratio(pi_periodic, baseline))
        modulated_ratio = np.abs(matched_filter_ratio(modulated, baseline))

        np.testing.assert_allclose(static_ratio, 1.0, atol=5e-3, rtol=5e-3)
        self.assertGreater(float(np.max(np.abs(modulated_ratio - 1.0))), 0.02)

    def test_non_plot_examples_run(self) -> None:
        commands = [
            ("examples/ideal_cpmg.py", "--numpts", "21"),
            ("examples/ideal_fid.py", "--numpts", "21"),
            ("examples/ideal_cpmg_train.py", "--numpts", "21", "--num-echoes", "3"),
            ("examples/ideal_time_varying_cpmg.py", "--numpts", "17", "--num-echoes", "4"),
            ("examples/compare_cpmg_fid.py", "--numpts", "21"),
            ("examples/tuned_probe_cpmg.py", "--numpts", "21"),
            ("examples/probe_cpmg_compare.py", "--numpts", "21"),
            ("examples/tuned_cpmg_train.py", "--numpts", "16", "--num-echoes", "2"),
            ("examples/untuned_cpmg_train.py", "--numpts", "16", "--num-echoes", "2"),
            ("examples/matched_cpmg_train.py", "--numpts", "9", "--num-echoes", "2"),
            ("examples/matched_cpmg_ir_train.py", "--numpts", "9", "--num-echoes", "2", "--num-tau", "2"),
            ("examples/finite_probe_train_sweeps.py", "--numpts", "9", "--num-echoes", "2"),
            ("examples/matched_diffusion_cpmg.py", "--numpts", "17", "--num-echoes", "2"),
            ("examples/radiation_damping_fid.py", "--points", "41"),
            ("examples/radiation_damping_cpmg_train.py", "--numpts", "9", "--num-echoes", "2"),
            ("examples/nmr_maser.py", "--points", "41", "--duration-trd", "3"),
            ("examples/heteronuclear_j_editing.py", "--points", "17"),
            ("examples/coupled_isochromat_fields.py", "--points", "9"),
            ("examples/received_signal_noise.py", "--numpts", "21"),
            ("examples/probe_parameter_sweeps.py", "--numpts", "9"),
            (
                "examples/diagnose_optimization_backends.py",
                "--numpts",
                "9",
                "--segments",
                "2",
                "--random-samples",
                "4",
                "--backend",
                "pattern",
            ),
        ]
        for command in commands:
            with self.subTest(command=command[0]):
                result = run_example(*command)
                self.assertTrue(result.stdout.strip())

    def test_examples_run_from_examples_directory(self) -> None:
        result = run_example("ideal_cpmg.py", "--numpts", "21", cwd=EXAMPLES)
        self.assertIn("Ideal CPMG example", result.stdout)

    def test_export_example_writes_npz(self) -> None:
        LOCAL_TMP.mkdir(exist_ok=True)
        output = LOCAL_TMP / f"arrays_test_{uuid.uuid4().hex}.npz"
        result = run_example(
            "examples/export_validation_arrays.py",
            str(output),
            "--numpts",
            "21",
        )
        self.assertIn("saved:", result.stdout)
        self.assertTrue(output.exists())

    def test_plot_examples_expose_cli_without_matplotlib(self) -> None:
        scripts = [
            "examples/plot_ideal_workflows.py",
            "examples/plot_ideal_imaging.py",
            "examples/plot_probe_cpmg.py",
            "examples/plot_probe_parameter_sweep.py",
            "examples/plot_optimization_workflows.py",
            "examples/plot_optimization_pipeline.py",
            "examples/diagnose_optimization_backends.py",
            "examples/plot_finite_train_workflows.py",
            "examples/plot_diffusion_sweep.py",
            "examples/plot_diffusion_absolute_phase_compare.py",
            "examples/plot_tuned_diffusion_absolute_phase_compare.py",
            "examples/plot_time_varying_sweep.py",
            "examples/plot_inverse_laplace.py",
            "examples/plot_motion_linear.py",
            "examples/plot_motion_diffusion_cpmg.py",
            "examples/plot_motion_diffusion_udd.py",
            "examples/plot_pgse_restricted_diffusion.py",
            "examples/plot_pgse_circular_pore_diffraction.py",
            "examples/plot_pgste_stimulated_echo.py",
            "examples/plot_phase_cycled_stimulated_echo.py",
            "examples/plot_pgse_double_encoding_elliptical_pore.py",
            "examples/plot_dexsy_exchange.py",
            "examples/plot_t2_t2_exchange.py",
            "examples/plot_internal_gradients.py",
            "examples/plot_ogse_frequency_diffusion.py",
            "examples/plot_rare_imaging.py",
            "examples/plot_imaging_inhomogeneity.py",
            "examples/plot_sensitive_slice.py",
            "examples/plot_wurst_flow.py",
            "examples/plot_radiation_damping.py",
            "examples/plot_radiation_damping_detuning.py",
            "examples/plot_radiation_damping_cpmg_train.py",
            "examples/plot_mandal2015_phase_step_sweep.py",
            "examples/plot_mandal2015_echo_modulation.py",
            "examples/plot_mandal2015_pulse_shapes.py",
            "examples/plot_nmr_maser.py",
            "examples/plot_j_editing_spectrum.py",
            "examples/plot_j_editing_field_spread.py",
            "examples/plot_tango_filter.py",
            "examples/plot_slic_two_spin.py",
            "examples/plot_udd_cpmg_filter.py",
            "examples/plot_esr_single_crystal.py",
            "examples/plot_esr_powder_spectrum.py",
            "examples/plot_esr_pulsed_echo.py",
            "examples/plot_esr_relaxation.py",
            "examples/plot_esr_hyperfine_doublet.py",
            "examples/plot_nqr_powder_nutation.py",
            "examples/plot_nqr_full_powder_nutation.py",
            "examples/plot_nqr_auto_model_selection.py",
            "examples/plot_chen2020_slse_relaxation.py",
            "examples/plot_nqr_population_transfer.py",
            "examples/plot_nqr_slse_offset.py",
            "examples/plot_nqr_slse_spacing.py",
            "examples/plot_nqr_efg_broadening.py",
            "examples/plot_nqr_temperature_broadening.py",
            "examples/plot_nqr_slse_efg_broadening.py",
            "examples/plot_nqr_weak_b0_spectrum.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                result = run_example(script, "--help")
                self.assertIn("usage:", result.stdout)
        result = run_example("examples/plot_probe_cpmg.py", "--help")
        self.assertIn("--masy-component", result.stdout)
        result = run_example("examples/plot_probe_parameter_sweep.py", "--help")
        self.assertIn("--workers", result.stdout)
        result = run_example("examples/plot_optimization_workflows.py", "--help")
        self.assertIn("--optimizer", result.stdout)
        self.assertIn("--inverse-starts", result.stdout)
        result = run_example("examples/plot_optimization_pipeline.py", "--help")
        self.assertIn("--refocusing-starts", result.stdout)
        self.assertIn("--inverse-starts", result.stdout)
        result = run_example("examples/diagnose_optimization_backends.py", "--help")
        self.assertIn("--backend", result.stdout)
        result = run_example("examples/plot_finite_train_workflows.py", "--help")
        self.assertIn("--probes", result.stdout)
        result = run_example("examples/plot_diffusion_sweep.py", "--help")
        self.assertIn("--q-values", result.stdout)
        result = run_example("examples/plot_diffusion_absolute_phase_compare.py", "--help")
        self.assertIn("--phase-step", result.stdout)
        self.assertIn("--diffusion-coefficient", result.stdout)
        result = run_example("examples/plot_tuned_diffusion_absolute_phase_compare.py", "--help")
        self.assertIn("--phase-step", result.stdout)
        self.assertIn("--diffusion-coefficient", result.stdout)
        result = run_example("examples/plot_time_varying_sweep.py", "--help")
        self.assertIn("--amplitudes", result.stdout)
        result = run_example("examples/plot_inverse_laplace.py", "--help")
        self.assertIn("--snr-levels", result.stdout)
        self.assertIn("--regularization", result.stdout)
        self.assertIn("--auto-regularization", result.stdout)
        result = run_example("examples/plot_motion_linear.py", "--help")
        self.assertIn("--velocity", result.stdout)
        result = run_example("examples/plot_motion_diffusion_cpmg.py", "--help")
        self.assertIn("--diffusion", result.stdout)
        result = run_example("examples/plot_motion_diffusion_udd.py", "--help")
        self.assertIn("--pulses", result.stdout)
        self.assertIn("--t2", result.stdout)
        self.assertIn("--fluctuation-amplitude", result.stdout)
        result = run_example("examples/plot_pgse_restricted_diffusion.py", "--help")
        self.assertIn("--walkers-per-cell", result.stdout)
        self.assertIn("--diffusion-time", result.stdout)
        result = run_example("examples/plot_pgse_circular_pore_diffraction.py", "--help")
        self.assertIn("--pore-radius", result.stdout)
        self.assertIn("--max-qa", result.stdout)
        result = run_example("examples/plot_pgste_stimulated_echo.py", "--help")
        self.assertIn("--fixed-b", result.stdout)
        self.assertIn("--t1", result.stdout)
        result = run_example("examples/plot_phase_cycled_stimulated_echo.py", "--help")
        self.assertIn("--t1-ms", result.stdout)
        self.assertIn("--t2-ms", result.stdout)
        self.assertIn("--offset-span-hz", result.stdout)
        result = run_example(
            "examples/plot_pgse_double_encoding_elliptical_pore.py", "--help"
        )
        self.assertIn("--semi-major", result.stdout)
        self.assertIn("--num-orientations", result.stdout)
        result = run_example("examples/plot_dexsy_exchange.py", "--help")
        self.assertIn("--mixing-time", result.stdout)
        self.assertIn("--exchange-rate", result.stdout)
        result = run_example("examples/plot_t2_t2_exchange.py", "--help")
        self.assertIn("--mixing-time-ms", result.stdout)
        self.assertIn("--exchange-rate", result.stdout)
        result = run_example("examples/plot_internal_gradients.py", "--help")
        self.assertIn("--grain-radius-um", result.stdout)
        self.assertIn("--echo-spacings-ms", result.stdout)
        self.assertIn("--susceptibility", result.stdout)
        result = run_example("examples/plot_ogse_frequency_diffusion.py", "--help")
        self.assertIn("--slab-widths", result.stdout)
        self.assertIn("--freq-max", result.stdout)
        result = run_example("examples/plot_rare_imaging.py", "--help")
        self.assertIn("--echo-train-length", result.stdout)
        self.assertIn("--readout-time", result.stdout)
        result = run_example("examples/plot_imaging_inhomogeneity.py", "--help")
        self.assertIn("--b0-spread-hz", result.stdout)
        self.assertIn("--num-offsets", result.stdout)
        result = run_example("examples/plot_sensitive_slice.py", "--help")
        self.assertIn("--b0-curvature-hz", result.stdout)
        self.assertIn("--excitation-duration", result.stdout)
        result = run_example("examples/plot_wurst_flow.py", "--help")
        self.assertIn("--sweep-width", result.stdout)
        result = run_example("examples/plot_radiation_damping.py", "--help")
        self.assertIn("--fill-factor", result.stdout)
        result = run_example("examples/plot_radiation_damping_detuning.py", "--help")
        self.assertIn("--max-detuning", result.stdout)
        result = run_example("examples/plot_radiation_damping_cpmg_train.py", "--help")
        self.assertIn("--apply-during-pulses", result.stdout)
        result = run_example("examples/plot_mandal2015_phase_step_sweep.py", "--help")
        self.assertIn("--probe", result.stdout)
        result = run_example("examples/plot_mandal2015_echo_modulation.py", "--help")
        self.assertIn("--phase-steps", result.stdout)
        result = run_example("examples/plot_mandal2015_pulse_shapes.py", "--help")
        self.assertIn("--absolute-phases", result.stdout)
        result = run_example("examples/plot_nmr_maser.py", "--help")
        self.assertIn("--t2-trd", result.stdout)
        self.assertIn("--pump-multipliers", result.stdout)
        result = run_example("examples/plot_j_editing_spectrum.py", "--help")
        self.assertIn("--max-time-ms", result.stdout)
        result = run_example("examples/plot_j_editing_field_spread.py", "--help")
        self.assertIn("--b0-spreads", result.stdout)
        self.assertIn("--b1-spreads", result.stdout)
        result = run_example("examples/plot_tango_filter.py", "--help")
        self.assertIn("--target", result.stdout)
        result = run_example("examples/plot_slic_two_spin.py", "--help")
        self.assertIn("--delta-hz", result.stdout)
        result = run_example("examples/plot_udd_cpmg_filter.py", "--help")
        self.assertIn("--min-omega-t", result.stdout)
        result = run_example("examples/plot_esr_single_crystal.py", "--help")
        self.assertIn("--microwave-ghz", result.stdout)
        self.assertIn("--broadening-mt", result.stdout)
        result = run_example("examples/plot_esr_powder_spectrum.py", "--help")
        self.assertIn("--n-chi", result.stdout)
        self.assertIn("--b1-b0-angle", result.stdout)
        self.assertIn("--detection-mode", result.stdout)
        self.assertIn("--g-strain", result.stdout)
        result = run_example("examples/plot_esr_pulsed_echo.py", "--help")
        self.assertIn("--nutation-mhz", result.stdout)
        self.assertIn("--detuning-span-mhz", result.stdout)
        result = run_example("examples/plot_esr_relaxation.py", "--help")
        self.assertIn("--t1-us", result.stdout)
        self.assertIn("--t2-us", result.stdout)
        result = run_example("examples/plot_esr_hyperfine_doublet.py", "--help")
        self.assertIn("--hyperfine-mhz", result.stdout)
        self.assertIn("--nuclear-gamma-mhz-per-t", result.stdout)
        result = run_example("examples/plot_nqr_powder_nutation.py", "--help")
        self.assertIn("--max-angle", result.stdout)
        self.assertIn("supports spin=1 only", result.stdout)
        result = run_example("examples/plot_nqr_full_powder_nutation.py", "--help")
        self.assertIn("--quadrupole-mhz", result.stdout)
        self.assertIn("--nutation-khz", result.stdout)
        result = run_example("examples/plot_nqr_auto_model_selection.py", "--help")
        self.assertIn("--offset-khz", result.stdout)
        self.assertIn("--t2-us", result.stdout)
        result = run_example("examples/plot_chen2020_slse_relaxation.py", "--help")
        self.assertIn("--max-field-g", result.stdout)
        result = run_example("examples/plot_nqr_population_transfer.py", "--help")
        self.assertIn("--perturb-angle", result.stdout)
        self.assertIn("supports spin=1 only", result.stdout)
        result = run_example("examples/plot_nqr_slse_offset.py", "--help")
        self.assertIn("--max-offset-khz", result.stdout)
        self.assertIn("--orientation", result.stdout)
        self.assertIn("supports spin=1 only", result.stdout)
        result = run_example("examples/plot_nqr_slse_spacing.py", "--help")
        self.assertIn("--min-spacing-us", result.stdout)
        self.assertIn("--orientation", result.stdout)
        self.assertIn("supports spin=1 only", result.stdout)
        result = run_example("examples/plot_nqr_efg_broadening.py", "--help")
        self.assertIn("--nuq-std-khz", result.stdout)
        self.assertIn("supports spin=1 only", result.stdout)
        result = run_example("examples/plot_nqr_temperature_broadening.py", "--help")
        self.assertIn("--nuq-slope-hz-per-k", result.stdout)
        self.assertIn("supports spin=1 only", result.stdout)
        result = run_example("examples/plot_nqr_slse_efg_broadening.py", "--help")
        self.assertIn("--echo-spacing-us", result.stdout)
        self.assertIn("--acq-us", result.stdout)
        self.assertIn("--noise-snr", result.stdout)
        self.assertIn("--deconvolve", result.stdout)
        self.assertIn("--rephase-action", result.stdout)
        self.assertIn("supports spin=1 only", result.stdout)
        result = run_example("examples/plot_nqr_weak_b0_spectrum.py", "--help")
        self.assertIn("--b0-mt", result.stdout)
        self.assertIn("--n-chi", result.stdout)
        self.assertIn("--b1-b0-angle", result.stdout)
        self.assertIn("supports spin=1 and spin=3/2", result.stdout)


if __name__ == "__main__":
    unittest.main()
