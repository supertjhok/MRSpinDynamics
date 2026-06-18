from __future__ import annotations

import subprocess
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
LOCAL_TMP = ROOT / ".tmp"


def run_example(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


class ExampleSmokeTests(unittest.TestCase):
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
            "examples/plot_time_varying_sweep.py",
            "examples/plot_inverse_laplace.py",
            "examples/plot_motion_linear.py",
            "examples/plot_motion_diffusion_cpmg.py",
            "examples/plot_wurst_flow.py",
            "examples/plot_radiation_damping.py",
            "examples/plot_radiation_damping_detuning.py",
            "examples/plot_radiation_damping_cpmg_train.py",
            "examples/plot_nmr_maser.py",
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
        result = run_example("examples/plot_wurst_flow.py", "--help")
        self.assertIn("--sweep-width", result.stdout)
        result = run_example("examples/plot_radiation_damping.py", "--help")
        self.assertIn("--fill-factor", result.stdout)
        result = run_example("examples/plot_radiation_damping_detuning.py", "--help")
        self.assertIn("--max-detuning", result.stdout)
        result = run_example("examples/plot_radiation_damping_cpmg_train.py", "--help")
        self.assertIn("--apply-during-pulses", result.stdout)
        result = run_example("examples/plot_nmr_maser.py", "--help")
        self.assertIn("--t2-trd", result.stdout)
        self.assertIn("--pump-multipliers", result.stdout)


if __name__ == "__main__":
    unittest.main()
