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
            ("examples/probe_parameter_sweeps.py", "--numpts", "9"),
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
        ]
        for script in scripts:
            with self.subTest(script=script):
                result = run_example(script, "--help")
                self.assertIn("usage:", result.stdout)
        result = run_example("examples/plot_probe_cpmg.py", "--help")
        self.assertIn("--masy-component", result.stdout)
        result = run_example("examples/plot_probe_parameter_sweep.py", "--help")
        self.assertIn("--workers", result.stdout)


if __name__ == "__main__":
    unittest.main()
