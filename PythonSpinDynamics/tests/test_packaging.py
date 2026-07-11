from __future__ import annotations

import importlib.metadata as metadata
import unittest
from pathlib import Path

import spin_dynamics
from spin_dynamics.experiment.cli import build_parser


class PackagingTests(unittest.TestCase):
    def test_installed_console_entry_point_targets_experiment_cli(self) -> None:
        distribution = metadata.distribution("python-spin-dynamics")
        entries = {
            entry.name: entry.value
            for entry in distribution.entry_points
            if entry.group == "console_scripts"
        }
        self.assertEqual(
            entries.get("spin-dynamics"),
            "spin_dynamics.experiment.cli:main",
        )
        self.assertEqual(build_parser().prog, "spin-dynamics")

    def test_typed_package_marker_is_present(self) -> None:
        package_dir = Path(spin_dynamics.__file__).resolve().parent
        self.assertTrue((package_dir / "py.typed").is_file())


if __name__ == "__main__":
    unittest.main()
