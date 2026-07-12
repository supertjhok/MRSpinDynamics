from __future__ import annotations

import importlib.metadata as metadata
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib

import spin_dynamics
from spin_dynamics.experiment.cli import build_parser


class PackagingTests(unittest.TestCase):
    def test_public_version_matches_distribution_metadata(self) -> None:
        self.assertEqual(
            spin_dynamics.__version__,
            metadata.version("python-spin-dynamics"),
        )

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

    def test_source_metadata_declares_license_and_project_urls(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        project = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        self.assertIn("LICENSE", project.get("license-files", []))
        self.assertTrue((project_root / "LICENSE").is_file())

        distribution = metadata.distribution("python-spin-dynamics")
        project_urls = distribution.metadata.get_all("Project-URL") or []
        labels = {entry.split(",", 1)[0] for entry in project_urls}
        self.assertTrue({"Documentation", "Source", "Issues", "Changelog"} <= labels)


if __name__ == "__main__":
    unittest.main()
