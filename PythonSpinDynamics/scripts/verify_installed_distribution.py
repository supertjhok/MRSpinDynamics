"""Verify metadata and files from an installed PythonSpinDynamics artifact."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import spin_dynamics


def main() -> int:
    distribution = metadata.distribution("python-spin-dynamics")
    entries = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    expected = "spin_dynamics.experiment.cli:main"
    if entries.get("spin-dynamics") != expected:
        raise RuntimeError(f"missing spin-dynamics={expected!r} console entry point")

    package_dir = Path(spin_dynamics.__file__).resolve().parent
    if not (package_dir / "py.typed").is_file():
        raise RuntimeError("installed wheel does not contain spin_dynamics/py.typed")
    files = tuple(distribution.files or ())
    if not any("license" in str(path).lower() for path in files):
        raise RuntimeError("installed distribution does not contain a license file")
    project_urls = distribution.metadata.get_all("Project-URL") or []
    labels = {entry.split(",", 1)[0] for entry in project_urls}
    required_urls = {"Documentation", "Source", "Issues", "Changelog"}
    if not required_urls <= labels:
        raise RuntimeError(
            "installed distribution is missing project URLs: "
            f"{sorted(required_urls - labels)}"
        )
    if spin_dynamics.__version__ != distribution.version:
        raise RuntimeError(
            f"package version {spin_dynamics.__version__!r} does not match "
            f"distribution {distribution.version!r}"
        )

    print(f"verified python-spin-dynamics {distribution.version} at {package_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
