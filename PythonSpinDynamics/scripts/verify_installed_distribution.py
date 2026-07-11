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

    print(f"verified python-spin-dynamics {distribution.version} at {package_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
