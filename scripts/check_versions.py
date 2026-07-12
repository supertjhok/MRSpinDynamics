#!/usr/bin/env python3
"""Fail if the workspace version is not identical across all release artifacts.

MRSpinDynamics ships as a single citable unit (see ``docs/release_process.md``),
so ``CITATION.cff`` and every subproject ``pyproject.toml`` must declare the same
version. This check runs in CI so a release can never go out with a split
version.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CITATION = REPO_ROOT / "CITATION.cff"
PYPROJECTS = [
    REPO_ROOT / "PythonSpinDynamics" / "pyproject.toml",
    REPO_ROOT / "QuadrupolarDFT" / "pyproject.toml",
    REPO_ROOT / "integration" / "pyproject.toml",
]


def citation_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"""^version:\s*["']?([^"'#\s]+)""", line)
        if match:
            return match.group(1)
    raise ValueError(f"no 'version:' field found in {path}")


def pyproject_version(path: Path) -> str:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return data["project"]["version"]


def integration_dependency_versions(path: Path) -> dict[str, str | None]:
    """Return exact workspace dependency pins from the integration metadata."""

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str | None] = {}
    for dependency in data["project"]["dependencies"]:
        for name in ("python-spin-dynamics", "quadrupolar-dft"):
            if dependency == name:
                result[name] = None
            elif dependency.startswith(name + "=="):
                result[name] = dependency.split("==", 1)[1]
    return result


def main() -> int:
    versions: dict[str, str] = {}
    try:
        versions[str(CITATION.relative_to(REPO_ROOT))] = citation_version(CITATION)
        for path in PYPROJECTS:
            versions[str(path.relative_to(REPO_ROOT))] = pyproject_version(path)
    except (OSError, KeyError, ValueError) as exc:
        print(f"error reading versions: {exc}", file=sys.stderr)
        return 2

    distinct = set(versions.values())
    width = max(len(name) for name in versions)
    for name, version in versions.items():
        print(f"  {name:<{width}}  {version}")

    if len(distinct) != 1:
        print(
            f"\nFAIL: versions disagree: {sorted(distinct)}\n"
            "Use scripts/bump_version.py <version> to set them together.",
            file=sys.stderr,
        )
        return 1

    version = distinct.pop()
    try:
        pins = integration_dependency_versions(PYPROJECTS[-1])
    except (OSError, KeyError, ValueError) as exc:
        print(f"error reading integration dependency pins: {exc}", file=sys.stderr)
        return 2
    expected_pins = {
        "python-spin-dynamics": version,
        "quadrupolar-dft": version,
    }
    if pins != expected_pins:
        print(
            f"\nFAIL: integration workspace dependency pins disagree: {pins}",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: workspace version is {version}")
    print(f"OK: integration dependency pins match {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
