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

    print(f"\nOK: workspace version is {distinct.pop()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
