#!/usr/bin/env python3
"""Set the workspace version across all release artifacts in one step.

Updates the ``version:`` field in ``CITATION.cff`` and the ``version`` field in
each subproject ``pyproject.toml`` so they stay identical (see
``docs/release_process.md``). Does not touch the CHANGELOG or create a tag.

Usage:
    python scripts/bump_version.py 0.2.0
    python scripts/bump_version.py 0.2.0 --date 2026-09-01   # also set CITATION date-released
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CITATION = REPO_ROOT / "CITATION.cff"
PYPROJECTS = [
    REPO_ROOT / "PythonSpinDynamics" / "pyproject.toml",
    REPO_ROOT / "QuadrupolarDFT" / "pyproject.toml",
    REPO_ROOT / "integration" / "pyproject.toml",
]

# A loose semver-ish check; allows pre-release suffixes like 0.2.0rc1.
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+([abrc].*)?$")


def _sub_once(text: str, pattern: str, replacement: str, path: Path) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"could not update pattern {pattern!r} in {path}")
    return new_text


def bump_citation(path: Path, version: str, date: str | None) -> None:
    text = path.read_text(encoding="utf-8")
    text = _sub_once(text, r'^version:.*$', f'version: {version}', path)
    if date is not None:
        text = _sub_once(text, r'^date-released:.*$', f'date-released: {date}', path)
    path.write_text(text, encoding="utf-8")


def bump_pyproject(path: Path, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = _sub_once(text, r'^version\s*=\s*".*"$', f'version = "{version}"', path)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="new workspace version, e.g. 0.2.0")
    parser.add_argument(
        "--date",
        default=None,
        help="optional date-released for CITATION.cff (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    if not VERSION_RE.match(args.version):
        print(f"error: {args.version!r} is not a valid version", file=sys.stderr)
        return 2

    try:
        bump_citation(CITATION, args.version, args.date)
        for path in PYPROJECTS:
            bump_pyproject(path, args.version)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Set workspace version to {args.version}.")
    print("Next: update CHANGELOG.md, commit, then tag v" + args.version + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
