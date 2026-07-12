#!/usr/bin/env python3
"""Validate workspace metadata and built artifacts before creating a release."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
CITATION = ROOT / "CITATION.cff"
PROJECTS = {
    "python_spin_dynamics": ROOT / "PythonSpinDynamics",
    "quadrupolar_dft": ROOT / "QuadrupolarDFT",
    "mr_integration": ROOT / "integration",
}
MANUALS = {
    "python_spin_dynamics_user_manual.pdf": (
        ROOT / "PythonSpinDynamics" / "docs" / "user_manual.pdf"
    ),
    "quadrupolar_dft_user_manual.pdf": (
        ROOT / "QuadrupolarDFT" / "docs" / "user_manual.pdf"
    ),
    "matlab_spin_dynamics_user_manual.pdf": (
        ROOT / "MATLABSpinDynamics" / "docs" / "user_manual.pdf"
    ),
}
REQUIRED_PROJECT_URLS = {"Documentation", "Source", "Issues", "Changelog"}


def _citation_version() -> str:
    for line in CITATION.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^version:\s*([^\s#]+)", line)
        if match:
            return match.group(1).strip("\"'")
    raise ValueError("CITATION.cff has no version")


def workspace_versions() -> dict[str, str]:
    """Return versions declared by the citation and three subprojects."""

    versions = {"CITATION.cff": _citation_version()}
    for name, root in PROJECTS.items():
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        versions[name] = str(data["project"]["version"])
    return versions


def _check_integration_pins(version: str) -> None:
    data = tomllib.loads(
        (PROJECTS["mr_integration"] / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = set(data["project"]["dependencies"])
    for name in ("python-spin-dynamics", "quadrupolar-dft"):
        requirement = f"{name}=={version}"
        if requirement not in dependencies:
            raise ValueError(f"integration dependency must be pinned as {requirement}")


def extract_changelog(version: str) -> str:
    """Return one version body without importing command-line script state."""

    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    header = re.compile(rf"^##\s+\[{re.escape(version)}\]")
    next_header = re.compile(r"^##\s+\[")
    body: list[str] = []
    capturing = False
    for line in lines:
        if header.match(line):
            capturing = True
            continue
        if capturing and next_header.match(line):
            break
        if capturing:
            body.append(line)
    out = "\n".join(body).strip()
    if not out:
        raise ValueError(f"CHANGELOG.md has no non-empty [{version}] section")
    return out


def _check_manuals() -> None:
    for path in MANUALS.values():
        if not path.is_file() or path.stat().st_size < 10_000:
            raise ValueError(f"missing or implausibly small manual: {path}")
        if not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError(f"manual is not a PDF: {path}")


def _wheel_metadata(path: Path) -> tuple[set[str], str]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"{path.name}: expected one METADATA file")
        text = archive.read(metadata_names[0]).decode("utf-8")
    return names, text


def _check_wheel(path: Path, version: str) -> None:
    names, metadata_text = _wheel_metadata(path)
    if not any("license" in name.lower() for name in names):
        raise ValueError(f"{path.name}: wheel contains no license file")
    metadata = Parser().parsestr(metadata_text)
    if metadata.get("Version") != version:
        raise ValueError(
            f"{path.name}: metadata version {metadata.get('Version')} != {version}"
        )
    if path.name.startswith("python_spin_dynamics-"):
        labels = {
            value.split(",", 1)[0]
            for value in (metadata.get_all("Project-URL") or [])
        }
        if not REQUIRED_PROJECT_URLS <= labels:
            raise ValueError(
                f"{path.name}: missing project URLs "
                f"{sorted(REQUIRED_PROJECT_URLS - labels)}"
            )
        if "spin_dynamics/py.typed" not in names:
            raise ValueError(f"{path.name}: wheel contains no py.typed marker")
    if path.name.startswith("mr_integration-"):
        requirements = set(metadata.get_all("Requires-Dist") or [])
        for name in ("python-spin-dynamics", "quadrupolar-dft"):
            requirement = f"{name}=={version}"
            if requirement not in requirements:
                raise ValueError(f"{path.name}: missing exact requirement {requirement}")


def _check_sdist(path: Path, version: str) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        names = set(archive.getnames())
    if not any(name.lower().endswith("/license") for name in names):
        raise ValueError(f"{path.name}: sdist contains no LICENSE")
    if f"-{version}" not in path.name:
        raise ValueError(f"{path.name}: filename does not contain version {version}")


def _check_artifacts(directory: Path, version: str) -> None:
    for prefix in PROJECTS:
        wheels = tuple(directory.glob(f"{prefix}-{version}-*.whl"))
        sdists = tuple(directory.glob(f"{prefix}-{version}.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ValueError(
                f"expected one wheel and sdist for {prefix} {version}; "
                f"found {len(wheels)} and {len(sdists)}"
            )
        _check_wheel(wheels[0], version)
        _check_sdist(sdists[0], version)
    for name in MANUALS:
        path = directory / name
        if not path.is_file() or not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError(f"release artifact is missing manual {name}")


def _check_generated() -> None:
    python_root = ROOT / "PythonSpinDynamics"
    api_path = python_root / "docs" / "python_api" / "api_reference.md"
    before = api_path.read_bytes()
    subprocess.run(
        [sys.executable, "docs/generate_api_reference.py"],
        cwd=python_root,
        check=True,
    )
    if api_path.read_bytes() != before:
        raise ValueError("generated API reference was stale")
    subprocess.run(
        [sys.executable, "docs/generate_validation_matrix.py", "--check"],
        cwd=python_root,
        check=True,
    )


def _check_clean() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise ValueError("tracked working tree is not clean")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version")
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--check-generated", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args(argv)
    try:
        versions = workspace_versions()
        distinct = set(versions.values())
        if len(distinct) != 1:
            raise ValueError(f"workspace versions disagree: {versions}")
        version = distinct.pop()
        if args.expected_version is not None and version != args.expected_version:
            raise ValueError(f"workspace version {version} != {args.expected_version}")
        _check_integration_pins(version)
        extract_changelog(version)
        _check_manuals()
        if not (ROOT / "PythonSpinDynamics" / "LICENSE").is_file():
            raise ValueError("PythonSpinDynamics/LICENSE is missing")
        if args.artifacts_dir is not None:
            _check_artifacts(args.artifacts_dir.resolve(), version)
        if args.check_generated:
            _check_generated()
        if args.require_clean:
            _check_clean()
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"release preflight failed: {exc}", file=sys.stderr)
        return 1
    print(f"release preflight passed for workspace {version}")
    if args.artifacts_dir is not None:
        print(f"verified artifacts in {args.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
