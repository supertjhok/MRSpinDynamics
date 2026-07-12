from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPOSITORY / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"release_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_preflight_reads_consistent_workspace_metadata() -> None:
    preflight = _load_script("release_preflight")
    versions = preflight.workspace_versions()
    assert len(set(versions.values())) == 1
    version = next(iter(versions.values()))
    assert preflight.extract_changelog(version)
    preflight._check_integration_pins(version)


def test_checksum_writer_is_stable_and_excludes_its_output(tmp_path) -> None:
    writer = _load_script("write_sha256s")
    (tmp_path / "b.txt").write_text("b", encoding="ascii")
    (tmp_path / "a.txt").write_text("a", encoding="ascii")
    output = tmp_path / "SHA256SUMS"
    output.write_text("old", encoding="ascii")
    lines = writer.checksum_lines(tmp_path, output)
    assert [line.split("  ", 1)[1] for line in lines] == ["a.txt", "b.txt"]
    assert writer.main([str(tmp_path)]) == 0
    assert output.is_file()


def test_version_bump_updates_integration_dependency_pins(tmp_path) -> None:
    bump = _load_script("bump_version")
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nversion = "0.2.0"\ndependencies = [\n'
        '  "python-spin-dynamics==0.2.0",\n'
        '  "quadrupolar-dft==0.2.0",\n]\n',
        encoding="utf-8",
    )
    bump.PYPROJECTS[-1] = project
    bump.bump_pyproject(project, "0.3.0")
    text = project.read_text(encoding="utf-8")
    assert 'version = "0.3.0"' in text
    assert '"python-spin-dynamics==0.3.0"' in text
    assert '"quadrupolar-dft==0.3.0"' in text
