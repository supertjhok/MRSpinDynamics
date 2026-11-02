"""Build the in-browser (JupyterLite) notebooks for the documentation site.

The notebook sources in ``docs/lite/notebooks`` are plain Python files with
``# %%`` cell markers, so they stay readable, lintable, and testable under
CPython. This script converts them to ``.ipynb`` (adding a first cell that
installs the package from a wheel bundled with the site) and runs ``jupyter
lite build``. Run it from ``PythonSpinDynamics/`` after ``pip install
".[lite]"``::

    python docs/lite/build_lite.py --out site/lite

``--notebooks-only DIR`` writes just the ``.ipynb`` files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

LITE_DIR = Path(__file__).resolve().parent
NOTEBOOK_SOURCES = LITE_DIR / "notebooks"
PROJECT_DIR = LITE_DIR.parents[1]

# Runs in the Pyodide kernel; piplite finds the package in the wheels bundled
# with the site and NumPy/Matplotlib in the Pyodide distribution.
INSTALL_CELL = "%pip install -q python-spin-dynamics matplotlib"


def _cell(cell_type: str, lines: list[str], index: int) -> dict:
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    source = [line + "\n" for line in lines[:-1]] + lines[-1:]
    cell: dict = {"cell_type": cell_type, "id": f"cell-{index}", "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    """Split ``# %%`` source into ``(cell_type, lines)`` pairs."""

    cells: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("# %%"):
            cells.append(("markdown" if "[markdown]" in line else "code", []))
            continue
        if not cells:
            if line.strip():
                raise ValueError("notebook source must start with a '# %%' cell marker")
            continue
        kind, lines = cells[-1]
        if kind == "markdown":
            if line.strip() and not line.startswith("#"):
                raise ValueError(f"markdown cell line must be a comment: {line!r}")
            line = line[2:] if line.startswith("# ") else line.lstrip("#")
        lines.append(line)
    return [(kind, lines) for kind, lines in cells if any(line.strip() for line in lines)]


def notebook_from_source(text: str) -> dict:
    """Return an nbformat 4.5 notebook with the install cell after the title."""

    cells = split_cells(text)
    if not cells or cells[0][0] != "markdown":
        raise ValueError("notebook source must open with a markdown cell")
    cells.insert(1, ("code", [INSTALL_CELL]))
    return {
        "cells": [_cell(kind, list(lines), index) for index, (kind, lines) in enumerate(cells)],
        "metadata": {
            "kernelspec": {
                "display_name": "Python (Pyodide)",
                "language": "python",
                "name": "python",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebooks(out_dir: Path) -> list[Path]:
    """Convert every notebook source; return the written paths."""

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for source in sorted(NOTEBOOK_SOURCES.glob("*.py")):
        notebook = notebook_from_source(source.read_text(encoding="utf-8"))
        target = out_dir / f"{source.stem}.ipynb"
        target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        written.append(target)
    return written


def _build_wheel(out_dir: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(PROJECT_DIR)],
        check=True,
    )
    (wheel,) = out_dir.glob("python_spin_dynamics-*.whl")
    return wheel


def build_site(out: Path, wheel: Path | None) -> None:
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        contents = scratch / "contents"
        write_notebooks(contents)
        if wheel is None:
            wheel = _build_wheel(scratch / "dist")
        lite_dir = scratch / "lite"
        lite_dir.mkdir()
        (lite_dir / "jupyter_lite_config.json").write_text(
            json.dumps({"PipliteAddon": {"piplite_urls": [str(wheel.resolve())]}}),
            encoding="utf-8",
        )
        (lite_dir / "jupyter-lite.json").write_text(
            json.dumps(
                {
                    "jupyter-lite-schema-version": 0,
                    "jupyter-config-data": {"appName": "PythonSpinDynamics in the browser"},
                }
            ),
            encoding="utf-8",
        )
        if out.exists():
            shutil.rmtree(out)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyterlite_core",
                "build",
                "--lite-dir",
                str(lite_dir),
                "--contents",
                str(contents),
                "--output-dir",
                str(out.resolve()),
                "--apps",
                "lab",
                "--apps",
                "notebooks",
                "--no-sourcemaps",
            ],
            check=True,
            cwd=lite_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("site/lite"), help="Site output directory.")
    parser.add_argument("--wheel", type=Path, default=None, help="Package wheel (built if omitted).")
    parser.add_argument("--notebooks-only", type=Path, default=None, metavar="DIR")
    args = parser.parse_args()
    if args.notebooks_only is not None:
        for path in write_notebooks(args.notebooks_only):
            print(f"wrote {path}")
        return
    build_site(args.out, args.wheel)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
