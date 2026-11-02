"""The in-browser (JupyterLite) notebooks convert cleanly and run under CPython.

``docs/lite/pyodide_smoke.mjs`` runs the same sources inside Pyodide in CI.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
LITE = ROOT / "docs" / "lite"
SOURCES = sorted((LITE / "notebooks").glob("*.py"))


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_lite", LITE / "build_lite.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiteNotebookTests(unittest.TestCase):
    def test_sources_convert_to_notebooks_with_an_install_cell(self) -> None:
        builder = _load_builder()
        self.assertGreaterEqual(len(SOURCES), 4)
        for source in SOURCES:
            with self.subTest(source.name):
                text = source.read_text(encoding="utf-8")
                notebook = builder.notebook_from_source(text)
                cells = notebook["cells"]
                self.assertEqual(notebook["nbformat"], 4)
                self.assertEqual(cells[0]["cell_type"], "markdown")
                self.assertTrue(cells[0]["source"][0].startswith("# "))
                self.assertEqual("".join(cells[1]["source"]), builder.INSTALL_CELL)
                self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
                # Every code line of the source reaches the notebook.
                code = "\n".join("".join(cell["source"]) for cell in cells[2:] if cell["cell_type"] == "code")
                for line in text.splitlines():
                    if line.strip() and not line.startswith("#"):
                        self.assertIn(line, code)

    def test_markdown_lines_must_be_comments(self) -> None:
        builder = _load_builder()
        with self.assertRaises(ValueError):
            builder.notebook_from_source("# %% [markdown]\n# Title\nx = 1\n")
        with self.assertRaises(ValueError):
            builder.notebook_from_source("x = 1\n# %%\n")

    def test_notebook_sources_run(self) -> None:
        try:
            import matplotlib
        except ImportError:  # pragma: no cover - plotting extra not installed
            self.skipTest("matplotlib is not installed")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        previous = os.environ.get("MPLBACKEND")
        os.environ["MPLBACKEND"] = "Agg"
        try:
            for source in SOURCES:
                with self.subTest(source.name), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
                    runpy.run_path(str(source), run_name="__main__")
                    plt.close("all")
        finally:
            if previous is None:
                os.environ.pop("MPLBACKEND", None)
            else:
                os.environ["MPLBACKEND"] = previous


if __name__ == "__main__":
    unittest.main()
