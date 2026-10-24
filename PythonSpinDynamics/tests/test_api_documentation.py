"""Public member discovery and lossless signature rendering in the API guide."""

import ast
import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "docs/generate_api_reference.py"
SPEC = importlib.util.spec_from_file_location("api_documentation_generator", PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class APIDocumentationTests(unittest.TestCase):
    def test_members_and_fields_are_found_without_importing_source(self):
        source = '''raise RuntimeError("source must not execute")
class Model:
    time: float | None = None
    def __init__(self, time, /, *, mode="cw"): pass
    @classmethod
    def from_rates(cls, rates): """Build | convert."""
    @staticmethod
    def convert(self): pass
    @property
    def result(self): """The result."""
    @result.setter
    def result(self, value): pass
    def _private(self): pass
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "module.py"
            path.write_text(source)
            symbols = GENERATOR._module_symbols(path)
        names = [symbol.name for symbol in symbols]
        self.assertEqual(
            names,
            [
                "Model",
                "Model.time",
                "Model.from_rates",
                "Model.convert",
                "Model.result",
            ],
        )
        self.assertEqual(symbols[0].signature, "(time, /, *, mode = 'cw')")
        self.assertEqual(symbols[2].signature, "(rates)")
        self.assertEqual(symbols[3].signature, "(self)")
        rendered = GENERATOR._render_module("test", symbols)
        self.assertIn(r"float \| None", rendered)
        self.assertIn(r"Build \| convert.", rendered)

    def test_positional_and_variadic_annotations_are_preserved(self):
        function = ast.parse(
            "def f(a, /, b=1, *values: float, flag: bool=True, **options: int) -> float: pass"
        ).body[0]
        self.assertEqual(
            GENERATOR._signature(function),
            "(a, /, b = 1, *values: float, flag: bool = True, **options: int) -> float",
        )

    def test_all_public_modules_are_in_inventory(self):
        self.assertEqual(GENERATOR._unlisted_public_modules(), [])


if __name__ == "__main__":
    unittest.main()
