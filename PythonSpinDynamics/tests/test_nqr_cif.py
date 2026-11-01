"""CIF reader shared with QuadrupolarDFT: lexing, symmetry parsing, and sync."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import _cif  # noqa: E402
from spin_dynamics.nqr import load_cif_structure  # noqa: E402

TWIN = ROOT.parent / "QuadrupolarDFT" / "src" / "quadrupolar_dft" / "_cif.py"

CIF_WITH_LEGAL_QUOTING = """data_quoting
_publ_author_name 'O'Brien, J.'
_chemical_name_common "it's a 'test'"   # trailing comment
_journal_name_full 'J#Chem'
_publ_section_title
;
 A title with 'quotes' and an O'Brien
;
_cell_length_a 5.0(1)
_cell_length_b 6.0
_cell_length_c 7.0
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x+1/2, y+1/2, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
N1 N 0.1 0.2 0.3
H1 H 0.15 0.25 0.35
"""


class CIFLexerTests(unittest.TestCase):
    def test_embedded_apostrophes_and_hashes_are_legal(self) -> None:
        # Regression: the posix-shlex tokenizer raised "No closing quotation" on
        # 'O'Brien, J.', which CIF 1.1 allows (a quote ends a string only when
        # followed by whitespace).
        items, loops = _cif.parse_cif(CIF_WITH_LEGAL_QUOTING)
        self.assertEqual(items["_publ_author_name"], "O'Brien, J.")
        self.assertEqual(items["_chemical_name_common"], "it's a 'test'")
        self.assertEqual(items["_journal_name_full"], "J#Chem")
        self.assertEqual(
            items["_publ_section_title"], " A title with 'quotes' and an O'Brien"
        )
        self.assertEqual(_cif.symmetry_operations_from_loops(loops), ["x, y, z", "-x+1/2, y+1/2, -z"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quoting.cif"
            path.write_text(CIF_WITH_LEGAL_QUOTING, encoding="utf-8")
            structure = load_cif_structure(path)
        self.assertEqual([atom.label for atom in structure.atoms], ["N1", "H1"])
        self.assertEqual(structure.cell_lengths, (5.0, 6.0, 7.0))

    def test_quoted_underscore_value_is_not_a_tag(self) -> None:
        items, _ = _cif.parse_cif("_name '_not_a_tag'\n_next 1\n")
        self.assertEqual(items, {"_name": "_not_a_tag", "_next": "1"})

    def test_unterminated_quote_is_reported(self) -> None:
        with self.assertRaises(ValueError):
            _cif.tokenize_cif("_name 'open\n")


class SymmetryExpressionTests(unittest.TestCase):
    def test_standard_operations(self) -> None:
        point = np.array([0.1, 0.2, 0.3])
        cases = {
            "x,y,z": [0.1, 0.2, 0.3],
            "-x+1/2, y+1/2, -z": [0.4, 0.7, -0.3],
            "1/2-x, -y, z+1/4": [0.4, -0.2, 0.55],
            "x-y, x, z+2/3": [-0.1, 0.1, 0.3 + 2 / 3],
            "(x+y)/2, +z, 0.25+x": [0.15, 0.3, 0.35],
            "'-X,-Y,-Z'": [-0.1, -0.2, -0.3],
        }
        for operation, expected in cases.items():
            with self.subTest(operation=operation):
                np.testing.assert_allclose(
                    _cif.apply_symmetry_operation(operation, point), expected, atol=1e-15
                )

    def test_eval_only_syntax_is_rejected(self) -> None:
        # Regression: the character-filtered eval accepted "**" (so "9**9**9"
        # could hang the reader) and evaluated "2**3" to 8.
        for operation in ("x,y,2**3", "x,y,9**9**9", "x,y,z.real", "x,y,abs(z)", "x,y,"):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                _cif.apply_symmetry_operation(operation, np.zeros(3))


class CellMatrixTests(unittest.TestCase):
    def test_hexagonal_cell(self) -> None:
        matrix = _cif.cell_matrix((5.0, 5.0, 8.0), (90.0, 90.0, 120.0))
        expected = [[5.0, 0.0, 0.0], [-2.5, 2.5 * np.sqrt(3.0), 0.0], [0.0, 0.0, 8.0]]
        np.testing.assert_allclose(matrix, expected, atol=1e-12)


class TwinSyncTests(unittest.TestCase):
    @unittest.skipUnless(TWIN.exists(), "QuadrupolarDFT checkout not present")
    def test_quadrupolar_dft_copy_is_identical(self) -> None:
        # The CIF reader is duplicated on purpose (neither package depends on
        # the other); keep the two copies byte-identical up to line endings.
        here = (ROOT / "src" / "spin_dynamics" / "nqr" / "_cif.py").read_bytes()
        twin = TWIN.read_bytes()
        self.assertEqual(here.replace(b"\r\n", b"\n"), twin.replace(b"\r\n", b"\n"))


if __name__ == "__main__":
    unittest.main()
