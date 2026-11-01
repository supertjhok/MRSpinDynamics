import unittest

import numpy as np

from quadrupolar_dft import (
    coupling_constant_hz,
    efg_tensor_from_record,
    format_abinit_efg_block,
    parse_abinit_efg,
)
from quadrupolar_dft.constants import EFG_AU_TO_SI


ABINIT_FRAGMENT = """
Electric Field Gradient Calculation

   atom :    3   typat :    2

   Nuclear quad. mom. (barns) :   -0.0256   Cq (MHz) :    6.6150   eta :    0.1403

      efg eigval (au) :     -1.100599 ; (1.0E+21 V/m^2) :     -10.69492247
-         eigvec :      0.707107    -0.707107     0.000000

      efg eigval (au) :      0.473085 ; (1.0E+21 V/m^2) :       4.59714077
-         eigvec :     -0.000000    -0.000000     1.000000
      efg eigval (au) :      0.627514 ; (1.0E+21 V/m^2) :       6.09778170
-         eigvec :     -0.707107    -0.707107    -0.000000

      total efg :     -0.236543     0.864057     0.000000
      total efg :      0.864057    -0.236543     0.000000
      total efg :      0.000000     0.000000     0.473085
"""

ABINIT_COMPACT_FRAGMENT = """
 Electric Field Gradient Calculation

 Atom   3, typat   2: Cq =      6.615041 MHz     eta =      0.140313

      efg eigval :     -1.100599
-         eigvec :      0.707107    -0.707107     0.000000
      efg eigval :      0.473085
-         eigvec :      0.000000     0.000000     1.000000
      efg eigval :      0.627514
-         eigvec :     -0.707107    -0.707107     0.000000

      total efg :     -0.236543     0.864057    -0.000000
      total efg :      0.864057    -0.236543    -0.000000
      total efg :     -0.000000    -0.000000     0.473085
"""


class AbinitParserTests(unittest.TestCase):
    def test_parse_tutorial_efg_fragment(self):
        records = parse_abinit_efg(ABINIT_FRAGMENT)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.atom_index, 3)
        self.assertEqual(record.typat, 2)
        self.assertAlmostEqual(record.quadrupole_moment_barns, -0.0256)
        self.assertAlmostEqual(record.cq_mhz, 6.6150)
        self.assertAlmostEqual(record.eta, 0.1403)
        self.assertTrue(np.allclose(record.eigvals_au, [-1.100599, 0.473085, 0.627514]))
        self.assertIsNotNone(record.tensor)

    def test_parse_compact_abinit_9_efg_fragment(self):
        records = parse_abinit_efg(ABINIT_COMPACT_FRAGMENT)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.atom_index, 3)
        self.assertEqual(record.typat, 2)
        self.assertAlmostEqual(record.cq_mhz, 6.615041)
        self.assertAlmostEqual(record.eta, 0.140313)
        self.assertTrue(np.isnan(record.quadrupole_moment_barns))
        self.assertTrue(np.allclose(record.eigvals_au, [-1.100599, 0.473085, 0.627514]))
        self.assertEqual(record.eigvals_1e21_v_per_m2.shape, (3,))
        self.assertTrue(np.all(np.isnan(record.eigvals_1e21_v_per_m2)))
        self.assertIsNotNone(record.tensor)

    def test_parses_fortran_d_exponents(self):
        # Regression: the number regex accepted "D" exponents but float() did not.
        fragment = (
            ABINIT_FRAGMENT.replace("-0.0256 ", "-2.56D-02 ")
            .replace("6.6150", "6.6150D+00")
            .replace("-1.100599", "-1.100599d0")
            .replace("     -0.236543     0.864057", "     -2.36543D-01     8.64057D-01")
        )
        self.assertIn("D-02", fragment)
        record = parse_abinit_efg(fragment)[0]
        reference = parse_abinit_efg(ABINIT_FRAGMENT)[0]
        self.assertAlmostEqual(record.quadrupole_moment_barns, -0.0256)
        self.assertAlmostEqual(record.cq_mhz, 6.6150)
        np.testing.assert_allclose(record.eigvals_au, reference.eigvals_au)
        np.testing.assert_allclose(record.total_tensor_au, reference.total_tensor_au)

    def test_eigenpair_fallback_reproduces_abinit_cq_and_eta(self):
        # Regression: without "total efg" rows the tensor rebuilt from ABINIT's
        # 6-decimal eigenpairs failed the 1e-8 trace check, so this always raised.
        compact = ABINIT_COMPACT_FRAGMENT.split("      total efg")[0]
        record = parse_abinit_efg(compact)[0]
        self.assertIsNone(record.tensor)

        tensor = efg_tensor_from_record(record)

        # Independent reference: ABINIT's own header values for this atom.
        self.assertAlmostEqual(tensor.eta, 0.140313, delta=2e-5)
        self.assertAlmostEqual(tensor.vzz_si / EFG_AU_TO_SI, -1.100599, delta=2e-6)
        # 17O quadrupole moment -25.58 mb (the header prints it rounded to -0.0256).
        cq_mhz = coupling_constant_hz(tensor.vzz_si, -0.02558) / 1e6
        self.assertAlmostEqual(cq_mhz, 6.615041, delta=1e-4)

    def test_eigenpair_fallback_rejects_non_traceless_eigenvalues(self):
        compact = ABINIT_COMPACT_FRAGMENT.split("      total efg")[0].replace(
            "0.627514", "0.727514"
        )
        record = parse_abinit_efg(compact)[0]
        with self.assertRaises(ValueError):
            efg_tensor_from_record(record)

    def test_format_abinit_efg_block(self):
        block = format_abinit_efg_block([0.0, -0.02558])

        self.assertIn("nucefg 2", block)
        self.assertIn("quadmom 0 -0.02558", block)
        self.assertIn("PAW", block)


if __name__ == "__main__":
    unittest.main()
