import unittest

import numpy as np

from quadrupolar_dft import format_abinit_efg_block, parse_abinit_efg


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

    def test_format_abinit_efg_block(self):
        block = format_abinit_efg_block([0.0, -0.02558])

        self.assertIn("nucefg 2", block)
        self.assertIn("quadmom 0 -0.02558", block)
        self.assertIn("PAW", block)


if __name__ == "__main__":
    unittest.main()
