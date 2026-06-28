import unittest

import numpy as np

from quadrupolar_dft import (
    anaddb_input,
    modes_from_arrays,
    parse_anaddb_modes,
    phonon_dfpt_input,
)

STATIC_INPUT = """nucefg 2
quadmom 0.104 0.02044 -0.02558
acell 3.557 5.569 5.384 Angstrom
ntypat 3
znucl 11 7 8
natom 4
typat 1 2 3 3
xred
  0.0 0.0 0.0
  0.0 0.3 0.25
  0.0 0.42 0.05
  0.0 0.42 0.45
ecut 25
ngkpt 4 4 4
"""

# Real ABINIT 9.10.4 anaddb eivec layout (captured from a NaNO2 run, reduced to
# 2 atoms / 6 modes). Frequencies are prefixed with '-'; eigendisplacements are
# ';'-lines, one real + one imaginary per atom, with a leading atom index. Modes
# 2 and 3 are near-zero acoustic modes that anaddb prints with NO displacement
# block -- this exercises the mode-number alignment (frequencies must not shift).
ANADDB_OUTPUT = """ Phonon frequencies in cm-1    :
- -5.000000E+00 -1.000000E-03  1.000000E-04  1.200000E+02  2.100000E+02  3.500000E+02

 Eigendisplacements
 (will be given, for each mode : in cartesian coordinates...)
  Mode number    1   Energy   -1.000000E-06
;  1  1.00000000E-01  0.00000000E+00  0.00000000E+00
;     0.00000000E+00  0.00000000E+00  0.00000000E+00
;  2 -1.00000000E-01  0.00000000E+00  0.00000000E+00
;     0.00000000E+00  0.00000000E+00  0.00000000E+00
  Mode number    2   Energy    1.000000E-09
 Attention : low frequency mode.
   (Could be unstable or acoustic mode)
  Mode number    3   Energy    2.000000E-09
 Attention : low frequency mode.
   (Could be unstable or acoustic mode)
  Mode number    4   Energy    5.000000E-04
;  1  0.00000000E+00  1.00000000E+00  0.00000000E+00
;     0.00000000E+00  0.00000000E+00  0.00000000E+00
;  2  0.00000000E+00 -1.00000000E+00  0.00000000E+00
;     0.00000000E+00  0.00000000E+00  0.00000000E+00
  Mode number    5   Energy    6.000000E-04
;  1  0.00000000E+00  0.00000000E+00  1.00000000E+00
;     0.00000000E+00  0.00000000E+00  0.00000000E+00
;  2  0.00000000E+00  0.00000000E+00 -1.00000000E+00
;     0.00000000E+00  0.00000000E+00  0.00000000E+00
  Mode number    6   Energy    7.000000E-04
;  1  1.00000000E+00  0.00000000E+00  0.00000000E+00
;     0.00000000E+00  0.00000000E+00  0.00000000E+00
;  2 -1.00000000E+00  0.00000000E+00  0.00000000E+00
;     0.00000000E+00  0.00000000E+00  0.00000000E+00
"""


class PhononInputTests(unittest.TestCase):
    def test_dfpt_input_strips_efg_and_keeps_structure(self):
        text = phonon_dfpt_input(STATIC_INPUT)
        self.assertIn("rfphon2 1", text)
        self.assertIn("rfatpol2 1 4", text)
        self.assertNotIn("nucefg", text)
        self.assertNotIn("quadmom", text)
        self.assertIn("ecut 25", text)
        self.assertIn("acell", text)

    def test_anaddb_input_requests_eigenvectors(self):
        self.assertIn("eivec", anaddb_input())


class ModesFromArraysTests(unittest.TestCase):
    def test_skips_nonpositive_and_normalizes(self):
        masses = np.array([23.0, 14.0, 16.0, 16.0])
        wavenumbers = [-1.0, 120.0, 210.0]
        displ = np.zeros((3, 4, 3))
        displ[1, 1, 1] = 1.0
        displ[2, 2, 2] = 1.0
        modes = modes_from_arrays(wavenumbers, displ, masses)
        self.assertEqual(len(modes), 2)  # the negative mode is dropped
        for mode in modes:
            self.assertAlmostEqual(float(np.sum(mode.eigenvector**2)), 1.0, places=6)

    def test_mass_weighting_applied(self):
        masses = np.array([1.0, 4.0])
        displ = np.zeros((1, 2, 3))
        displ[0, 0, 0] = 1.0
        displ[0, 1, 0] = 1.0  # equal displacement; heavier atom gets larger eps
        modes = modes_from_arrays([100.0], displ, masses)
        eps = modes[0].eigenvector
        self.assertGreater(abs(eps[1, 0]), abs(eps[0, 0]))  # sqrt(4) > sqrt(1)


class ParseAnaddbTests(unittest.TestCase):
    def test_parses_real_modes(self):
        masses = np.array([23.0, 14.0])
        modes = parse_anaddb_modes(ANADDB_OUTPUT, masses, natom=2)
        # One imaginary (-5) and two acoustic (~0) modes dropped; 3 real kept.
        self.assertEqual(len(modes), 3)
        self.assertAlmostEqual(modes[0].wavenumber_cm_inv, 120.0, places=2)
        self.assertAlmostEqual(modes[1].wavenumber_cm_inv, 210.0, places=2)
        self.assertAlmostEqual(modes[2].wavenumber_cm_inv, 350.0, places=2)

    def test_mode_number_alignment_across_missing_blocks(self):
        # Modes 2 and 3 have no eigendisplacement block. The 120 cm^-1 mode is
        # mode #4 (y-motion); a positional zip would mis-assign it. Check each
        # kept frequency carries its own distinct eigenvector axis.
        masses = np.array([23.0, 14.0])
        modes = parse_anaddb_modes(ANADDB_OUTPUT, masses, natom=2)
        dominant_axis = [int(np.argmax(np.abs(m.eigenvector[0]))) for m in modes]
        self.assertEqual(dominant_axis, [1, 2, 0])  # 120->y, 210->z, 350->x

    def test_raises_without_blocks(self):
        with self.assertRaises(ValueError):
            parse_anaddb_modes("no modes here", np.array([23.0, 14.0]), natom=2)


if __name__ == "__main__":
    unittest.main()
