import unittest

import numpy as np

from quadrupolar_dft import (
    parse_abinit_structure,
    parse_relaxed_structure,
    relax_input,
    relaxed_input,
)

# A converged static EFG input (orthorhombic NaNO2 starter, abridged).
BASE_INPUT = """# static EFG input
nucefg 2
quadmom 0.104 0.02044 -0.02558
acell 3.557 5.569 5.384 Angstrom
chkprim 0
nsym 1
ntypat 3
znucl 11 7 8
natom 8
typat
  1 1
  2 2
  3 3 3 3
xred
  0.000000  0.000000  0.000000  # Na
  0.500000  0.500000  0.500000  # Na
  0.000000  0.300000  0.250000  # N
  0.500000  0.800000  0.750000  # N
  0.000000  0.421000  0.054000  # O
  0.000000  0.421000  0.446000  # O
  0.500000  0.921000  0.554000  # O
  0.500000  0.921000  0.946000  # O
ecut 25
tolvrs 1.0d-14
nstep 80
"""

# Relaxed reduced coordinates: the two N atoms (and their O neighbours) move off
# the starter positions; everything else is unchanged.
RELAXED_XRED = np.array([
    [0.000000, 0.000000, 0.000000],
    [0.500000, 0.500000, 0.500000],
    [0.000000, 0.312000, 0.250000],  # N moved in y
    [0.500000, 0.788000, 0.750000],  # N moved in y
    [0.000000, 0.418000, 0.060000],
    [0.000000, 0.418000, 0.440000],
    [0.500000, 0.918000, 0.560000],
    [0.500000, 0.918000, 0.940000],
])

_ANG_TO_BOHR = 1.8897259886


def _xred_block(name, xred):
    rows = [f"            {name}" + "".join(f"  {v: .10E}" for v in xred[0])]
    rows += ["                 " + "".join(f"  {v: .10E}" for v in row) for row in xred[1:]]
    return "\n".join(rows)


def _make_abo(relaxed_xred):
    """A minimal ABINIT relaxation output: header echo + post-computation footer.

    The header echoes the *initial* geometry; the footer (the block parsers
    read) echoes the *relaxed* geometry, with acell in Bohr as ABINIT prints it.
    """

    acell_bohr = np.array([3.557, 5.569, 5.384]) * _ANG_TO_BOHR
    initial_xred = parse_abinit_structure(BASE_INPUT).cart_angstrom @ np.linalg.inv(
        parse_abinit_structure(BASE_INPUT).lattice_angstrom
    )
    common = f"""            acell      {acell_bohr[0]:.10E}  {acell_bohr[1]:.10E}  {acell_bohr[2]:.10E} Bohr
            natom         8
            ntypat        3
            rprim      1.0000000000E+00  0.0000000000E+00  0.0000000000E+00
                       0.0000000000E+00  1.0000000000E+00  0.0000000000E+00
                       0.0000000000E+00  0.0000000000E+00  1.0000000000E+00
            typat      1  1  2  2  3  3  3  3
            znucl      1.10000000E+01  7.00000000E+00  8.00000000E+00"""
    return f"""
 -outvars: echo values of preprocessed input variables --------
{common}
{_xred_block('xred', initial_xred)}
================================================================================

   ... iterations / Broyd/MD steps elided ...

== END DATASET(S) ==============================================================
================================================================================

 -outvars: echo values of variables after computation  --------
{common}
{_xred_block('xred', relaxed_xred)}
================================================================================
"""


class RelaxInputTests(unittest.TestCase):
    def test_strips_efg_and_adds_optimization(self):
        text = relax_input(BASE_INPUT)
        self.assertIn("ionmov 2", text)
        self.assertIn("optcell 0", text)
        self.assertIn("tolmxf", text)
        # EFG-only keywords gone; structure/electronic settings preserved.
        self.assertNotIn("nucefg", text)
        self.assertNotIn("quadmom", text)
        self.assertIn("ecut 25", text)
        self.assertIn("tolvrs", text)
        # Still a parseable structure.
        crystal = parse_abinit_structure(text)
        self.assertEqual(crystal.natom, 8)


class ParseRelaxedStructureTests(unittest.TestCase):
    def test_reads_relaxed_positions_from_footer(self):
        crystal = parse_relaxed_structure(_make_abo(RELAXED_XRED), BASE_INPUT)
        # Cell and species come from the base input.
        np.testing.assert_allclose(
            np.diag(crystal.lattice_angstrom), [3.557, 5.569, 5.384]
        )
        self.assertEqual(crystal.species_z, (11, 11, 7, 7, 8, 8, 8, 8))
        # Positions are the relaxed ones (footer), not the initial ones (header).
        recovered_xred = crystal.cart_angstrom @ np.linalg.inv(crystal.lattice_angstrom)
        np.testing.assert_allclose(recovered_xred, RELAXED_XRED, atol=1e-8)

    def test_missing_footer_raises(self):
        truncated = "ABINIT crashed before finishing.\n no footer here\n"
        with self.assertRaises(ValueError):
            parse_relaxed_structure(truncated, BASE_INPUT)

    def test_atom_count_mismatch_raises(self):
        smaller_base = BASE_INPUT.replace("natom 8", "natom 7")
        with self.assertRaises(ValueError):
            parse_relaxed_structure(_make_abo(RELAXED_XRED), smaller_base)


class RelaxedInputTests(unittest.TestCase):
    def test_emits_static_input_at_relaxed_geometry(self):
        text = relaxed_input(BASE_INPUT, _make_abo(RELAXED_XRED))
        # EFG keywords survive -- it is a static EFG input again.
        self.assertIn("nucefg 2", text)
        self.assertIn("quadmom", text)
        # Positions are the relaxed ones.
        reparsed = parse_abinit_structure(text)
        recovered_xred = reparsed.cart_angstrom @ np.linalg.inv(
            reparsed.lattice_angstrom
        )
        np.testing.assert_allclose(recovered_xred, RELAXED_XRED, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
