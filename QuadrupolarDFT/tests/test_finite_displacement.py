import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quadrupolar_dft import (
    EFGTensor,
    PhononMode,
    abinit_input_with_positions,
    efg_temperature_sweep,
    generate_displacement_jobs,
    parse_abinit_structure,
    vibrational_modes_from_efg,
    write_jobs,
)

BASE_INPUT = """# test cell
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
"""


def _n_mode(natom):
    vector = np.zeros((natom, 3))
    vector[2, 1] = 1.0
    vector[3, 1] = -1.0
    return PhononMode(wavenumber_cm_inv=210.0, eigenvector=vector, label="N libration")


class StructureParseTests(unittest.TestCase):
    def test_parses_positions_and_species(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        self.assertEqual(crystal.natom, 8)
        self.assertEqual(crystal.species_z, (11, 11, 7, 7, 8, 8, 8, 8))
        np.testing.assert_allclose(
            np.diag(crystal.lattice_angstrom), [3.557, 5.569, 5.384]
        )
        # N atom 3 at xred (0, 0.3, 0.25).
        np.testing.assert_allclose(
            crystal.cart_angstrom[2], [0.0, 0.3 * 5.569, 0.25 * 5.384], atol=1e-9
        )

    def test_displaced_input_round_trips(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        shifted = crystal.with_positions(crystal.cart_angstrom + 0.01)
        text = abinit_input_with_positions(BASE_INPUT, shifted)
        self.assertIn("xred", text)
        self.assertNotIn("xangst", text)
        reparsed = parse_abinit_structure(text)
        np.testing.assert_allclose(
            reparsed.cart_angstrom, shifted.cart_angstrom, atol=1e-9
        )
        # Untouched settings survive.
        self.assertIn("nucefg 2", text)
        self.assertIn("ecut 25", text)


class DisplacementGeometryTests(unittest.TestCase):
    def test_step_hits_requested_max_displacement(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        mode = _n_mode(crystal.natom)
        jobs = generate_displacement_jobs(
            crystal, [mode], max_displacement_angstrom=0.05
        )
        self.assertEqual(len(jobs), 3)  # equilibrium + plus + minus
        plus = next(j for j in jobs if j.name == "mode000_plus")
        max_disp = np.max(
            np.linalg.norm(plus.crystal.cart_angstrom - crystal.cart_angstrom, axis=1)
        )
        self.assertAlmostEqual(max_disp, 0.05, places=6)

    def test_equilibrium_job_unchanged(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        jobs = generate_displacement_jobs(crystal, [_n_mode(crystal.natom)])
        eq = next(j for j in jobs if j.name == "equilibrium")
        np.testing.assert_allclose(eq.crystal.cart_angstrom, crystal.cart_angstrom)


class CurvatureRecoveryTests(unittest.TestCase):
    def test_central_difference_recovers_known_curvature(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        mode = _n_mode(crystal.natom)
        jobs = generate_displacement_jobs(crystal, [mode])

        # Synthetic EFG: V(Q) = V_eq + B Q + 1/2 C Q^2 along the normal coord.
        v_eq = np.array([[1.0e21, 0.0, 0.0], [0.0, -0.4e21, 0.0], [0.0, 0.0, -0.6e21]])
        gradient = np.array(
            [[2.0e44, 0.0, 0.0], [0.0, -1.0e44, 0.0], [0.0, 0.0, -1.0e44]]
        )
        curvature = np.array(
            [[3.0e68, 1.0e68, 0.0], [1.0e68, -2.0e68, 0.0], [0.0, 0.0, -1.0e68]]
        )

        efg_by_job = {}
        for job in jobs:
            q = job.sign * job.delta_q_si
            matrix = v_eq + gradient * q + 0.5 * curvature * q**2
            efg_by_job[job.name] = EFGTensor.from_components(matrix, unit="si")

        modes = vibrational_modes_from_efg([mode], jobs, efg_by_job)
        self.assertEqual(len(modes), 1)
        np.testing.assert_allclose(
            modes[0].efg_curvature_si, curvature, rtol=1e-6
        )

    def test_recovered_mode_drives_temperature_sweep(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        mode = _n_mode(crystal.natom)
        jobs = generate_displacement_jobs(crystal, [mode])
        # V_zz is the largest-magnitude component (the zz slot here).
        v_eq = np.diag([-0.35e22, -0.65e22, 1.0e22])
        # A curvature that reduces |V_zz| (negative on the zz slot).
        curvature = np.diag([0.6 * 4.0e69, 0.4 * 4.0e69, -4.0e69])
        efg_by_job = {}
        for job in jobs:
            q = job.sign * job.delta_q_si
            efg_by_job[job.name] = EFGTensor.from_components(
                v_eq + 0.5 * curvature * q**2, unit="si"
            )
        modes = vibrational_modes_from_efg([mode], jobs, efg_by_job)
        equilibrium = EFGTensor.from_components(v_eq, unit="si")
        points = efg_temperature_sweep(
            equilibrium, modes, [0.0, 150.0, 300.0],
            spin=1.0, quadrupole_moment_barns=0.02044,
        )
        nu_plus = [np.sort(p.frequencies_hz)[-1] for p in points]
        self.assertTrue(np.all(np.diff(nu_plus) < 0.0))


class WriteJobsTests(unittest.TestCase):
    def test_writes_inputs_and_manifest(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        jobs = generate_displacement_jobs(crystal, [_n_mode(crystal.natom)])
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_jobs(
                BASE_INPUT, jobs, tmp, target_atom_index=2
            )
            files = sorted(p.name for p in Path(directory).glob("*.abi"))
            self.assertEqual(
                files, ["equilibrium.abi", "mode000_minus.abi", "mode000_plus.abi"]
            )
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["target_atom_index"], 2)
            self.assertEqual(len(manifest["jobs"]), 3)


if __name__ == "__main__":
    unittest.main()
