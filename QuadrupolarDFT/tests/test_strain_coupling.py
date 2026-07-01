import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quadrupolar_dft import (
    EFGTensor,
    abinit_input_with_lattice_and_positions,
    collect_strain_derivatives,
    coupling_constant_hz,
    generate_strain_jobs,
    glycine_static_efg_input_from_cif,
    parse_abinit_structure,
    quadrupole_hamiltonian_hz_from_efg,
    space_group_is_likely_centrosymmetric,
    strain_manifest_dict,
    strain_transition_couplings,
    strained_crystal,
    write_strain_jobs,
)
from quadrupolar_dft.constants import EFG_AU_TO_SI


BASE_INPUT = """# minimal static EFG input
nucefg 2
quadmom 0.02044
acell 4.0 5.0 6.0 Angstrom
chkprim 0
ntypat 2
znucl 7 1
natom 2
typat 1 2
xred
  0.25 0.25 0.25
  0.50 0.50 0.50
ecut 20
"""


def _abo_from_tensor_au(matrix, *, atom_index=1):
    rows = "\n".join(
        f" total efg : {row[0]: .12e} {row[1]: .12e} {row[2]: .12e}"
        for row in matrix
    )
    return f"""
 Electric Field Gradient Calculation
 atom : {atom_index} typat : 1
 Nuclear quad. mom. (barns) : 0.02044 Cq (MHz) : 1.0 eta : 0.1
{rows}
"""


class StrainGeometryTests(unittest.TestCase):
    def test_glycine_cif_static_input_expands_p21_cell(self):
        root = Path(__file__).resolve().parents[2]
        cif = root / "QuadrupolarDFT" / "structures" / "Glycine" / "189379.cif"

        text, atoms = glycine_static_efg_input_from_cif(cif)
        crystal = parse_abinit_structure(text)
        nitrogen_indices = [
            index for index, atom in enumerate(atoms) if atom["element"] == "N"
        ]

        self.assertEqual(crystal.natom, 20)
        self.assertEqual(nitrogen_indices, [1, 11])
        self.assertIn('pseudos "H.xml, C.xml, N.xml, O.xml"', text)
        self.assertIn("pp_dirpath \"$ABI_PSPDIR/Pseudodojo_paw_pw_standard\"", text)
        self.assertIn("# job label: glycine_beta_p21", text)
        self.assertNotIn("\ntbase ", text)
        self.assertIn("quadmom 0 0 0.02044 -0.02558", text)
        self.assertNotIn("pawovlp", text)

        text_with_overlap, _ = glycine_static_efg_input_from_cif(cif, pawovlp=25)
        self.assertIn("pawovlp 25", text_with_overlap)

    def test_affine_strain_preserves_fractional_coordinates(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        strain = np.array([[0.01, 0.002, 0.0], [0.002, 0.0, 0.0], [0.0, 0.0, -0.01]])

        deformed = strained_crystal(crystal, strain)
        original_xred = crystal.cart_angstrom @ np.linalg.inv(crystal.lattice_angstrom)
        deformed_xred = (
            deformed.cart_angstrom @ np.linalg.inv(deformed.lattice_angstrom)
        )

        np.testing.assert_allclose(deformed_xred, original_xred, atol=1e-12)

    def test_strained_input_round_trips_lattice_and_positions(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        job = generate_strain_jobs(
            crystal,
            strain_amplitude=0.01,
            components=("xx",),
        )[1]

        text = abinit_input_with_lattice_and_positions(BASE_INPUT, job.crystal)
        reparsed = parse_abinit_structure(text)

        np.testing.assert_allclose(
            reparsed.lattice_angstrom,
            job.crystal.lattice_angstrom,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            reparsed.cart_angstrom,
            job.crystal.cart_angstrom,
            atol=1e-9,
        )

    def test_write_strain_jobs_creates_manifest(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        jobs = generate_strain_jobs(crystal, strain_amplitude=0.002, components=("xy",))
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_strain_jobs(BASE_INPUT, jobs, tmp, target_atom_index=0)
            files = sorted(path.name for path in Path(directory).glob("*.abi"))
            self.assertEqual(
                files,
                ["equilibrium.abi", "strain_xy_minus.abi", "strain_xy_plus.abi"],
            )
            manifest = json.loads(
                (Path(directory) / "strain_manifest.json").read_text()
            )
            self.assertEqual(manifest["target_atom_index"], 0)
            self.assertEqual(manifest["jobs"][1]["component"], "xy")


class StrainCouplingMathTests(unittest.TestCase):
    def test_general_tensor_hamiltonian_matches_spin_one_pas_formula(self):
        vzz = 1.0e21
        eta = 0.4
        vxx = 0.5 * (eta - 1.0) * vzz
        vyy = -0.5 * (eta + 1.0) * vzz
        tensor = np.diag([vxx, vyy, vzz])
        h = quadrupole_hamiltonian_hz_from_efg(
            tensor,
            spin=1.0,
            quadrupole_moment_barns=0.02044,
        )
        energies = np.sort(np.linalg.eigvalsh(h))
        frequencies = sorted(
            abs(float(energies[j] - energies[i]))
            for i in range(3)
            for j in range(i + 1, 3)
            if abs(float(energies[j] - energies[i])) > 1e-9
        )

        cq = coupling_constant_hz(vzz, 0.02044)
        expected = sorted(
            [
                0.75 * cq * (1.0 - eta / 3.0),
                0.75 * cq * (1.0 + eta / 3.0),
                0.5 * cq * eta,
            ]
        )
        np.testing.assert_allclose(frequencies, expected, rtol=1e-12)

    def test_off_diagonal_strain_derivative_drives_transition(self):
        equilibrium = EFGTensor.from_components(
            np.diag([-0.3e21, -0.7e21, 1.0e21]),
            unit="si",
        )
        derivative = np.array([[0.0, 2.0e21, 0.0], [2.0e21, 0.0, 0.0], [0.0, 0.0, 0.0]])
        couplings = strain_transition_couplings(
            equilibrium,
            [
                type(
                    "Derivative",
                    (),
                    {
                        "component": "xy",
                        "derivative_si_per_strain": derivative,
                    },
                )()
            ],
        )

        self.assertEqual(len(couplings), 3)
        self.assertGreater(max(item.rwa_rabi_hz_per_strain for item in couplings), 0.0)

    def test_collect_derivative_from_synthetic_abinit_outputs(self):
        crystal = parse_abinit_structure(BASE_INPUT)
        jobs = generate_strain_jobs(crystal, strain_amplitude=0.001, components=("xx",))
        manifest = strain_manifest_dict(jobs, target_atom_index=0)
        v0 = np.diag([1.0, -0.4, -0.6])
        dvdexx = np.diag([0.2, -0.1, -0.1])

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "strain_manifest.json").write_text(json.dumps(manifest))
            for job in manifest["jobs"]:
                matrix = v0.copy()
                if job["component"] == "xx":
                    matrix = matrix + job["sign"] * job["strain_amplitude"] * dvdexx
                (directory / f"{job['name']}.abo").write_text(
                    _abo_from_tensor_au(matrix),
                    encoding="utf-8",
                )
            equilibrium, derivatives = collect_strain_derivatives(manifest, directory)

        np.testing.assert_allclose(equilibrium.matrix_si, v0 * EFG_AU_TO_SI)
        self.assertEqual(len(derivatives), 1)
        np.testing.assert_allclose(
            derivatives[0].derivative_si_per_strain,
            dvdexx * EFG_AU_TO_SI,
            rtol=1e-12,
        )

    def test_glycine_space_group_guard_flags_alpha_like_groups(self):
        self.assertFalse(space_group_is_likely_centrosymmetric("P 21"))
        self.assertTrue(space_group_is_likely_centrosymmetric("P 21/n"))
        self.assertTrue(space_group_is_likely_centrosymmetric("P -1"))


if __name__ == "__main__":
    unittest.main()
