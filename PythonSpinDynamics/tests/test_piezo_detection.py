from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    PiezoNQRCoupling,
    PiezoelectricCrystal,
    default_glycine_nqr_lines,
    glycine_crystal_from_cif,
    glycine_site_from_qcc,
    load_glycine_nqr_lines_from_sqlite,
    nuclear_absorbed_power,
    resonant_strain_peak,
    simulate_piezoelectric_nqr_detection,
)
from spin_dynamics.nqr.hamiltonians import diagonalize_site  # noqa: E402


class PiezoDetectionTests(unittest.TestCase):
    def test_glycine_site_matches_database_transition_frequencies(self) -> None:
        site = glycine_site_from_qcc(1.193e6, 0.528)
        transitions = {
            item.label: item.frequency_hz
            for item in diagonalize_site(site).transitions
        }

        self.assertAlmostEqual(transitions["x"] / 1e3, 1052.3, delta=0.5)
        self.assertAlmostEqual(transitions["y"] / 1e3, 737.2, delta=0.5)
        self.assertAlmostEqual(transitions["z"] / 1e3, 315.0, delta=0.5)

    def test_default_glycine_lines_include_measured_database_lines(self) -> None:
        lines = default_glycine_nqr_lines()
        by_label = {line.transition_label: line for line in lines}

        self.assertEqual(set(by_label), {"x", "y"})
        self.assertAlmostEqual(by_label["x"].frequency_hz, 1052.0e3)
        self.assertAlmostEqual(by_label["y"].frequency_hz, 737.0e3)
        self.assertAlmostEqual(by_label["x"].t1_seconds, 43.4)
        self.assertAlmostEqual(by_label["y"].t2_seconds, 17.1)

    def test_load_glycine_lines_from_sqlite_deduplicates_sources(self) -> None:
        db_path = REPO_ROOT / "NQRDatabase" / "data" / "exports" / "nqr.sqlite"
        lines = load_glycine_nqr_lines_from_sqlite(db_path)

        self.assertEqual(len(lines), 2)
        self.assertEqual({line.frequency_hz for line in lines}, {737.0e3, 1052.0e3})
        self.assertTrue(all(line.qcc_hz == 1.193e6 for line in lines))
        self.assertTrue(all(line.eta == 0.528 for line in lines))

    def test_glycine_crystal_from_cif_uses_density_metadata(self) -> None:
        cif = REPO_ROOT / "QuadrupolarDFT" / "structures" / "Glycine" / "189379.cif"
        crystal = glycine_crystal_from_cif(
            cif,
            electrode_area_m2=1e-6,
            thickness_m=1e-3,
        )

        self.assertAlmostEqual(crystal.density_kg_m3, 1576.0)
        self.assertAlmostEqual(crystal.molar_mass_kg_mol, 0.07507)
        self.assertGreater(crystal.quadrupolar_spin_count, 1e19)

    def test_resonant_strain_scales_linearly_with_voltage(self) -> None:
        crystal = PiezoelectricCrystal()

        strain_1 = resonant_strain_peak(crystal, 1.0)
        strain_2 = resonant_strain_peak(crystal, 2.0)

        self.assertGreater(strain_1, 0.0)
        self.assertAlmostEqual(strain_2, 2.0 * strain_1)

    def test_absorbed_power_saturates_with_drive_strength(self) -> None:
        weak = nuclear_absorbed_power(
            1e20,
            1e6,
            50.0,
            saturation_parameter=1e-6,
        )
        saturated = nuclear_absorbed_power(
            1e20,
            1e6,
            50.0,
            saturation_parameter=1e6,
        )

        self.assertGreater(saturated, weak)
        self.assertAlmostEqual(
            saturated,
            nuclear_absorbed_power(1e20, 1e6, 50.0, saturation_parameter=np.inf),
        )

    def test_detection_model_reports_positive_powers_and_saturation(self) -> None:
        crystal = PiezoelectricCrystal(electrode_area_m2=1e-6, thickness_m=1e-3)
        line = default_glycine_nqr_lines()[0]
        result = simulate_piezoelectric_nqr_detection(
            crystal,
            line,
            PiezoNQRCoupling(drive_hz_per_strain=1.0e6),
            voltage_rms=5.0,
        )

        self.assertGreater(result.strain_peak, 0.0)
        self.assertGreater(result.rabi_hz, 0.0)
        self.assertGreaterEqual(result.saturation_parameter, 0.0)
        self.assertGreater(result.spin_absorbed_power_w, 0.0)
        self.assertGreater(result.mechanical_drive_power_w, 0.0)
        self.assertGreaterEqual(result.fractional_q_shift, 0.0)
        self.assertGreater(result.capacitance_f, 0.0)


if __name__ == "__main__":
    unittest.main()
