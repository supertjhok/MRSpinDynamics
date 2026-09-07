"""Physical checks for Phase 2 loading, independent material solve and circuit."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "studies/nqr_mail_screening/phase2"
sys.path.insert(0, str(PATH))
from aperture import Region, candidates
from spatial_loading import (
    dielectric_bounds,
    combined_incident_integral,
    TunedCircuit,
    EPS0,
)
from material_validation import depolarization_matrix, solve_regions

sys.path.remove(str(PATH))


class LoadingTests(unittest.TestCase):
    def test_complex_sphere_and_slab_response_within_bounds(self):
        for eps, sigma in ((1.0, 1e-4), (1.5, 0.0), (3.0, 1e-4), (6.0, 1e-8)):
            f = 3.3043e6
            integral = 1e-5
            chi = eps - 1 - 1j * sigma / (2 * np.pi * f * EPS0)
            bounds = dielectric_bounds(integral, eps, sigma, f)
            for t in (0.0, 1 / 3, 1.0):
                c = EPS0 * integral * (chi / (1 + t * chi)).real
                g = sigma * integral / abs(1 + t * chi) ** 2
                for value, interval in (
                    (c, bounds["delta_capacitance_f"]),
                    (g, bounds["shunt_conductance_s"]),
                ):
                    self.assertGreaterEqual(value + 1e-25, interval[0])
                    self.assertLessEqual(value, interval[1] + 1e-25)

    def test_conductive_metal_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            dielectric_bounds(1.0, 3.0, 1e7, 3.3e6)

    def test_combined_electric_field_includes_phase_cross_term(self):
        e = np.array([1.0, 2.0, -3.0])
        a = np.array([2e-7, -1e-7, 4e-7])
        f = 3.3e6
        z = 0.3 + 200j
        expected = np.sum(np.abs(e + 2j * np.pi * f * a / z) ** 2)
        actual = combined_incident_integral(e @ e, a @ a, e @ a, f, z)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_tuned_network_and_passive_loading(self):
        circuit = TunedCircuit(1e-5, 0.3, 1e-11, 3.3043e6)
        bare = circuit.evaluate()
        loaded = circuit.evaluate(delta_c=1e-12, conductance=1e-5)
        self.assertAlmostEqual(
            bare["resonance_hz"] / circuit.frequency_hz, 1.0, places=10
        )
        self.assertLess(loaded["resonance_hz"], bare["resonance_hz"])
        self.assertLess(loaded["loaded_q_during_drive"], bare["loaded_q_during_drive"])
        self.assertGreater(loaded["receiver_port_johnson_psd_v2_hz"], 0.0)


@unittest.skipUnless(
    importlib.util.find_spec("magpylib"), "optional Phase 2 validation dependency"
)
class MaterialSolverTests(unittest.TestCase):
    def test_exact_cube_self_depolarization(self):
        matrix = depolarization_matrix(np.zeros((1, 3)), np.ones((1, 3)) * 0.01)
        np.testing.assert_allclose(matrix, -np.eye(3) / 3, atol=1e-13)

    def test_nonzero_uniform_parcel_respects_electrostatic_bounds(self):
        from spatial_loading import ElectrostaticReference

        region = Region(
            "dielectric",
            (0.01, 0.01, 0.01),
            (0.0, 0.0, 0.0),
            0.001,
            1.0,
            293.15,
            3.0,
            1e-4,
        )
        coil = candidates()["enclosing_helix"]
        network = {"test": {"L": [1e-5, 1e-5], "R": [0.3, 0.3]}}
        result = solve_regions([region], {"test": coil}, network, shape=(3, 3, 3))[
            "test"
        ]
        points, _ = region.voxels((3, 3, 3))
        electric = ElectrostaticReference(coil).field(points)
        integral = float(np.sum(electric**2) * np.prod(region.size_m) / len(points))
        interval = dielectric_bounds(integral, 3.0, 1e-4, 3.3043e6)[
            "delta_capacitance_f"
        ]
        self.assertGreater(result["delta_capacitance_f"], interval[0])
        self.assertLess(result["delta_capacitance_f"], interval[1])
        self.assertGreater(result["total_conductance_s"], 0.0)
        self.assertLess(result["relative_linear_residual"], 1e-12)

    def test_vacuum_limit_and_material_domain(self):
        region = Region(
            "vacuum", (0.01, 0.002, 0.01), (0.0, 0.0, 0.0), 0.0, 0.0, 293.15, 1.0, 0.0
        )
        coil = {"test": candidates()["enclosing_helix"]}
        network = {"test": {"L": [1e-5, 1e-5], "R": [0.3, 0.3]}}
        result = solve_regions([region], coil, network, shape=(2, 1, 2))["test"]
        self.assertEqual(result["delta_capacitance_f"], 0.0)
        self.assertEqual(result["total_conductance_s"], 0.0)
        with self.assertRaises(ValueError):
            solve_regions(
                [replace(region, conductivity_s_m=1e6)], coil, network, shape=(2, 1, 2)
            )
