"""Physical invariants and rejection cases for the moving NQR study."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.linalg import expm

STUDY = Path(__file__).resolve().parents[1] / "studies/nqr_mail_screening/phase1"
sys.path.insert(0, str(STUDY))
import pulsed_study as model
from geometry import for_config

sys.path.remove(str(STUDY))


class PulsedStudyTests(unittest.TestCase):
    def test_thermal_source_preserves_equilibrium_without_rf(self):
        eig, line, values = model.material_reference()
        eq = model.thermal_deviation(eig.levels_hz, 293.15)
        relaxation = model.NQRRelaxationModel(
            t1_seconds=values["t1_s"], t2_seconds=0.0144
        )
        h = model.static_hamiltonian_rotating(eig, line.frequency_hz)
        state = np.r_[eq.reshape(9, order="F"), 1.0]
        after = expm(model.affine_generator(h, relaxation, eq) * 0.1) @ state
        np.testing.assert_allclose(after, state, atol=1e-20)
        hp = model.pulse_hamiltonian(
            eig,
            nutation_hz=1000.0,
            rf_frequency_hz=line.frequency_hz,
            b1_direction_pas=(1.0, 0.0, 0.0),
        )
        driven = expm(model.affine_generator(hp, relaxation, eq) * 100e-6) @ state
        self.assertGreater(np.linalg.norm(driven - state), 1e-10)

    def test_zero_rf_no_signal_and_moving_receive_position(self):
        cfg = model.Config(
            current_peak_a=0.0, echoes=2, powder_theta=2, powder_phi=4, offset_points=9
        )
        row, times, voltage, valid = model.simulate(cfg)
        self.assertLess(np.max(np.abs(voltage)), 1e-25)
        self.assertGreater(
            row["readout_end_z_relative_coil_m"], row["readout_start_z_relative_coil_m"]
        )
        self.assertTrue(np.any(valid))
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertLess(row["max_density_trace_error"], 1e-18)

    def test_zeeman_excludes_close_magnet_even_when_rf_off(self):
        self.assertFalse(for_config(model.Config(magnet_coil_spacing_m=0.55))["passes"])
        self.assertTrue(for_config(model.Config(magnet_coil_spacing_m=1.2))["passes"])

    def test_preparation_uses_transport_and_preserves_density(self):
        cfg = model.Config(prepolarization=True)
        state, detail = model.prepared_density(cfg)
        self.assertGreater(detail["incoming_proton_polarization_fraction"], 0.0)
        self.assertGreater(detail["preparation_and_transport_s"], 0.0)
        self.assertAlmostEqual(np.trace(state).real, 0.0, places=15)
        self.assertGreater(np.linalg.eigvalsh(state + np.eye(3) / 3).min(), 0.0)
        _, slower = model.prepared_density(replace(cfg, transport_velocity_m_s=1.0))
        self.assertGreater(
            slower["preparation_and_transport_s"], detail["preparation_and_transport_s"]
        )

    def test_train_cannot_outlast_coil_passage(self):
        with self.assertRaisesRegex(ValueError, "coil exit"):
            model.simulate(model.Config(transport_velocity_m_s=100.0))


if __name__ == "__main__":
    unittest.main()
