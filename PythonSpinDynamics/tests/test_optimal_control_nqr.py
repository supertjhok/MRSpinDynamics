"""Tests for the NQR/quadrupolar GRAPE control-model builder.

The builder works in the rotating frame the real NQR density-matrix engine
uses (``nqr.full_dynamics``), so a rectangular pulse pushed through the GRAPE
model propagator must reproduce the engine's own pulse Hamiltonian evolution to
machine precision -- the parity anchor for the whole milestone.
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.evolution import evolve_density  # noqa: E402
from spin_dynamics.nqr.full_dynamics import (  # noqa: E402
    _default_carrier_hz,
    pulse_hamiltonian,
)
from spin_dynamics.nqr.hamiltonians import diagonalize_site  # noqa: E402
from spin_dynamics.nqr.isotopes import quadrupolar_site  # noqa: E402
from spin_dynamics.nqr.orientations import powder_average_grid  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import (  # noqa: E402
    propagate_state_numpy,
)
from spin_dynamics.optimal_control.hamiltonians import (  # noqa: E402
    nqr_fundamental_states,
    nqr_powder_control_batch,
    nqr_site_control_model,
)


def _site(spin_iso="63Cu", nu_q_hz=2.0e6, eta=0.1):
    return quadrupolar_site(spin_iso, nu_q_hz=nu_q_hz, eta=eta)


class NqrModelBuilderTests(unittest.TestCase):
    def test_dimension_and_hermiticity(self) -> None:
        for iso, nu_q in (("14N", 3.0e6), ("63Cu", 2.0e6), ("27Al", 1.0e6), ("51V", 0.8e6)):
            site = quadrupolar_site(iso, nu_q_hz=nu_q, eta=0.1)
            model = nqr_site_control_model(site)
            self.assertEqual(model.dimension, site.dimension)
            for op in (model.h_drift, model.h_x, model.h_y):
                self.assertEqual(op.shape, (site.dimension, site.dimension))
                np.testing.assert_allclose(op, op.conj().T, atol=1e-9)

    def test_drift_is_diagonal_in_eigenbasis(self) -> None:
        model = nqr_site_control_model(_site())
        off_diagonal = model.h_drift - np.diag(np.diag(model.h_drift))
        np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-9)

    def test_fundamental_states_are_distinct_valid_indices(self) -> None:
        site = _site()
        lower, upper = nqr_fundamental_states(site)
        self.assertNotEqual(lower, upper)
        self.assertTrue(0 <= lower < site.dimension)
        self.assertTrue(0 <= upper < site.dimension)

    def test_rectangular_pulse_matches_engine_to_machine_precision(self) -> None:
        # A rectangular pulse (constant amplitude/phase) via the GRAPE model must
        # equal evolving the density under the engine's own pulse_hamiltonian:
        # by construction pulse_hamiltonian == h_drift + w1(cos.hx + sin.hy).
        site = _site()
        eig = diagonalize_site(site, None)
        rf = _default_carrier_hz(eig)
        model = nqr_site_control_model(site, rf_frequency_hz=rf)
        lower, _upper = nqr_fundamental_states(site)
        d = model.dimension
        psi0 = np.zeros(d, dtype=np.complex128)
        psi0[lower] = 1.0

        w1, phi, n = 40e3, 0.6, 5
        dt = np.full(n, 2e-6)
        psi = propagate_state_numpy(
            model.h_drift, model.h_x, model.h_y, np.full(n, w1), np.full(n, phi), dt, psi0
        )
        rho_model = np.outer(psi, psi.conj())

        h_engine = pulse_hamiltonian(eig, nutation_hz=w1, rf_frequency_hz=rf, phase=phi)
        rho_engine = evolve_density(np.outer(psi0, psi0.conj()), h_engine, float(np.sum(dt)))
        np.testing.assert_allclose(rho_model, rho_engine, atol=1e-12)

    def test_h_x_h_y_reconstruct_pulse_hamiltonian(self) -> None:
        site = _site()
        eig = diagonalize_site(site, None)
        rf = _default_carrier_hz(eig)
        model = nqr_site_control_model(site, rf_frequency_hz=rf)
        w1, phi = 25e3, 1.3
        reconstructed = model.h_drift + w1 * (np.cos(phi) * model.h_x + np.sin(phi) * model.h_y)
        engine = pulse_hamiltonian(eig, nutation_hz=w1, rf_frequency_hz=rf, phase=phi)
        np.testing.assert_allclose(reconstructed, engine, atol=1e-6)

    def test_satellite_carrier_warns(self) -> None:
        # A spin-5/2 nucleus has higher lines; forcing the carrier onto the
        # 2*nu_Q satellite must warn (the single-carrier RWA is fundamental-only).
        site = quadrupolar_site("27Al", nu_q_hz=1.0e6, eta=0.0)
        eig = diagonalize_site(site, None)
        satellite = max(eig.transitions, key=lambda t: t.frequency_hz)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nqr_site_control_model(site, rf_frequency_hz=satellite.frequency_hz)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_powder_control_batch_shapes(self) -> None:
        site = _site()
        orientations = powder_average_grid(n_theta=3, n_phi=4)
        batch = nqr_powder_control_batch(site, orientations)
        self.assertEqual(len(batch), len(orientations))
        for h_x, h_y in batch:
            self.assertEqual(h_x.shape, (site.dimension, site.dimension))
            self.assertEqual(h_y.shape, (site.dimension, site.dimension))

    def test_powder_control_operators_actually_vary_with_orientation(self) -> None:
        site = _site()
        orientations = powder_average_grid(n_theta=3, n_phi=4)
        batch = nqr_powder_control_batch(site, orientations)
        # At least two crystallites must have genuinely different drive operators.
        self.assertGreater(np.max(np.abs(batch[0][0] - batch[-1][0])), 1e-6)


if __name__ == "__main__":
    unittest.main()
