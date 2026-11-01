"""Regression tests: powder GRAPE objectives use the orientation quadrature weights.

``nqr_powder_control_batch`` used to drop ``OrientationSample.weight`` and the
ensemble objective took an unweighted mean, so on a Gauss-Legendre powder grid
the score was not the powder average (14N, 4x4 grid, rectangular pulse:
unweighted 0.258 vs the weighted powder average 0.185).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr.full_dynamics import _default_carrier_hz  # noqa: E402
from spin_dynamics.nqr.hamiltonians import diagonalize_site  # noqa: E402
from spin_dynamics.nqr.isotopes import quadrupolar_site  # noqa: E402
from spin_dynamics.nqr.orientations import powder_average_grid  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import (  # noqa: E402
    JAX_AVAILABLE,
    propagate_state_numpy,
)
from spin_dynamics.optimal_control.hamiltonians import (  # noqa: E402
    nqr_fundamental_states,
    nqr_powder_control_batch,
    nqr_powder_control_weights,
    nqr_site_control_model,
)

if JAX_AVAILABLE:
    from spin_dynamics.optimal_control.objectives import (  # noqa: E402
        make_grape_objective,
        robust_ensemble_fidelity,
    )

N_THETA = 4
N_PHI = 4
N_SEG = 16
NUTATION_HZ = 20e3


def _powder_case():
    site = quadrupolar_site("14N", nu_q_hz=1e6, eta=0.0)
    rf = _default_carrier_hz(diagonalize_site(site, None))
    model = nqr_site_control_model(site, rf_frequency_hz=rf)
    lower, upper = nqr_fundamental_states(site)
    psi0 = np.zeros(model.dimension, dtype=np.complex128)
    psi0[lower] = 1.0
    target = np.zeros(model.dimension, dtype=np.complex128)
    target[upper] = 1.0
    orientations = powder_average_grid(n_theta=N_THETA, n_phi=N_PHI)
    batch = nqr_powder_control_batch(site, orientations, rf_frequency_hz=rf)
    dt = np.full(N_SEG, 1.0 / (2.0 * NUTATION_HZ) / N_SEG)
    return model, psi0, target, orientations, batch, dt


def _gauss_legendre_reference_weights() -> np.ndarray:
    # Independent reference: powder_average_grid is a Gauss-Legendre product
    # rule in cos(theta) times a uniform azimuthal rule, theta-major order.
    _nodes, mu_weights = np.polynomial.legendre.leggauss(N_THETA)
    return np.repeat(mu_weights / (2.0 * N_PHI), N_PHI)


class PowderWeightHelperTests(unittest.TestCase):
    def test_weights_match_gauss_legendre_quadrature_in_batch_order(self) -> None:
        orientations = powder_average_grid(n_theta=N_THETA, n_phi=N_PHI)
        weights = nqr_powder_control_weights(orientations)
        np.testing.assert_allclose(weights, _gauss_legendre_reference_weights(), rtol=1e-13)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-13)
        # The grid is genuinely non-uniform, so dropping the weights matters.
        self.assertGreater(weights.max() / weights.min(), 1.5)

    def test_rejects_empty_orientations(self) -> None:
        with self.assertRaises(ValueError):
            nqr_powder_control_weights(())


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class WeightedPowderObjectiveTests(unittest.TestCase):
    def test_objective_is_the_quadrature_weighted_powder_average(self) -> None:
        model, psi0, target, orientations, batch, dt = _powder_case()
        phase = np.zeros(N_SEG)
        amplitude = np.full(N_SEG, NUTATION_HZ)
        fids = np.array([
            abs(np.vdot(target, propagate_state_numpy(
                model.h_drift, hx, hy, amplitude, phase, dt, psi0
            ))) ** 2
            for hx, hy in batch
        ])
        reference_weights = _gauss_legendre_reference_weights()
        expected = float(np.sum(reference_weights * fids) / np.sum(reference_weights))
        unweighted = float(np.mean(fids))
        self.assertAlmostEqual(expected, 0.185, places=3)
        self.assertAlmostEqual(unweighted, 0.258, places=3)

        common = dict(
            n_segments=N_SEG, dt=dt, target=target, psi0=psi0,
            fixed_amplitude=NUTATION_HZ, control_operator_batch=batch,
            propagator="expm",
        )
        weighted_value, grad = make_grape_objective(
            model, ensemble_weights=nqr_powder_control_weights(orientations), **common
        )(phase)
        np.testing.assert_allclose(weighted_value, expected, rtol=1e-9, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(grad)))
        # Weights are normalized internally; scaling them changes nothing.
        scaled_value, _ = make_grape_objective(
            model, ensemble_weights=7.0 * reference_weights, **common
        )(phase)
        np.testing.assert_allclose(scaled_value, expected, rtol=1e-9, atol=1e-12)
        # Backward compatible default: no weights keeps the unweighted mean.
        default_value, _ = make_grape_objective(model, **common)(phase)
        np.testing.assert_allclose(default_value, unweighted, rtol=1e-9, atol=1e-12)

    def test_weighted_reductions(self) -> None:
        fids = np.array([0.2, 0.6, 1.0])
        weights = np.array([0.5, 0.25, 0.0])
        mean = float(robust_ensemble_fidelity(fids, weights=weights))
        np.testing.assert_allclose(mean, (0.5 * 0.2 + 0.25 * 0.6) / 0.75)
        worst = float(
            robust_ensemble_fidelity(fids, reduction="worst_case", weights=[0.0, 1.0, 1.0])
        )
        np.testing.assert_allclose(worst, 0.6)

    def test_rejects_invalid_ensemble_weights(self) -> None:
        model, psi0, target, _orientations, batch, dt = _powder_case()
        common = dict(
            n_segments=N_SEG, dt=dt, target=target, psi0=psi0,
            fixed_amplitude=NUTATION_HZ, propagator="expm",
        )
        bad = (
            np.ones(len(batch) - 1),
            -np.ones(len(batch)),
            np.zeros(len(batch)),
        )
        for weights in bad:
            with self.assertRaises(ValueError):
                make_grape_objective(
                    model, control_operator_batch=batch, ensemble_weights=weights, **common
                )
        with self.assertRaises(ValueError):
            make_grape_objective(model, ensemble_weights=[1.0], **common)


if __name__ == "__main__":
    unittest.main()
