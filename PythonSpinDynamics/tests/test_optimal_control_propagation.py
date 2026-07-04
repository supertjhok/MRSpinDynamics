"""Parity tests for the GRAPE differentiable propagator chain.

Cross-checks the new optimal_control engine against two independent existing
implementations for a hand-fixed (non-optimized) waveform: the plain-NumPy
reference in the same module, and the pre-existing, already-validated
core.rotations single-spin rotation-axis machinery (after converting hertz
nutation/offset to the rad/s convention rotations.py expects).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.operators import spin_operator  # noqa: E402
from spin_dynamics.coupling.systems import coupled_spin_system  # noqa: E402
from spin_dynamics.core.rotations import sim_spin_dynamics_exc  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import (  # noqa: E402
    JAX_AVAILABLE,
    propagate_state_numpy,
    propagate_state_numpy_grad,
)
from spin_dynamics.optimal_control.hamiltonians import (  # noqa: E402
    ControlHamiltonianModel,
    coupled_spin_control_model,
    gradient_control_operator,
    position_gradient_batch,
)

if JAX_AVAILABLE:
    import jax.numpy as jnp  # noqa: E402

    from spin_dynamics.optimal_control._jax_propagation import (  # noqa: E402
        iterate_unitary,
        propagate_batched_controls,
        propagate_batched_grad,
        propagate_state_scan,
        propagate_state_scan_grad,
        propagate_unitary_scan,
    )
    from spin_dynamics.optimal_control.objectives import (  # noqa: E402
        bloch_vector_to_ket,
        su2_effective_axis,
    )

TAU = 2.0 * np.pi

_SIGMA_X = 2 * spin_operator(1, 0, "x")
_SIGMA_Y = 2 * spin_operator(1, 0, "y")
_SIGMA_Z = 2 * spin_operator(1, 0, "z")


def _bloch_vector(psi: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.real(np.conj(psi) @ _SIGMA_X @ psi),
            np.real(np.conj(psi) @ _SIGMA_Y @ psi),
            np.real(np.conj(psi) @ _SIGMA_Z @ psi),
        ]
    )


def _single_spin_model(offset_hz: float) -> ControlHamiltonianModel:
    system = coupled_spin_system([offset_hz], [[0.0]])
    return coupled_spin_control_model(system)


_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)


class NumpyEngineVsRotationsParityTests(unittest.TestCase):
    """propagate_state_numpy vs. the pre-existing rotations.py machinery."""

    def test_single_on_resonance_pulse(self) -> None:
        w1_hz, tp_s, phi = 5000.0, 1.0 / (4 * 5000.0), 0.7
        model = _single_spin_model(offset_hz=0.0)
        psi_final = propagate_state_numpy(
            model.h_drift, model.h_x, model.h_y, [w1_hz], [phi], [tp_s], _PSI_UP
        )
        got = _bloch_vector(psi_final)

        mvect = sim_spin_dynamics_exc(
            np.array([tp_s]), np.array([phi]), np.array([TAU * w1_hz]), np.array([0.0])
        )
        expected = np.real(mvect[:, 0])
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_multi_segment_off_resonance(self) -> None:
        offset_hz, w1_hz, n_seg = 800.0, 4000.0, 5
        rng = np.random.default_rng(0)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        phi = rng.uniform(0, 2 * np.pi, size=n_seg)
        model = _single_spin_model(offset_hz)

        psi_final = propagate_state_numpy(
            model.h_drift, model.h_x, model.h_y, np.full(n_seg, w1_hz), phi, tp, _PSI_UP
        )
        got = _bloch_vector(psi_final)

        mvect = sim_spin_dynamics_exc(
            tp, phi, np.full(n_seg, TAU * w1_hz), np.array([TAU * offset_hz])
        )
        expected = np.real(mvect[:, 0])
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_zero_amplitude_segment_is_free_precession(self) -> None:
        offset_hz = 250.0
        tp_s = 3e-4
        model = _single_spin_model(offset_hz)
        psi_final = propagate_state_numpy(
            model.h_drift, model.h_x, model.h_y, [0.0], [0.0], [tp_s], _PSI_UP
        )
        got = _bloch_vector(psi_final)
        # Free precession about z leaves a purely +z Bloch vector unchanged.
        np.testing.assert_allclose(got, [0.0, 0.0, 1.0], atol=1e-9)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class JaxEngineMatchesNumpyReferenceTests(unittest.TestCase):
    """The JAX lax.scan path must be bit-for-bit consistent with the NumPy path."""

    def test_state_propagation_matches(self) -> None:
        offset_hz, w1_hz, n_seg = 300.0, 3000.0, 6
        rng = np.random.default_rng(1)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        model = _single_spin_model(offset_hz)

        expected = propagate_state_numpy(
            model.h_drift, model.h_x, model.h_y, np.full(n_seg, w1_hz), phi, tp, _PSI_UP
        )
        got = propagate_state_scan(
            jnp.asarray(model.h_drift),
            jnp.asarray(model.h_x),
            jnp.asarray(model.h_y),
            jnp.asarray(np.full(n_seg, w1_hz)),
            jnp.asarray(phi),
            jnp.asarray(tp),
            jnp.asarray(_PSI_UP),
        )
        np.testing.assert_allclose(np.asarray(got), expected, atol=1e-10)

    def test_unitary_propagation_is_consistent_with_state_propagation(self) -> None:
        offset_hz, w1_hz, n_seg = 150.0, 2500.0, 4
        rng = np.random.default_rng(2)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        model = _single_spin_model(offset_hz)
        amplitude = jnp.asarray(np.full(n_seg, w1_hz))
        phase = jnp.asarray(phi)
        dt = jnp.asarray(tp)

        u_final = propagate_unitary_scan(
            jnp.asarray(model.h_drift), jnp.asarray(model.h_x), jnp.asarray(model.h_y),
            amplitude, phase, dt,
        )
        psi_via_unitary = np.asarray(u_final) @ _PSI_UP
        psi_direct = np.asarray(
            propagate_state_scan(
                jnp.asarray(model.h_drift), jnp.asarray(model.h_x), jnp.asarray(model.h_y),
                amplitude, phase, dt, jnp.asarray(_PSI_UP),
            )
        )
        np.testing.assert_allclose(psi_via_unitary, psi_direct, atol=1e-10)

    def test_unitary_is_unitary(self) -> None:
        model = _single_spin_model(offset_hz=500.0)
        n_seg = 5
        rng = np.random.default_rng(3)
        u_final = propagate_unitary_scan(
            jnp.asarray(model.h_drift), jnp.asarray(model.h_x), jnp.asarray(model.h_y),
            jnp.asarray(rng.uniform(1000.0, 4000.0, size=n_seg)),
            jnp.asarray(rng.uniform(-np.pi, np.pi, size=n_seg)),
            jnp.asarray(rng.uniform(1e-5, 5e-5, size=n_seg)),
        )
        u = np.asarray(u_final)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(2), atol=1e-9)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class IterateUnitaryTests(unittest.TestCase):
    def test_trajectory_shape_and_first_entry(self) -> None:
        u = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)  # X gate
        psi0 = jnp.asarray(_PSI_UP)
        trajectory = iterate_unitary(u, psi0, 4)
        self.assertEqual(np.asarray(trajectory).shape, (5, 2))
        np.testing.assert_allclose(np.asarray(trajectory[0]), _PSI_UP, atol=1e-12)

    def test_x_gate_alternates(self) -> None:
        u = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
        trajectory = np.asarray(iterate_unitary(u, jnp.asarray(_PSI_UP), 3))
        np.testing.assert_allclose(trajectory[0], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(trajectory[1], [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(trajectory[2], [1.0, 0.0], atol=1e-12)

    def test_axis_aligned_state_is_a_fixed_point(self) -> None:
        model = _single_spin_model(offset_hz=310.0)
        n_seg = 5
        rng = np.random.default_rng(9)
        u = propagate_unitary_scan(
            jnp.asarray(model.h_drift), jnp.asarray(model.h_x), jnp.asarray(model.h_y),
            jnp.asarray(rng.uniform(1000.0, 4000.0, size=n_seg)),
            jnp.asarray(rng.uniform(-np.pi, np.pi, size=n_seg)),
            jnp.asarray(rng.uniform(1e-5, 5e-5, size=n_seg)),
        )
        axis, _angle = su2_effective_axis(u)
        ket = bloch_vector_to_ket(axis)
        trajectory = np.asarray(iterate_unitary(u, ket, 10))
        # Every step is the same ray as psi0 (fixed up to global phase).
        overlaps = np.abs(np.sum(np.conj(ket)[np.newaxis, :] * trajectory, axis=1)) ** 2
        np.testing.assert_allclose(overlaps, 1.0, atol=1e-9)

    def test_generic_state_is_not_a_fixed_point(self) -> None:
        model = _single_spin_model(offset_hz=310.0)
        n_seg = 5
        rng = np.random.default_rng(9)
        u = propagate_unitary_scan(
            jnp.asarray(model.h_drift), jnp.asarray(model.h_x), jnp.asarray(model.h_y),
            jnp.asarray(rng.uniform(1000.0, 4000.0, size=n_seg)),
            jnp.asarray(rng.uniform(-np.pi, np.pi, size=n_seg)),
            jnp.asarray(rng.uniform(1e-5, 5e-5, size=n_seg)),
        )
        trajectory = np.asarray(iterate_unitary(u, jnp.asarray(_PSI_UP), 10))
        overlaps = np.abs(np.sum(np.conj(_PSI_UP)[np.newaxis, :] * trajectory, axis=1)) ** 2
        self.assertLess(np.min(overlaps), 1.0 - 1e-6)


class GradientChannelNumpyTests(unittest.TestCase):
    """Physics of the gradient control channel via the NumPy reference."""

    def test_grad_reduces_to_rf_only_when_gradient_is_zero(self) -> None:
        model = _single_spin_model(offset_hz=200.0)
        n_seg = 6
        rng = np.random.default_rng(4)
        amp = rng.uniform(1000.0, 4000.0, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        h_grad = gradient_control_operator(coupled_spin_system([200.0], [[0.0]]))
        rf_only = propagate_state_numpy(
            model.h_drift, model.h_x, model.h_y, amp, phi, tp, _PSI_UP
        )
        with_zero_grad = propagate_state_numpy_grad(
            model.h_drift, model.h_x, model.h_y, h_grad, amp, phi, np.zeros(n_seg), tp, _PSI_UP
        )
        np.testing.assert_allclose(with_zero_grad, rf_only, atol=1e-12)

    def test_pure_gradient_segment_dephases_by_analytic_phase(self) -> None:
        # RF off, on-resonance drift zero: a spin at position ``r`` under a
        # gradient waveform accrues relative phase 2*pi * sum(g*dt) * r about z.
        system = coupled_spin_system([0.0], [[0.0]])
        model = coupled_spin_control_model(system, include_gradient=True)
        n_seg = 8
        rng = np.random.default_rng(5)
        g = rng.uniform(-2000.0, 2000.0, size=n_seg)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        r = 0.73
        h_grad = r * model.h_grad
        psi_plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        got = propagate_state_numpy_grad(
            model.h_drift, model.h_x, model.h_y, h_grad,
            np.zeros(n_seg), np.zeros(n_seg), g, tp, psi_plus,
        )
        theta = 2 * np.pi * np.sum(g * tp) * r
        expected = np.array(
            [np.exp(-1j * theta / 2), np.exp(1j * theta / 2)], dtype=np.complex128
        ) / np.sqrt(2)
        np.testing.assert_allclose(got, expected, atol=1e-12)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class GradientChannelJaxParityTests(unittest.TestCase):
    """The JAX _grad path must match the NumPy _grad reference, batched too."""

    def test_state_grad_scan_matches_numpy(self) -> None:
        system = coupled_spin_system([300.0], [[0.0]])
        model = coupled_spin_control_model(system, include_gradient=True)
        n_seg = 7
        rng = np.random.default_rng(6)
        amp = rng.uniform(1000.0, 4000.0, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        g = rng.uniform(-1500.0, 1500.0, size=n_seg)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        h_grad = 0.9 * model.h_grad
        expected = propagate_state_numpy_grad(
            model.h_drift, model.h_x, model.h_y, h_grad, amp, phi, g, tp, _PSI_UP
        )
        got = propagate_state_scan_grad(
            jnp.asarray(model.h_drift), jnp.asarray(model.h_x), jnp.asarray(model.h_y),
            jnp.asarray(h_grad), jnp.asarray(amp), jnp.asarray(phi), jnp.asarray(g),
            jnp.asarray(tp), jnp.asarray(_PSI_UP),
        )
        np.testing.assert_allclose(np.asarray(got), expected, atol=1e-10)

    def test_batched_grad_matches_per_case_loop(self) -> None:
        system = coupled_spin_system([0.0], [[0.0]])
        model = coupled_spin_control_model(system, include_gradient=True)
        n_seg = 6
        rng = np.random.default_rng(7)
        amp = rng.uniform(1000.0, 4000.0, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        g = rng.uniform(-1500.0, 1500.0, size=n_seg)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        positions = np.array([-1.2, -0.4, 0.3, 1.1])
        hgb = position_gradient_batch(model.h_grad, positions)
        drift_b = jnp.broadcast_to(jnp.asarray(model.h_drift), (positions.size, 2, 2))
        psi0_b = jnp.broadcast_to(jnp.asarray(_PSI_UP), (positions.size, 2))
        batched = np.asarray(
            propagate_batched_grad(
                drift_b, jnp.asarray(model.h_x), jnp.asarray(model.h_y),
                jnp.stack([jnp.asarray(h) for h in hgb]),
                jnp.asarray(amp), jnp.asarray(phi), jnp.asarray(g),
                jnp.asarray(tp), psi0_b,
            )
        )
        loop = np.stack(
            [
                propagate_state_numpy_grad(
                    model.h_drift, model.h_x, model.h_y, hgb[k], amp, phi, g, tp, _PSI_UP
                )
                for k in range(positions.size)
            ]
        )
        np.testing.assert_allclose(batched, loop, atol=1e-10)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class BatchedControlsTests(unittest.TestCase):
    """Per-case RF drive operators (the NQR powder primitive)."""

    def test_batched_controls_matches_per_case_loop(self) -> None:
        # Distinct drive operators per case, shared RF waveform + drift.
        rng = np.random.default_rng(21)
        n_seg = 6
        amp = rng.uniform(1000.0, 4000.0, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        model = _single_spin_model(offset_hz=200.0)
        # Build a handful of rotated (h_x, h_y) pairs to stand in for a batch.
        angles = [0.0, 0.4, 1.1, 2.0]
        hx_list, hy_list = [], []
        for a in angles:
            hx_list.append(np.cos(a) * model.h_x - np.sin(a) * model.h_y)
            hy_list.append(np.sin(a) * model.h_x + np.cos(a) * model.h_y)
        drift_b = jnp.broadcast_to(jnp.asarray(model.h_drift), (len(angles), 2, 2))
        psi0_b = jnp.broadcast_to(jnp.asarray(_PSI_UP), (len(angles), 2))
        batched = np.asarray(
            propagate_batched_controls(
                drift_b,
                jnp.stack([jnp.asarray(h) for h in hx_list]),
                jnp.stack([jnp.asarray(h) for h in hy_list]),
                jnp.asarray(amp), jnp.asarray(phi), jnp.asarray(tp), psi0_b,
            )
        )
        loop = np.stack(
            [
                propagate_state_numpy(model.h_drift, hx_list[k], hy_list[k], amp, phi, tp, _PSI_UP)
                for k in range(len(angles))
            ]
        )
        np.testing.assert_allclose(batched, loop, atol=1e-10)

    def test_expm_method_matches_eigh_for_nondegenerate(self) -> None:
        # For a non-degenerate spin-1/2 system the expm and eigh segment
        # propagators agree (their gradients differ only under degeneracy).
        rng = np.random.default_rng(22)
        n_seg = 5
        amp = rng.uniform(1000.0, 4000.0, size=n_seg)
        phi = rng.uniform(-np.pi, np.pi, size=n_seg)
        tp = rng.uniform(1e-5, 5e-5, size=n_seg)
        model = _single_spin_model(offset_hz=300.0)
        args = (
            jnp.asarray(model.h_drift), jnp.asarray(model.h_x), jnp.asarray(model.h_y),
            jnp.asarray(amp), jnp.asarray(phi), jnp.asarray(tp), jnp.asarray(_PSI_UP),
        )
        via_eigh = np.asarray(propagate_state_scan(*args, method="eigh"))
        via_expm = np.asarray(propagate_state_scan(*args, method="expm"))
        np.testing.assert_allclose(via_expm, via_eigh, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
