"""Tests for the PGSE diffusion optimal-control pieces (Milestone 5).

Covers the combined (position x B0 x B1) gradient propagator, the q-vector
moments (validated against ``workflows.pgse.gradient_moment_b_value``), the
detected-echo SNR, and the weighted PGSE objective (build, autodiff vs finite
difference, and a tiny end-to-end improvement).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.systems import coupled_spin_system  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE  # noqa: E402
from spin_dynamics.optimal_control.hamiltonians import (  # noqa: E402
    coupled_spin_control_model,
    gradient_control_operator,
)
from spin_dynamics.workflows.pgse import gradient_moment_b_value  # noqa: E402

if JAX_AVAILABLE:
    import jax.numpy as jnp  # noqa: E402

    from spin_dynamics.optimal_control._jax_propagation import (  # noqa: E402
        propagate_batched_grad_controls,
        propagate_batched_grad_controls_numpy,
    )
    from spin_dynamics.optimal_control.control_response import (  # noqa: E402
        GradientDriverResponse,
        ReceiverResponse,
        TailSpec,
        TunedProbeResponse,
    )
    from spin_dynamics.optimal_control.diffusion import (  # noqa: E402
        detected_echo_signal,
        detected_echo_snr,
        gradient_moments,
        make_pgse_objective,
    )

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_I_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)


def _ensemble(positions, offsets, b1s, *, g_scale=1.0):
    # g_scale folds a characteristic offset scale into the gradient OPERATOR so
    # the optimized gradient control is O(1) (well-conditioned against the O(1)
    # phase block), the same trick examples/grape_slice_selective_pulse.py uses.
    base = coupled_spin_control_model(coupled_spin_system([0.0], [[0.0]]))
    hx0, hy0 = base.h_x, base.h_y
    hgrad0 = gradient_control_operator(coupled_spin_system([0.0], [[0.0]])) * g_scale
    drift_b, hx_b, hy_b, hgrad_b, weights, offs = [], [], [], [], [], []
    for off in offsets:
        drift = coupled_spin_control_model(coupled_spin_system([off], [[0.0]])).h_drift
        for b1 in b1s:
            for r in positions:
                drift_b.append(drift)
                hx_b.append(b1 * hx0)
                hy_b.append(b1 * hy0)
                hgrad_b.append(r * hgrad0)
                weights.append(1.0)
                offs.append(off)
    return drift_b, hx_b, hy_b, hgrad_b, np.array(weights), np.array(offs)


class GradientMomentTests(unittest.TestCase):
    """q-vector moments -- no JAX needed for the convention check via NumPy."""

    @unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
    def test_balanced_lobes_refocus(self) -> None:
        grad = np.zeros(40)
        grad[5:15] = 1.0
        grad[25:35] = -1.0
        _q, q_enc, q_res, _b = gradient_moments(jnp.asarray(grad), 1e-6, encode_end=15)
        self.assertAlmostEqual(float(q_enc), 10 * 1e-6, places=12)
        self.assertAlmostEqual(float(q_res), 0.0, places=12)

    @unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
    def test_b_value_matches_workflow_convention(self) -> None:
        # gradient_moments uses q = integral g dt (gamma folded into g); compare
        # against workflows.pgse.gradient_moment_b_value with gamma = 1.
        dt = 1e-6
        grad = np.zeros(30)
        grad[2:10] = 0.7
        grad[18:26] = -0.7
        _q, _qe, _qr, b_ours = gradient_moments(jnp.asarray(grad), dt, encode_end=10)
        segments = [(dt, float(g)) for g in grad]
        b_ref = gradient_moment_b_value(segments, gamma=1.0)
        self.assertAlmostEqual(float(b_ours), float(b_ref), places=10)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class PropagatorParityTests(unittest.TestCase):
    def test_combined_grad_controls_jax_matches_numpy(self) -> None:
        drift_b, hx_b, hy_b, hgrad_b, _w, _o = _ensemble([-1.0, 0.0, 1.0], [-200.0, 200.0], [0.9, 1.1])
        n = 16
        dt = 5e-6
        rng = np.random.default_rng(0)
        amp = rng.uniform(0, 3000, n)
        phase = rng.uniform(-np.pi, np.pi, n)
        grad = rng.uniform(-500, 500, n)
        psi0_batch = np.broadcast_to(_PSI_UP, (len(drift_b), 2))
        got = np.asarray(propagate_batched_grad_controls(
            jnp.asarray(drift_b), jnp.asarray(hx_b), jnp.asarray(hy_b), jnp.asarray(hgrad_b),
            jnp.asarray(amp), jnp.asarray(phase), jnp.asarray(grad), dt, jnp.asarray(psi0_batch),
        ))
        ref = propagate_batched_grad_controls_numpy(
            np.asarray(drift_b), np.asarray(hx_b), np.asarray(hy_b), np.asarray(hgrad_b),
            amp, phase, grad, dt, psi0_batch,
        )
        self.assertLess(np.max(np.abs(got - ref)), 1e-12)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class DetectedEchoTests(unittest.TestCase):
    def test_refocused_beats_dephased(self) -> None:
        offsets = jnp.asarray([-300.0, 0.0, 300.0])
        weights = jnp.ones(3)
        # Refocused: all coherences aligned in phase.
        refocused = jnp.asarray([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
        # Dephased: random relative phases (poor refocusing).
        dephased = jnp.asarray([np.exp(1j * 2.0), 1.0 + 0j, np.exp(-1j * 2.3)])
        _t, s_ref = detected_echo_signal(refocused, weights, offsets, n_acq=41, dt_acq=2e-6)
        _t, s_deph = detected_echo_signal(dephased, weights, offsets, n_acq=41, dt_acq=2e-6)
        snr_ref = float(detected_echo_snr(s_ref, 2e-6))
        snr_deph = float(detected_echo_snr(s_deph, 2e-6))
        self.assertGreater(snr_ref, snr_deph)

    def test_receiver_filter_changes_snr(self) -> None:
        offsets = jnp.asarray([-300.0, 0.0, 300.0])
        weights = jnp.ones(3)
        coh = jnp.asarray([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
        _t, s = detected_echo_signal(coh, weights, offsets, n_acq=41, dt_acq=2e-6)
        # A detuned receiver attenuates the (near-DC) coherent echo -> lower SNR.
        rx = ReceiverResponse(tau=1e-4, detuning_hz=8000.0)
        snr_raw = float(detected_echo_snr(s, 2e-6))
        snr_rx = float(detected_echo_snr(s, 2e-6, rx_response=rx))
        self.assertLess(snr_rx, 0.9 * snr_raw)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class PgseObjectiveTests(unittest.TestCase):
    def _objective(self, **overrides):
        # Gradient operator carries a 1000 Hz/position scale, so the gradient
        # control is O(1) and q is in units of integral(g_control) dt.
        drift_b, hx_b, hy_b, hgrad_b, weights, offs = _ensemble(
            [-1.0, 0.0, 1.0], [-200.0, 0.0, 200.0], [0.9, 1.0], g_scale=1000.0
        )
        n = 16
        amp_template = np.zeros(n)
        amp_template[:3] = 4000.0
        amp_template[7:10] = 4000.0
        grad_mask = np.zeros(n)
        grad_mask[3:7] = 1.0
        grad_mask[10:14] = 1.0
        coherence_sign = np.ones(n)
        coherence_sign[10:] = -1.0  # 180 pulse at [7:10] flips the coherence sign
        kwargs = dict(
            drift_batch=drift_b, hx_batch=hx_b, hy_batch=hy_b, hgrad_batch=hgrad_b,
            psi0=_PSI_UP, detection_operator=_I_PLUS, weights=weights, offsets_hz=offs,
            dt=5e-6, n_segments=n, amplitude_template=amp_template, gradient_mask=grad_mask,
            coherence_sign=coherence_sign,
            encode_end_segment=7, q_target=2.0e-5, acquisition_points=33, acquisition_dt=2e-6,
            q_weight=1e6, refocus_weight=1e6,
        )
        kwargs.update(overrides)
        return make_pgse_objective(**kwargs), n

    def test_grad_matches_finite_difference(self) -> None:
        probe = TunedProbeResponse.from_quality_factor(f0_hz=1e6, quality_factor=100.0, oversample=2)
        driver = GradientDriverResponse(tau_rl=2e-6, eddy_terms=((0.2, 3e-5),), oversample=2,
                                        tail=TailSpec(factor=3.0))
        vg, n = self._objective(rf_response=probe, gradient_response=driver,
                                rx_response=ReceiverResponse(tau=probe.tau))
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.uniform(-1, 1, n), rng.uniform(-1.0, 1.0, n)])
        _v, grad = vg(x)
        h = 1e-5
        for i in (n + 4, n + 11):  # gradient-block params
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            fd = (vg(xp)[0] - vg(xm)[0]) / (2 * h)
            rel = abs(fd - grad[i]) / max(abs(fd), 1e-9)
            self.assertLess(rel, 5e-3, msg=f"param {i}: fd={fd:.3e} ad={grad[i]:.3e}")

    def test_optimization_improves_objective(self) -> None:
        from spin_dynamics.optimization._bounded import scipy_maximize_with_grad

        driver = GradientDriverResponse(tau_rl=2e-6, eddy_terms=((0.25, 4e-5),), oversample=2,
                                        tail=TailSpec(factor=3.0))
        vg, n = self._objective(gradient_response=driver)
        # Poor random-phase start with a refocused unit gradient: plenty of SNR
        # room, well-conditioned (both blocks O(1)).
        rng = np.random.default_rng(4)
        grad0 = np.ones(n)
        x0 = np.concatenate([rng.uniform(-np.pi, np.pi, n), grad0])
        v0 = vg(x0)[0]
        bounds = [(-4 * np.pi, 4 * np.pi)] * n + [(-3.0, 3.0)] * n
        run = scipy_maximize_with_grad(vg, x0, bounds=bounds, options={"maxiter": 100})
        self.assertGreater(run.best_score, v0)


if __name__ == "__main__":
    unittest.main()
