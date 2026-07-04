"""Tests for the hardware-response actuator layer (probe + gradient driver).

Covers the design-note invariants: identity fallback reproduces the ideal
objective bit-for-bit, autodiff through the filter matches finite differences,
a band-limited probe both degrades an uncompensated pulse and is recovered by
GRAPE, tuned and matched probes differ, the gradient driver has DC gain 1 with
an explicit eddy tail, the eddy-fit helper round-trips, and the receiver
resamples onto the caller's grid.
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
from spin_dynamics.optimal_control.control_response import (  # noqa: E402
    GradientDriverResponse,
    MatchedProbeResponse,
    ReceiverResponse,
    TailSpec,
    TunedProbeResponse,
    UntunedProbeResponse,
    eddy_terms_from_step_response,
)
from spin_dynamics.optimal_control.hamiltonians import (  # noqa: E402
    coupled_spin_control_model,
)

if JAX_AVAILABLE:
    import jax.numpy as jnp  # noqa: E402

    from spin_dynamics.optimal_control.objectives import make_grape_objective  # noqa: E402
    from spin_dynamics.optimal_control.solvers import grape_optimize  # noqa: E402

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_PSI_DOWN = np.array([0.0, 1.0], dtype=np.complex128)


def _model(offset_hz: float = 300.0):
    return coupled_spin_control_model(coupled_spin_system([offset_hz], [[0.0]]))


class ResponsePrimitiveTests(unittest.TestCase):
    """Pure NumPy behaviour of the response filters (no JAX required)."""

    def test_tuned_probe_dc_gain_unity_on_resonance(self) -> None:
        probe = TunedProbeResponse(tau=1e-5)
        self.assertAlmostEqual(abs(probe.transfer(np.array([0.0]))[0]), 1.0, places=12)

    def test_tuned_probe_rolls_off_above_cutoff(self) -> None:
        tau = 1e-5
        probe = TunedProbeResponse(tau=tau)
        cutoff = 1.0 / (2 * np.pi * tau)
        mag = abs(probe.transfer(np.array([cutoff]))[0])
        self.assertAlmostEqual(mag, 1 / np.sqrt(2), places=6)  # -3 dB point

    def test_mistuned_probe_attenuates_carrier(self) -> None:
        probe = TunedProbeResponse(tau=1e-5, detuning_hz=2e4)
        self.assertLess(abs(probe.transfer(np.array([0.0]))[0]), 1.0)

    def test_matched_differs_from_tuned(self) -> None:
        tuned = TunedProbeResponse.from_quality_factor(f0_hz=1e6, quality_factor=100.0)
        matched = MatchedProbeResponse.from_loaded_q(f0_hz=1e6, loaded_q=100.0)
        freqs = np.linspace(-5e4, 5e4, 101)
        self.assertFalse(np.allclose(tuned.transfer(freqs), matched.transfer(freqs)))

    def test_matched_unity_at_resonance(self) -> None:
        matched = MatchedProbeResponse.from_loaded_q(f0_hz=1e6, loaded_q=100.0)
        self.assertAlmostEqual(abs(matched.transfer(np.array([0.0]))[0]), 1.0, places=12)

    def test_untuned_probe_flatter_than_tuned(self) -> None:
        # Over a normal NMR bandwidth the low-Q untuned coil is far flatter than
        # a Q=100 tuned probe (its fractional bandwidth is ~the carrier).
        untuned = UntunedProbeResponse(inductance_h=1e-6, resistance_ohm=1.0, carrier_hz=1e6)
        tuned = TunedProbeResponse.from_quality_factor(f0_hz=1e6, quality_factor=100.0)
        freqs = np.linspace(-1e4, 1e4, 51)
        untuned_min = np.abs(untuned.transfer(freqs)).min()
        tuned_min = np.abs(tuned.transfer(freqs)).min()
        self.assertGreater(untuned_min, 0.98)
        self.assertLess(tuned_min, 0.9)
        self.assertGreater(untuned_min, tuned_min + 0.05)

    def test_gradient_driver_dc_gain_unity(self) -> None:
        gd = GradientDriverResponse(
            tau_rl=2e-6, eddy_terms=((0.3, 4e-5),), tail=TailSpec(factor=0.0)
        )
        out = gd.apply(np.ones(2000), 1e-6, xp=np)
        self.assertAlmostEqual(float(out[-1]), 1.0, places=4)

    def test_gradient_driver_droops_then_recovers(self) -> None:
        gd = GradientDriverResponse(tau_rl=2e-6, eddy_terms=((0.4, 5e-5),))
        out = gd.apply(np.ones(3000), 1e-6, xp=np)
        self.assertLess(float(out[5]), 0.9)  # slew + eddy dip early
        self.assertGreater(float(out[-1]), float(out[5]))  # recovers toward command

    def test_eddy_fit_round_trips(self) -> None:
        t = np.arange(0.0, 3e-4, 1e-6)
        step = 1.0 * (1.0 - 0.25 * np.exp(-t / 4e-5))
        (alpha, tau), = eddy_terms_from_step_response(t, step, n_terms=1)
        self.assertAlmostEqual(alpha, 0.25, places=2)
        self.assertAlmostEqual(tau, 4e-5, delta=4e-6)

    def test_receiver_resamples_onto_grid(self) -> None:
        rx = ReceiverResponse.from_samples([-1e5, 0.0, 1e5], [0.5, 1.0, 0.5])
        h = rx.transfer(np.array([-5e4, 0.0, 5e4]))
        self.assertAlmostEqual(h[1].real, 1.0, places=12)
        self.assertAlmostEqual(h[0].real, 0.75, places=12)  # linear interp midpoint

    def test_tail_spec_switch(self) -> None:
        self.assertEqual(TailSpec(factor=5.0).seconds(2e-5), 1e-4)
        self.assertEqual(TailSpec(window_seconds=3e-4).seconds(2e-5), 3e-4)

    def test_invalid_params_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TunedProbeResponse(tau=-1.0)
        with self.assertRaises(ValueError):
            GradientDriverResponse(tau_rl=1e-6, eddy_terms=((1.5, 1e-5),))  # alpha >= 1


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class ResponseJaxParityTests(unittest.TestCase):
    def test_gradient_driver_jax_matches_numpy(self) -> None:
        gd = GradientDriverResponse(tau_rl=3e-6, eddy_terms=((0.2, 2e-5), (0.1, 8e-5)))
        wave = np.sin(np.linspace(0, 6, 500))
        ref = gd.apply(wave, 1e-6, xp=np)
        got = np.asarray(gd.apply(jnp.asarray(wave), 1e-6, xp=jnp))
        self.assertLess(np.max(np.abs(ref - got)), 1e-9)


@unittest.skipUnless(JAX_AVAILABLE, "jax not installed")
class ResponseObjectiveTests(unittest.TestCase):
    def _base_kwargs(self, n, dt):
        return dict(
            n_segments=n, dt=dt, target=_PSI_DOWN, psi0=_PSI_UP, fixed_amplitude=3000.0
        )

    def test_identity_when_probe_bandwidth_is_huge(self) -> None:
        model = _model()
        n, dt = 10, 2e-5
        phase = np.linspace(0, 1.5, n)
        vg_ideal = make_grape_objective(model, **self._base_kwargs(n, dt))
        wide = TunedProbeResponse(tau=1e-13, oversample=2, tail=TailSpec(factor=0.0))
        vg_wide = make_grape_objective(model, rf_response=wide, **self._base_kwargs(n, dt))
        self.assertAlmostEqual(float(vg_ideal(phase)[0]), float(vg_wide(phase)[0]), places=9)

    def test_probe_changes_objective(self) -> None:
        model = _model()
        n, dt = 10, 2e-5
        phase = np.linspace(0, 1.5, n)
        narrow = TunedProbeResponse.from_quality_factor(
            f0_hz=1e6, quality_factor=150.0, oversample=4
        )
        vg_ideal = make_grape_objective(model, **self._base_kwargs(n, dt))
        vg_narrow = make_grape_objective(model, rf_response=narrow, **self._base_kwargs(n, dt))
        self.assertNotAlmostEqual(float(vg_ideal(phase)[0]), float(vg_narrow(phase)[0]), places=4)

    def test_rf_filter_grad_matches_finite_difference(self) -> None:
        model = _model()
        n, dt = 8, 2e-5
        probe = TunedProbeResponse.from_quality_factor(
            f0_hz=1e6, quality_factor=120.0, oversample=4
        )
        vg = make_grape_objective(model, rf_response=probe, **self._base_kwargs(n, dt))
        rng = np.random.default_rng(3)
        phase = rng.uniform(-np.pi, np.pi, size=n)
        _v, grad = vg(phase)
        h = 1e-6
        fd = np.zeros_like(phase)
        for i in range(n):
            pp, pm = phase.copy(), phase.copy()
            pp[i] += h
            pm[i] -= h
            fd[i] = (vg(pp)[0] - vg(pm)[0]) / (2 * h)
        rel = np.linalg.norm(grad - fd) / max(np.linalg.norm(fd), 1e-12)
        self.assertLess(rel, 5e-3, msg=f"grad rel error {rel:.2e}")

    def test_ring_down_tail_changes_result(self) -> None:
        model = _model()
        n, dt = 8, 2e-5
        phase = np.linspace(0, 1.0, n)
        probe_notail = TunedProbeResponse.from_quality_factor(
            f0_hz=1e6, quality_factor=150.0, oversample=4, tail=TailSpec(factor=0.0)
        )
        probe_tail = TunedProbeResponse.from_quality_factor(
            f0_hz=1e6, quality_factor=150.0, oversample=4, tail=TailSpec(factor=6.0)
        )
        v_notail = float(make_grape_objective(
            model, rf_response=probe_notail, **self._base_kwargs(n, dt)
        )(phase)[0])
        v_tail = float(make_grape_objective(
            model, rf_response=probe_tail, **self._base_kwargs(n, dt)
        )(phase)[0])
        self.assertNotAlmostEqual(v_notail, v_tail, places=4)

    def test_grape_recovers_fidelity_under_band_limit(self) -> None:
        # A band-limited probe hurts a rectangular pulse; GRAPE optimizing the
        # command (bounded) against the delivered pulse recovers fidelity.
        model = _model(offset_hz=1500.0)
        n, dt = 24, 1e-5
        probe = TunedProbeResponse.from_quality_factor(
            f0_hz=1e6, quality_factor=120.0, oversample=4, tail=TailSpec(factor=5.0)
        )
        rect = np.zeros(n)
        base_kwargs = dict(
            dt=dt, target=_PSI_DOWN, psi0=_PSI_UP, fixed_amplitude=6000.0,
            rf_response=probe,
        )
        vg = make_grape_objective(model, n_segments=n, **base_kwargs)
        rect_fid = float(vg(rect)[0])
        result = grape_optimize(
            model, rect, **base_kwargs,
            scipy_options={"maxiter": 200},
        )
        self.assertGreater(result.best_fidelity, rect_fid + 0.05)

    def test_non_uniform_dt_rejected_with_response(self) -> None:
        model = _model()
        n = 6
        dt = np.array([1e-5, 2e-5, 1e-5, 1e-5, 1e-5, 1e-5])
        probe = TunedProbeResponse(tau=1e-5)
        with self.assertRaises(ValueError):
            make_grape_objective(
                model, n_segments=n, dt=dt, target=_PSI_DOWN, psi0=_PSI_UP,
                fixed_amplitude=3000.0, rf_response=probe,
            )

    def test_gradient_response_without_channel_rejected(self) -> None:
        model = _model()
        gd = GradientDriverResponse(tau_rl=2e-6)
        with self.assertRaises(ValueError):
            make_grape_objective(
                model, n_segments=6, dt=1e-5, target=_PSI_DOWN, psi0=_PSI_UP,
                fixed_amplitude=3000.0, gradient_response=gd,
            )


if __name__ == "__main__":
    unittest.main()
