"""Tests for spin_dynamics.flow (Phase A: kinematics and washout)."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.flow import (
    FlowModel,
    apply_washout,
    flow_enhanced_rate,
    washout_density,
    washout_fraction,
)


def _flow(regime="plug", q=1e-6, r=5e-3, length=20e-3):
    return FlowModel(
        volumetric_flow_rate=q, pipe_radius=r, sensitive_length=length, regime=regime
    )


class TestFlowKinematics:
    def test_derived_quantities(self) -> None:
        f = _flow(q=1e-6, r=5e-3, length=20e-3)
        area = np.pi * 5e-3**2
        assert f.cross_section == pytest.approx(area)
        assert f.sensitive_volume == pytest.approx(area * 20e-3)
        assert f.mean_velocity == pytest.approx(1e-6 / area)
        assert f.mean_residence_time == pytest.approx(area * 20e-3 / 1e-6)
        # tau = L / v_mean too.
        assert f.mean_residence_time == pytest.approx(20e-3 / f.mean_velocity)

    def test_centerline_velocity(self) -> None:
        assert _flow("plug").centerline_velocity() == pytest.approx(_flow("plug").mean_velocity)
        lam = _flow("laminar")
        assert lam.centerline_velocity() == pytest.approx(2.0 * lam.mean_velocity)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            FlowModel(volumetric_flow_rate=-1.0, pipe_radius=1e-3, sensitive_length=1e-2)
        with pytest.raises(ValueError):
            FlowModel(volumetric_flow_rate=1e-6, pipe_radius=1e-3, sensitive_length=1e-2,
                      regime="turbulent")


class TestWashout:
    def test_plug_is_linear_with_cutoff(self) -> None:
        f = _flow("plug")
        tau = f.mean_residence_time
        t = np.linspace(0.0, 1.5 * tau, 151)
        w = washout_fraction(f, t)
        # W(0)=1, W(tau)=0, linear in between, zero past tau.
        assert w[0] == pytest.approx(1.0)
        assert washout_fraction(f, np.array([tau]))[0] == pytest.approx(0.0, abs=1e-12)
        mid = washout_fraction(f, np.array([0.5 * tau]))[0]
        assert mid == pytest.approx(0.5)
        assert np.all(w[t > tau] == 0.0)

    def test_laminar_matches_derived_piecewise(self) -> None:
        f = _flow("laminar")
        tau = f.mean_residence_time
        # t <= tau/2: 1 - t/tau; continuous at tau/2 (=0.5); tail tau/(4t).
        assert washout_fraction(f, np.array([0.25 * tau]))[0] == pytest.approx(0.75)
        assert washout_fraction(f, np.array([0.5 * tau]))[0] == pytest.approx(0.5)
        assert washout_fraction(f, np.array([tau]))[0] == pytest.approx(0.25)
        assert washout_fraction(f, np.array([2.0 * tau]))[0] == pytest.approx(0.125)

    def test_both_regimes_share_initial_slope(self) -> None:
        tau = _flow("plug").mean_residence_time
        dt = 1e-4 * tau
        for regime in ("plug", "laminar"):
            f = _flow(regime)
            slope = (washout_fraction(f, np.array([dt]))[0] - 1.0) / dt
            assert slope == pytest.approx(-1.0 / tau, rel=1e-3)

    def test_density_is_negative_derivative_of_fraction(self) -> None:
        for regime in ("plug", "laminar"):
            f = _flow(regime)
            tau = f.mean_residence_time
            t = np.linspace(1e-4 * tau, 3.0 * tau, 4000)
            w = washout_fraction(f, t)
            e = washout_density(f, t)
            num = -np.gradient(w, t)
            # Compare away from the kink points.
            interior = (t > 0.1 * tau) & (np.abs(t - tau / 2) > 0.05 * tau) & (t < 0.95 * tau)
            np.testing.assert_allclose(e[interior], num[interior], rtol=0.02, atol=1e-3)

    def test_density_integral_matches_fraction_identity(self) -> None:
        # Exact identity: int_0^T E(t) dt = W(0) - W(T) = 1 - W(T).
        for regime in ("plug", "laminar"):
            f = _flow(regime)
            tau = f.mean_residence_time
            for horizon in (5.0, 50.0):
                t = np.linspace(0.0, horizon * tau, 200_000)
                integral = np.trapezoid(washout_density(f, t), t)
                expected = 1.0 - washout_fraction(f, np.array([horizon * tau]))[0]
                assert integral == pytest.approx(expected, abs=2e-3)

    def test_monte_carlo_cross_check_laminar(self) -> None:
        # Independent check: sample spins uniformly in the pipe volume, assign
        # the parabolic velocity, and count those still inside at each time.
        f = _flow("laminar")
        tau = f.mean_residence_time
        rng = np.random.default_rng(0)
        n = 400_000
        # Uniform in area: u = (r/R)^2 uniform on [0,1]; uniform axial x in [0,L].
        u = rng.uniform(0.0, 1.0, n)
        x = rng.uniform(0.0, f.sensitive_length, n)
        v = 2.0 * f.mean_velocity * (1.0 - u)
        exit_time = (f.sensitive_length - x) / v
        t = np.linspace(0.0, 3.0 * tau, 25)
        w_mc = np.array([(exit_time > ti).mean() for ti in t])
        w_analytic = washout_fraction(f, t)
        np.testing.assert_allclose(w_mc, w_analytic, atol=3e-3)

    def test_monte_carlo_cross_check_plug(self) -> None:
        f = _flow("plug")
        tau = f.mean_residence_time
        rng = np.random.default_rng(1)
        n = 400_000
        x = rng.uniform(0.0, f.sensitive_length, n)
        exit_time = (f.sensitive_length - x) / f.mean_velocity
        t = np.linspace(0.0, 1.2 * tau, 25)
        w_mc = np.array([(exit_time > ti).mean() for ti in t])
        np.testing.assert_allclose(w_mc, washout_fraction(f, t), atol=3e-3)


class TestApplyWashout:
    def test_apply_to_exponential_decay(self) -> None:
        f = _flow("plug")
        tau = f.mean_residence_time
        t2 = tau  # comparable relaxation and flow scales
        t = np.linspace(0.0, 0.8 * tau, 200)
        clean = np.exp(-t / t2)
        flowing = apply_washout(f, t, clean)
        np.testing.assert_allclose(flowing, clean * (1.0 - t / tau))
        # Flow always lowers the signal (extra washout).
        assert np.all(flowing <= clean + 1e-15)

    def test_fast_flow_dominates_slow_flow_negligible(self) -> None:
        t2 = 0.5
        t = np.linspace(0.0, 0.2, 200)
        clean = np.exp(-t / t2)
        fast = apply_washout(_flow("plug", q=1e-5), t, clean)  # short tau
        slow = apply_washout(_flow("plug", q=1e-9), t, clean)  # tau >> window
        # Fast flow strips much more signal than slow flow.
        assert np.trapezoid(fast, t) < np.trapezoid(slow, t)
        np.testing.assert_allclose(slow, clean, rtol=5e-2)

    def test_apply_washout_shape_mismatch(self) -> None:
        f = _flow()
        with pytest.raises(ValueError):
            apply_washout(f, np.linspace(0, 1, 10), np.ones(9))


class TestFlowEnhancedRate:
    def test_adds_reciprocal_residence(self) -> None:
        f = _flow("plug")
        t2 = 0.1
        assert flow_enhanced_rate(f, t2) == pytest.approx(1.0 / t2 + 1.0 / f.mean_residence_time)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            flow_enhanced_rate(_flow(), -1.0)
