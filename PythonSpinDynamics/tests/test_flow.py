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


class TestTransitPolarization:
    def test_plug_matches_prepolarized_flow_state(self) -> None:
        from spin_dynamics.flow import inflow_polarization, mean_transit_time
        from spin_dynamics.prepolarization import prepolarized_flow_state

        f = _flow("plug", q=2e-7, r=5e-3, length=20e-3)
        l_pre = 0.1
        b_pol, b_det, t1 = 0.5, 0.05, 2.0
        mz = inflow_polarization(
            f, polarizing_field_tesla=b_pol, detection_field_tesla=b_det,
            prepolarizer_length_meters=l_pre, t1_seconds=t1,
        )
        ref = prepolarized_flow_state(
            b_pol, b_det, l_pre, f.mean_velocity, t1,
        )
        assert mz == pytest.approx(float(ref.m0), rel=1e-9)
        # Sanity: transit time = L_pre / v_mean.
        assert mean_transit_time(f, l_pre) == pytest.approx(l_pre / f.mean_velocity)

    def test_transit_distribution_normalized_and_mean(self) -> None:
        from spin_dynamics.flow import mean_transit_time, transit_time_distribution

        f = _flow("laminar")
        l_pre = 0.1
        tau = mean_transit_time(f, l_pre)
        t = np.linspace(0.0, 300.0 * tau, 300_000)
        e = transit_time_distribution(f, l_pre, t)
        assert np.trapezoid(e, t) == pytest.approx(1.0, abs=3e-3)
        # Laminar exit-age RTD has finite mean = tau (converges, unlike washout).
        assert np.trapezoid(t * e, t) == pytest.approx(tau, rel=1e-2)

    def test_laminar_polarization_between_limits(self) -> None:
        from spin_dynamics.flow import inflow_polarization

        f = _flow("laminar", q=2e-7)
        kw = dict(
            polarizing_field_tesla=0.5, detection_field_tesla=0.05,
            prepolarizer_length_meters=0.1, t1_seconds=2.0,
        )
        mz = inflow_polarization(f, **kw)
        # Full polarizer equilibrium in detection units = B_pol/B_det = 10.
        assert 0.0 < mz < 10.0

    def test_slower_flow_polarizes_more(self) -> None:
        from spin_dynamics.flow import inflow_polarization

        kw = dict(
            polarizing_field_tesla=0.5, detection_field_tesla=0.05,
            prepolarizer_length_meters=0.1, t1_seconds=2.0,
        )
        fast = inflow_polarization(_flow("laminar", q=2e-6), **kw)
        slow = inflow_polarization(_flow("laminar", q=2e-8), **kw)
        assert slow > fast  # longer transit -> more build-up

    def test_slow_flow_approaches_full_equilibrium(self) -> None:
        from spin_dynamics.flow import inflow_polarization

        # tau_pre >> T1 -> spins reach the polarizing-field equilibrium (=10).
        mz = inflow_polarization(
            _flow("laminar", q=2e-10),
            polarizing_field_tesla=0.5, detection_field_tesla=0.05,
            prepolarizer_length_meters=0.1, t1_seconds=0.5,
        )
        assert mz == pytest.approx(10.0, rel=1e-3)

    def test_monte_carlo_flux_weighted_cross_check(self) -> None:
        from spin_dynamics.flow import inflow_polarization
        from spin_dynamics.prepolarization import longitudinal_recovery

        f = _flow("laminar", q=5e-7)
        l_pre, b_pol, b_det, t1 = 0.08, 0.5, 0.05, 1.5
        analytic = inflow_polarization(
            f, polarizing_field_tesla=b_pol, detection_field_tesla=b_det,
            prepolarizer_length_meters=l_pre, t1_seconds=t1,
        )
        rng = np.random.default_rng(3)
        n = 2_000_000
        # Flux weighting: probability of a spin ~ v(r) * area. Sample u=(r/R)^2
        # uniform (area), then accept with prob proportional to v ~ (1-u).
        u = rng.uniform(0.0, 1.0, n)
        keep = rng.uniform(0.0, 1.0, n) < (1.0 - u)  # thin by velocity
        u = u[keep]
        transit = (l_pre / (2.0 * f.mean_velocity)) / (1.0 - u)
        eq = 10.0  # b_pol/b_det
        mz = longitudinal_recovery(0.0, eq, transit, t1)
        assert mz.mean() == pytest.approx(analytic, rel=5e-3)
