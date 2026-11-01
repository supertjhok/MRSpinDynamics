from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.core.kernels import (
    _matrix_elements_power,
    sim_spin_dynamics_arb10_radiation_damping,
)
from spin_dynamics.core.rotations import rf_matrix_elements
from spin_dynamics.parameters import set_params_matched_orig, set_params_tuned_orig
from spin_dynamics.radiation_damping import (
    MU0,
    RadiationDampingProbe,
    RadiationDampingSpec,
    analytic_radiation_damping_envelope,
    initial_state_from_flip_angle,
    hyperpolarized_proton_sample,
    normalized_radiation_damping_weights,
    proton_thermal_magnetization_density,
    radiation_damping_probe_from_matched,
    radiation_damping_probe_from_tuned,
    radiation_damping_time,
    simulate_nmr_maser,
    simulate_radiation_damping,
    simulate_radiation_damping_fid,
    water_proton_sample,
)
from spin_dynamics.workflows import (
    acquisition,
    run_matched_cpmg_train,
    run_radiation_damping_fid,
    run_tuned_cpmg_train,
)


class RadiationDampingTests(unittest.TestCase):
    def test_radiation_damping_time_matches_textbook_formula(self) -> None:
        gamma = 2.0
        eta = 0.5
        mth = 3.0
        q = 4.0
        expected = 2.0 / (gamma * MU0 * eta * mth * q)
        self.assertAlmostEqual(radiation_damping_time(gamma, eta, mth, q), expected)

    def test_probe_constructors_use_existing_probe_q(self) -> None:
        _params, tuned_sp, _tuned_pp = set_params_tuned_orig(numpts=9)
        matched_sp, _matched_pp = set_params_matched_orig(numpts=9)
        tuned = radiation_damping_probe_from_tuned(
            tuned_sp,
            fill_factor=0.6,
            equilibrium_magnetization=0.01,
        )
        matched = radiation_damping_probe_from_matched(
            matched_sp,
            fill_factor=0.6,
            equilibrium_magnetization=0.01,
        )
        self.assertEqual(tuned.name, "tuned")
        self.assertEqual(matched.name, "matched")
        self.assertAlmostEqual(tuned.q, tuned_sp.Q)
        self.assertAlmostEqual(matched.q, matched_sp.Q)
        self.assertGreater(tuned.trd, 0.0)
        self.assertGreater(matched.resonator_time_constant, 0.0)

    def test_initial_state_from_x_flip_points_along_minus_y(self) -> None:
        mxy, mz = initial_state_from_flip_angle(np.pi / 2)
        np.testing.assert_allclose(mxy, -1j)
        self.assertAlmostEqual(mz, 0.0, places=15)

    def test_instant_fid_matches_analytic_envelope(self) -> None:
        probe = RadiationDampingProbe(
            gamma=1.0,
            omega0=100.0,
            q=10.0,
            fill_factor=0.5,
            equilibrium_magnetization=1.0,
        )
        time = np.linspace(0.0, 3.0 * probe.trd, 501)
        theta = np.deg2rad(45.0)
        result = simulate_radiation_damping_fid(time, probe, flip_angle=theta)
        expected = analytic_radiation_damping_envelope(time, theta, probe.trd)
        np.testing.assert_allclose(result.envelope, expected, rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(
            result.envelope**2 + result.mz**2,
            np.ones_like(time),
            rtol=2e-5,
            atol=2e-7,
        )

    def test_damping_is_independent_of_transverse_phase(self) -> None:
        # The Bloom-form feedback (-mxy/Trd) damps every transverse azimuth at
        # the same rate. The textbook Eq. 10.29/10.30 conjugate form would
        # anti-damp the x quadrature and drive mz downward for x-phase
        # magnetization; this test pins the phase-independent convention.
        probe = RadiationDampingProbe(
            gamma=1.0,
            omega0=100.0,
            q=10.0,
            fill_factor=0.5,
            equilibrium_magnetization=1.0,
        )
        time = np.linspace(0.0, 3.0 * probe.trd, 501)
        theta = np.deg2rad(45.0)
        reference = simulate_radiation_damping_fid(time, probe, flip_angle=theta)
        for pulse_phase in (np.pi / 4, np.pi / 2, np.pi, 5.0):
            rotated = simulate_radiation_damping_fid(
                time, probe, flip_angle=theta, pulse_phase=pulse_phase
            )
            np.testing.assert_allclose(
                rotated.envelope, reference.envelope, rtol=1e-10, atol=1e-12
            )
            np.testing.assert_allclose(
                rotated.mz, reference.mz, rtol=1e-10, atol=1e-12
            )
            # Energy flows spins -> circuit: mz recovers monotonically.
            self.assertTrue(np.all(np.diff(rotated.mz) > 0.0))

    def test_circuit_model_lags_instant_feedback_for_high_q_probe(self) -> None:
        probe = RadiationDampingProbe(
            gamma=1e6,
            omega0=1e5,
            q=50.0,
            fill_factor=0.7,
            equilibrium_magnetization=1.0,
        )
        time = np.linspace(0.0, 0.2 * probe.trd, 101)
        circuit = simulate_radiation_damping_fid(time, probe, model="circuit")
        instant = simulate_radiation_damping_fid(time, probe, model="instant")
        self.assertLess(circuit.envelope[-1], circuit.envelope[0])
        self.assertGreater(circuit.envelope[-1], instant.envelope[-1])

    def test_proton_magnetization_density_scales_with_field(self) -> None:
        low = proton_thermal_magnetization_density(1.0)
        high = proton_thermal_magnetization_density(2.0)
        self.assertGreater(low, 0.0)
        self.assertAlmostEqual(high / low, 2.0)

    def test_sample_presets_scale_magnetization(self) -> None:
        water = water_proton_sample(1.0)
        boosted = hyperpolarized_proton_sample(1.0, polarization_scale=100.0)
        self.assertEqual(water.name, "water protons")
        self.assertGreater(water.equilibrium_magnetization, 0.0)
        self.assertAlmostEqual(
            boosted.equilibrium_magnetization / water.equilibrium_magnetization,
            100.0,
        )

    def test_normalized_radiation_damping_weights_use_sensitivity_squared(self) -> None:
        density = np.array([1.0, 1.0, 2.0])
        sensitivity = np.array([1.0, 2.0, 0.5])
        weights = normalized_radiation_damping_weights(density, sensitivity)
        expected = density * np.abs(sensitivity) ** 2
        expected = expected / np.sum(expected)
        np.testing.assert_allclose(weights, expected)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_tuned_cpmg_train_accepts_radiation_damping_mapping(self) -> None:
        clean = run_tuned_cpmg_train(
            numpts=9,
            num_echoes=2,
            rephase_action="ignore",
        )
        damped = run_tuned_cpmg_train(
            numpts=9,
            num_echoes=2,
            rephase_action="ignore",
            radiation_damping={
                "fill_factor": 0.7,
                "equilibrium_magnetization": 0.8,
            },
        )
        self.assertIsNotNone(damped.radiation_damping)
        self.assertEqual(damped.mrx.shape, clean.mrx.shape)
        self.assertGreater(damped.radiation_damping.trd, 0.0)
        self.assertGreater(
            float(np.max(np.abs(damped.mrx - clean.mrx))),
            0.0,
        )

    def test_tuned_cpmg_train_can_apply_radiation_damping_during_pulses(self) -> None:
        common = {
            "numpts": 9,
            "num_echoes": 2,
            "rephase_action": "ignore",
            "radiation_damping": {
                "fill_factor": 0.7,
                "equilibrium_magnetization": 0.8,
            },
        }
        free_only = run_tuned_cpmg_train(**common)
        during_pulses = run_tuned_cpmg_train(
            **{
                **common,
                "radiation_damping": {
                    **common["radiation_damping"],
                    "apply_during_pulses": True,
                },
            }
        )
        self.assertIsNotNone(during_pulses.radiation_damping)
        self.assertTrue(during_pulses.radiation_damping.apply_during_pulses)
        self.assertEqual(during_pulses.mrx.shape, free_only.mrx.shape)
        self.assertGreater(
            float(np.max(np.abs(during_pulses.mrx - free_only.mrx))),
            0.0,
        )

    def test_radiation_damping_fid_workflow_matches_analytic_envelope(self) -> None:
        result = run_radiation_damping_fid(
            probe="matched",
            equilibrium_magnetization=0.8,
            fill_factor=0.7,
            flip_angle=np.deg2rad(60.0),
            num_points=301,
        )
        self.assertEqual(result.time_seconds.shape, result.mxy.shape)
        self.assertGreater(result.probe.trd, 0.0)
        np.testing.assert_allclose(
            result.envelope,
            result.analytic_envelope,
            rtol=3e-5,
            atol=3e-7,
        )

    def test_radiation_damping_fid_workflow_supports_tuned_probe(self) -> None:
        result = run_radiation_damping_fid(
            probe="tuned",
            equilibrium_magnetization=0.8,
            fill_factor=0.7,
            flip_angle=np.deg2rad(45.0),
            num_points=31,
        )
        self.assertEqual(result.probe.name, "tuned")
        self.assertEqual(result.normalized_time.shape, result.time_seconds.shape)
        self.assertGreater(result.envelope[0], result.envelope[-1])

    def test_radiation_damping_fid_workflow_exposes_circuit_detuning(self) -> None:
        common = {
            "probe": "matched",
            "equilibrium_magnetization": 0.8,
            "fill_factor": 0.7,
            "flip_angle": np.deg2rad(60.0),
            "model": "circuit",
            "num_points": 81,
        }
        tuned = run_radiation_damping_fid(**common)
        detuned = run_radiation_damping_fid(
            **{
                **common,
                "phase": 0.25,
                "detuning": 1.0e5,
            }
        )
        self.assertAlmostEqual(detuned.probe.phase, 0.25)
        self.assertAlmostEqual(detuned.probe.detuning, 1.0e5)
        self.assertGreater(
            float(np.max(np.abs(detuned.mxy - tuned.mxy))),
            0.0,
        )

    def test_nmr_maser_threshold_growth(self) -> None:
        probe = RadiationDampingProbe(
            gamma=1.0,
            omega0=100.0,
            q=10.0,
            fill_factor=0.5,
            equilibrium_magnetization=1.0,
        )
        time = np.linspace(0.0, 20.0 * probe.trd, 501)
        t2 = 10.0 * probe.trd
        t1 = 10.0 * probe.trd
        below = simulate_nmr_maser(
            time,
            probe,
            seed_mxy=-1e-5j,
            initial_mz=-0.05,
            pump_mz=-0.05,
            t1=t1,
            t2=t2,
            model="instant",
        )
        above = simulate_nmr_maser(
            time,
            probe,
            seed_mxy=-1e-5j,
            initial_mz=-0.8,
            pump_mz=-0.8,
            t1=t1,
            t2=t2,
            model="instant",
        )
        self.assertLess(below.envelope[-1], below.envelope[0])
        self.assertGreater(above.envelope[-1], 1e3 * above.envelope[0])
        self.assertGreater(above.mz[-1], -0.8)


class RadiationDampingConventionTests(unittest.TestCase):
    """M-05 (drive sign) and M-06 (time_scale direction) conventions."""

    def test_documented_drive_reproduces_package_pulse_phase_convention(self) -> None:
        gamma = 2 * np.pi * 42.58e6
        # Negligible feedback: fill factor and magnetization far below RD regime.
        probe = RadiationDampingProbe(
            gamma=gamma,
            omega0=2 * np.pi * 400e6,
            q=1e-6,
            fill_factor=1e-6,
            equilibrium_magnetization=1e-9,
        )
        omega1 = 2 * np.pi * 10e3
        t90 = 0.5 * np.pi / omega1
        for phase in (0.0, 0.7, np.pi / 2):
            result = simulate_radiation_damping(
                np.linspace(0.0, t90, 5),
                probe,
                initial_mxy=0.0,
                initial_mz=1.0,
                drive=-(omega1 / gamma) * np.exp(1j * phase),
                max_step=t90 / 400,
            )
            mxy, mz = initial_state_from_flip_angle(np.pi / 2, pulse_phase=phase)
            self.assertAlmostEqual(complex(result.mxy[-1]), complex(mxy), places=8)
            self.assertAlmostEqual(float(result.mz[-1]), float(mz), places=8)
            # Same state as the core pulse matrix (M+ = R_p0 * Mz).
            core = rf_matrix_elements(np.array([0.0]), 1.0, np.pi / 2, phase)
            self.assertAlmostEqual(complex(result.mxy[-1]), complex(core.R_p0[0]), places=8)

    def test_spec_time_scale_is_normalized_units_per_second(self) -> None:
        probe = RadiationDampingProbe(
            gamma=2 * np.pi * 42.58e6,
            omega0=2 * np.pi * 400e6,
            q=50.0,
            fill_factor=0.5,
            equilibrium_magnetization=1.0,
        )
        t90 = 25e-6
        spec = RadiationDampingSpec.from_t90(probe, t90, apply_during_pulses=True)
        self.assertAlmostEqual(spec.time_scale, (np.pi / 2) / t90)
        # Trd in normalized units equals Trd / T90 * pi/2 (90-degree pulse = pi/2).
        self.assertAlmostEqual(spec.trd, probe.trd / t90 * (np.pi / 2))
        self.assertTrue(spec.apply_during_pulses)
        for bad in (0.0, -1.0, np.inf):
            with self.assertRaises(ValueError):
                RadiationDampingSpec(probe=probe, time_scale=bad)


def _reference_pulsed_rd_bloch(
    del_w: np.ndarray,
    pulses: list[tuple[float, float, float]],
    trd: float,
    dt: float,
) -> np.ndarray:
    """Independent Cartesian RK4 of Bloch + instant RD feedback during pulses.

    Package convention: dM/dt = Omega x M with Omega = (w1 cos phi, w1 sin phi,
    del_w) (a phase-0 pulse tips +z to -y; M+ precesses as exp(+i del_w t)),
    plus the Bloom feedback F = -<M+>/Trd acting as dM+/dt += Mz F and
    dMz/dt -= Re(conj(M+) F). Returns M- = Mx - i My after all pulses.
    """

    m = np.zeros((3, del_w.size))
    m[2] = 1.0

    def rhs(state: np.ndarray, w1: float, phi: float) -> np.ndarray:
        omega = np.vstack(
            [
                np.full(del_w.size, w1 * np.cos(phi)),
                np.full(del_w.size, w1 * np.sin(phi)),
                del_w,
            ]
        )
        deriv = np.cross(omega.T, state.T).T
        feedback = -np.mean(state[0] + 1j * state[1]) / trd
        deriv[0] += state[2] * feedback.real
        deriv[1] += state[2] * feedback.imag
        deriv[2] -= np.real((state[0] - 1j * state[1]) * feedback)
        return deriv

    for duration, w1, phi in pulses:
        steps = int(np.ceil(duration / dt))
        h = duration / steps
        for _ in range(steps):
            k1 = rhs(m, w1, phi)
            k2 = rhs(m + 0.5 * h * k1, w1, phi)
            k3 = rhs(m + 0.5 * h * k2, w1, phi)
            k4 = rhs(m + h * k3, w1, phi)
            m = m + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return m[0] - 1j * m[1]


def _reference_arb10_rd_bloch(params: dict, spec: RadiationDampingSpec, dt: float) -> np.ndarray:
    """Independent RK4 of a whole arb10 sequence with instant RD feedback.

    Pulses are integrated from their ``pulse_segments`` generators (no
    relaxation, as in the pulse matrices); free intervals add offset
    precession and T1/T2 relaxation. Negative (rewind) durations integrate
    the same ODE backward. Returns M- at each acquisition, like the kernel.
    """

    del_w0 = np.asarray(params["del_w"], dtype=np.float64)
    del_wg = np.asarray(params["del_wg"], dtype=np.float64)
    n = del_w0.size
    t1 = np.asarray(params["T1n"], dtype=np.float64)
    t2 = np.asarray(params["T2n"], dtype=np.float64)
    r1 = np.where(np.isfinite(t1), 1.0 / t1, 0.0)
    r2 = np.where(np.isfinite(t2), 1.0 / t2, 0.0)
    mth = np.real(np.asarray(params["mth"], dtype=np.complex128)) * np.ones(n)
    weights = (
        np.full(n, 1.0 / n)
        if spec.weights is None
        else np.asarray(spec.weights, dtype=np.float64) / np.sum(spec.weights)
    )
    probe_phase = np.exp(1j * spec.probe.phase)
    m = np.zeros((3, n))
    m[2] = np.real(np.asarray(params["m0"], dtype=np.complex128))

    def rhs(state: np.ndarray, omega: np.ndarray, relax: bool) -> np.ndarray:
        deriv = np.cross(omega.T, state.T).T
        feedback = -probe_phase * np.sum(weights * (state[0] + 1j * state[1])) / spec.trd
        deriv[0] += state[2] * feedback.real
        deriv[1] += state[2] * feedback.imag
        deriv[2] -= np.real((state[0] - 1j * state[1]) * feedback)
        if relax:
            deriv[0] -= r2 * state[0]
            deriv[1] -= r2 * state[1]
            deriv[2] += r1 * (mth - state[2])
        return deriv

    def integrate(state: np.ndarray, duration: float, omega: np.ndarray, relax: bool) -> np.ndarray:
        if duration == 0.0:
            return state
        steps = max(1, int(np.ceil(abs(duration) / dt)))
        h = duration / steps
        for _ in range(steps):
            k1 = rhs(state, omega, relax)
            k2 = rhs(state + 0.5 * h * k1, omega, relax)
            k3 = rhs(state + 0.5 * h * k2, omega, relax)
            k4 = rhs(state + h * k3, omega, relax)
            state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return state

    acquired = []
    for tp, pul, amp, acq, grad in zip(
        params["tp"], params["pul"], params["amp"], params["acq"], params["grad"]
    ):
        dw = del_w0 + grad * del_wg
        if amp > 0:
            segment = params["pulse_segments"][int(pul) - 1]
            assert segment.get("post_rotation") is None
            w1 = np.asarray(segment["w_1"], dtype=np.float64) * np.ones(n)
            for tk, pk, ak in zip(segment["tp"], segment["phi"], segment["amp"]):
                omega = np.vstack([ak * w1 * np.cos(pk), ak * w1 * np.sin(pk), dw])
                m = integrate(m, float(tk), omega, relax=False)
        else:
            m = integrate(m, float(tp), np.vstack([np.zeros(n), np.zeros(n), dw]), relax=True)
        if acq:
            acquired.append(m[0] - 1j * m[1])
    return np.asarray(acquired)


class RadiationDampingDuringPulsesTests(unittest.TestCase):
    """Regression: RD during pulses must follow the physical in-pulse path.

    The earlier eigen-decomposition matrix root mapped the half step of an
    on-resonance pi pulse to Mz -> i*Mz (not a rotation), so a single 180
    matrix gave M- = -0.587+0.793j where the explicit sub-pulse limit is
    0.778+0.553j.
    """

    def setUp(self) -> None:
        self.probe = RadiationDampingProbe(
            gamma=2 * np.pi * 42.58e6,
            omega0=2 * np.pi * 400e6,
            q=50.0,
            fill_factor=0.5,
            equilibrium_magnetization=1e-3,
        )
        self.del_w = np.linspace(-0.2, 0.2, 5)

    def _params(self, segments: bool) -> dict:
        n = self.del_w.size
        params = {
            "tp": [np.pi / 2, np.pi],
            "pul": [1, 2],
            "Rtot": [
                rf_matrix_elements(self.del_w, 1.0, np.pi / 2, 0.0),
                rf_matrix_elements(self.del_w, 1.0, np.pi, np.pi / 2),
            ],
            "amp": [1.0, 1.0],
            "acq": [0, 1],
            "grad": [0.0, 0.0],
            "del_w": self.del_w,
            "del_wg": np.zeros(n),
            "T1n": np.full(n, np.inf),
            "T2n": np.full(n, np.inf),
            "m0": np.ones(n),
            "mth": np.ones(n),
        }
        if segments:
            params["pulse_segments"] = [
                {"tp": [np.pi / 2], "phi": [0.0], "amp": [1.0]},
                {"tp": [np.pi], "phi": [np.pi / 2], "amp": [1.0]},
            ]
        return params

    def _spec(self, max_step: float | None = None) -> RadiationDampingSpec:
        # Trd = 2 normalized time units: strong feedback inside a pi pulse.
        return RadiationDampingSpec(
            probe=self.probe,
            time_scale=2.0 / self.probe.trd,
            apply_during_pulses=True,
            max_step=max_step,
        )

    def test_single_pi_matrix_matches_independent_bloch_rd_reference(self) -> None:
        spec = self._spec(max_step=2e-3)
        got = sim_spin_dynamics_arb10_radiation_damping(self._params(True), spec)[0]
        expected = _reference_pulsed_rd_bloch(
            self.del_w,
            [(np.pi / 2, 1.0, 0.0), (np.pi, 1.0, np.pi / 2)],
            trd=spec.trd,
            dt=1e-3,
        )
        np.testing.assert_allclose(got, expected, atol=2e-3)

    def test_single_pi_matrix_matches_explicit_subpulse_limit(self) -> None:
        got = sim_spin_dynamics_arb10_radiation_damping(self._params(True), self._spec())[0]
        self.assertAlmostEqual(complex(got[0]).real, 0.778, places=2)
        self.assertAlmostEqual(complex(got[0]).imag, 0.553, places=2)

    def test_matrix_root_fallback_refuses_ambiguous_pi_rotation(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "no pulse_segments"):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                sim_spin_dynamics_arb10_radiation_damping(self._params(False), self._spec())

    def test_matrix_power_of_rotation_is_rotation_by_scaled_angle(self) -> None:
        # Offsets small enough that every net angle stays below pi, where the
        # shortest-path root is the physical one.
        del_w = np.linspace(-0.01, 0.01, 5)
        for angle in (0.9 * np.pi, np.pi - 1e-3, 0.3):
            mat = rf_matrix_elements(del_w, 1.0, angle, 0.7)
            half = _matrix_elements_power(mat, 0.5)
            ref = rf_matrix_elements(del_w, 1.0, 0.5 * angle, 0.7)
            for name in ("R_00", "R_0p", "R_p0", "R_pp", "R_pm", "R_mm"):
                np.testing.assert_allclose(
                    getattr(half, name), getattr(ref, name), atol=1e-9, err_msg=name
                )

    def test_cpmg_trains_during_pulses_match_independent_bloch_reference(self) -> None:
        """Workflow-level check: tuned/matched CPMG pass exact pulse generators.

        The workflow's kernel call is intercepted; its acquired M- must match an
        independent RK4 Bloch + RD integration of the same shaped pulses. The
        pre-fix shortest-path matrix root (still the no-segment fallback) is
        off by about 0.2 here, the generator path by under 1e-3.
        """

        real_kernel = acquisition.sim_spin_dynamics_arb10_radiation_damping
        for runner in (run_tuned_cpmg_train, run_matched_cpmg_train):
            with self.subTest(runner=runner.__name__):
                captured = []

                def spy(params, spec):
                    out = real_kernel(params, spec)
                    captured.append((params, spec, out))
                    return out

                with mock.patch.object(
                    acquisition, "sim_spin_dynamics_arb10_radiation_damping", spy
                ), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    runner(
                        numpts=7,
                        num_echoes=2,
                        rephase_action="ignore",
                        radiation_damping={
                            "fill_factor": 0.7,
                            # Trd ~ 1.3 normalized units: strong in-pulse feedback.
                            "equilibrium_magnetization": 8.0,
                            "apply_during_pulses": True,
                            "max_step": 5e-3,
                        },
                    )
                self.assertFalse(
                    [w for w in caught if "pulse_segments" in str(w.message)]
                )
                params, spec, out = captured[0]
                self.assertIn("pulse_segments", params)
                expected = _reference_arb10_rd_bloch(params, spec, dt=2e-3)
                np.testing.assert_allclose(out, expected, atol=2e-3)

                fallback = {k: v for k, v in params.items() if k != "pulse_segments"}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    shortest = real_kernel(fallback, spec)
                self.assertGreater(float(np.max(np.abs(shortest - expected))), 0.05)

    def test_tuned_cpmg_phase_kick_pulses_carry_post_rotation(self) -> None:
        """Longitudinal phase kicks travel as zero-duration ``post_rotation``.

        The kernel recomposes segments plus kick and raises on any mismatch
        with ``Rtot``, so a clean run with no fallback warning checks the
        generator bookkeeping for absolute-phase refocusing matrices.
        """

        captured = []
        real_kernel = acquisition.sim_spin_dynamics_arb10_radiation_damping

        def spy(params, spec):
            captured.append(params)
            return real_kernel(params, spec)

        with mock.patch.object(
            acquisition, "sim_spin_dynamics_arb10_radiation_damping", spy
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_tuned_cpmg_train(
                numpts=7,
                num_echoes=2,
                rephase_action="ignore",
                radiation_damping={
                    "fill_factor": 0.7,
                    "equilibrium_magnetization": 0.8,
                    "apply_during_pulses": True,
                },
                absolute_phase={
                    "rf_frequency_hz": 0.25 / 200e-6,
                    "transient_model": {
                        "kind": "longitudinal_phase_kick",
                        "phase_amplitude_rad": 0.3,
                    },
                },
            )
        self.assertFalse([w for w in caught if "pulse_segments" in str(w.message)])
        self.assertTrue(np.all(np.isfinite(result.mrx)))
        kicks = [
            entry["post_rotation"]
            for entry in captured[0]["pulse_segments"]
            if entry["post_rotation"] is not None
        ]
        self.assertTrue(kicks)

    def test_matrix_power_raises_for_on_resonance_pi_pulse(self) -> None:
        mat = rf_matrix_elements(np.array([0.0]), 1.0, np.pi, 0.0)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _matrix_elements_power(mat, 0.5)


if __name__ == "__main__":
    unittest.main()
