"""Independent ODE check of the tuned-probe analytic transient solution.

``tuned_probe_lp`` / ``tuned_probe_lp_orig`` propagate the coil current of a
parallel-tuned tank driven through a source resistance ``Rs`` with a
piecewise closed-form solution. In normalized time ``tau = t / sqrt(L C)`` the
coil current obeys

    I'' + 2 gamma I' + wn^2 I = (amp / Rs) cos(w tau + phi),

with ``gamma = (R/Z0 + Z0/Rs)/2``, ``wn^2 = 1 + R/Rs`` and ``Z0 = sqrt(L/C)``.
These tests integrate that ODE numerically (segment by segment, so each drive
discontinuity is a fresh initial-value problem) and require the analytic
current to agree. The MATLAB Version 3 port carried an extra ``1/(2 alpha)``
on one homogeneous coefficient, which made the current non-zero at pulse
start and left a ~50 % transient error; these tolerances reject it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.probes.tuned import tuned_probe_lp, tuned_probe_lp_orig

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - scipy is a core dependency
    solve_ivp = None


F0 = 1.0e6
L_COIL = 1.0e-6
C_TANK = 1.0 / ((2.0 * np.pi * F0) ** 2 * L_COIL)
R_COIL = 2.0 * np.pi * F0 * L_COIL / 50.0  # unloaded Q = 50
N_STEPS = 32


def _probe(rs_values, amps, phases, durations):
    sp = {"L": L_COIL, "R": R_COIL, "C": C_TANK, "w0": 2.0 * np.pi * F0}
    pp = {
        "tref": np.asarray(durations, dtype=np.float64),
        "pref": np.asarray(phases, dtype=np.float64),
        "aref": np.asarray(amps, dtype=np.float64),
        "Rsref": np.asarray(rs_values, dtype=np.float64),
        "w": 2.0 * np.pi * F0,
        "N": N_STEPS,
    }
    return sp, pp


def _ode_reference(sp, pp):
    """Integrate the tank ODE on the analytic solver's quantized time grid."""

    wp = 1.0 / np.sqrt(sp["L"] * sp["C"])
    z0 = np.sqrt(sp["L"] / sp["C"])
    w = pp["w"] / wp
    n = pp["N"]
    step = 2.0 * np.pi / (w * n)
    counts = np.rint(wp * pp["tref"] * n * w / (2.0 * np.pi)).astype(int)

    state = np.zeros(2)
    start = 0.0
    samples = []
    for count, phase, amp, rs in zip(counts, pp["pref"], pp["aref"], pp["Rsref"]):
        if count <= 0:
            continue
        gamma = 0.5 * (sp["R"] / z0 + z0 / rs)
        wn2 = 1.0 + sp["R"] / rs

        def rhs(tau, y, gamma=gamma, wn2=wn2, amp=amp, rs=rs, phase=phase):
            drive = (amp / rs) * np.cos(w * tau + phase)
            return [y[1], drive - 2.0 * gamma * y[1] - wn2 * y[0]]

        times = start + step * np.arange(1, count + 1)
        solution = solve_ivp(
            rhs,
            (start, times[-1]),
            state,
            t_eval=times,
            method="DOP853",
            rtol=1e-11,
            atol=1e-15,
        )
        samples.append(solution.y[0])
        state = solution.y[:, -1]
        start = times[-1]
    return np.concatenate(samples)


@unittest.skipIf(solve_ivp is None, "scipy is required for the ODE reference")
class TunedProbeTransientTests(unittest.TestCase):
    # Two drive segments (with a phase step and an amplitude change), a
    # lightly damped ringdown, and an over-damped (real alpha) Q-switch.
    CASES = {
        "underdamped_pulse_and_ringdown": (
            [50.0, 50.0, 2.0e3],
            [1.0, 0.5, 0.0],
            [0.0, np.pi / 2, 0.0],
            [10e-6, 5e-6, 10e-6],
        ),
        "overdamped_drive_and_qswitch": (
            [1.0, 1.0],
            [1.0, 0.0],
            [0.3, 0.0],
            [8e-6, 4e-6],
        ),
    }

    def _check(self, func):
        for name, case in self.CASES.items():
            with self.subTest(case=name):
                sp, pp = _probe(*case)
                _t_avg, _icr_avg, _t_raw, current = func(sp, pp)
                reference = _ode_reference(sp, pp)
                scale = np.max(np.abs(reference))
                self.assertEqual(current.size, reference.size)
                # The coil current is a real waveform.
                self.assertLess(np.max(np.abs(current.imag)) / scale, 1e-9)
                error = np.max(np.abs(current.real - reference)) / scale
                self.assertLess(error, 1e-7)
                # The current starts from rest: no jump at the first sample.
                first = np.abs(current[: N_STEPS // 4]) / scale
                ref_first = np.abs(reference[: N_STEPS // 4]) / scale
                np.testing.assert_allclose(first, ref_first, atol=1e-8)

    def test_tuned_probe_lp_orig_matches_ode(self) -> None:
        self._check(tuned_probe_lp_orig)

    def test_tuned_probe_lp_matches_ode(self) -> None:
        self._check(tuned_probe_lp)

    def test_rotating_frame_average_matches_ode(self) -> None:
        sp, pp = _probe(*self.CASES["underdamped_pulse_and_ringdown"])
        _t_avg, icr_avg, t_raw, _current = tuned_probe_lp_orig(sp, pp)
        reference = _ode_reference(sp, pp)
        demod = reference * np.exp(-1j * sp["w0"] * t_raw)
        windows = icr_avg.size * N_STEPS
        expected = demod[:windows].reshape(-1, N_STEPS).mean(axis=1)
        scale = np.max(np.abs(expected))
        self.assertLess(np.max(np.abs(icr_avg - expected)) / scale, 1e-7)


if __name__ == "__main__":
    unittest.main()
