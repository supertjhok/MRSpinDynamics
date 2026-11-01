"""Independent ODE check of the untuned-probe analytic coil current.

``untuned_probe_lp`` propagates the current of a series L-(R+Rs) loop driven by
``amp * cos(w t + phi)`` segment by segment:

    L dI/dt + (R + Rs) I = amp cos(w t + phi).

These tests integrate that ODE numerically on the solver's own time grid. In
particular a leading segment that rounds to zero RF steps (e.g. ``tfp = 0`` in
``calc_rot_axis_untuned_probe_lp``) must not change the drive phase of the
segment that follows it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.probes.untuned import untuned_probe_lp

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - scipy is a core dependency
    solve_ivp = None

F0 = 1.0e6
L_COIL = 1.0e-6
R_COIL = 1.0
N_STEPS = 32


def _probe(durations, phases, amps, rs_values):
    sp = {"L": L_COIL, "R": R_COIL, "w0": 2.0 * np.pi * F0}
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
    w = pp["w"]
    step = 2.0 * np.pi / (w * pp["N"])
    counts = np.rint(pp["tref"] * pp["N"] * w / (2.0 * np.pi)).astype(int)
    state = np.zeros(1)
    start = 0.0
    samples = []
    for count, phase, amp, rs in zip(counts, pp["pref"], pp["aref"], pp["Rsref"]):
        if count <= 0:
            continue

        def rhs(t, y, phase=phase, amp=amp, rs=rs):
            return [(amp * np.cos(w * t + phase) - (sp["R"] + rs) * y[0]) / sp["L"]]

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
class UntunedProbeTransientTests(unittest.TestCase):
    CASES = {
        "zero_length_leading_segment": (
            [0.0, 10e-6, 5e-6],
            [0.0, np.pi / 2, 0.0],
            [0.0, 1.0, 0.0],
            [50.0, 50.0, 50.0],
        ),
        "sub_step_leading_segment": (
            [1e-12, 10e-6],
            [0.3, 1.1],
            [1.0, 1.0],
            [5.0, 5.0],
        ),
        "regular_train": (
            [2e-6, 8e-6, 4e-6, 6e-6],
            [0.0, 0.7, 0.0, -1.2],
            [0.0, 1.0, 0.0, 0.5],
            [5.0, 5.0, 50.0, 5.0],
        ),
    }

    def test_untuned_probe_lp_matches_ode(self) -> None:
        for name, case in self.CASES.items():
            with self.subTest(case=name):
                sp, pp = _probe(*case)
                _t_avg, _icr_avg, _t_raw, current = untuned_probe_lp(sp, pp)
                reference = _ode_reference(sp, pp)
                scale = np.max(np.abs(reference))
                self.assertEqual(current.size, reference.size)
                self.assertLess(np.max(np.abs(current.imag)) / scale, 1e-9)
                self.assertLess(np.max(np.abs(current.real - reference)) / scale, 1e-7)


if __name__ == "__main__":
    unittest.main()
