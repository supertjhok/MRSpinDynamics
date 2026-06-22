"""Analytic physics regression tests for constant-gradient CPMG diffusion.

These tests give the diffusion kernel a real reference value (the textbook
Carr-Purcell free-diffusion law) rather than only checking that outputs stay
finite. They need no MATLAB fixtures: for a constant background gradient the
exact attenuation of the n-th echo is

    A(N) = exp(-(1/12) * gamma**2 * G**2 * D * t_E**3 * N)

where ``t_E`` is the free-evolution time per echo (echo spacing minus the
finite refocusing-pulse width) and N is the echo number.
"""

from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.core.kernels import sim_spin_dynamics_arb10_diffusion
from spin_dynamics.core.rotations import calc_rotation_matrix


GAMMA = 2.675e8  # rad/s/T (proton)
T90 = 100e-6     # s


def _on_resonance_cpmg_echoes(
    num_echoes: int,
    echo_spacing_seconds: float,
    gradient: float,
    diffusion_coefficient: float,
) -> np.ndarray:
    """Return |M| at each echo center for an ideal on-resonance CPMG train."""

    del_w = np.array([0.0])
    time_scale = 2.0 * T90 / np.pi
    exc = calc_rotation_matrix(
        del_w, np.ones_like(del_w), np.array([np.pi / 2]),
        np.array([np.pi / 2]), np.array([1.0]),
    )
    ref = calc_rotation_matrix(
        del_w, np.ones_like(del_w), np.array([np.pi]),
        np.array([0.0]), np.array([1.0]),
    )
    # Free half-echo time in normalized units, excluding the 180 pulse width.
    tfp = (np.pi / 2) * (echo_spacing_seconds - 2 * T90) / (2 * T90)
    tp = [np.pi / 2] + list(np.tile([tfp, np.pi, tfp], num_echoes))
    pul = [1] + list(np.tile([1, 2, 1], num_echoes))
    amp = [1.0] + list(np.tile([0.0, 1.0, 0.0], num_echoes))
    acq = [0] + list(np.tile([0, 0, 1], num_echoes))
    params = dict(
        tp=np.array(tp, float), pul=np.array(pul, int), amp=np.array(amp, float),
        acq=np.array(acq, int), grad=np.zeros(len(tp)), Rtot=[exc, ref],
        del_w=del_w, del_wg=np.zeros_like(del_w),
        T1n=np.array([np.inf]), T2n=np.array([np.inf]),
        m0=np.array([1.0 + 0j]), mth=np.array([1.0 + 0j]),
        gamma=GAMMA, gradient=gradient, diffusion_coefficient=diffusion_coefficient,
        time_scale=time_scale,
    )
    return np.abs(sim_spin_dynamics_arb10_diffusion(params)[:, 0])


class DiffusionPhysicsTests(unittest.TestCase):
    def test_matches_textbook_carr_purcell_law(self) -> None:
        gradient, diffusion = 0.05, 2.3e-9
        for num_echoes in (1, 2, 4, 8):
            for echo_spacing in (0.4e-3, 0.8e-3):
                echoes = _on_resonance_cpmg_echoes(
                    num_echoes, echo_spacing, gradient, diffusion
                )
                t_e_free = echo_spacing - 2 * T90
                expected = np.exp(
                    -(1.0 / 12.0) * GAMMA**2 * gradient**2 * diffusion
                    * t_e_free**3 * np.arange(1, num_echoes + 1)
                )
                np.testing.assert_allclose(echoes, expected, rtol=1e-9, atol=1e-12)

    def test_attenuation_scales_as_gradient_squared(self) -> None:
        diffusion, echo_spacing = 2.3e-9, 1.0e-3
        a1 = _on_resonance_cpmg_echoes(1, echo_spacing, 0.05, diffusion)[0]
        a2 = _on_resonance_cpmg_echoes(1, echo_spacing, 0.10, diffusion)[0]
        # ln A ~ G^2, so doubling G multiplies the exponent by 4.
        self.assertAlmostEqual(np.log(a2) / np.log(a1), 4.0, places=6)

    def test_attenuation_increases_with_echo_spacing_and_diffusion(self) -> None:
        base = _on_resonance_cpmg_echoes(4, 0.6e-3, 0.05, 2.3e-9)[-1]
        longer_te = _on_resonance_cpmg_echoes(4, 1.0e-3, 0.05, 2.3e-9)[-1]
        faster_d = _on_resonance_cpmg_echoes(4, 0.6e-3, 0.05, 5.0e-9)[-1]
        self.assertLess(longer_te, base)   # longer echo spacing -> more loss
        self.assertLess(faster_d, base)    # larger D -> more loss

    def test_zero_diffusion_is_lossless(self) -> None:
        echoes = _on_resonance_cpmg_echoes(5, 0.8e-3, 0.05, 0.0)
        np.testing.assert_allclose(echoes, np.ones_like(echoes), rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
