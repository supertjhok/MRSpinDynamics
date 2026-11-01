"""The receiver noise used for probe SNR and for injected probe noise must agree.

``spin_dynamics.noise.{tuned,untuned}_probe_output_noise_density`` feed the
probe noise model, the field-referred detectors and imaging noise, while
``tuned_probe_rx``/``untuned_probe_rx`` compute the probe SNR. Both now come
from one receiver model that includes the damping resistor's ``4kT/Rd`` noise
(review item L-32). Checked against each other and against the Nyquist/Bode
theorem (node noise ``4kT Re(Z_node)`` when only thermal resistors are present).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.noise import (
    tuned_probe_output_noise_density,
    untuned_probe_output_noise_density,
)
from spin_dynamics.parameters import set_params_tuned_orig, set_params_untuned_orig
from spin_dynamics.probes.tuned import tuned_probe_rx
from spin_dynamics.probes.untuned import untuned_probe_rx

K_B = 1.380649e-23


def _matched_filter_snr(mrx, pnoise, del_w, f):
    mf = np.conj(mrx)
    mf = mf / np.sqrt(trapezoid(np.abs(mf) ** 2, del_w))
    vsig = trapezoid(mrx * mf, del_w)
    return float(np.real(vsig) / np.sqrt(trapezoid(pnoise * np.abs(mf) ** 2, f)))


@pytest.mark.parametrize(
    ("setter", "density", "receiver"),
    [
        (set_params_tuned_orig, tuned_probe_output_noise_density, tuned_probe_rx),
        (set_params_untuned_orig, untuned_probe_output_noise_density, untuned_probe_rx),
    ],
)
@pytest.mark.parametrize("rd", [None, 1.0e4])
def test_noise_density_matches_receiver_snr_noise(setter, density, receiver, rd) -> None:
    _params, sp, pp = setter(numpts=31)
    sp = replace(sp, numpts=31, del_w=np.linspace(-5.0, 5.0, 31), mf_type=1)
    if rd is not None:
        sp = replace(sp, Rd=rd)
    macq = np.exp(-0.1 * sp.del_w**2).astype(np.complex128)
    mrx, snr, _ = receiver(sp, pp, macq)
    pnoise, f = density(sp, pp)
    assert snr == pytest.approx(_matched_filter_snr(mrx, pnoise, sp.del_w, f), rel=1e-12)


def _base_sp(**extra):
    sp = {
        "k": K_B,
        "T": 300.0,
        "L": 1.0e-6,
        "R": 0.2,
        "Cin": 5e-12,
        "Rin": 1e30,  # amplifier input treated as noiseless (vn = in = 0)
        "Rd": 1.0e3,
        "vn": 0.0,
        "in": 0.0,
        "mf_type": 1,
        "w0": 2 * np.pi * 1.0e6,
        "del_w": np.linspace(-5.0, 5.0, 41),
    }
    sp.update(extra)
    return sp


def test_tuned_noise_density_obeys_nyquist_theorem() -> None:
    sp = _base_sp(C=1.0 / ((2 * np.pi * 1.0e6) ** 2 * 1.0e-6))
    pp = {"T_90": 25e-6}
    pnoise, f = tuned_probe_output_noise_density(sp, pp)
    s = 2j * np.pi * f
    y_node = s * sp["C"] + 1 / sp["Rd"] + 1 / (s * sp["L"] + sp["R"]) + s * sp["Cin"]
    np.testing.assert_allclose(pnoise, 4 * K_B * sp["T"] * np.real(1 / y_node), rtol=1e-10)


def test_untuned_noise_density_obeys_nyquist_theorem() -> None:
    # Ideal-transformer limit, where the untuned model is an exact passive network.
    sp = _base_sp(C=0.0, Rdup=0.0, Nrx=3.0, krx=1.0, L1=2e-6, R1=0.0)
    pp = {"T_90": 25e-6}
    pnoise, f = untuned_probe_output_noise_density(sp, pp)
    s = 2j * np.pi * f
    nv = sp["krx"] * sp["Nrx"] * sp["L1"] / (sp["L"] + sp["L1"])
    y_node = 1 / ((s * sp["L"] + sp["R"]) * nv**2) + s * sp["Cin"] + 1 / sp["Rd"]
    np.testing.assert_allclose(pnoise, 4 * K_B * sp["T"] * np.real(1 / y_node), rtol=1e-10)
