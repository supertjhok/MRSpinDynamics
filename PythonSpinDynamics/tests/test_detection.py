"""Tests for the field-referred detector layer (``spin_dynamics.detection``).

The key guarantees this increment must hold:

* the field-referred SNR is invariant to the volts<->field reference, and
* an ``InductiveCoilDetector`` reproduces ``estimate_matched_filter_snr``'s
  ``predicted_snr`` bit-for-bit, so the new layer changes no existing numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.detection import (
    InductiveCoilDetector,
    detected_field_snr,
    field_referred_from_output,
)
from spin_dynamics.noise import estimate_matched_filter_snr


def _synthetic_signal(n: int = 129):
    """A Lorentzian-ish complex line plus a smooth positive noise density."""

    f = np.linspace(-5.0e3, 5.0e3, n)
    clean = 1.0 / (1.0 + 1j * (f - 300.0) / 400.0)
    pnoise = 1.0e-3 * (1.0 + (f / 4.0e3) ** 2)  # frequency-dependent, positive
    return f, clean, pnoise


def _predicted_snr(clean, pnoise, freqs):
    result = estimate_matched_filter_snr(
        clean,
        clean[None, :],
        pnoise=pnoise,
        frequencies=freqs,
        offsets=freqs,
    )
    return result.predicted_snr


def test_inductive_matches_predicted_snr_bit_for_bit():
    f, clean, pnoise = _synthetic_signal()
    det = InductiveCoilDetector.from_output_density(f, pnoise)  # flat H -> field == output
    got = detected_field_snr(clean, f, det).snr
    expected = _predicted_snr(clean, pnoise, f)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_snr_equals_closed_form_integral():
    f, clean, pnoise = _synthetic_signal()
    det = InductiveCoilDetector.from_output_density(f, pnoise)
    got = detected_field_snr(clean, f, det).snr
    closed = np.sqrt(np.trapezoid(np.abs(clean) ** 2 / pnoise, f))
    assert got == pytest.approx(float(closed), rel=1e-12)


def test_reference_invariance_volts_to_field():
    # A non-trivial transfer H folds signal and noise together; the field-
    # referred SNR must be identical whether we work in volts (clean_out, pnoise)
    # or in field (S = clean_out / H, N^2 = pnoise / |H|^2).
    f, clean_out, pnoise = _synthetic_signal()
    h = 0.5 + 0.3j + 0.2j * (f / 5.0e3)  # arbitrary finite non-zero transfer
    det = InductiveCoilDetector(freqs_hz=f, output_noise_psd=pnoise, rx_transfer=h)
    s_field = clean_out / h
    got = detected_field_snr(s_field, f, det).snr
    expected = _predicted_snr(clean_out, pnoise, f)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_field_referred_from_output_divides_by_h_squared():
    out = np.array([1.0, 2.0, 4.0])
    h = np.array([1.0, 2.0, 0.5])
    n2 = field_referred_from_output(out, h)
    assert np.allclose(n2, out / np.abs(h) ** 2)


def test_field_referred_from_output_rejects_zero_transfer():
    with pytest.raises(ValueError):
        field_referred_from_output([1.0, 2.0], [1.0, 0.0])


def test_transfer_interpolates_off_grid():
    f = np.array([0.0, 1.0, 2.0])
    h = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    det = InductiveCoilDetector(freqs_hz=f, output_noise_psd=np.ones_like(f), rx_transfer=h)
    mid = det.transfer(np.array([0.5, 1.5]))
    assert np.allclose(mid, [1.5, 2.5])


def test_field_noise_psd_uses_transfer():
    f = np.array([0.0, 1.0, 2.0])
    out = np.array([4.0, 4.0, 4.0])
    h = np.array([2.0 + 0j, 2.0 + 0j, 2.0 + 0j])
    det = InductiveCoilDetector(freqs_hz=f, output_noise_psd=out, rx_transfer=h)
    assert np.allclose(det.field_noise_psd(f), out / 4.0)


def test_df_rectangular_matches_uniform_sum():
    f = np.linspace(0.0, 10.0, 11)
    clean = np.ones_like(f, dtype=complex)
    pnoise = np.full_like(f, 2.0)
    det = InductiveCoilDetector.from_output_density(f, pnoise)
    got = detected_field_snr(clean, f, det, df=1.0).snr
    assert got == pytest.approx(np.sqrt(np.sum(np.abs(clean) ** 2 / pnoise) * 1.0))


def test_tuned_classmethod_wraps_noise_density():
    # Minimal circuit params exercising the tuned_probe_output_noise_density path.
    del_w = np.linspace(-3.0, 3.0, 65)
    sp = {
        "k": 1.380649e-23,
        "T": 300.0,
        "L": 1.0e-6,
        "R": 0.5,
        "C": 1.0e-10,
        "Cin": 1.0e-12,
        "Rin": 1.0e6,
        "Rd": 1.0e4,
        "vn": 1.0e-9,
        "in_": 1.0e-12,
        "w0": 2 * np.pi * 5.0e6,
        "del_w": del_w,
    }
    pp = {"T_90": 5.0e-6}
    det = InductiveCoilDetector.tuned(sp, pp)
    assert det.name == "tuned coil"
    n2 = det.field_noise_psd(det._f)
    assert np.all(np.isfinite(n2)) and np.all(n2 > 0)


def test_rejects_nonpositive_noise():
    f = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        InductiveCoilDetector.from_output_density(f, np.array([1.0, 0.0, 1.0]))


def test_detected_field_snr_validates_length():
    f, clean, pnoise = _synthetic_signal()
    det = InductiveCoilDetector.from_output_density(f, pnoise)
    with pytest.raises(ValueError):
        detected_field_snr(clean[:-1], f, det)
