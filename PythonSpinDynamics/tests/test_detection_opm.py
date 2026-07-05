"""Tests for the OPM (atomic magnetometer) detector.

Covers the two operating modes and, crucially, the atomic-bandwidth roll-off:
the RF/Mx magnetometer is quiet only within its bandwidth of the tuned carrier,
and the SERF (zero-field) magnetometer is rolled off -- useless -- at MHz NQR.
"""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.detection import (
    OPMMagnetometer,
    SQUIDMagnetometer,
    detected_field_snr,
)


def _line(fc, hwhm, *, n=6001, span=40.0):
    fg = np.linspace(fc - span * hwhm, fc + span * hwhm, n)
    s = (hwhm / np.pi) / ((fg - fc) ** 2 + hwhm**2)
    return fg, s


def test_rf_opm_floor_on_resonance():
    fc = 423.0e3
    opm = OPMMagnetometer.rf(fc, field_noise_T_per_rtHz=0.24e-15, bandwidth_hz=300.0)
    assert opm.field_noise_amplitude(np.array([fc]))[0] == pytest.approx(0.24e-15)


def test_rf_opm_rolls_off_away_from_carrier():
    fc = 423.0e3
    bw = 300.0
    opm = OPMMagnetometer.rf(fc, field_noise_T_per_rtHz=0.24e-15, bandwidth_hz=bw)
    # one bandwidth off resonance -> N^2 doubles (factor sqrt(2) in amplitude)
    off = opm.field_noise_amplitude(np.array([fc + bw]))[0]
    assert off == pytest.approx(0.24e-15 * np.sqrt(2.0))


def test_serf_floor_at_dc_and_rolloff_at_mhz():
    serf = OPMMagnetometer.serf(field_noise_T_per_rtHz=0.16e-15, bandwidth_hz=200.0)
    assert serf.field_noise_amplitude(np.array([0.0]))[0] == pytest.approx(0.16e-15)
    # at 423 kHz the SERF field noise is ~ f/df worse than at DC
    penalty = (
        serf.field_noise_amplitude(np.array([423e3]))[0]
        / serf.field_noise_amplitude(np.array([0.0]))[0]
    )
    assert penalty == pytest.approx(np.sqrt(1.0 + (423e3 / 200.0) ** 2), rel=1e-6)
    assert penalty > 1000.0


def test_rf_opm_beats_serf_and_matches_regime_on_mhz_line():
    fc = 423.0e3
    fg, s = _line(fc, 100.0)
    rf = OPMMagnetometer.rf(fc, field_noise_T_per_rtHz=0.24e-15, bandwidth_hz=300.0)
    serf = OPMMagnetometer.serf(field_noise_T_per_rtHz=0.16e-15, bandwidth_hz=200.0)
    squid = SQUIDMagnetometer.berkeley_2007()
    snr_rf = detected_field_snr(s, fg, rf).snr
    snr_serf = detected_field_snr(s, fg, serf).snr
    snr_squid = detected_field_snr(s, fg, squid).snr
    # RF-OPM (0.24 fT) beats the 1 fT SQUID; SERF is far worse at MHz.
    assert snr_rf > snr_squid > snr_serf
    assert snr_rf / snr_serf > 100.0


def test_snr_degrades_when_line_exceeds_opm_bandwidth():
    fc = 423.0e3
    bw = 300.0
    rf = OPMMagnetometer.rf(fc, field_noise_T_per_rtHz=0.24e-15, bandwidth_hz=bw)
    fg_narrow, s_narrow = _line(fc, 0.1 * bw)
    fg_wide, s_wide = _line(fc, 10.0 * bw)
    snr_narrow = detected_field_snr(s_narrow, fg_narrow, rf).snr
    snr_wide = detected_field_snr(s_wide, fg_wide, rf).snr
    assert snr_narrow > snr_wide


def test_transfer_is_normalized_lorentzian():
    fc = 1.0e6
    bw = 500.0
    rf = OPMMagnetometer.rf(fc, bandwidth_hz=bw)
    assert rf.transfer(np.array([fc]))[0] == pytest.approx(1.0 + 0j)
    # |H|^2 = 1/2 one bandwidth off resonance
    h_off = rf.transfer(np.array([fc + bw]))[0]
    assert abs(h_off) ** 2 == pytest.approx(0.5)


def test_opm_rejects_bad_parameters():
    with pytest.raises(ValueError):
        OPMMagnetometer.rf(0.0)  # rf needs positive carrier
    with pytest.raises(ValueError):
        OPMMagnetometer(field_noise_T_per_rtHz=1e-15, bandwidth_hz=0.0)
    with pytest.raises(ValueError):
        OPMMagnetometer(field_noise_T_per_rtHz=1e-15, bandwidth_hz=100.0, mode="bogus")
