"""Tests for the SQUID detector and the ideal-Faraday crossover baseline.

Validates the Clarke 2007 physics in the field-referred framework: a flat SQUID
floor vs a ``1/f`` Faraday floor, their crossover, and the prepolarized detected
SNR scaling (``B0^-1/2`` for the SQUID, ``B0^+1/2`` for the coil).
"""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.detection import (
    IdealFaradayCoil,
    SQUIDMagnetometer,
    detected_field_snr,
)

GAMMA_H = 42.5774e6  # Hz/T


def _line(b0, *, frac=0.01, t2=1.0, n=4001, span=20.0):
    f0 = GAMMA_H * b0
    hwhm = max(frac * f0, 1.0 / (np.pi * t2))
    fg = np.linspace(f0 - span * hwhm, f0 + span * hwhm, n)
    s = (hwhm / np.pi) / ((fg - f0) ** 2 + hwhm**2)  # unit area
    return fg, s


def test_squid_floor_is_flat_at_configured_value():
    squid = SQUIDMagnetometer.berkeley_2007()
    f = np.array([10.0, 5.6e3, 1.0e6, 1.0e7])
    n = squid.field_noise_amplitude(f)
    assert np.allclose(n, 1.0e-15)


def test_squid_one_over_f_knee_raises_at_low_frequency():
    squid = SQUIDMagnetometer(field_noise_T_per_rtHz=1e-15, one_over_f_knee_hz=100.0)
    f = np.array([1.0, 10.0, 100.0, 1000.0, 1.0e4])
    n2 = squid.field_noise_psd(f)
    # N^2 = S0^2 (1 + fk/f): monotonically decreasing toward high f, flat above fk.
    assert np.all(np.diff(n2) < 0)
    assert n2[2] == pytest.approx((1e-15) ** 2 * 2.0)  # at f = fk, factor 2


def test_squid_projected_is_lower_than_baseline():
    assert (
        SQUIDMagnetometer.projected().field_noise_T_per_rtHz
        < SQUIDMagnetometer.berkeley_2007().field_noise_T_per_rtHz
    )


def test_faraday_noise_scales_as_one_over_f():
    coil = IdealFaradayCoil(field_noise_T_per_rtHz_ref=1e-15, f_ref_hz=1e6)
    assert coil.field_noise_amplitude(np.array([1e6]))[0] == pytest.approx(1e-15)
    # one decade down in frequency -> one decade up in field noise
    lo = coil.field_noise_amplitude(np.array([1e5]))[0]
    assert lo == pytest.approx(1e-14, rel=1e-12)


def test_noise_floor_crossover_at_reference():
    # Faraday anchored to equal the SQUID floor at 1 MHz -> crossover there.
    squid = SQUIDMagnetometer(field_noise_T_per_rtHz=1e-15)
    coil = IdealFaradayCoil(field_noise_T_per_rtHz_ref=1e-15, f_ref_hz=1e6)
    f = np.logspace(3, 8, 20000)
    cross = f[np.argmin(np.abs(squid.field_noise_amplitude(f) - coil.field_noise_amplitude(f)))]
    assert cross == pytest.approx(1e6, rel=0.05)
    # SQUID quieter below crossover, coil quieter above.
    assert squid.field_noise_amplitude(np.array([1e4]))[0] < coil.field_noise_amplitude(np.array([1e4]))[0]
    assert squid.field_noise_amplitude(np.array([1e7]))[0] > coil.field_noise_amplitude(np.array([1e7]))[0]


def test_prepolarized_snr_scalings_match_clarke():
    squid = SQUIDMagnetometer.berkeley_2007()
    coil = IdealFaradayCoil(field_noise_T_per_rtHz_ref=1e-15, f_ref_hz=1e6)
    b0 = np.array([1.8e-3, 1.8e-4, 1.8e-5])
    snr_sq, snr_fa = [], []
    for value in b0:
        fg, s = _line(value)
        snr_sq.append(detected_field_snr(s, fg, squid).snr)
        snr_fa.append(detected_field_snr(s, fg, coil).snr)
    p_sq = np.polyfit(np.log(b0), np.log(snr_sq), 1)[0]
    p_fa = np.polyfit(np.log(b0), np.log(snr_fa), 1)[0]
    assert p_sq == pytest.approx(-0.5, abs=0.05)
    assert p_fa == pytest.approx(+0.5, abs=0.05)


def test_prepolarized_squid_snr_beats_coil_below_crossover():
    squid = SQUIDMagnetometer.berkeley_2007()
    coil = IdealFaradayCoil(field_noise_T_per_rtHz_ref=1e-15, f_ref_hz=1e6)
    fg, s = _line(1.32e-4)  # 132 uT, 5.6 kHz -- well below the 1 MHz crossover
    assert detected_field_snr(s, fg, squid).snr > detected_field_snr(s, fg, coil).snr


def test_squid_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SQUIDMagnetometer(field_noise_T_per_rtHz=0.0)
    with pytest.raises(ValueError):
        SQUIDMagnetometer(field_noise_T_per_rtHz=1e-15, one_over_f_knee_hz=-1.0)
    with pytest.raises(ValueError):
        IdealFaradayCoil(field_noise_T_per_rtHz_ref=1e-15, f_ref_hz=0.0)
