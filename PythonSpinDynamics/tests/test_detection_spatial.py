"""Tests for wiring a gradiometer pickup into a field-sensor detector.

Covers the spatial detected-SNR path: a sample couples through the pickup while
ambient sources couple through the *same* pickup, so a gradiometer rejects a
uniform ambient field (Clarke's ~1000x common-mode rejection) at a small
sample-coupling cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.detection import (
    AmbientFieldSource,
    Gradiometer,
    OPMMagnetometer,
    SQUIDMagnetometer,
    detected_field_snr_spatial,
    pickup_signal_spectrum,
    spatial_field_noise_psd,
)

R, B = 0.025, 0.05


def _mag():
    return Gradiometer.magnetometer(radius_m=R)


def _grad():
    return Gradiometer.second_order_axial(radius_m=R, baseline_m=B)


def _line(freqs):
    return (5.0 / np.pi) / (freqs**2 + 5.0**2)


def test_detectors_accept_pickup_and_reject_bad_pickup():
    squid = SQUIDMagnetometer.berkeley_2007(pickup=_grad())
    assert isinstance(squid.pickup, Gradiometer)
    opm = OPMMagnetometer.rf(423e3, pickup=_mag())
    assert isinstance(opm.pickup, Gradiometer)
    with pytest.raises(ValueError):
        SQUIDMagnetometer(field_noise_T_per_rtHz=1e-15, pickup="not a gradiometer")


def test_uniform_coupling_magnetometer_one_gradiometer_zero():
    assert _mag().uniform_coupling() == pytest.approx(1.0)
    assert _grad().uniform_coupling() == pytest.approx(0.0, abs=1e-12)


def test_normalized_sensitivity_is_one_at_reference():
    g = _grad()
    assert g.normalized_sensitivity(np.array([g.reference_point]))[0] == pytest.approx(1.0)


def test_pickup_signal_spectrum_is_linear_combination():
    freqs = np.linspace(-100, 100, 51)
    pu = _mag()
    positions = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.03]])
    moments = np.array([_line(freqs), 2.0 * _line(freqs)])
    s = pickup_signal_spectrum(pu, positions, moments)
    g = pu.normalized_sensitivity(positions)
    expected = g[0] * _line(freqs) + g[1] * 2.0 * _line(freqs)
    assert np.allclose(s, expected)


def test_uniform_ambient_rejected_by_gradiometer_not_magnetometer():
    freqs = np.linspace(-500, 500, 1001)
    sample_pos = np.array([[0.0, 0.0, 0.01]])
    sample_m = _line(freqs)[None, :]
    ambient = AmbientFieldSource(psd=np.full(freqs.size, (1e-12) ** 2))  # 1 pT/rtHz uniform

    squid_mag = SQUIDMagnetometer.berkeley_2007(pickup=_mag())
    squid_grad = SQUIDMagnetometer.berkeley_2007(pickup=_grad())

    mag_only = detected_field_snr_spatial(squid_mag, sample_pos, sample_m, freqs).snr
    mag_amb = detected_field_snr_spatial(squid_mag, sample_pos, sample_m, freqs, ambient_sources=[ambient]).snr
    grad_only = detected_field_snr_spatial(squid_grad, sample_pos, sample_m, freqs).snr
    grad_amb = detected_field_snr_spatial(squid_grad, sample_pos, sample_m, freqs, ambient_sources=[ambient]).snr

    # magnetometer is crushed by the ambient field (~1 pT / 1 fT = 1000x);
    # the balanced gradiometer is essentially unaffected.
    assert mag_only / mag_amb == pytest.approx(1000.0, rel=0.1)
    assert grad_only / grad_amb == pytest.approx(1.0, rel=1e-6)
    # under ambient, the gradiometer wins by a large margin
    assert grad_amb > 100.0 * mag_amb


def test_localized_ambient_coupling_scales_with_position():
    freqs = np.linspace(-50, 50, 21)
    pu = _grad()
    squid = SQUIDMagnetometer.berkeley_2007(pickup=pu)
    near = AmbientFieldSource(psd=np.ones(freqs.size) * 1e-30, position_m=(0.0, 0.0, 0.02))
    far = AmbientFieldSource(psd=np.ones(freqs.size) * 1e-30, position_m=(0.0, 0.0, 1.0))
    n2_near = spatial_field_noise_psd(squid, freqs, [near])
    n2_far = spatial_field_noise_psd(squid, freqs, [far])
    # a nearer source couples more strongly -> larger added noise
    assert np.all(n2_near >= n2_far)
    assert n2_near.mean() > n2_far.mean()


def test_spatial_requires_pickup():
    freqs = np.linspace(-50, 50, 21)
    squid = SQUIDMagnetometer.berkeley_2007()  # no pickup
    with pytest.raises(ValueError):
        detected_field_snr_spatial(squid, np.array([[0.0, 0.0, 0.01]]), _line(freqs)[None, :], freqs)
