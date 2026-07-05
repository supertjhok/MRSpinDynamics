"""Tests for gradiometer pickup geometry and its reciprocal sensitivity.

Validates the Clarke 2007 gradiometer physics: order-n balancing nulls a uniform
field (and lower gradients), the distant-source sensitivity falls as 1/r^{3+n},
and a nearby sample still couples like a magnetometer.
"""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.detection import Gradiometer

R = 0.025
B = 0.05


def _axial_on_axis(g, z):
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=-1)
    return np.abs(g.axial_sensitivity(pts))


def _far_slope(g, r1=20.0):
    s1 = _axial_on_axis(g, np.array([r1]))[0]
    s2 = _axial_on_axis(g, np.array([2.0 * r1]))[0]
    return np.log(s1 / s2) / np.log(2.0)


def test_preset_orders():
    assert Gradiometer.magnetometer(radius_m=R).order == 0
    assert Gradiometer.first_order_axial(radius_m=R, baseline_m=B).order == 1
    assert Gradiometer.second_order_axial(radius_m=R, baseline_m=B).order == 2


def test_uniform_field_nulled_by_gradiometers():
    mag = Gradiometer.magnetometer(radius_m=R)
    g1 = Gradiometer.first_order_axial(radius_m=R, baseline_m=B)
    g2 = Gradiometer.second_order_axial(radius_m=R, baseline_m=B)
    assert abs(mag.uniform_field_response()) == pytest.approx(np.pi * R * R)
    assert g1.uniform_field_response() == pytest.approx(0.0, abs=1e-18)
    assert g2.uniform_field_response() == pytest.approx(0.0, abs=1e-18)


def test_far_field_falloff_exponents():
    assert _far_slope(Gradiometer.magnetometer(radius_m=R)) == pytest.approx(3.0, abs=0.02)
    assert _far_slope(Gradiometer.first_order_axial(radius_m=R, baseline_m=B)) == pytest.approx(4.0, abs=0.05)
    assert _far_slope(Gradiometer.second_order_axial(radius_m=R, baseline_m=B)) == pytest.approx(5.0, abs=0.05)


def test_gradiometer_rejects_distant_source_relative_to_near():
    # Near(10 mm)/far(3 m) contrast grows monotonically with gradiometer order:
    # higher order rejects distant sources more strongly for the same near coupling.
    near, far = np.array([0.01]), np.array([3.0])
    mag = Gradiometer.magnetometer(radius_m=R)
    g1 = Gradiometer.first_order_axial(radius_m=R, baseline_m=B)
    g2 = Gradiometer.second_order_axial(radius_m=R, baseline_m=B)
    ratios = [
        _axial_on_axis(g, near)[0] / _axial_on_axis(g, far)[0] for g in (mag, g1, g2)
    ]
    assert ratios[0] < ratios[1] < ratios[2]
    # each added order buys roughly (far/near) extra rejection; be conservative.
    assert ratios[2] > 100.0 * ratios[0]


def test_near_sample_couples_like_magnetometer():
    # At the bottom loop the gradiometer is not strongly suppressed vs the bare
    # magnetometer (localization, not attenuation, of the sample region).
    near = np.array([0.01])
    mag = _axial_on_axis(Gradiometer.magnetometer(radius_m=R), near)[0]
    g2 = _axial_on_axis(Gradiometer.second_order_axial(radius_m=R, baseline_m=B), near)[0]
    assert 0.3 < g2 / mag < 1.0


def test_sensitivity_shape_and_axis():
    g = Gradiometer.first_order_axial(radius_m=R, baseline_m=B, axis="x")
    pts = np.zeros((4, 5, 3))
    pts[..., 0] = 0.5  # 0.5 m along x (the axis)
    s = g.sensitivity(pts)
    assert s.shape == (4, 5, 3)
    ax = g.axial_sensitivity(pts)
    assert ax.shape == (4, 5)


def test_validation_errors():
    with pytest.raises(ValueError):
        Gradiometer(radii_m=(), positions_m=(), weights=())
    with pytest.raises(ValueError):
        Gradiometer(radii_m=(R, R), positions_m=(0.0,), weights=(1.0, -1.0))
    with pytest.raises(ValueError):
        Gradiometer(radii_m=(-R,), positions_m=(0.0,), weights=(1.0,))
    with pytest.raises(ValueError):
        Gradiometer(radii_m=(R,), positions_m=(0.0,), weights=(1.0,), axis="q")
    with pytest.raises(ValueError):
        Gradiometer.magnetometer(radius_m=R).sensitivity(np.zeros((3, 2)))
