"""Validation of electrothermal electromagnets used as B0 sources."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import coils  # noqa: E402
from spin_dynamics.fields.coil_properties import ConductorMaterial  # noqa: E402
from spin_dynamics.fields.magnetostatics import biot_savart  # noqa: E402
from spin_dynamics.thermal import (  # noqa: E402
    ElectromagnetControl,
    ElectrothermalElectromagnet,
)


def _model(**updates) -> ElectrothermalElectromagnet:
    values = {
        "inductance_h": 0.02,
        "reference_resistance_ohm": 0.25,
        "field_sensitivity_t_per_a": 1.0e-3,
        "heat_capacity_j_per_k": 1200.0,
        "thermal_conductance_w_per_k": 10.0,
    }
    values.update(updates)
    return ElectrothermalElectromagnet(**values)


def test_voltage_step_matches_exact_rl_response_when_resistance_is_constant() -> None:
    material = ConductorMaterial(
        "temperature independent",
        resistivity=1.0e-8,
        temp_coefficient=0.0,
    )
    model = _model(
        inductance_h=0.1,
        reference_resistance_ohm=1.0,
        material=material,
    )
    times = np.linspace(0.0, 0.8, 81)
    result = model.simulate(times, 2.0)
    expected = 2.0 * (1.0 - np.exp(-times / 0.1))
    np.testing.assert_allclose(result.current_a, expected, rtol=2.0e-6, atol=2.0e-8)
    np.testing.assert_allclose(result.field_t, 1.0e-3 * expected)


def test_constant_voltage_reaches_coupled_electrothermal_fixed_point() -> None:
    model = _model(heat_capacity_j_per_k=100.0)
    voltage = 10.0
    times = np.linspace(0.0, 120.0, 241)
    result = model.simulate(times, voltage)

    alpha = model.material.temp_coefficient
    gain = alpha * voltage / model.thermal_conductance_w_per_k
    expected_current = (-1.0 + np.sqrt(1.0 + 4.0 * gain * voltage / 0.25)) / (
        2.0 * gain
    )
    expected_temperature = (
        model.ambient_temperature_k
        + voltage * expected_current / model.thermal_conductance_w_per_k
    )
    assert result.current_a[-1] == pytest.approx(expected_current, rel=2.0e-5)
    assert result.temperature_k[-1] == pytest.approx(
        expected_temperature,
        rel=2.0e-6,
    )
    assert result.deposited_energy_j > 0.0


def test_temperature_current_and_field_feedback_remove_b0_drift() -> None:
    model = _model(heat_capacity_j_per_k=100.0)
    times = np.linspace(0.0, 120.0, 241)
    target_current = 40.0
    target_field = model.field_sensitivity_t_per_a * target_current
    voltage = target_current * model.reference_resistance_ohm

    constant_voltage = model.simulate(times, voltage)
    temperature_compensated = model.simulate(
        times,
        target_current,
        control=ElectromagnetControl(
            "temperature_compensated",
            voltage_limits_v=(0.0, 20.0),
        ),
    )
    current_feedback = model.simulate(
        times,
        target_current,
        control=ElectromagnetControl(
            "current",
            response_time_s=0.1,
            voltage_limits_v=(-20.0, 20.0),
        ),
    )
    field_lock = model.simulate(
        times,
        target_field,
        control=ElectromagnetControl(
            "field",
            response_time_s=0.1,
            voltage_limits_v=(-20.0, 20.0),
        ),
    )

    assert constant_voltage.field_t[-1] < 0.95 * target_field
    assert temperature_compensated.current_a[-1] == pytest.approx(
        target_current,
        rel=2.0e-6,
    )
    assert current_feedback.current_a[-1] == pytest.approx(target_current, rel=2.0e-5)
    assert field_lock.field_t[-1] == pytest.approx(target_field, rel=2.0e-5)
    assert field_lock.uniform_b0().field_tesla == pytest.approx(target_field)


def test_realized_coil_produces_spatial_b0_and_motion_maps() -> None:
    segments = coils.solenoid(radius=0.04, length=0.08, turns=9, n_segments=32)
    model = ElectrothermalElectromagnet.from_segments(
        segments,
        inductance_h=1.0e-3,
        reference_resistance_ohm=0.5,
        heat_capacity_j_per_k=500.0,
        thermal_conductance_w_per_k=5.0,
    )
    times = np.asarray([0.0, 0.01])
    result = model.simulate(
        times,
        0.0,
        initial_current_a=2.0,
        max_step_s=1.0e-4,
    )
    points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    expected = biot_savart(points, segments, current=2.0)
    np.testing.assert_allclose(
        result.field_vectors(points, time_index=0),
        expected,
        rtol=1.0e-12,
        atol=1.0e-15,
    )

    x_axis = np.linspace(-0.01, 0.01, 5)
    z_axis = np.linspace(-0.015, 0.015, 7)
    projected = result.projected_field_map(
        (x_axis, z_axis),
        time_index=0,
    )
    assert projected.shape == (5, 7)
    maps = result.to_motion_field_maps((x_axis, z_axis), time_index=0)
    assert maps.domain.shape == (5, 7)
    assert maps.b0_map.shape == (5, 7)
    assert np.all(np.isfinite(maps.b0_map))


def test_voltage_limits_and_additional_losses_are_applied() -> None:
    model = _model(heat_capacity_j_per_k=100.0)
    times = np.linspace(0.0, 2.0, 101)
    result = model.simulate(
        times,
        100.0,
        control=ElectromagnetControl(
            "current",
            response_time_s=0.05,
            voltage_limits_v=(-3.0, 3.0),
        ),
        additional_power_w=5.0,
    )
    assert np.max(np.abs(result.voltage_v)) <= 3.0
    assert np.all(result.power_w >= 5.0)
    assert result.temperature_k[-1] > model.ambient_temperature_k
