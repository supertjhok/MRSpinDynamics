from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.composition import (
    FieldSolutionAdapter,
    FlowFieldAdapter,
    HardwareResponseAdapter,
    SequenceTimelineAdapter,
    SpatialGrid,
    ThermalStateAdapter,
    TimeAxis,
    convert_units,
)
from spin_dynamics.fields.magnetostatics import MagnetFieldMaps
from spin_dynamics.sequences import GradientWaveform, SequenceBlock, SequenceIR, compile_sequence
from spin_dynamics.thermal.network import ThermalTransientResult


def test_field_solution_preserves_named_axes_units_and_affine_resampling():
    source = SpatialGrid((np.array([0.0, 1.0]), np.array([0.0, 2.0])), ("x", "z"))
    x, z = source.domain.meshgrid()
    field = FieldSolutionAdapter(source, {"b0": 1.0 + 2.0 * x - z}, {"b0": "T"})
    target = SpatialGrid((np.linspace(0, 1, 3), np.linspace(0, 2, 5)), ("x", "z"))
    adapted = field.resample(target)
    xt, zt = target.domain.meshgrid()
    np.testing.assert_allclose(adapted.channels["b0"], 1.0 + 2.0 * xt - zt)
    assert adapted.units["b0"] == "T"
    with pytest.raises(ValueError, match="axis names"):
        field.resample(SpatialGrid(target.axes_m, ("x", "y")))


def test_magnet_field_adapter_assigns_physical_units():
    zeros = np.zeros((2, 3))
    result = MagnetFieldMaps(
        np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]),
        np.zeros((2, 3, 3)), zeros, zeros, zeros, None, None,
    )
    field = FieldSolutionAdapter.from_magnet_field_maps(result)
    assert field.grid.axis_names == ("x", "y")
    assert field.units["b0_gradient"] == "T/m"


def test_thermal_state_resamples_all_nodes_on_one_time_axis():
    result = ThermalTransientResult(
        times=np.array([0.0, 2.0]),
        temperatures={"coil": np.array([300.0, 304.0]), "sample": np.array([299.0, 301.0])},
    )
    state = ThermalStateAdapter.from_transient(result).at(TimeAxis(np.array([0.5, 1.5])))
    np.testing.assert_allclose(state.temperatures_kelvin["coil"], [301.0, 303.0])
    np.testing.assert_allclose(state.temperatures_kelvin["sample"], [299.5, 300.5])


def test_flow_field_interpolates_space_and_time_in_si_units():
    grid = SpatialGrid((np.array([0.0, 1.0]),), ("x",))
    time = TimeAxis(np.array([0.0, 2.0]))
    velocity = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
    flow = FlowFieldAdapter(grid, velocity, time)
    got = flow.sample(np.array([[0.25], [0.75]]), time_seconds=1.0)
    np.testing.assert_allclose(got[:, 0], [1.25, 1.75])
    uniform = FlowFieldAdapter.uniform([0.1, -0.2])
    np.testing.assert_allclose(uniform.sample(np.zeros((3, 2))), [[0.1, -0.2]] * 3)


class _Gain:
    def apply(self, waveform, dt, *, xp=np):
        return waveform * (2.0 * dt)


def test_compiled_timeline_aligns_channels_and_applies_typed_hardware():
    sequence = SequenceIR(
        blocks=(SequenceBlock(
            duration_seconds=2e-3,
            gradients=(GradientWaveform([1.0, 2.0], dwell_seconds=1e-3), None, None),
        ),)
    )
    timeline = SequenceTimelineAdapter.from_compiled(compile_sequence(sequence))
    assert timeline.units == {"rf": "Hz", "gradient": "Hz/m"}
    assert timeline.time.dwell_seconds == pytest.approx(1e-3)
    response = HardwareResponseAdapter(_Gain(), "Hz/m", "Hz/m", "gradient")
    realized = timeline.with_hardware_response("gradient", response)
    np.testing.assert_allclose(realized.channels["gradient"], timeline.channels["gradient"] * 2e-3)
    with pytest.raises(ValueError, match="expected"):
        timeline.with_hardware_response(
            "gradient", HardwareResponseAdapter(_Gain(), "T/m", "T/m", "gradient")
        )


def test_time_axis_rejects_extrapolation_and_nonuniform_hardware_time():
    source = TimeAxis(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="outside"):
        source.resample(np.array([1.0, 2.0]), TimeAxis(np.array([0.0, 1.1])))
    adapter = HardwareResponseAdapter(_Gain(), "A", "A", "gradient")
    with pytest.raises(ValueError, match="uniform"):
        adapter.apply(np.ones(3), TimeAxis(np.array([0.0, 1.0, 3.0])))


def test_unit_conversion_handles_angular_frequency_and_temperature_offsets():
    np.testing.assert_allclose(convert_units([1.0], "Hz", "rad/s"), [2.0 * np.pi])
    np.testing.assert_allclose(convert_units([0.0, 100.0], "degC", "K"), [273.15, 373.15])
    with pytest.raises(ValueError, match="cannot convert"):
        convert_units([1.0], "s", "m")
