# Cross-Component Composition

Integrated studies often fail at the boundaries between otherwise correct
models: axes are reordered, Hz is mistaken for rad/s, a command waveform is
interpolated like a physical state, or a field map is evaluated on the wrong
grid. `spin_dynamics.composition` makes those boundaries explicit.

This is an advanced integration page, not the first place to learn an
individual field, thermal, flow, or sequence model. Use it after the component
models work independently and you need typed spatial/time resampling and unit
checks while preserving each engine's native result type.

The shared conventions are:

- `SpatialGrid`: one to three named rectilinear axes, in metres. Array dimension
  order is axis order, so `(x, z)` cannot be mistaken for `(x, y)`.
- `TimeAxis`: strictly increasing absolute times, in seconds. Linear
  interpolation is used for physical states and zero-order hold for sequence
  commands. Extrapolation raises an error.
- every field and sequence channel has an explicit unit. `convert_units`
  handles the package-boundary units, including Hz to rad/s and Hz/m to
  rad/s/m; incompatible conversions raise.

## Adapters

`FieldSolutionAdapter` holds scalar or vector channels whose leading dimensions
match a `SpatialGrid`. `from_magnet_field_maps` and
`from_spatial_field_maps` adapt the existing field-map families. `resample`
uses multilinear interpolation and checks the named-axis contract.

`ThermalStateAdapter` gives lumped nodes and conduction profiles one leading
time dimension. It adapts `ThermalTransientResult`, `ConductionResult`, and
`ThermalCouplingResult`, then aligns trajectories with `at()`.

`FlowFieldAdapter` represents velocity in m/s as a uniform vector, a steady or
time-varying Eulerian grid, or a callable. `sample()` returns velocities at
particle positions and an absolute time. `from_flow_model` exposes a pipe
model's bulk velocity along an explicit direction.

`HardwareResponseAdapter` wraps the existing RF, gradient, or receiver response
objects. It makes input/output units explicit and rejects nonuniform time axes,
because the response implementations require a single dwell.

`SequenceTimelineAdapter.from_compiled` turns a `CompiledSequence` into aligned
interval-centered RF and gradient channels, preserving exact ADC times and
source metadata. `at()` applies zero-order hold to another time base;
`with_hardware_response()` realizes one channel after checking its unit.

## Example

```python
from spin_dynamics.composition import (
    FieldSolutionAdapter,
    HardwareResponseAdapter,
    SequenceTimelineAdapter,
    SpatialGrid,
)
from spin_dynamics.optimal_control import GradientDriverResponse
from spin_dynamics.sequences import compile_sequence

# Existing solver/compiler results remain their native types.
field = FieldSolutionAdapter.from_magnet_field_maps(magnet_result)
study_grid = SpatialGrid((x_m, y_m), ("x", "y"))
field_on_study_grid = field.resample(study_grid)

timeline = SequenceTimelineAdapter.from_compiled(compile_sequence(sequence))
gradient_response = HardwareResponseAdapter(
    GradientDriverResponse(tau_rl=80e-6),
    input_unit="Hz/m",
    output_unit="Hz/m",
    channel_kind="gradient",
)
realized = timeline.with_hardware_response("gradient", gradient_response)
```

The adapters reject mismatched names, shapes, units, and time ranges early.
This is deliberate: a failed boundary check is safer than a numerically valid
study assembled in the wrong coordinate or frequency convention.
