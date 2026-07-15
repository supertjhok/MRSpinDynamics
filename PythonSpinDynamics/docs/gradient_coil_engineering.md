# Gradient-coil engineering and integration

The engineering API operates on `GradientWinding` or
`ActivelyShieldedWinding`, so every calculation uses the same oriented contour
geometry that is plotted and passed downstream.

## Combined engineering report

```python
from spin_dynamics.fields import gradient_coil_engineering_metrics

metrics = gradient_coil_engineering_metrics(
    winding,
    target_points,
    target_field_t=target_bz,
    shield_points=exterior_points,
    field_direction=(0, 0, 1),
    wire_radius=0.5e-3,
    background_field=(0, 0, 3.0),
)
```

The report contains:

| Group | Returned quantities |
| --- | --- |
| Field | fitted Cartesian gradient and offset, efficiency per ampere, target RMS error/correlation, exterior RMS/peak field |
| Electrical | contour count, wire length, DC resistance and power, approximate series-equivalent inductance and stored energy |
| Mechanical | net and per-contour Lorentz force/torque plus peak segment force |

Force uses `dF = I dl x B_background` at segment midpoints.  The background can
be a constant 3-vector or a callback evaluated at all midpoints.  Torque is
reported about `force_origin`.  These are static filament-load estimates; they
do not include support deformation, vibration modes, conductor stress, or the
Maxwell stress of magnetic material.

DC resistance uses the selected `ConductorMaterial` and optional temperature.
Self inductance uses an equal-perimeter circular-loop approximation, while
mutual terms use the existing Neumann/vector-potential calculation on actual
paths.  Contours are treated as series-connected with their extracted
orientation.  This is appropriate for early trade studies, not a substitute
for extraction of the final routed conductor.

## PEEC handoff

```python
from spin_dynamics.fields import winding_peec_conductors

conductors = winding_peec_conductors(winding, wire_radius=0.5e-3)
```

This returns one existing `coil_peec.Conductor` per closed contour and preserves
orientation.  Each loop is opened at its first point to define terminals.  Add
explicit crossover, lead, and terminal paths before combining loops into a
single manufacturable series conductor and running detailed AC resistance,
inductance, capacitance, Q, or self-resonance extraction.  The adapter does not
silently create geometrically impossible connections.

## Eddy-current and pre-emphasis handoff

```python
from spin_dynamics.fields import EddyModes, winding_to_gradient_driver

modes = EddyModes(
    shield_radii,
    shield_centers,
    wire_radius=1e-3,
    resistivity=1.7e-8,
    axis="z",
)
driver = winding_to_gradient_driver(winding, modes, tau_rl=1e-3)
```

The adapter gives the combined oriented primary/shield segment set to the
existing `EddyModes.to_gradient_driver` calculation, returning the standard
`GradientDriverResponse` used by gradient optimization and pre-emphasis.  Both
surfaces must use a common turn current.  `EddyModes` is a coaxial-ring model;
its quantitative assumptions remain most appropriate for axisymmetric shield
structures and should be validated separately for strongly transverse designs.

## Imaging and motion maps

```python
import numpy as np
from spin_dynamics.fields import winding_imaging_field_map

x = np.linspace(-0.02, 0.02, 41)
z = np.linspace(-0.03, 0.03, 61)
field_map = winding_imaging_field_map(winding, (x, z))
motion_maps = field_map.to_motion_field_maps()
```

The result retains projected field in tesla and converts it to angular
off-resonance in rad/s using `gamma_rad_s_t`.  Missing physical coordinates are
zero: defaults are `z` for one-dimensional maps, `(x, z)` for two-dimensional
maps, and `(x, y, z)` for three-dimensional maps.  Override with
`cartesian_axes` when another plane is required.  `to_motion_field_maps()`
returns the existing N-dimensional motion/imaging field-map container and can
accept matching transmit and receive B1 maps.

## Design checks before manufacture

1. Recompute target and exterior field from extracted contours, not only the
   continuous current sheet.
2. Route crossovers and leads, then repeat field and PEEC extraction.
3. Evaluate force/torque over the actual background-field map and relevant
   current polarities.
4. Use the realized paths in the eddy model and include its returned driver in
   waveform optimization or pre-emphasis.
5. Sample the final field on the exact imaging domain and gyromagnetic-ratio
   convention used by the sequence workflow.
