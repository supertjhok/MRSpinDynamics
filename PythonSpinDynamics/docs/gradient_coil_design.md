# Stream-function gradient-coil design

PythonSpinDynamics implements cylindrical stream-function gradient-coil design,
closed-winding extraction, active shielding, and adapters to the library's coil
engineering, eddy-current, and imaging tools.  The formulation follows Section
10.3.3 of *Sensors, Circuits, and Systems for Scientific Instruments: Back-ends
and Applications*.

## Inverse design model

For a cylindrical winding surface, a divergence-free surface current is written
in terms of a stream function, `psi`,

\[
J_\phi=-\frac{\partial\psi}{\partial z}, \qquad
J_z=\frac{1}{r}\frac{\partial\psi}{\partial\phi}.
\]

The cylinder is discretized into short azimuthal filament elements.  Applying
Biot-Savart to each element gives the projected-field sensitivity matrix `S`.
The KCL-constrained Tikhonov solve is

\[
\underset{I}{\operatorname{minimize}}\quad
\sum_q w_q (S I-b)_q^2+\alpha\sum_j p_j I_j^2,
\qquad C I=0.
\]

`C I = 0` is enforced by eliminating one current per azimuthal column, not by a
soft penalty.  The returned stream function therefore obeys
`segment_currents_a == -np.diff(stream_function_a, axis=1)` and its nonzero
periodic contours close.  SciPy `lsmr` is used when available; small problems
also have a NumPy dense fallback.  `solve_regularization_path` reuses `S`,
reports the L-curve, and chooses its maximum-curvature interior point.

```python
import numpy as np
from spin_dynamics.fields import (
    CylindricalWindingSurface,
    build_cylindrical_gradient_system,
    extract_winding_contours,
    linear_gradient_target,
    solve_regularization_path,
    spherical_target_points,
)

surface = CylindricalWindingSurface(0.08, 0.18, 32, 17)
points = spherical_target_points(0.03, points_per_axis=7)
target_bz = linear_gradient_target(points, gradient=(0.0, 10e-3, 0.0))
system = build_cylindrical_gradient_system(surface, points)
path = solve_regularization_path(system, target_bz, np.logspace(-20, -9, 23))
result = path.selected_result
contours = extract_winding_contours(result, current_per_turn_a=1.0)
```

Contour levels are half-step centered and separated by the specified current
per turn.  They are stitched across the periodic azimuth seam and oriented with
the optimized current before conversion to the library's common straight-line
segment representation.

## Active shielding

An active design jointly solves independent current sheets on a primary and a
larger concentric shield cylinder.  If `S_t` samples the target volume and `S_e`
samples a surface outside the shield, the combined problem is

\[
\underset{I_p,I_s}{\operatorname{minimize}}\quad
\left\|W_t(S_t[I_p;I_s]-b)\right\|_2^2+
\left\|W_eS_e[I_p;I_s]\right\|_2^2+
\alpha(p_p\|I_p\|_2^2+p_s\|I_s\|_2^2),
\]

with a separate exact KCL constraint on each surface.  `shield_weights` controls
the relative exterior-field objective, while
`surface_regularization_weights` can discourage current on either cylinder.

```python
from spin_dynamics.fields import (
    build_actively_shielded_gradient_system,
    cylindrical_shield_points,
    extract_actively_shielded_winding,
    solve_actively_shielded_gradient_coil,
)

primary = CylindricalWindingSurface(0.05, 0.12, 24, 13)
shield = CylindricalWindingSurface(0.065, 0.15, 24, 13)
exterior = cylindrical_shield_points(0.085, 0.18, n_phi=32, n_z=17)
target_bz = linear_gradient_target(points, gradient=(0.0, 0.0, 10e-3))
system = build_actively_shielded_gradient_system(
    primary, shield, points, exterior
)
result = solve_actively_shielded_gradient_coil(
    system,
    target_bz,
    regularization=1e-15,
    shield_weights=5.0,
)
winding = extract_actively_shielded_winding(
    result,
    current_per_turn_a=1.0,
)
```

The result reports target error, exterior RMS and peak field, primary and shield
current norms, both stream functions, and the common closure error.  Use a
symmetric cylindrical control surface for a coaxial shield.  The separate
`spherical_shell_points` helper or an application-specific point set can be used
when that geometry better represents the exclusion region.

## Realization and validation

The continuous current sheet is the inverse-design result.  Contour extraction
quantizes that sheet into physical turns and generally changes target error and
shielding.  Always re-evaluate the realized winding with
`gradient_coil_engineering_metrics`; decrease current per turn or refine the
source mesh when quantization error is excessive.

The implementation tests sensitivity columns against direct Biot-Savart,
independent KCL closure, target scaling, L-curve behavior, periodic contour
closure, and active suppression relative to the same unshielded primary.  The
book's dimensions can be reproduced, but its exact point quadrature,
regularization, and contour current are not specified, so its reported error is
not used as an exact regression fixture.

Run these plotting examples:

```bash
python examples/plot_gradient_coil_regularization.py --output results/lcurve.png
python examples/plot_stream_function_gradient_coil.py --output results/winding.png
python examples/plot_actively_shielded_gradient_coil.py --output results/shielded.png
```

See [Gradient-coil engineering and integration](gradient_coil_engineering.md)
for electrical, force/torque, PEEC, eddy-current, and imaging adapters.

## Scope

The current inverse solver supports regular concentric cylinders.  Extracted
contours are independent closed loops; automatic crossover, terminal, and lead
routing is not invented.  Electrical extraction therefore documents its
series-equivalent assumption and exposes individual contours to PEEC for a
later routed design.  Planar surfaces, arbitrary triangular surfaces,
manufacturing constraints, and force/torque constraints in the inverse solve
remain future extensions; force and torque evaluation is available now.
