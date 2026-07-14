# Stream-function gradient-coil design

## Purpose

PythonSpinDynamics will support numerical MRI gradient-coil design using the
stream-function method described in Section 10.3.3 of *Sensors, Circuits, and
Systems for Scientific Instruments: Back-ends and Applications*. The first
implementation targets regular cylindrical winding surfaces, for which the
surface current can be represented by a rectangular azimuth/axial mesh of
thin-wire elements.

The feature is intended to connect coil geometry to the rest of the library:

- the finite-segment Biot-Savart solver for field prediction;
- PEEC extraction of winding resistance and inductance;
- gradient eddy-current and pre-emphasis models;
- imaging workflows that consume sampled gradient-field maps.

## Mathematical model

A divergence-free surface current density can be written in terms of a stream
function, `psi`, normal to the winding surface. On a cylinder,

\[
J_\phi = -\frac{\partial \psi}{\partial z}, \qquad
J_z = \frac{1}{r}\frac{\partial \psi}{\partial \phi}.
\]

For a regular cylinder, the initial implementation discretizes the azimuthal
current into short filamentary segments. The axial segments needed to close a
physical winding do not contribute to `B_z`, so they are omitted from the
inverse field solve. Applying the Biot-Savart law independently to every source
segment produces the sensitivity matrix

\[
B_z = S I.
\]

Given a target field `b`, the segment currents are found from the constrained
Tikhonov problem

\[
\mathop{\mathrm{minimize}}_I
\left\|W_f(SI-b)\right\|_2^2
+ \alpha\left\|W_p I\right\|_2^2,
\qquad C I = 0.
\]

`W_f` contains field-point or quadrature weights. The first implementation
uses `W_p = I`, matching the textbook current-norm penalty. A later electrical
extension will allow segment-resistance weights so the second term estimates
actual `I^2 R` loss. `C I = 0` enforces Kirchhoff current balance along every
azimuthal column of the mesh, which causes the recovered stream-function
contours to close.

The regularization value has physical units of `T^2/A^2` for the unnormalized
form above. There is no universal optimum: users must choose it from the field
error/current trade-off, or use a regularization-path selector when that phase
is implemented.

## Public API direction

The design API is split into a reusable field system and a solve:

```python
surface = CylindricalWindingSurface(
    radius=0.40,
    length=0.90,
    n_phi=56,
    n_z=61,
)
points = spherical_target_points(radius=0.20, points_per_axis=11)
target_bz = linear_gradient_target(points, gradient=(0.0, 1.0e-3, 0.0))

system = build_cylindrical_gradient_system(surface, points)
result = solve_gradient_coil(
    system,
    target_bz,
    regularization=1.0e-14,
)
```

Separating system construction from the solve makes the expensive sensitivity
matrix reusable for regularization sweeps and alternative targets.

## Implementation phases

### Phase 1: cylindrical inverse solve

- Generate a z-axis cylindrical mesh of azimuthal source segments.
- Calculate a chunked `B_z`-per-ampere sensitivity matrix.
- Enforce the axial KCL constraints by eliminating one current per azimuthal
  column.
- Solve the weighted, regularized least-squares problem with SciPy `lsmr` when
  available and a NumPy dense fallback for small systems.
- Recover the discrete stream function by integrating the segment currents
  along `z`.
- Report field error, current norm, closure error, and solver diagnostics.

### Phase 2: winding extraction and regularization paths

- Extract equally spaced periodic contours of the stream function without
  making Matplotlib a runtime dependency.
- Stitch contours across the `0/2 pi` seam and map them to 3-D segment paths.
- Define contour spacing through an explicit current-per-turn value.
- Add alpha sweeps and L-curve diagnostics.

### Phase 3: active shielding and mechanical constraints

- Concatenate main- and shield-cylinder source matrices.
- Add zero-field targets on a surface outside the shield.
- Add net-force and net-torque calculations in a specified static field.
- Support zero-torque equality constraints and optional torque penalties.

### Phase 4: electrical and workflow integration

- Convert winding contours to the common straight-segment representation.
- Feed realizable windings to the PEEC resistance/inductance solver.
- Feed the same paths to the eddy-mode and gradient-driver models.
- Sample the optimized fields directly for imaging workflows.
- Add planar surfaces; reserve arbitrary triangular-surface boundary-element
  design for a separate extension.

## Validation strategy

Fast tests will verify:

- each sensitivity column against a direct one-segment Biot-Savart result;
- exact current closure within numerical tolerance;
- linear scaling with target gradient amplitude;
- the expected parity of transverse and longitudinal gradients;
- lower current norm as regularization is increased;
- agreement between the returned predicted field and direct Biot-Savart
  evaluation of all independently weighted source segments.

A slower reference test will use the textbook's unshielded cylinder dimensions:
40 cm radius, 90 cm height, a `56 x 61` source mesh, and a 20 cm spherical target
volume. The book does not state its exact target-point sampling, regularization
value, or contour current, so these inputs must be recorded in the test before
the reported approximately 2 percent field-error result can be used as a fixed
regression threshold.

An actively shielded reference will use the textbook's concentric 40 cm and
50 cm winding cylinders and a zero-field target at 60 cm radius.

## Current status

Phase 1 is implemented: the package can build cylindrical source meshes,
calculate segment sensitivities, solve the KCL-constrained regularized inverse
problem, and return the segment-current grid and discrete stream function. The
remaining phases above define intended scope, not yet-available public behavior.
