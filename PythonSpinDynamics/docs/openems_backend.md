# openEMS full-wave backend

The optional openEMS backend connects PythonSpinDynamics coil geometry to a
full-wave EC-FDTD solve without making openEMS a package dependency. It writes a
versioned JSON project and a standalone Python driver, runs that driver in a
user-selected external Python environment, and imports complex frequency-domain
E and B into `HarmonicEMSolution`.

This backend was introduced in Phase 3. Phase 4 now supplies explicit numerical
termination, mesh, E/J loss, reciprocity, and analytical-limit checks. The
resolved-strip reference passes these checks in the declared loaded-sample
interior region; see [full-wave validation](full_wave_validation.md).

## Capabilities

The adapter provides:

- conversion of the package's common straight-segment coil geometry into one
  or more contiguous CSXCAD wire paths;
- finite-radius PEC wire primitives;
- mesh-resolved planar PEC polygons for strip conductors;
- homogeneous lossy cylindrical sample materials with relative permittivity,
  conductivity, and relative permeability;
- curve-based or box-based lumped excitation/measurement ports;
- Gaussian excitation, six configurable PEC/PMC/MUR/PML boundaries, and a
  rectilinear mesh containing requested field-sampling coordinates;
- frequency-domain HDF5 E and B dumps;
- current-, voltage-, or accepted-power-normalized field import; and
- a stable model hash plus backend, material, port, and requested-termination
  provenance.

The generated driver uses SI metres throughout. This avoids drawing-unit
ambiguity when fields are brought back into PythonSpinDynamics.

## Installation boundary

Core project generation only needs NumPy. Reading openEMS HDF5 output needs the
package's `fullwave` extra:

```bash
python -m pip install "python-spin-dynamics[fullwave]"
```

The external interpreter used to run the generated driver must separately
contain the openEMS and CSXCAD Python bindings and their native executables.
Follow the [official openEMS Python installation
instructions](https://docs.openems.de/python/install.html). The adapter checks a
candidate interpreter with `check_openems_python()` before a long run.

openEMS is intentionally not declared as a PyPI dependency. Its Python bindings
are distributed with platform-specific native solver installations, and users
may keep those installations isolated from PythonSpinDynamics.

## Loaded-loop reference problem

`loaded_loop_openems_reference()` creates the first wave-sensitive case:

- a 65 mm mean-radius, 6 mm wide planar PEC strip loop in the y-z plane;
- a geometry-aligned 50 ohm lumped feed across a symmetric 5 mm gap;
- a 45 mm radius by 100 mm long sample cylinder along x, enclosed by and not
  intersecting the loop;
- relative permittivity 80 and conductivity 0.5 S/m;
- 128 MHz excitation and PML on all six simulation boundaries; and
- a three-dimensional field region covering the sample.

The loop's central magnetic field is along x, transverse to the conventional +z
B0 stored in the project metadata. At 128 MHz the dielectric wavelength in the
sample is about 0.26 m, so this geometry is intentionally outside a comfortably
uniform-phase regime.

Generate the project without needing openEMS:

```bash
python examples/openems_loaded_loop.py --output .tmp/openems_loaded_loop
```

This writes `openems_project.json` and a self-contained `run_openems.py`.
Execute it through an environment containing openEMS:

```bash
python examples/openems_loaded_loop.py \
  --output .tmp/openems_loaded_loop \
  --python /path/to/openems-python \
  --run
```

A successful run produces raw `E_fd.h5` and `B_fd.h5`, port and run metadata,
solver logs, the CSXCAD XML model, and normalized `harmonic_solution.npz`.

## Geometry conversion

The coil generators in `spin_dynamics.fields.coils` return flat lists of
straight segment endpoint pairs. Use `OpenEMSWire.from_segments()` or
`segments_to_polylines()` to join contiguous segments. A discontinuity starts a
new path, so stacked solenoid loops do not acquire fictitious axial connections.

```python
from spin_dynamics.fields import OpenEMSWire, coils

segments = coils.conducting_ring(radius=0.04, n_segments=96)
wire = OpenEMSWire.from_segments(
    "loop",
    segments,
    radius_m=1.0e-3,
)
```

The mesh includes each wire path's endpoints and Cartesian extrema, rather than
every tessellation vertex. The complete path is still passed to CSXCAD, while
avoiding nearly coincident mesh planes near curved-wire extrema that can force
an impractically small FDTD timestep.

Ports are not inferred from a closed polyline. The caller must provide an
intentional gap and an `OpenEMSLumpedPort` connecting its endpoints. This makes
feed placement, reference impedance, direction, and excitation state explicit.
The loaded reference instead uses an `OpenEMSPlanarConductor` annular-sector
polygon. Its 6 mm width provides three cells across the strip on the 2 mm
validation mesh; narrower, under-resolved strips produced incorrect B/I
normalization and are not the reference geometry.

## HDF5 import and phasor convention

`load_openems_field_dump()` supports both current openEMS complex `NXYZ` vector
datasets and legacy split-real/imaginary `NZYX` datasets. It rejects
non-Cartesian meshes and ambiguous or missing frequencies.

openEMS writes Cartesian HDF5 mesh axes in physical units; the `mesh_scaling`
attribute records how drawing units were converted and must not be applied a
second time. The openEMS frequency transform uses the representation compatible
with PythonSpinDynamics' canonical `exp(+iwt)` phasor convention.

`load_openems_solution()` divides both E and B by the same complex port current
or voltage, preserving phase, or by the positive square root of accepted power.
The original port voltage, current, and power remain in the solution metadata.

## Convergence policy

A zero process exit code proves that the external driver completed; it does not
by itself prove mesh convergence or that the energy-decay termination criterion
was reached. Accordingly, the runner parses the retained logs into `HarmonicConvergence`, but
keeps `converged=False` until an explicit Phase 4 report also establishes mesh
convergence and the required physical checks. Solver termination alone never
promotes a field map. The current validation claim covers the sample interior
with a 3 mm dielectric-interface exclusion. Electric field values at the
staircased material interface and points outside that region are not covered.

The generated driver selects openEMS's multithreaded engine and explicitly uses
up to eight host CPUs. An explicit positive thread count avoids platform builds
where openEMS's nominal automatic value (`numThreads=0`) falls back to one
active worker.

The official openEMS documentation describes the underlying [ports and their
limitations](https://docs.openems.de/concepts/ports.html) and [frequency-domain
field dumps](https://docs.openems.de/concepts/dump.html).
