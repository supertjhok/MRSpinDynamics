# Quasistatic validity and a path to full-wave field modeling

PythonSpinDynamics currently uses static or quasistatic electromagnetic field
models.  These models are efficient and appropriate across much of NMR, NQR,
gradient-coil, and permanent-magnet work, but they can become inaccurate for
large RF structures, high operating frequencies, high-permittivity samples,
strong conductive shielding, or operation near a distributed resonance.

This note records the evaluated options, the immediate validity-safeguard
design, and the recommended path to full-wave support.

## Existing field-model boundary

The current field stack contains several related but distinct approximations:

- `fields.magnetostatics.biot_savart` evaluates the instantaneous field of
  prescribed filament currents.  Complex current phasors are supported by RF
  coil layers, but propagation delay and material scattering are absent.
- `fields.quasistatic` computes inductive E fields and a first-order conductive
  response.  Its Born approximation ignores the eddy currents' reaction field,
  displacement current, and wave propagation.
- `fields.eddy_modes` is self-consistent within a discrete conducting-loop
  model, but remains magnetoquasistatic.
- `fields.coil_peec` resolves conductor skin/proximity effects and lumped
  capacitance.  Its inductive and electrostatic kernels remain quasistatic; its
  radiation term is only a leading small-loop approximation.
- birdcage circuit and multiport models determine complex branch currents from
  lumped networks, after which their B1 fields are still evaluated by
  Biot--Savart.

The downstream spin dynamics are already field-map based.  This is the key
architectural advantage: a full-wave backend need not replace the spin solver,
only produce normalized complex E and B/H maps in a common representation.

## Independent failure modes

A single free-space wavelength check is insufficient.  Validity screening must
report at least the following quantities separately.

### Propagation phase and attenuation

For a homogeneous region with propagation constant

`gamma = alpha + i beta = sqrt(i omega mu (sigma + i omega epsilon))`,

the relevant dimensionless spans are `beta L` and `alpha L`.  The first measures
spatial phase change; the second measures amplitude attenuation across a
characteristic material path.  Dielectric wavelength is shortened by roughly
`sqrt(epsilon_r)` in a low-loss material, so a sample can be electrically large
even when the coil looks small in free space.

For example, a 0.30 m region with relative permittivity 80 at 128 MHz is about
1.15 low-loss dielectric wavelengths across.  A quasistatic B1 map cannot
represent the resulting interference or dielectric loading.

### Displacement versus conduction current

The first-order conductive solver assumes magnetoquasistatic behavior.  Its
material screen is `omega epsilon / sigma`; when this is not small,
displacement current cannot be discarded in favor of `J = sigma E`.

### Reaction fields and skin depth

Even when displacement current is negligible, the Born conductive solver fails
when the conductor is no longer weakly perturbing.  `L/delta`, with
`delta = sqrt(2/(omega mu sigma))`, screens this behavior.  A large value calls
for a self-consistent MQS eddy-current solver; it does not automatically imply
that a full-wave solver is needed.

### Distributed coil behavior

Free-space electrical extent and the operating-frequency/self-resonance ratio
screen the assumptions of instantaneous prescribed current and lumped circuit
behavior.  PEEC can estimate the first self-resonance, but it cannot reproduce
the full distributed field near that resonance.

These are engineering screens, not certified error estimates.  Geometry,
interfaces, resonant modes, and source topology can make a particular problem
more sensitive.

## Options evaluated

### Extend Biot--Savart with retardation

A retarded filament Green function could add propagation phase and leading
radiation for prescribed currents at modest implementation cost.  It would not
solve the motivating case: sample polarization and scattering, conductor/sample
coupling, and distributed redistribution of coil current would remain absent.
It is useful as a benchmark or intermediate impressed-current model, not as the
package's general answer to wave effects.

### Add a self-consistent MQS A--phi solver

A volume or surface impedance solver could capture reaction fields, shielding,
and skin effects while still neglecting displacement current.  This is a
valuable future addition for metal shields and strong conductive loading, and
it is cheaper than full wave in its natural regime.  It does not address
high-permittivity wave propagation.

### Write an in-package full-wave FDTD or FEM solver

This provides maximum control but is not recommended.  A credible general
solver requires mesh generation, stable material averaging, ports, lumped
elements, open-boundary treatment, dispersive media, scalable linear algebra or
time stepping, field normalization, power/SAR postprocessing, convergence
tooling, and a large independent validation program.  Maintaining that stack
would compete directly with the spin-dynamics project.

### Integrate openEMS

openEMS is the recommended first automated backend.  Its EC-FDTD engine has a
Python interface, Cartesian and cylindrical grids, curve/wire geometry, lossy
materials, lumped ports and RLC elements, PML, SAR, and HDF5/VTK field output.
The existing PythonSpinDynamics segment and PEEC polyline geometry maps
naturally to its conductor representation.  It is GPLv3, matching this
repository's license.

The main trade-offs are the usual FDTD costs: small conductor details and short
wavelengths constrain the grid and timestep, while narrowband high-Q structures
can require long settling times.

### Integrate Palace

Palace is the strongest second backend.  Its frequency-domain finite-element
formulation, unstructured meshes, lumped and wave ports, conductive/lossy
materials, and parallel CPU/GPU solvers are attractive for narrowband high-Q
RF structures and complex material interfaces.  It requires an external mesh,
has a heavier installation workflow, and currently offers absorbing boundaries
rather than PML.  A Palace adapter is best added after an openEMS reference case
shows a concrete FDTD limitation.

### Meep and commercial solvers

Meep is a mature Python-controlled FDTD solver, but its workflow is less
RF-port and circuit oriented than openEMS.  CST, HFSS, COMSOL, and other
commercial tools are common in RF-coil work and should be supported through a
solver-neutral import format without becoming required dependencies.

## Recommended sequence

### Phase 1: validity safeguards

1. Add a structured quasistatic assessment containing phase, attenuation,
   displacement-current, skin/reaction, and self-resonance metrics.
2. Add a visible package-specific warning and `warn`/`error`/`ignore` policy.
3. Keep generic DC and gradient Biot--Savart calls quiet.
4. Check RF-specific and conductive-loading entry points where frequency,
   geometry, and material information are available.
5. Report missing material or frequency context rather than silently declaring
   a result valid.
6. Retain the assessment in result objects where the API returns a structured
   result.

### Phase 2: solver-neutral harmonic field maps

Define a `HarmonicEMSolution` carrying frequency, phasor convention, complex E
and B/H, coordinates, normalization, ports, material metadata, backend/version,
and convergence provenance.  It should compute B1+/B1- and adapt into existing
spatial field maps.  Add generic HDF5/NPZ import before binding to one solver.

**Status (2026-08-08): complete.** The backend-neutral implementation is in
`spin_dynamics.fields.harmonic`, including the versioned NPZ/HDF5 schema and
the magnitude-preserving adapter described in
[Harmonic electromagnetic field interchange](harmonic_em_fields.md).

### Phase 3: openEMS reference backend

Convert existing coil polylines to CSXCAD, add sample materials and lumped
ports, run openEMS as an optional external executable, and import normalized
complex fields.  The first reference problem should be a loop or birdcage
loaded by a homogeneous high-permittivity, lossy cylinder.

**Status (2026-08-08): implementation and live execution complete.** The
optional adapter in `spin_dynamics.fields.openems` generates standalone
openEMS/CSXCAD projects, supports current and legacy frequency-domain HDF5
field dumps, and includes the [loaded-loop reference configuration](openems_backend.md).
The Linux solver is installed in the WSL validation environment and has been
exercised by the Phase 4 suite.

### Phase 4: validation and selective expansion

Require low-frequency convergence to Biot--Savart, mesh convergence, accepted
power/loss balance, transmit/receive reciprocity, and at least one independent
cross-solver or published benchmark. The reusable checks and live openEMS suite
are implemented in [full-wave validation](full_wave_validation.md). A resolved
6 mm strip reference with a non-intersecting sample now passes termination,
sample-interior mesh convergence, E/J loss consistency, reciprocity, and the
Biot--Savart limit. Dielectric-interface E and an independent high-frequency
cross-solver or published benchmark remain outstanding before expanding the
claim. Develop a self-consistent MQS solver separately if conductive shielding
and skin-effect applications justify it.

## Decision

Do not build a general full-wave solver inside PythonSpinDynamics.  Make
quasistatic validity visible immediately, preserve a solver-neutral field-map
boundary, use openEMS as the first optional full-wave backend, and retain Palace
as the preferred narrowband/unstructured alternative.

## References and implementation resources

- Yang et al., [Dependence of B1+ and B1- field patterns on sample electrical
  properties and MR frequency](https://pmc.ncbi.nlm.nih.gov/articles/PMC5082994/).
- [openEMS introduction and Python interface](https://docs.openems.de/intro.html),
  including [ports and boundary conditions](https://docs.openems.de/python/openEMS/openEMS.html)
  and [frequency-domain field/SAR dumps](https://docs.openems.de/python/CSXCAD/CSProperties/CSPropDumpBox.html).
- [Palace problem types](https://awslabs.github.io/palace/stable/guide/problem/),
  [material models](https://awslabs.github.io/palace/dev/guide/model/), and
  [boundary conditions](https://awslabs.github.io/palace/stable/guide/boundaries/).
- [Meep Python interface](https://meep.readthedocs.io/en/master/Python_User_Interface/).
- Rappetti and Rousseaux, [On quasi-static models hidden in Maxwell's
  equations](https://doi.org/10.1016/j.apnum.2012.11.007).
