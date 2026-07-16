# Coupled Thermal Modeling: Design and Validation

Magnetic-resonance hardware converts electrical power into heat, and that heat
feeds back into resistance, Q, noise, field strength, sample properties, and
safety margins. Treating temperature as a fixed input can therefore make a
long sequence or high-duty device prediction internally inconsistent.

Use this page to connect known heat sources to lumped thermal networks or
one-dimensional conduction models, including RF/gradient coils, samples,
flowing fluids, and voltage-driven electromagnets. The package predicts thermal
transients from supplied geometry and material parameters; it is not a general
three-dimensional CFD or bioheat planning system.

> **Status (audited 2026-07-15): implemented.** Material, source, network,
> conduction, RF/gradient coupling, flowing-sample paths, and coupled
> electrothermal B0 electromagnets are available with tests and examples.

## Motivation

Three heating paths matter in real experiments and motivate the implemented models:

1. **RF coil self-heating.** Transmit pulses dissipate `I^2 R` in the coil;
   the coil warms, its resistivity rises, `Q` drops, Johnson noise grows, and
   `B1` per unit drive falls. For cryoprobes the coil-to-sample temperature
   gap is the entire design point.
2. **Gradient coil heating.** Gradient waveforms deposit large average power
   (duty-cycled `I^2 R` plus eddy losses in nearby conductors); the heat
   conducts and radiates toward shims, the magnet bore, and the sample.
3. **Direct sample heating (SAR).** The RF E-field drives eddy currents in a
   conductive sample, depositing `P = (1/2) integral sigma |E|^2 dV`; long
   high-duty sequences (CPMG/SLSE trains, decoupling, spin-lock) measurably
   warm lossy samples.
4. **Electromagnet B0 drift.** A voltage-driven B0 coil warms under its own
   `I^2 R(T)` loss. Its rising resistance reduces current and field unless the
   supply compensates from a temperature, current, or direct field sensor.

Temperature then feeds back into nearly every quantity the package computes:
coil `R(T)` and `Q(T)` (SNR, probe transfer functions), sample `M0` (Curie
law), `T1`/`T2` (correlation times), sample conductivity `sigma(T)` (loading,
SAR), and the two-temperature noise models. The spin-noise work
([spin_noise.md](spin_noise.md)) made the sample and coil temperatures
first-class, independent parameters -- a thermal solver is what makes them
*dynamic and consistent* instead of user-supplied constants.

## How the implemented pieces connect

Thermal calculations have three parts: a heat **source**, a model that
**transports or stores** that heat, and a temperature-dependent **consumer**.
The package implements all three. Users choose either a lumped thermal network
or a one-dimensional conduction model, then opt into a coupling loop when
temperature must update electrical or sample properties.

Sources:

- `fields.coil_properties` (QOIL port): solenoid AC resistance `R(f)` with
  skin/proximity effects, built on `ConductorMaterial.resistivity_at(T)` --
  a linear temperature coefficient plus an optional RRR/Matthiessen model
  valid to cryogenic temperatures. Coil Joule power at a given drive and duty
  cycle is a one-liner on top of this.
- `fields.coil_peec`: arbitrary-geometry coils with per-sub-filament
  resistances and currents, i.e. a **spatially resolved** Joule-heating
  density along the winding, already parameterized by `rho(T)`.
- `fields.quasistatic`: the induced-E/eddy solver computes the sample power
  deposition `P = (1/2) integral sigma |E|^2 dV` (first-order/Born, valid for
  skin depth >> sample size) -- this **is** the SAR source. Sample loading
  resistance (the `coil_loading_degrades_q_and_raises_noise` path) comes from
  the same machinery.
- `fields.eddy_modes`: self-consistent L/R eddy modes of shields give the
  gradient-switching loss deposited in conducting structures.
- Pulse-sequence timing (`pp` parameters, CPMG/SLSE train definitions,
  gradient waveforms in the imaging/diffusion workflows) defines the duty
  cycles that convert peak powers to average thermal loads.

Consumers:

- `sample.Sample.temperature` -> `M0`, polarization, spin-noise `m_rms`.
- `sp.T` (coil/circuit temperature) -> Johnson terms in all three probe
  noise densities; `spin_noise.SampleCoupling.temperature` and
  `SpinNoiseSource.coil_temperature`/`sample_temperature` -> bump/dip physics.
- `ConductorMaterial.resistivity_at(T)` -> coil `R`, `Q`, probe transfer.
- BPP-style relaxation estimates -> `T1`/`T2` via temperature-dependent
  correlation times (Arrhenius `tau_c(T)`), currently taking `tau_c` as a
  direct input.

Transport and coupling:

- `thermal.network`: lumped heat capacities, fixed-temperature baths, and
  conductive, convective, radiative, or flowing-fluid links.
- `thermal.conduction`: one-dimensional slab, cylinder, and sphere conduction,
  including fixed-temperature, convection, and perfusion boundaries.
- `thermal.coupling`: steady and time-marched quasi-static feedback from
  temperature to coil resistance and sample SAR.
- `thermal.electromagnet`: coupled current, winding temperature, resistance,
  and B0 drift with voltage, current, temperature-compensated, or field control.

## Physics and time scales

Heat equation with sources: `rho_m c_p dT/dt = div(k grad T) + q_v`, with
`q_v` the volumetric source (coil Joule density, sample SAR density) and
boundary conditions of fixed temperature (cryostat, water jacket), convection
(`-k dT/dn = h (T - T_amb)`), and radiation (`epsilon sigma_SB (T^4 - T_amb^4)`,
linearizable for small excursions).

The time-scale hierarchy justifies weak (staged) coupling everywhere:

- RF period: ns-us. Pulse: us-ms. Sequence/train: ms-s.
- Thermal diffusion `L^2/alpha`: seconds (thin wires) to many minutes (sample
  volumes, formers). Convective equilibration: minutes.

So the electromagnetic problem never needs to be re-solved *within* a thermal
step; sources are duty-cycle-averaged powers, and temperature updates feed
back into the EM/spin layers between steps (quasi-static two-way coupling).
The one fast path -- a single long pulse adiabatically heating a small sample
-- is the lumped limit `dT = E_dep / (m c_p)` and needs no solver at all.

## Choose a thermal model

### Lumped thermal network (recommended starting point)

Represent the probe as a nodal RC network: nodes (coil, former, shield,
sample, optionally sample shells or coil sections) with heat capacities
`C_i = m_i c_p,i`, connected by thermal resistances (conduction paths,
convection films, radiation links), driven by per-node average powers, with
one or more bath nodes at fixed temperature. Solve
`C dT/dt = -G(T) (T - T_bath) + P(t)` as a small stiff ODE system (the same
numerical scale as the radiation-damping integrator; `scipy.integrate` or the
package's fixed-step RK4).

- Pros: matches the package's lumped-circuit philosophy (probes are already
  `L,R,C` networks); parameters are measurable (thermal resistances from
  datasheets or a step-response fit); handles cryogenic and room-temperature
  probes alike; steady state is a linear solve; trivially couples to the
  duty-cycle sources and the temperature consumers.
- Cons: no internal temperature *profiles* (hot center vs. cooled surface of
  a lossy sample; hottest-spot SAR); network topology and coefficients must
  be supplied or fitted rather than derived from geometry.
- Validation: analytic 1-2 node step responses; energy conservation; known
  coil warm-up curves.

### One-dimensional finite-difference conduction

A radial (cylinder/sphere) or 1D slab conduction solver for the two places
profiles actually matter: the **sample interior** (SAR density from the eddy
solver peaks off-center; surface is cooled) and the **coil-former-shield
stack**. Crank-Nicolson on a small 1D grid, spatially varying `k`, `rho c_p`,
source density sampled from the quasistatic solver, mixed boundary conditions.

- Pros: cheap, robust, analytically validatable (Fourier/Bessel series
  solutions for slab/cylinder with uniform or parabolic sources); gives
  hot-spot and center-line temperatures the network cannot; slots naturally
  under the network as a "distributed node."
- Cons: only layered/axisymmetric geometry.
- Validation: steady cylinder with uniform source
  (`dT = q_v (a^2 - r^2) / 4k`), transient slab series solution, comparison
  with the lumped limit at low Biot number.

### Electrothermal electromagnet B0 sources

`ElectrothermalElectromagnet` implements the coupled block diagram from
Section 11.2 of the measurements textbook:

```text
L dI/dt = V - I R(T)                 B0 = (B/I)coil I
Cth dT/dt = I^2 R(T) + Ploss - Gth(T - Tamb)
R(T) = Rref rho(T) / rho(Tref)
```

`from_segments` obtains `(B/I)coil` from any existing Biot-Savart coil path.
The time-domain solver returns current, temperature, resistance, voltage,
power, deposited energy, and B0 for four supply modes:

- `voltage`: direct CV operation, so heating produces current and B0 drift;
- `temperature_compensated`: commands `V = Iset R(T)` without sensing current;
- `current`: nominal-pole-canceling PI current feedback;
- `field`: the same PI structure around a direct B0 sensor or field lock.

```python
from spin_dynamics.fields import coils
from spin_dynamics.thermal import ElectromagnetControl, ElectrothermalElectromagnet

paths = coils.solenoid(radius=0.12, length=0.24, turns=144)
magnet = ElectrothermalElectromagnet.from_segments(
    paths,
    inductance_h=25e-3,
    reference_resistance_ohm=0.20,
    heat_capacity_j_per_k=15e3,
    thermal_conductance_w_per_k=8.0,
)
result = magnet.simulate(
    times,
    30e-3,
    control=ElectromagnetControl(
        "field", response_time_s=0.5, voltage_limits_v=(-18, 18)
    ),
)
hardware_b0 = result.uniform_b0()
motion_maps = result.to_motion_field_maps((x_axis, z_axis))
```

The adapters preserve existing hardware and imaging conventions:
`uniform_b0()` returns `experiment.UniformB0`, while
`to_motion_field_maps()` samples the realized coil field and converts its
spatial deviation from the reference field to angular off-resonance. Additional
measured core or eddy loss can be supplied as `additional_power_w`.

This core model is a uniform-temperature, one-thermal-pole representation with
linear `B/I`. It does not infer ferromagnetic hysteresis, saturation,
frequency-dependent core loss, sensor dynamics, or detailed switching-regulator
ripple. Use the existing thermal network/conduction solvers for multiple
thermal nodes and supply measured or separately calculated core loss.

### Three-dimensional conduction (not implemented)

Voxelized `k`, `rho c_p`, and source maps on the existing `fields.domain`
grids (so PEEC winding losses and eddy-solver SAR maps deposit directly onto
the thermal grid), assembled as a scipy sparse system; implicit Euler or
Crank-Nicolson in time, conjugate-gradient solves.

- Pros: arbitrary geometry; reuses grid/interpolation infrastructure; the
  natural endpoint for "thermal solver alongside the other field solvers."
- Cons: an order of magnitude more code and runtime; boundary conditions on
  voxelized surfaces need care; convection still parameterized by film
  coefficients (no CFD).
- Position: a possible future backend after measured cases establish that the
  lumped and one-dimensional models are insufficient.

### Option D -- out of scope (documented boundaries)

- CFD / natural-convection flow solving (film coefficients `h` remain inputs;
  correlations for common geometries can be tabulated).
- Pennes bioheat perfusion term for in-vivo samples -- straightforward to add
  to B/C later (`+ w_b c_b (T_a - T)`), noted for future imaging use cases.
- Thermo-mechanical expansion and permanent-magnet temperature drift remain
  outside this solver. Electromagnet B0 drift from winding `R(T)` is modeled;
  core hysteresis and saturation require separate magnetic material models.
- Full-wave SAR at high field (`sigma ~ omega eps`, wavelength ~ sample): the
  quasistatic Born solver's stated limits apply to its thermal use too.

## Package layout and coupling architecture

The `spin_dynamics.thermal` package separates material data, power sources,
heat transport, and electrothermal feedback:

```text
spin_dynamics/thermal/
  materials.py    # ThermalMaterial (k, rho, c_p, emissivity) + presets
                  # (copper, sapphire, PTFE, water, glass, air film)
  sources.py      # duty-cycled power sources: coil Joule from (sp, pp) and
                  # coil_properties/PEEC; sample SAR from quasistatic eddy
                  # solve; gradient-waveform RMS power; explicit constants
  network.py      # Option A: ThermalNode / ThermalLink / ThermalNetwork,
                  # transient + steady-state solves
  conduction.py   # 1D slab/cylinder/sphere conduction,
                  # same Source/BC dataclasses as network.py
  coupling.py     # quasi-static loop: T -> rho(T), sigma(T), M0(T), tau_c(T)
                  # -> updated sources/consumers -> T ...
  electromagnet.py # coupled RL + thermal B0 source and feedback controllers
```

The coupling loop is explicit and opt-in, like radiation damping:

1. Build sources at current temperatures (coil `R(T_coil)` from
   `ConductorMaterial`, SAR at `sigma(T_sample)`), duty-cycle-averaged over
   the pulse sequence.
2. Advance the thermal model over a macro-step (or solve steady state).
3. Push temperatures back: `sp.T <- T_coil` (noise densities, transfer
   functions via `R(T)`), `Sample.with_temperature(T_sample)` (`M0`,
   spin-noise temperatures), optional Arrhenius `tau_c(T) -> T1/T2`.
4. Repeat; converged fixed point for steady state, or time-march for
   experiment-duration drift studies (e.g. SNR and frequency drift across a
   long CPMG averaging session).

Use `examples/plot_nmr_mouse_thermal.py` for a coupled sample-and-coil case and
`examples/plot_electrothermal_electromagnet.py` for B0 drift and feedback. The
validation suite checks analytic one- and two-node responses, energy
conservation, steady/transient agreement, conduction limits, and coupled
electromagnet fixed points.

## Boundaries and caveats

- Convection enters only through film coefficients; no flow solving.
- The SAR source inherits the quasistatic solver's Born limits (no skin-depth
  shielding inside the sample); high-conductivity samples over-estimate
  deposition, as documented in
  [field_solvers_quasistatic.md](field_solvers_quasistatic.md).
- Thermo-mechanical expansion and permanent-magnet B0 drift remain out of
  scope. Electromagnet drift due to winding `R(T)` is included; nonlinear core
  magnetics are not.
- Temperature dependence of `T1`/`T2` requires a `tau_c(T)` model
  (Arrhenius parameters are sample-specific inputs, not predictions).
