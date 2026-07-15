# Image-Guided Magnetic Therapy

This guide collects the imaging, particle-transport, feedback-control, and
dynamic-trapping workflows into one system-level path. These models use field
sources and hardware models, but they are not field solvers: their purpose is
to answer whether a measured particle distribution can be reconstructed,
steered toward a target, and kept localized under explicit flow and actuator
constraints.

Start with [electropermanent magnet hardware](electropermanent_magnets.md) when
you need rod geometry, retained remanence, programming circuits, hysteresis, or
array-state synthesis. Start here when the scientific question concerns image
formation, particle delivery, closed-loop control, or trapping stability.

## System workflow

The implemented system is organized as five successive layers:

1. **Actuator state.** A coil, EPM array, or hybrid EPM-plus-coil system produces
   a spatial field and gradient from an explicit hardware state.
2. **Image formation.** Multiple nonlinear EPM operating states encode a tissue
   or particle distribution, including transmit and receive sensitivity.
3. **Transport.** Superparamagnetic particles move under magnetic force,
   background flow, Brownian diffusion, and reflecting boundaries.
4. **Feedback control.** Reconstructed particle position and target occupancy
   select the next transport state instead of assuming perfect open-loop
   delivery.
5. **Dynamic trapping.** Ferromagnetic rods retain their polarization briefly
   while an opposing gradient produces repulsion. Cycling the source direction
   creates time-averaged central localization without claiming a stable static
   magnetic trap.

The layers can be studied independently. A transport study does not require a
synthetic imaging result, and a hardware assessment can be run without a
particle simulation.

## Choose the particle model first

The transport and dynamic-inversion models represent different magnetic
regimes and should not be substituted silently.

| Scientific question | Particle model | Decisive parameter |
|---|---|---|
| Can a dilute injectable particle cloud be moved through a field gradient? | Superparamagnetic sphere or effective particle | Susceptibility, hydrodynamic size, viscosity, and flow |
| Can an already magnetized particle be repelled before it rotates? | Ferromagnetic rod or sphere with remanent moment | Body rotational diffusion and internal moment-relaxation time |
| Does a repeated pulse sequence concentrate particles? | Dynamic-inversion ensemble | Polarize-delay-gradient timing relative to orientation memory |
| Can particles remain localized under continuing flow? | Closed-loop or dynamic-inversion simulation | Concentration gain, target retention, escape fraction, and controller cadence |

For a sphere, the simulator recovers Stokes translational and rotational drag.
For a rod, it uses the Tirado-Martinez-de la Torre finite-cylinder friction
expressions, including separate mobility parallel and perpendicular to the rod.
An optional internal moment-relaxation time exposes the important failure mode
in which magnetization follows the inverted field faster than the body can
rotate.

## Nonlinear EPM imaging

`simulate_epm_nonlinear_imaging` constructs state-dependent encoding matrices
from the EPM array field basis. It separates magnetic-field encoding,
transmit/receive sensitivity, tissue parameters, regularization, and noise so a
reconstruction failure is not mistaken for a transport failure.

Run the reference example with:

```bash
python examples/plot_epm_nonlinear_tissue_imaging.py \
  --output results/epm_nonlinear_tissue.png
```

The resulting state set can be passed to the controller as a measured or
synthetic localization source.

## Image-guided particle transport

`simulate_image_guided_transport` propagates a dilute, noninteracting particle
ensemble through a spatial magnetic force field. It includes deterministic
background velocity, Brownian displacement, a target region, and reflecting
domain boundaries. The output preserves the trajectory history and reports
target occupancy and capture gain.

```bash
python examples/plot_epm_image_guided_transport.py \
  --output results/epm_image_guided_transport.png
```

This is a mechanism model. It does not yet include aggregation, adhesion,
vascular permeability, particle-particle magnetic interaction, or a full
pharmacokinetic compartment model.

## Alternating imaging and therapy control

`run_epm_image_guided_controller` alternates among imaging, EPM programming,
and transport intervals. Each cycle reconstructs the particle distribution,
identifies uncaptured particles, synthesizes or selects a transport field, and
records the resulting target capture and programming effort.

```bash
python examples/plot_epm_closed_loop_controller.py \
  --output results/epm_closed_loop_controller.png
```

The controller is useful for studying cadence and observability. It should not
be read as evidence that tissue concentration can be inferred perfectly from a
single image; measurement noise, model mismatch, and biological transport need
experiment-specific calibration.

## Dynamic-inversion trapping

`simulate_dynamic_inversion` implements the pulse mechanism reported by Nacev
et al.: a uniform field polarizes a ferromagnetic particle, a short delay
follows, and an oppositely directed gradient repels the still-antialigned
moment. Repeating the sequence from four directions produces an average inward
drift. The preset preserves the reported 600 microsecond polarization pulse,
5 microsecond delay, 50 microsecond gradient pulse, and 60.6 millisecond element
period; field magnitudes and source geometry remain visible inferred inputs.

```bash
python examples/plot_epm_dynamic_inversion.py \
  --output results/epm_dynamic_inversion.png
```

The example sweeps particle shape and size, compares a rigid long rod with a
rapidly relaxing moment, and reports concentration and escape metrics. A
concentration gain above one indicates contraction of the ensemble RMS radius;
it is not by itself a biological capture prediction.

## Hardware architecture consequences

`assess_dynamic_inversion_hardware` compares the switching burden without
pretending that programming-count metrics are electrical energy.

| Architecture | Strength | Principal limitation |
|---|---|---|
| Fast coils | Highest pulse-shape fidelity and straightforward polarity reversal | Repetitive copper loss, driver stress, and cooling |
| EPM only | Retains fields without continuous power | Every fast element needs calibrated polarize, gradient, and field-off states; programming transients dominate |
| Hybrid EPM plus coils | EPM supplies slow bias and spatial shaping while coils supply fast inversion | Still requires coil power and a calibrated superposition of retained and driven fields |

For the default 72-channel, 9.1-minute illustration, coil-only operation uses
18,018 fast pulses. EPM-only operation requires 27,027 retained-state changes,
or about 1.95 million channel programming pulses. The hybrid uses two EPM
states, 144 EPM channel pulses, and 18,018 coil inversion pulses. These are
switching counts, not thermal or energy predictions.

The present engineering interpretation is therefore:

- use coils when pulse fidelity is the dominant requirement and continuous
  electrical/thermal support is acceptable;
- use EPMs for slowly changed bias, shaping, imaging, or parked states;
- use a hybrid for a dynamic-inversion prototype unless measured EPM
  programming data demonstrate repeatable sub-millisecond, multistate operation.

## Evidence boundary and next measurements

The mechanism and software invariants are covered by analytical and regression
tests. Quantitative system prediction still requires:

- measured three-dimensional coil and EPM field maps;
- calibrated programming transients, state uncertainty, and channel coupling;
- particle remanence and internal relaxation across the relevant size range;
- viscosity, flow, adhesion, and permeability measurements for the target
  tissue or phantom; and
- electrical pulse energy, driver limits, and temperature rise for the chosen
  hardware.

See the [validation evidence matrix](validation_matrix.md) for the exact claims
and reproducible tests. The dynamic-inversion mechanism is based on
[Nacev et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4296920/).
