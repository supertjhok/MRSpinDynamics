# Image-Guided Magnetic Therapy

Magnetic particles can act as carriers, local heat sources, or contrast agents.
An external magnet can pull them toward a target, but applying a force is only
half of an image-guided treatment: the system must also estimate where the
particles are, decide how the next field should change, and determine whether
the particles remain localized after they arrive.

In this documentation, a **particle distribution** means the spatial
concentration of particles. The transport model represents it internally with
continuous particle positions. A calibrated particle-sensitive channel turns
those hidden positions into concentration signal, passes that signal through
the nonlinear EPM acquisition, and reconstructs an image. The state estimator
then obtains particle count, image-resolved positions, centroids, and target
occupancy from that reconstructed image.

The controller also reconstructs a **tissue-contrast image** to identify the
target. Tissue contrast and particle concentration are separate measurement
channels in the present model; neither is silently treated as the other.

This guide connects four questions that are otherwise easy to confuse:

1. Can the hardware produce useful imaging and force fields?
2. Can measurements reveal both the target and particle concentration well
   enough to guide a decision?
3. Can magnetic force overcome diffusion and background flow to move particles
   toward a target?
4. Can the particles remain concentrated without relying on an impossible
   static magnetic-field trap?

The current implementation is a mechanism and engineering study, not a
clinical-treatment model. It uses simplified tissue phantoms, dilute
noninteracting particles, prescribed flow, and inferred actuator fields.

Start with [electropermanent magnet hardware](electropermanent_magnets.md) for
rod geometry, retained remanence, programming circuits, hysteresis, or array
state synthesis. Continue here for image formation, transport, feedback, and
trapping stability.

## From delivery concept to simulation

A complete simulated cycle reconstructs the tissue target and the particle
concentration, estimates the uncaptured-particle centroid, programs a force
field from that estimate, transports the hidden physical particles, and then
acquires a verification image. *Target occupancy* is the fraction of recovered
particle signal assigned to the target region. The controller stops from that
image-derived occupancy rather than from simulator truth.

> **Important model boundary:** the particle channel is a calibrated synthetic
> contrast model with a supplied signal per particle. Its estimated positions
> are representative locations at the image resolution; they describe the
> recovered distribution and do not preserve individual particle identities.
> True simulator positions appear only in diagnostics that score centroid and
> occupancy error.

## System workflow

The implemented system is organized as five successive layers:

1. **Actuator state.** A coil, EPM array, or hybrid EPM-plus-coil system produces
   a spatial field and gradient from an explicit hardware state.
2. **Image formation.** Multiple nonlinear EPM operating states separately
   encode a tissue target and a particle-sensitive concentration image.
3. **Transport.** Superparamagnetic particles move under magnetic force,
   background flow, Brownian diffusion, and reflecting boundaries.
4. **Feedback control.** The reconstructed tissue image supplies the target;
   the reconstructed particle image supplies the uncaptured centroid and target
   occupancy used for steering and stopping.
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

The resulting tissue image supplies target localization to the controller.

## Particle-sensitive imaging and state estimation

`particle_distribution_image` uses bilinear cloud-in-cell deposition to convert
continuous hidden positions into calibrated concentration signal while
preserving total signal and the ensemble centroid. `run_epm_particle_imaging`
encodes that image, adds complex acquisition noise, performs the nonnegative
reconstruction, and passes only the reconstructed image to
`estimate_particle_state_from_image`.

The estimate contains:

- recovered particle count from total signal and the supplied per-particle
  calibration;
- representative image-resolved positions obtained by deterministic resampling
  of the recovered concentration;
- whole-cloud and outside-target centroids;
- target occupancy; and
- voxel resolution and support-threshold diagnostics.

The transport model immobilizes a particle at first target entry, so captured
particles accumulate near the circular boundary. A hard voxel-center mask would
systematically lose the portion of each bilinear footprint outside that mask.
For this controller, `particle_boundary_capture_correction=True` calibrates the
boundary response from the known grid and target geometry and corrects the
occupancy. The calibration never reads the hidden particle positions and can be
disabled for transport models that do not capture at a boundary.

The result keeps the hidden simulator positions in explicitly named
`ground_truth_*` diagnostics so centroid and occupancy error can be plotted.
The controller never uses those fields for a decision.

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

`run_epm_image_guided_controller` alternates among initial imaging, EPM
programming, transport, and verification imaging. Each cycle reconstructs the
tissue target and particle concentration, aims a transport field from the
estimated outside-target centroid, advances the physical transport model, and
then reconstructs the new particle state. The post-transport image supplies the
reported capture estimate and stopping decision.

```bash
python examples/plot_epm_closed_loop_controller.py \
  --output results/epm_closed_loop_controller.png
```

The controller is useful for studying observability, image resolution,
cadence, switching burden, and estimator error. The bundled example plots
estimated positions over hidden truth, image-derived versus true capture, and
centroid error by cycle. It is still a synthetic proof of mechanism: particle
contrast, noise, model mismatch, and biological transport need
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
- particle-sensitive MR contrast and signal-per-particle calibration;
- viscosity, flow, adhesion, and permeability measurements for the target
  tissue or phantom; and
- electrical pulse energy, driver limits, and temperature rise for the chosen
  hardware.

See the [validation evidence matrix](validation_matrix.md) for the exact claims
and reproducible tests. The dynamic-inversion mechanism is based on
[Nacev et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4296920/).
