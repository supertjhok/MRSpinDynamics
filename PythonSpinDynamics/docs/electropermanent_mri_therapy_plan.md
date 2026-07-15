# Electropermanent MRI and image-guided therapy plan

## Status and scope

This is a design document for a future PythonSpinDynamics feature.  It was
refined on 2026-07-15 after reviewing the local Weinberg Medical Physics
archive.  The first implementation phase models and validates individual
electropermanent magnets (EPMs) before adding array synthesis, nonlinear
imaging, particle transport, or closed-loop therapy.

Implementation progress as of 2026-07-15:

- **Complete:** evidence-tagged AlNiCo material and retained-state records;
  arbitrary-axis finite rods; documented one-rod and 37-rod presets; explicit
  close-packed bundles; area-equivalent cylinder reduction; exact on-axis and
  cubature field solvers; imaging/motion field-map adapters; static tests,
  documentation, and the magnet-only plotting example.
- **Complete:** capacitor/H-bridge/series-RLC programming pulses, passive
  recovery, electrical/thermal loss accounting, state-dependent-inductance and
  neighbor-field hooks, and configuration-specific regression against the
  archived 220/400/600 V peak-current traces.
- **Complete at empirical tier:** a protocol-scoped saturate-then-demagnetize
  interpolation with branch/polarity guards, uncertainty, and reversal-point
  recording. The published preset intentionally uses only three defensible
  anchors because the raw demagnetization samples remain unavailable.
- **Complete at mathematical tier:** weighted play-operator return-point memory,
  wiping-out, temperature-scaled saturation, geometry-derived neighbor
  coupling, optional self-demagnetizing diagonals, and fixed-point coupling of
  state-dependent inductance, pulse waveform, bias field, and retained state.
  The built-in play distribution remains explicitly illustrative because raw
  minor-loop calibration data are unavailable.
- **Complete at transient two-element tier:** mutually coupled winding currents,
  induced neighbor voltage, closed recovery paths, winding-field leakage,
  sample-by-sample disturbance of every return-point state, state-dependent
  inductance iteration, array-channel losses, temperatures, and energy checks.
  The mutual and leakage matrices remain explicit inferred inputs pending
  measurement.
- **Next:** hybrid NdFeB/AlNiCo sub-units, field-basis assembly, constrained
  state synthesis, and then the full 18-sub-unit/72-element array. Measured
  minor loops and coupling matrices remain parallel calibration work.

The textbook starting point is Section 11.2, printed pages 603--604, of
`References/Measurements_Book_2_Final_WEB.pdf`.  Figure 11.6 describes an
AlNiCo-5 electropermanent B0 assembly, RF coil, pulse-power unit, and
spectrometer.  Short high-current pulses program the remanence of individual
magnet elements.  Multiple programmed field distributions can support imaging
and field-guided therapy, but nonlinear profiles require general encoding and
reconstruction rather than an FFT-only workflow.

All numerical inputs used by the implementation must carry provenance and one
of four evidence tags: **measured**, **simulated**, **specified**, or
**inferred**.  Revisions and hardware generations must remain separate unless
there is evidence that they describe the same device.

## Literature anchors

- Nacev et al., *Tunable Electropermanent System for Magnetic Resonance Imaging
  and Magnetic Particle Propulsion*, ISMRM 2017.  The reported prototype was a
  two-sided array with 72 independently adjustable elements, an adjustable
  1--40 cm gap, and approximately 0.5--150 mT field range.  Each sub-unit
  combined fixed NdFeB with coil-programmed AlNiCo 5-7.
  <https://cds.ismrm.org/protected/17MProceedings/PDFfiles/4434.html>
- Ropp et al., *Electropermanent magnets for variable-field NMR*, Journal of
  Magnetic Resonance 303 (2019), 82--90.  An AlNiCo-5 array was individually
  programmed for variable-field NMR over 0.5--2 MHz proton frequency.
  <https://doi.org/10.1016/j.jmr.2019.04.010>
- Weinberg et al., *Platform for Image-Guided Noninvasive Brain Delivery of
  Magnetic Particles: Concept and Technical Progress*, IEEE Magnetics Letters 9
  (2018), 1--5.  <https://doi.org/10.1109/LMAG.2018.2837649>
- Jafari et al., *Opening the Blood Brain Barrier with an Electropermanent
  Magnet System*, Pharmaceutics 14 (2022), 1503.  The therapeutic experiment
  used four EPM assemblies.  Each assembly was a 3 x 3 x 20 cm bundle of 200
  AlNiCo-5 rods with a 40-turn programming coil; approximately 1 kA, 50 us
  pulses generated fields up to 150 mT.  This is useful pulse and material
  evidence, but is not the same geometry as the 72-element MRI prototype.
  <https://doi.org/10.3390/pharmaceutics14071503>
- Nacev et al., *Dynamic Inversion Enables External Magnets To Concentrate
  Ferromagnetic Rods to a Central Target*, Nano Letters 15 (2015), 359--364.
  <https://doi.org/10.1021/nl503654t>

## Findings from the local project archive

The most useful source directory is
`C:/Users/super/Dropbox/Weinberg_Medical_Physics`.  The custom MRI front-end
directory mostly concerns RF electronics; its magnet-pulser design review is
still useful for electrical stress limits.

| Local source | Evidence recovered | Intended use |
| --- | --- | --- |
| `documents/Tunable Electropermanent System for Magnetic Resonance Imaging Poster ii.pdf` | **Specified/presented:** two-sided hybrid EPM system; 18 sub-units and 72 adjustable AlNiCo elements; 4 cm spherical imaging/control volume; 15 cm small-animal gap; 0.5--150 mT system range.  The poster describes 600 A programming current, 0.5 T inside the AlNiCo cores, 30 ms saturation pulses, and fourteen 12 V batteries capable of 800 A.  It explicitly notes nonlinear hysteresis, recoil permeability greater than one at low actuation, and the need to model or measure these effects. | Array hierarchy, system acceptance targets, and a warning not to use a memoryless remanence law. |
| `NQR_proposal_2019/figures/demag_curve_alnico.*` and `electropermanent_magnet_system.*` | **Published:** four one-inch-diameter, six-inch-long AlNiCo-5 rods; about 150 mT on-axis at 1 mm from a fully magnetized rod; 50 turns of 14 AWG wire and approximately 25 uH per coil.  Pulse polarity can magnetize, demagnetize, or reverse a rod. | First simple geometry and field benchmark. |
| `NQR_proposal_2019/figures/demag_data.*` | **Measured/published figure:** a strongly nonlinear, monotonic field transition from approximately +150 to -150 mT as programming-pulse duty cycle increases.  The zero crossing is near 17 percent in the plotted experiment. | Regression target for an empirical programming law.  Do not treat pixels digitized from this cropped figure as raw measurements without uncertainty metadata. |
| `MRI/IGBT_board/P10801-001-A02 Magnet bundle design(2).docx` and `Magnet Cabling and other physical attributes 05Jan2020.xlsx` | **Specified/CAD:** a bundle contains 37 close-packed AlNiCo-5 rods, each 1/8 inch diameter and 4 inches long; approximately 60 turns of 16 AWG magnet wire; estimated inductance about 50 uH.  The drawings show the rod bundle, winding former, shielding, Hall-sensor end board, and a 3 x 3 bundle enclosure. | Second, more detailed element benchmark and engineering geometry. |
| `update_reports/Magnet Coil Calculator.xlsx` | **Calculated design cases:** AC500 inputs Hc = 0.64 kOe and Br = 12.7 kG.  A 35 mm x 150 mm, 84-turn, 16 AWG design gives 51.3 uH and 0.547 ohm in the workbook; alternative voltage, winding, and capacitor cases are retained in the sheet. | Driver sanity checks only; these dimensions are not silently assigned to the 37-rod bundle. |
| `MRI/IGBT_board/experimental results.pdf` | **Measured:** examples include about 330 A from 220 V/50 us, 170 A from 400 V/20 us, and 147 A from 600 V/10 us.  The traces show the programming current ramp and recovery transient. | Pulse-driver waveform regression with configuration-specific labels. |
| `update_reports/20200525_IGBT_update.docx` | **Measured:** 600 V, 20 us bundle tests; current/sensor sweeps in both polarities; neighboring magnets affect the achievable state, especially when an adjacent magnet is saturated; reversal produces additional transients. | Mandatory mutual-interaction test and evidence that isolated-element calibration is insufficient for the final array. |
| `update_reports/WMP_pulse_generator_070620.pptx` | **Specified/simulated:** design goal up to 300 A for 50 us; distributed local H-bridges; 18--36 V bus and nominal 200 V local capacitor bank; current feedback; state-dependent L and C.  One design example uses L about 27 uH and R about 0.24 ohm at 0.1 MHz. | Circuit-model topology, closed-loop current control, and electrothermal coupling. |
| `simulations/RK4_2020_06_04_H_bridge.m` | **Simulated:** a switched series-RLC pulse calculation with capacitor voltage, coil current, inductor voltage, and collector-emitter stress. | Cross-check for the first circuit integrator, not a production reference model. |
| `NQR_proposal_2019/simulations/*.FEM` and `*.ans` | **Simulated:** solved FEMM cases for a two-magnet gap and a four-element Halbach-like layout.  The files contain tabulated AlNiCo-5 and AlNiCo-8 B-H data; the AlNiCo-5 table reaches about 1.254 T at 50.99 kA/m. | Static-field regression fixtures and material-curve import tests.  These files substitute AlNiCo for fixed magnets and do not themselves model pulse history. |
| `200415 Weinberg Custom MRI Front-End/MPR Magnet Pulser PCB Design Review 2021_11_21.pdf` | **Measured/design review:** later electronics exhibited approximately 1460 V overshoot for less than 5 us and used a 1200 A surge requirement. | Optional hardware-stress envelope.  It must not be confused with the nominal pulse of the earlier 300 A element driver. |

### Consequences for the model

1. Geometry, magnetic history, pulse electronics, and temperature must be
   separate model objects.  A static B-H curve is useful to a field solver but
   cannot predict the retained state after a pulse.
2. The model must represent reversal points or an equivalent return-point
   memory.  Saturating, partially demagnetizing, and reversing an element are
   distinct operations.
3. The programming field is not simply `N I / length`: demagnetizing field,
   recoil permeability, rod aspect ratio, and fields from neighboring elements
   change the internal H field.
4. Coil inductance may change with magnet state.  The circuit and magnetic
   update therefore require either a coupled integration or a documented
   fixed-point approximation.
5. Multiple historical prototypes are valuable validation configurations, not
   interchangeable parameter sets.

## Reference configurations

### Single-element benchmark A: published rod

- One AlNiCo-5 cylinder, diameter 25.4 mm and length 152.4 mm.
- 50-turn 14 AWG programming coil, nominal L = 25 uH.
- Full-state field target of approximately 150 mT on axis, 1 mm from the face.
- Four-element arrangement for comparison with the variable-field NMR figure.

This deliberately simple benchmark should be implemented first because it can
separate the permanent-magnet field calculation from pulse programming.

### Single-element benchmark B: local rod bundle

- 37 close-packed AlNiCo-5 rods, each 3.175 mm diameter and 101.6 mm long.
- Approximately 60 turns of 16 AWG wire, nominal L about 50 uH.
- Hall-sensor endpoint, shielding, and winding former represented as engineering
  metadata initially; conductive shielding is included when transient eddy
  currents are enabled.
- Measured 600 V/20 us programming and current/sensor sweeps used as the first
  state-transition regression target.

### Hybrid sub-unit and imaging array

The array reference geometry is revised to follow the poster hierarchy:

| Component | Proposed model |
| --- | --- |
| Magnet panels | Two opposing panels, each containing a 3 x 3 grid of hybrid sub-units |
| Hybrid sub-unit | One fixed NdFeB contribution with four independently programmable AlNiCo elements arranged around it |
| Total programmable elements | 18 sub-units x 4 AlNiCo elements = 72 |
| Initial validation ROI | 40 mm diameter spherical volume at a 150 mm panel gap |
| Scaled therapy ROI | Parameterized head-scale ROI; dimensions remain explicitly inferred until a drawing or measurement is selected |
| Imaging state | Constrained uniform-field state between 0.5 and 150 mT, followed by nonlinear encoding configurations |
| Transport states | Directional force configurations with the uniform imaging component reduced or disabled |
| RF hardware | Central broadband solenoid or saddle coil |
| Driver | Per-element or multiplexed H-bridge with configurable voltage/current limits, pulse duration, capacitor droop, switching energy, and coil heating |

The array field can use a precomputed basis in the linear fast path,

```text
B(r) = B_NdFeB(r) + sum_i Br_i K_i(r),
```

where `K_i` is the unit-remanence field of programmable element `i`.  A coupled
field solve is required when magnetic interaction materially changes `Br_i` or
the internal programming field.

## EPM-first implementation phase

### 1. Data and state representation

Add immutable, unit-explicit records for:

- tabulated AlNiCo major curves and recoil data;
- `RemanenceState` containing magnetization vector, branch/reversal history,
  temperature, calibration identifier, and uncertainty;
- cylindrical rods and close-packed rod bundles;
- programming coils, sensors, yokes/shields, and fixed NdFeB components; and
- evidence/provenance metadata for every preset.

Candidate public types are `AlNiCoMaterial`, `RemanenceState`,
`ElectropermanentRod`, `ElectropermanentBundle`, `HybridEPMSubunit`, and
`ElectropermanentArray`.

### 2. Magnetostatic field model

Implement finite-cylinder fields for the rod benchmark using either stable
quadrature of bound surface currents or the existing finite-magnet cubature.
Build a rod bundle by superposition, retaining individual rods when internal
demagnetizing fields are requested and allowing an equivalent-cylinder fast
mode.  Add FEMM import/comparison helpers for the saved `.FEM`/`.ans` fixtures.

Acceptance tests cover axial and transverse symmetry, far-field dipole limits,
convergence with quadrature order, bundle-versus-explicit-rod error, and the
published approximately 150 mT surface-field benchmark.

### 3. Pulse-power model

Represent the capacitor bank, H-bridge polarity and switching state, winding
R/L, state-dependent inductance, current/voltage limits, parasitic resistance,
recovery path, switching energy, and coil/device heating.  The primary output
is the actual `I(t)` and applied `H(t)`, not merely the requested pulse width.

Start with a piecewise RLC state-space integrator and reproduce the archived
220/400/600 V traces.  Add an optional closed-loop current controller after the
open-loop waveforms agree.  Keep 50 us and 30 ms pulse families as different
driver presets rather than forcing one circuit to explain both.

Candidate public types are `PulsePowerDriver`, `ProgrammingPulse`,
`PulseWaveform`, and `PulseThermalState`.

### 4. History-dependent programming law

Implement the state transition

```text
state_new = program(state_old, H_internal(t), temperature)
```

in increasing fidelity tiers:

1. **Empirical monotonic map:** saturate to a known endpoint, then interpolate a
   calibrated pulse-amplitude/duration/duty table.  This is sufficient to
   reproduce the published demagnetization sweep but is intentionally limited
   to its calibration protocol.
2. **Return-point model:** add reversal memory using a Preisach-, play-, or
   stop-operator representation calibrated to major and minor loops.
3. **Coupled internal-field model:** update the state with self-demagnetizing
   and neighbor fields, iterating with the magnetostatic solver when needed.

The model must expose saturation, partial programming, reversal, retention,
temperature sensitivity, and an uncertainty estimate.  No default should
invent unmeasured minor loops while appearing quantitative.

### 5. Validation order

1. Static field of one published rod.
2. Static field of the 37-rod bundle and equivalent-cylinder error.
3. RLC current traces at the three archived voltage/pulse-width examples.
4. Saturate-then-demagnetize transfer curve from the published duty-cycle plot.
5. Bipolar current/sensor sweep from the May 2020 bundle test.
6. Two-element neighbor-interaction case, including the saturated-neighbor
   failure mode described in the test report.
7. Saved FEMM gap and four-element field maps.
8. Four-rod variable-field NMR assembly, then one hybrid sub-unit.
9. Only after these pass, the 18-sub-unit/72-element array.

The next implementation task is therefore a small `epm` module containing the
material/state records, a finite AlNiCo cylinder, a close-packed bundle
builder, and static-field validation fixtures.  Pulse-driven state updates
follow once the geometry benchmark is stable.

## Later system phases

After the EPM model is validated:

1. Add constrained state synthesis for uniform imaging, nonlinear encoding,
   field-off, and directional transport objectives.
2. Add nonlinear MR encoding,
   `s_k = sum_j rho_j exp(-i gamma integral B_k(r_j, t) dt)`, and regularized
   reconstruction.  Conventional FFT reconstruction is not sufficient for the
   deliberately nonlinear profiles.
3. Add magnetophoretic particle/cluster transport.  In the linear
   superparamagnetic regime,
   `F = V DeltaChi grad(|B|^2) / (2 mu0)`, with a Langevin or saturation cap at
   higher fields, plus drag, background flow, Brownian diffusion, and boundary
   interactions.
4. Add an integrated controller alternating image/localization windows, EPM
   programming, and transport bursts.  Static central trapping must not be
   implied; dynamic inversion is a separate protocol for central focusing.
5. Connect switching transients to the existing eddy-current, imaging, motion,
   engineering-metrics, and electrothermal workflows.

## Plotting example plan

The first magnet-only example, `examples/plot_electropermanent_magnet.py`,
should precede the full therapy example.  It should plot:

1. an exploded rod/bundle/coil geometry;
2. axial and transverse static-field profiles for partial remanence states;
3. the programming current and capacitor voltage versus time;
4. retained field versus pulse command, including reversal history;
5. coil temperature and pulse energy over a pulse train; and
6. the field error caused by a saturated neighboring element.

It should print the provenance of the selected preset, surface field, magnetic
moment, peak current, peak internal H, pulse energy, maximum temperature,
retained remanence, and field-model error relative to its validation fixture.

The later integrated
`examples/plot_epm_image_guided_transport.py` should create a 2 x 3 figure:

1. opposing 3 x 3 hybrid-sub-unit panels, ROI, RF coil, and color-coded AlNiCo
   remanence;
2. imaging-mode Bz map with ROI homogeneity and Larmor-frequency contours;
3. synthetic phantom and reconstruction from nonlinear EPM encoding profiles;
4. transport-mode `|B|`, `grad(|B|^2)`, and magnetic-force vectors;
5. particle trajectories through a branching vessel or channel toward an
   image-selected off-center target, with capture fraction; and
6. a multiscale timeline showing programming pulses, central B0, transport
   gradient, switching energy, magnet temperature, and alternating imaging and
   transport windows.

The integrated example should print achieved field homogeneity, off-state
residual, encoding-matrix condition number, peak and median gradient, force
range, switching energy, maximum temperature, and target capture fraction.

## Open data-recovery tasks

- Recover the raw data behind `demag_data.*`; until then, any digitization must
  include graphical-extraction uncertainty.
- Extract centerline and ROI field samples from the solved FEMM `.ans` files
  for versioned regression fixtures.
- Determine whether the 37-rod bundle test used the stated approximately
  60-turn/50-uH winding in every archived oscilloscope trace.
- Locate measured minor loops, temperature sweeps, or repeated-pulse retention
  data.  These are required before a quantitative default Preisach calibration
  can be claimed.
- Resolve the 30 ms/600 A poster driver, the 50 us/300 A distributed driver,
  and later approximately 1 kA therapeutic driver as separate hardware
  generations.
