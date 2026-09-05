# 14N NQR mail-screening system design study

> **Status (2026-09-05): Phase 0 revised for pharmaceutical envelopes; Gate 0 open.** The
> versioned requirements, evidence classes, development material library,
> result-record schema, test-set policy, approval record, and validator live in `phase0/`.
> Fifteen blocking requirements and stakeholder sign-off remain pending;
> second-target spectroscopy is incomplete for subsequent modeling. This document
> does not claim a scanner design, detection limit, or screening performance.

## Purpose and reference concept

The study asks a system-level question: for a specified class of mail parcels
and a curated library of crystalline 14N-bearing pharmaceutical target and benign materials,
which coil, resonator, pulse/acquisition schedule, receiver, RFI mitigation,
and decision rule gives the best screening performance within time, power,
thermal, voltage, current, and size constraints?

The user scope is typical USPS/UPS/FedEx envelopes at 0�50 degrees C,
approximately 5 mm access clearance, and illicit pharmaceuticals with synthetic
opioids prioritized. Pulsed SLSE/SORC and a pre-polarization comparison are
required study branches. See `phase0/user_requirements.md` and
`phase0/envelope_scope.md` for exact inputs and unresolved interpretations.

The provisional reference concept is a stop-and-scan envelope station. A
parcel is placed inside or passed through a shielded RF aperture; an inductive
transmit/receive resonator applies one or more pulsed NQR interrogations, and a
receiver combines echo-train evidence across selected 14N lines. External
reference coils support RFI cancellation. OPM or SQUID readout may be evaluated
as comparison branches, but the first baseline is an inductive transceiver
because it provides the most direct path to an integrated system model.

This reference concept is provisional. The study must not assign a detection
limit until the parcel envelope, allowed scan time, target-material test set,
false-alarm requirement, and hardware limits have been supplied and the
absolute signal chain has been validated.

## Decision problem

Let \(d\) contain controllable design choices:

- aperture, coil geometry, segmentation or array layout, shield geometry, and
  conductor;
- tuning, matching, loaded quality factor, damping or Q switching, and receive
  bandwidth;
- pulse family, carrier frequencies, RF waveform, pulse length and amplitude,
  echo spacing, echo count, phase cycle, repetitions, recovery time, and line
  ordering;
- transmit/receive switching, blanking, gain, filtering, ADC settings, and RFI
  reference layout;
- signal features, multi-line fusion, threshold, and adaptive stopping policy.

Let ξ collect nuisance variables that the system cannot control exactly:

- target identity, mass, morphology, crystalline fraction, location, and pose;
- parcel size and contents, dielectric and conductive loading, and temperature;
- NQR frequency, linewidth, EFG distribution, \(T_1\), \(T_{2e}\), and their
  uncertainty;
- component tolerances, tuning drift, transmit-field variation, receiver
  drift, ambient RFI, and random noise.

For a simulated record \(y \sim p(y\mid d,\xi,H)\), \(H_1\) denotes a target-
present parcel and \(H_0\) a target-absent parcel drawn from a realistic benign
population. The primary performance measures are:

\[
P_D(d)=\Pr[\delta(y;d)=1\mid H_1],
\qquad
P_{FA}(d)=\Pr[\delta(y;d)=1\mid H_0].
\]

The user-directed primary objectives are to maximize ROC AUC and minimize full
physical time per envelope. Compare AUC at matched physical times and retain an
AUC-versus-time Pareto frontier with uncertainty. Population definitions and
amount distributions must be frozen so that AUC comparisons mean the same thing.
PD/PFA at selected thresholds remain diagnostics and future deployment criteria;
no fixed operating-point or scan-time cap was supplied. No user-imposed RF power
cap applies, while component, thermal and installation feasibility still applies.
SNR is an intermediate diagnostic, not a replacement for ROC evaluation.

A Pareto front should be retained rather than collapsing all engineering goals
into an arbitrary weighted sum. The front should expose at least scan time,
detection performance, peak RF power, average dissipation, aperture or active
volume, and implementation complexity.

## Components of the study

| Component | Required outputs | Existing MRSpinDynamics foundation | Study-specific work |
| --- | --- | --- | --- |
| 1. Requirements and scenario model | Envelope geometry, operating mode, AUC/time objectives, target and benign populations, environmental and hardware limits | Experiment specifications and physical acquisition-cost abstractions | Versioned requirement/scenario schema; explicit provisional versus measured inputs |
| 2. Spectral and relaxation library | Lines, strengths, linewidths, \(T_1\), \(T_{2e}\), temperature coefficients, uncertainty, provenance | `NQRDatabase`, `mr_integration`, NQR Hamiltonians and temperature/EFG distributions | Screening-oriented material records, confidence grades, benign-confuser set, missing-data priors |
| 3. Parcel digital phantom | Material mass and crystalline fraction, 3-D position/pose, packing, temperature, dielectric/conductive loading | Spatial grids and composition adapters | Voxel or region-based parcel model and reproducible scenario sampler |
| 4. RF aperture and fields | \(B_1^+(\mathbf r)/I\), receive sensitivity, filling factor, active volume, homogeneity, inductance, resistance, capacitance, SRF, shield loss | Biot-Savart, coil properties, PEEC, grounded-box and shield models, gradiometers | Parcel-scale coil candidates, Tx/Rx reciprocity bridge, loading/tolerance parameterization |
| 5. Resonator and transmit chain | Delivered current and \(B_1\), bandwidth, ringdown, detuning, peak voltage/current, pulse energy and amplifier demand | Tuned/matched/untuned probes, receiver networks, resonator time constant, control-response models | Q switching or damping model if required; parcel-dependent retuning and amplifier constraints |
| 6. NQR spin response | Orientation- and offset-resolved FID/SLSE/SORC response, multi-line behavior, relaxation and temperature dependence | Spin-1 14N Hamiltonians, powder averaging, reduced/full engines, SLSE/SORC, EFG and temperature distributions | Spatially varying \(B_1\), material-scale normalization, recovery/history across a multi-line schedule |
| 7. Absolute signal transduction | Magnetic moment or field spectrum at the pickup, detector output in volts, calibration uncertainty | Boltzmann populations, detector spatial coupling, inductive/OPM/SQUID detectors, field-referred SNR | Validated bridge from nuclei count and density-matrix response to moment, pickup flux/EMF, network gain, and ADC units |
| 8. Receiver, noise, and interference | Noise PSD/covariance, saturation and recovery, valid acquisition windows, residual RFI distribution | Probe-noise PSDs, receiver front end and networks, acquisition masks, multi-reference and adaptive RFI cancellers | ADC and overload model; measured-site RFI priors; freeze/adapt rules that protect NQR windows |
| 9. Signal processing and decision | Matched-filter or likelihood scores, line/echo features, multi-line evidence, calibrated decision and stopping rule | Matched-filter SNR, Bayesian design utilities, NQR/RFI parameter estimators | Target-present/absent likelihoods, nuisance marginalization, threshold calibration, sequential multi-line policy |
| 10. Thermal and safety constraints | Coil, capacitor, shield, amplifier and parcel temperature trajectories; duty-cycle limits | Lumped and conduction thermal models, electro-thermal coupling | RF loss-to-thermal network, component ratings, exposure constraints supplied as requirements |
| 11. Robust optimization | Feasible Pareto set and sensitivity to uncertain assumptions | Sweeps, bounded/scipy/JAX optimization, GRAPE, robust ensembles, Bayesian adaptive design | Mixed discrete/continuous system optimizer, surrogate or staged evaluation, confidence-aware constraints |
| 12. Verification and experimental validation | Traceable evidence from unit checks through blind parcel trials | Cross-validation, fixtures, evidence records, experiment provenance | Absolute calibration experiments, loading phantoms, blind target/benign test protocol, domain-shift checks |

## End-to-end model boundary

The study should make every conversion between normalized spin dynamics and an
instrument decision explicit:

```text
requirements + target/benign library
                |
                v
      parcel scenario sampler -----> loading + temperature
                |                          |
                v                          v
       spatial B1 / Rx coupling <---- coil + shield + resonator
                |
                v
     14N powder spin response -----> absolute magnetic moment
                |
                v
  pickup + receiver transfer ------> ADC record
                ^                          |
                |                          v
       noise + RFI scenarios -----> processing + decision
                                           |
                                           v
                              ROC/AUC, PD, PFA, time, power,
                              temperature, and robustness
```

Each arrow must carry declared units and uncertainty. Predicted quantities,
measured calibration inputs, fitted nuisance parameters, and held-out
validation targets must remain distinct in the result record.

## Initial model hierarchy

Three fidelity levels keep the design search computationally tractable.

### Level A: screening surrogate

Use analytic or interpolated coil sensitivity, a single effective line per
interrogation, reduced spin-1 dynamics, analytic relaxation, stationary
Gaussian noise, and a matched filter. This level supports broad sweeps and
eliminates infeasible designs. It must preserve absolute units even when its
physics is simplified.

### Level B: engineering simulation

Use PEEC-derived coil and shield response, parcel loading, spatial \(B_1\) and
receive sensitivity, powder and EFG distributions, full acquisition masks,
colored receiver noise, sampled RFI, network transfer, thermal transients, and
multi-line decisions. This is the default level for Pareto optimization.

### Level C: validation cases

Use the highest-fidelity available field/circuit model, full density-matrix
dynamics where model-selection rules require it, measured material and RFI
distributions, explicit component tolerances, and recorded instrument
waveforms. Level C is applied to a small set of finalists and failure cases,
not every optimizer evaluation.

Agreement limits between adjacent levels are acceptance criteria, not informal
visual checks.

## Baseline design variables and constraints

The first parameterization should be compact enough for exhaustive and
uncertainty-aware sweeps.

### Hardware variables

- coil family and aperture dimensions;
- turns, conductor dimensions, winding length, and optional segmentation;
- shield dimensions, material, and coil-to-wall clearance;
- series or parallel tuning, match network, target loaded Q, damping state;
- one receive coil versus a receive array or gradiometric combination;
- number, axes, and positions of RFI reference stations;
- amplifier source limit, receiver noise parameters, blanking, and recovery.

### Acquisition variables

- chosen target lines and their order;
- SLSE, SORC, or a staged combination;
- pulse duration, amplitude or nutation rate, phase cycle, carrier offset;
- echo or half spacing, echo/pulse count, acquisition window, and repetitions;
- recovery delay and optional off-resonance or baseline calibration shots;
- fixed schedule versus posterior-driven next-line selection and early stop.

### Mandatory constraints

- geometric clearance for every parcel in scope;
- coil SRF margin and realizable tuning/matching values;
- realizable amplifier output, coil current, terminal voltage, electric field, and
  component ratings;
- transmit/receive switch isolation, receiver headroom, blanking and recovery;
- temperature rise and average/peak duty cycle;
- minimum useful receive bandwidth and maximum permitted ringdown;
- total physical scan time and switching overhead;
- numerical-resolution and model-validity checks, including powder, spatial,
  EFG, and time-step convergence.

Requirement values are deliberately not invented here. A checked-in study
configuration should fail clearly when a required operational or safety limit
has not been supplied.

## Statistical study design

The unit of evaluation is a complete target-present or target-absent parcel
trial, not a noiseless echo. Scenario generation should be stratified over:

- material and line uncertainty;
- target mass or amount and crystalline fraction;
- parcel geometry, target location and orientation;
- temperature and parcel loading;
- hardware tolerance and drift;
- RFI environment and receiver-noise realization.

Common random numbers should compare candidate designs on identical scenarios.
Report stratum-specific performance as well as aggregates so that a high mean
does not conceal a weak parcel region or material class. Confidence intervals
must include finite Monte Carlo uncertainty. All optimization runs need a
separate held-out scenario set; the final blind test set must not be used for
model fitting, threshold selection, or design ranking.

The first detector should be interpretable: a complex matched-filter or
generalized-likelihood score built from a physically simulated echo template,
with nuisance amplitude, phase, baseline, small frequency shift, and decay
marginalized or profiled. Multi-line evidence can then combine calibrated log
likelihood ratios. More flexible classifiers are a later comparison and must
demonstrate calibration and out-of-distribution behavior, not merely accuracy
on synthetic data.

## Implementation plan

### Phase 0: freeze the question and evidence rules

> **Implementation status:** the four artifacts below are present in `phase0/`
> with the test-set policy and hash-bound approval record; see
> `phase0/completion_audit.md`. Gate 0 remains open until stakeholder values
> and approvals replace the explicit unresolved requirements.

1. Add a versioned requirement worksheet covering parcel dimensions, scan
   mode, AUC/time objectives, target and benign populations, environmental range,
   hardware limits, and optional deployment \(P_D/P_{FA}\) criteria.
2. Define provenance labels: `predicted`, `literature`, `measured_calibration`,
   `fitted`, and `held_out_validation`.
3. Define a canonical result schema containing scenario ID, design ID, random
   seed, model fidelity, signal/noise/RFI components, constraint margins,
   decision score, and outcome.
4. Select a small development library with at least two 14N target materials,
   representative benign nitrogenous crystalline materials, and null parcels.

**Gate 0:** stakeholders approve the requirement worksheet and material/test
set; every numerical claim can identify its evidence class.

### Phase 1: absolute 14N signal-chain reference

1. Implement or audit the high-temperature spin-1 equilibrium population
   scale for a specified isotope count and temperature.
2. Convert the simulated density-matrix observable into magnetic moment per
   spatial element with an explicit operator convention.
3. Couple that moment through a simple calibrated solenoid by reciprocity to
   flux, EMF, receiver output, and ADC samples.
4. Compute coil Johnson noise and receiver noise in the same units and verify
   matched-filter SNR analytically and with seeded Monte Carlo trials.
5. Add limiting checks: signal linearity with target amount, inverse-
   temperature population scaling, reciprocity consistency, √averages SNR,
   and unit invariance between field- and voltage-referred calculations.

6. Define an optional pre-polarization branch with explicit state preparation,
   material-specific transfer and relaxation, field settling, energy and timing;
   compare against the unpolarized baseline without assuming enhancement.

**Gate 1:** an end-to-end single-line reference case predicts absolute signal
and noise with a closed budget; two independent calculation paths agree within
a declared tolerance. No system optimization begins before this gate passes.

### Phase 2: parcel and aperture model

1. Add parcel regions or voxels with material identity, density or amount,
   crystalline fraction, pose, temperature, and loading properties.
2. Define initial coil candidates: enclosing solenoid, split or saddle-like
   aperture, and gradiometric or array receive variants as geometry permits.
3. Generate Tx and Rx maps, active-volume and worst-region coupling metrics,
   and PEEC/network properties over the target frequency span.
4. Couple dielectric/conductive parcel loading and component tolerances to
   resonance, Q, delivered current, noise, and ringdown.
5. Establish Level-A surrogate errors against Level-B field/circuit cases.

**Gate 2:** spatial and loading convergence pass; the surrogate bounds or
predicts the engineering model over the declared parcel envelope.

### Phase 3: realistic acquisition and interference

1. Compose spatial \(B_1\), powder orientation, EFG/temperature distributions,
   and relaxation into FID, SLSE, and SORC candidate records.
2. Model recovery and state history across repetitions and multi-line schedules.
3. Apply resonator ringdown, switching, blanking, receiver transfer, overload
   recovery, colored noise, and ADC sampling.
4. Build site-specific RFI scenario families and compare passive shielding,
   gradiometry, multi-reference cancellation, and protected-window adaptive
   cancellation.
5. Couple RF losses and acquisition schedules to the thermal network.

**Gate 3:** clean, noisy, and RFI-contaminated records conserve timing and
units; injected signals are not spuriously learned or removed by cancellers;
thermal and receiver constraint margins are reported for every trial.

### Phase 4: detector and adaptive protocol

1. Implement the matched-filter/likelihood baseline and calibrate thresholds on
   target-absent development scenarios.
2. Fuse multiple lines and echo-decay evidence while accounting for correlated
   nuisance and RFI residuals.
3. Use the existing Bayesian design layer to choose the next carrier, sequence,
   or repetition by expected decision-risk reduction per physical second.
4. Add a stopping rule based on posterior decision risk or calibrated evidence,
   over a declared physical-time sweep; deployment caps remain optional inputs.
5. Test sensitivity to prior misspecification and missing library records.

**Gate 4:** target-absent scores are calibrated, ROC/AUC uncertainty is reported on held-out synthetic scenarios (and any
subsequently declared false-alarm bound is checked), and adaptive acquisition is compared
with fixed schedules under identical physical-time accounting.

### Phase 5: robust co-design optimization

1. Screen broad hardware and acquisition combinations with Level A.
2. Optimize continuous variables within surviving discrete architectures with
   Level B, using common random numbers and confidence-aware constraints.
3. Retain the Pareto front and quantify which assumptions move its knee.
4. Re-evaluate finalists and adversarial failure scenarios at Level C.
5. Retain the AUC-versus-time frontier under engineering constraints; select a
   baseline only after a time/AUC tradeoff is chosen. Apply any later supplied
   PD/PFA operating-point criteria separately.

**Gate 5:** the selected design remains feasible under held-out uncertainties;
its advantage is larger than Monte Carlo error and model-fidelity discrepancy.

### Phase 6: experimental calibration and blind validation

1. Measure empty-coil and loaded impedance, \(B_1/I\), receive sensitivity,
   ringdown, noise PSD, receiver recovery, and RFI reference transfer paths.
2. Fit only parameters designated as calibration inputs and preserve raw data.
3. Validate signal scaling with position, temperature, and controlled material
   amount using safe laboratory phantoms.
4. Run randomized blind target/benign parcel trials across the requirement
   envelope, including hard negatives and out-of-distribution contents.
5. Compare observed and predicted score distributions, recalibrate if allowed,
   and repeat the held-out evaluation after any design change.

**Gate 6:** report empirical \(P_D/P_{FA}\) confidence intervals, scan-time
distribution, constraint margins, calibration drift, and model discrepancy.

## Proposed software boundary

The system layer should be a consumer of existing NQR, field, probe, detection,
interference, thermal, and Bayesian-design APIs rather than embedding duplicate
physics in an example script.

```text
spin_dynamics/screening/
    requirements.py   # Operational requirements and constraint validation
    materials.py      # Screening material records and uncertainty adapters
    parcel.py         # Parcel regions/voxels and scenario sampling
    hardware.py       # Coil/resonator/receiver assembly and calibration inputs
    protocol.py       # Multi-line shots, schedules, costs, and state history
    forward.py        # Absolute spin-to-ADC composition
    decision.py       # Scores, fusion, thresholds, calibration, stopping
    objectives.py     # PD/PFA/time/power/thermal metrics and robust constraints
    study.py          # Seeded sweeps, optimization, replay, and result records
    reporting.py      # Budgets, Pareto tables, failure strata, evidence labels
```

The exact package name can change after the Phase-1 reference identifies the
smallest stable public abstraction. The first implementation should remain a
library-backed capstone with a reproducible configuration and report, not an
instrument-control interface.

## First executable study slice

The smallest useful vertical slice is one measured 14N line, one benign/null
population, one enclosing solenoid in a shield box, an inductive receiver, an
SLSE train, white plus measured-shaped noise, and a matched-filter decision.
It should sweep:

- target position at the center and representative edge/corner locations;
- a small temperature and line-frequency uncertainty grid;
- loaded-Q and ringdown tradeoffs;
- pulse amplitude/duration, echo spacing/count, and repetitions;
- target-present and target-absent Monte Carlo trials.

The slice must produce an absolute signal budget, constraint table, ROC curve,
time-to-decision distribution, and a small Pareto plot. It is a software and
calibration proof, not a performance claim for all mail or all 14N materials.

## Definition of success

The study is successful when it can answer, with traceable assumptions:

1. Which physical effect limits detection in each parcel/material stratum?
2. How much does each hardware or protocol change improve calibrated decision
   performance, not merely normalized echo amplitude?
3. Which design is robustly preferred after accounting for loading, location,
   temperature, line uncertainty, RFI, tolerances, and model discrepancy?
4. What ROC AUC is attainable versus full physical scan time, with confidence
   intervals, and how does pre-polarization change that frontier?
5. Which conclusions are simulation predictions, which depend on measured
   calibration, and which have survived blind validation?

Until the absolute-signal and blind-validation gates pass, outputs should be
described as design-study predictions rather than scanner specifications.
