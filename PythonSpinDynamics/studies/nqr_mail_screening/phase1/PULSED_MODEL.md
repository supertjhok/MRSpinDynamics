# Moving-sample Phase 1 engineering model

The previous ideal reference is a normalization regression. It does not establish
scanner feasibility or close the engineering gate. `pulsed_study.py` is the
working comparison; `geometry.py` constrains inline magnet/coil placement.

## Geometry and motion

The magnet and coil share a transport axis. Their dimensions, centre spacing and
sample speed are design variables. The example values are **candidates, not selected
hardware**: 0.18 m coil radius, 0.46 m length, six turns of 3 mm wire; four square
Halbach rods, 0.23 m centre radius, 0.46 m length, 0.08 m rod width, 1.15 T remanence;
1.2 m centre spacing and 5 m/s transport. Both apertures clear the mail cross section.

The sample moves during the pulse train. Biot-Savart fields along the path set
local pulse amplitude and receive reciprocity sensitivity. Readout begins one
quarter of the way through the coil by default. A train extending beyond the
coil exit is rejected. Coil sensitivity is held constant within each short pulse
and interpolated along the trajectory between pulses and receive samples.
The current model represents a small sample packet on the centreline; it is not
a full spatial distribution of target mass throughout the envelope.

Parcel time includes travel from magnet entrance to coil exit, any requested
magnet-centre dwell and the provisional 2 s handling time. Settling occurs during
the final approach to readout; its distance and T1 loss are included without
adding a stationary delay.
This is a serial timing comparison, not an optimized conveyor/pipeline model.

## Stray-field constraint

`geometry.py` evaluates the finite Halbach field at 45 points covering the
430 x 340 x 35 mm clear mail volume at the coil. It diagonalizes the package's
full quadrupole-plus-Zeeman Hamiltonian for a powder grid plus the principal axes.
The addressed aniline line's largest absolute shift must be below a provisional
10% allocation of the smallest of native linewidth, inverse pulse duration and
loaded receiver bandwidth. For the baseline this is 52.18 Hz. This is a sampled
spectral constraint, not a certified worst-case bound. It must be revisited for
other material lines, geometry, bandwidth and field orientations.

Do not equate gamma*B directly to NQR splitting: the exact eigenvalue calculation
includes the asymmetric spin-1 level structure. For this candidate magnet, the
coarse spacing sweep rejects 0.8 m (about 105 Hz maximum perturbation) and accepts
1.0 m (about 5 Hz). There is no universal minimum spacing. Closer designs fail
eligibility even if they show an attractive idealized signal. The readout solver
uses the zero-field basis only after this small-perturbation screen; it does not
simulate the Zeeman-distorted signal of rejected placements.

## Spin dynamics and electronics

- Full three-level rotating-frame Hamiltonians from `nqr.full_dynamics` drive
  the density matrix. No selective ideal pulse is used in the working model.
- The package Liouvillian includes T1 recovery and transverse decay during RF
  and free intervals. An affine source `-R rho_equilibrium` supplies the physical
  thermal fixed point. Using `-L rho_equilibrium` during RF would incorrectly
  cancel excitation; a regression test guards this distinction.
- SLSE uses a phase-x excitation and phase-y refocusing train. SORC uses repeated
  same-phase finite pulses, starting from the prepared density rather than an
  assigned steady state. These are explicit candidate schedules, not optimized
  sequence parameters or a claim of measured SLSE/SORC performance.
- A Gaussian static frequency ensemble supplies reversible dephasing; its width
  and the assumed homogeneous T2 reproduce the imported T2-star at its 1/e time.
  The 14.4 ms CPMG decay is a **provisional T2 prior**, not an intrinsic T2 or a
  measured SLSE/SORC decay. A shorter-T2 sensitivity case is included.
- The coil-property engine supplies inductance, AC resistance and self-resonance.
  A loaded-Q equivalent series resistance gives the RF envelope time constant.
  Finite rise and ringdown are integrated as piecewise-constant RF Hamiltonians;
  ringdown continues to act on spins while the receiver is blanked.
- Current is peak linear RF current: its co-rotating field is half the peak field.
  Reactive voltage, dissipated RF energy, 50 A and 2 kV provisional study limits
  are reported. Infeasible cases are excluded by `eligible_for_comparison`.
- Receive voltage uses absolute Gibbs populations, one aniline nitrogen per
  fentanyl-HCl molecule, and reciprocity. Only the addressed line is detected.
  Noise is an **input-referred flat-receiver bound** with loaded-resistance Johnson
  noise and 1 nV/sqrt(Hz) amplifier noise. Tuned transfer, correlated receiver noise,
  ADC overload and environmental RFI remain to be integrated. The reported
  Gaussian oracle AUC is not a screening ROC or the Phase 0 H0 mixture result.

## Pre-polarization

This calls `simulate_adiabatic_polarization_transfer`, the same built-in model used
by `examples/plot_nqr_database_prepolarization.py`, with a finite Halbach field
and controllable transport speed. It includes crossing gradient, adiabatic transfer
efficiency, nitrogen T1 loss after crossing, and an additional settling loss.
Incoming proton buildup is integrated from the magnet entrance to its centre
using the local field and T1H, then combined with an optional centre dwell into
an effective preparation time for the worked model. Starting with zero proton
polarization at the entrance and neglecting additional proton relaxation before
the crossing are approximations; the model is not a microscopic H-N avoided-crossing
propagation. The transport field is sampled on axis over a small packet.

The single-line enhancement changes the addressed pair's population difference
while preserving trace, pair population sum and positivity. The third population
is unchanged. Competing-site and correlated three-line transfers are unresolved.
A continuous-motion case with pre-polarization enabled and zero dwell is included;
thermal cases are explicit controls, not a claim that passage through a magnet
produces exactly no polarization.

The temperature setting controls thermal polarization/noise only; temperature-dependent
line shifts are currently explored as detuning scenarios, not a validated 0–50 C
material model.

Fentanyl-specific proton T1, dipolar coupling and linewidth are not established by
the existing worked examples. Defaults of 1 s, 1 kHz and 80 kHz are **assumptions**;
29 protons and two nitrogen reservoirs are used. A glycine CIF is not substituted
for a fentanyl structure. Coupling and velocity sensitivity cases are included.
Long transport after the crossing can erase the gain because the imported NQR T1
is only 39 ms. Speed and spacing must therefore be optimized together.

## Run and verification

From `PythonSpinDynamics` in the documented Ubuntu-24.04 environment:

```bash
OPENBLAS_NUM_THREADS=1 python studies/nqr_mail_screening/phase1/pulsed_study.py
OPENBLAS_NUM_THREADS=1 python studies/nqr_mail_screening/phase1/check_convergence.py
python -m unittest tests.test_nqr_mail_screening_pulsed
```

`--config overrides.json` runs one scenario using Config field names; `--quick`
runs the four thermal/dwell sequence comparisons. Outputs go to `.tmp/nqr_pulsed`:
JSON budgets, spacing sweep, per-case complex receive waveforms and masks, and a
comparison plot. `pulsed_report.json` and `convergence.json` here are snapshots.
The convergence script independently refines powder, disorder, receive time,
RF envelope and magnet/spectral resolution and requires each change below 3%.
Its pulse tests use a three-cycle baseline; this does not certify every candidate.

Next engineering closure requires joint geometry/speed/sequence optimization,
material-specific relaxation/transfer evidence, receiver/RFI integration and
held-out H0/H1 decision statistics. The present result is a more useful physical
comparison with explicit rejection rules, not a validated mail-screening design.
