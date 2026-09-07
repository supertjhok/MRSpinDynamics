# Phase 3 — acquisition, interference and thermal records

This stage composes the package's full three-level Hamiltonians, affine
Liouvillian relaxation, Biot–Savart/PEEC results, protected reference cancellers
and nodal thermal network. It does not use ideal pulses or replace spin evolution
with synthetic echo envelopes. Hardware settings remain design candidates.

The numerical Gate 3 applies to the explicitly declared synthetic, retuned
operating domain in `phase3_report.json`. It is not a site validation or a
claim about detectable drug mass. The vector gradiometer is an engineering
comparison; the fine spatial/powder convergence audit is on the enclosing coil.

## What is implemented

- A packet held stationary in the coil executes SLSE-x, FID-y, SORC-x and repeated SLSE-x, with 1 ms
  intervening recovery. Every crystallite/disorder density matrix survives the
  handoff. Laboratory-frame conversion preserves coherences when the carrier's
  phase is reset. The packet stays at the same coil position during recovery and acquisition.
- Phase 2 converged maps provide transmit and reciprocal receive coupling;
  converged L/R/C values set current-envelope ringdown, voltage and loss.
  Vector fields use all three RF operators and the third powder Euler angle.
  The enclosing-coil central path uses its fixed-axis approximation only while
  transverse/axial field is below 5%; arbitrary paths are not silently admitted.
- Temperature shifts both upper energy levels with the Phase 0 literature
  slopes (-43 and -72 Hz/K). The temperature mixture has weights 0.25/0.5/0.25
  at 0/20/50 C. An 8 x 8 x 20 mm uniform packet is integrated with 27 Gauss
  voxels; its total target mass is 1 g, not 1 g per voxel.
- A causal resonator pole and two receiver poles process source signal, TX
  leakage, RFI and Johnson/LNA noise before gain, clipping, overload recovery
  and ADC quantization. Spin EMF is sampled during free evolution; RF intervals
  themselves are omitted from that receiver input and excluded by blanking.
  Spin dynamics still propagate through every RF step and tail. The switch leaves finite leakage; blanked samples still
  affect receiver state. Raw arrays retain every mask on one physical clock.
- RFI families include uniform axial pickup, a nearby magnetic dipole,
  gain drift, narrowband/comb/impulsive components and deliberate overload.
  Actual signed conductor areas and reciprocal dipole coupling determine the
  magnetic pickup ratios. Straight port-return leads are an explicit assumption.
  Comparisons cover no cancellation, a hypothetical 20 dB shield, one reference,
  three references and protected NLMS. The shield's effect on coil Q is not
  assumed known. No scenarios are labelled as measured site data.
- Cancellation calibration uses only the quiet pre-RF interval; the entire
  acquisition is protected independently of the target's amplitude. A separate
  injected template checks preservation. A 10% reference-signal-pickup challenge
  quantifies bias and flags the need to reject or calibrate such references.
  Training protection alone does not remove this bias.
- The built-in thermal network receives the analytic current-squared energy of
  each pulse and its ringdown tail. Copper and damping losses are separated.
  A Phase 2 uniform-contrast upper loss bound supplies parcel heating. Pulse-wise
  averaging conserves energy, with an adiabatic bound on within-pulse ripple.
  Repetition steady state uses the actual serial parcel cycle, including travel
  and the frozen 2 s handling allowance. Every trial has receiver and thermal
  margins, including trials that intentionally fail receiver limits.

## Stop–transfer–stop motion

The Phase 3 default now has six explicit stages: entry to the magnet centre,
stationary magnet dwell, moving polarization transfer to the coil stop,
stationary coil dwell, exit, and handling. `motion` in each report record gives
absolute stage times and positions, the coil-arrival time, and the RF interval.
Receiver-record time zero is coil arrival. The serial cycle includes both dwells
exactly once; all thermal repetition calculations use that cycle.

`schedule` in `acquisition.py` exposes `magnet_dwell_s`, `coil_dwell_s`,
`transfer_velocity_m_s`, `coil_stop_fraction`, `settling_s`, and `repetitions`.
For example:

```python
record = schedule(
    Path(".tmp/nqr_phase2_gate"), prepolarization=True,
    magnet_dwell_s=2.0, coil_dwell_s=0.080,
    transfer_velocity_m_s=3.0, coil_stop_fraction=0.25,
    settling_s=0.005, repetitions=4,
)
```

Repetitions repeat the four-block x/y schedule while retaining the density state;
changing dwell alone does not silently insert more RF pulses. A too-short coil
dwell is rejected. Remaining dwell time continues free spin evolution and receiver sampling,
and remains in the physical cycle; it is not padded with zero signal.
The magnet dwell builds proton polarization at the magnet centre. The existing
adiabatic-transfer solver operates only on the subsequent moving segment, which
ends at the actual coil stop. Relaxation during stationary settling/calibration
then precedes the first pulse. Subsequent block recovery uses the retained full
spin state. Transfer-disabled controls keep the same motion timing.

Stops are currently ideal, with constant-speed travel between them; acceleration
and deceleration ramps are not represented. The old continuous-motion mode remains
available in the Phase 1 engine for regressions, but is no longer Phase 3's default.
The gradiometer still uses vector fields; the enclosing-coil packet's convergence
is regenerated for stationary acquisition rather than inherited from the moving case.

## Explicit numerical assumptions

The candidate uses 8 A peak, 30/60 us excitation/train pulses, 600 us spacing,
4 echoes per block, Q=30, 5 m/s transfer speed and 1.2 m magnet–coil spacing. The default magnet dwell
is 1 s and the coil dwell is 25 ms, including 5 ms mechanical settling, 5 ms
quiet receiver calibration, the pulse schedule and at least 2 ms final guard. The existing
exact quadrupole-plus-Zeeman volume check remains attached to each block.
These values have not been optimized for detection or throughput.

Receiver defaults are 200 ksample/s complex IQ, 15 kHz per receiver pole,
1000 V/V gain, +/-1 V per quadrature, 18-bit ADC, 140 dB TX isolation and
200 us overload recovery. The resonator pole is 3.3043 MHz/(2 Q). Its normalized
complex transfer is checked against the Phase 2 tuned network on both lines
within +/-15 kHz. Gain and noise are input-referred; this does not claim a
particular LNA impedance-match implementation. Per-line retuning and controlled
current are assumed. Unretuned load/component corners remain outside this gate.

Noise uses peak-phasor IQ convention: E|n|^2 = S_one-sided * sample_rate before
filtering, with half this variance in each real quadrature. Both the discrete
filter norm and an independent seeded Monte Carlo check the normalization.
The physical 1 g source is below an individual ADC LSB with these unoptimized pulse parameters;
noise dithers the quantizer. Digital injection uses an explicitly scaled template
to isolate cancellation bias from quantization, and is not a mass-sensitivity test.

The frequency grid grows with the complete observation time, including repeated
blocks and post-train free evolution. Its artificial discrete-spectrum recurrence
is kept beyond 1.5 observation windows, with at least 81 points at base resolution.
An analytic Gaussian-decay regression checks short and long dwell windows; coupled
refinement doubles the frequency intervals as well as refining pulses and powder.

The two lines share the x-line Gaussian static EFG-disorder and 39 ms T1 prior;
14.4 ms effective CPMG decay remains a provisional homogeneous T2 prior, not a
measured microscopic SLSE/SORC relaxation model. Optional prepolarization retains
the worked adiabatic-transfer machinery and the existing single-line population
embedding. A complete measured two-line preparation model is still unavailable.

Assumed thermal parameters: copper density 8960 kg/m3 and heat capacity
385 J/(kg K), damping-node capacity 50 J/K, 21 g parcel capacity 1200 J/(kg K),
and coil/damper/parcel-to-ambient conductances 0.5/1/0.05 W/K. Parcel conductivity
is bounded by the Phase 2 1e-4 S/m dry-dielectric domain. The component limit is
80 C and parcel-rise limit 2 K. Contacts, heat capacities and shielding require
bench calibration; the numbers are engineering priors, not measurements.

## Reproduce

From `PythonSpinDynamics`, using the Ubuntu-24.04 scientific environment:

```bash
# Phase 2's full report and converged maps must already exist; see its README.
OPENBLAS_NUM_THREADS=1 python studies/nqr_mail_screening/phase3/validate_resolution.py
python studies/nqr_mail_screening/phase3/validate_circuit.py
OPENBLAS_NUM_THREADS=1 python studies/nqr_mail_screening/phase3/run_phase3.py
python studies/nqr_mail_screening/phase3/plot_phase3.py
python -m unittest tests.test_nqr_mail_screening_phase3
```

Full records and validation reports are in `.tmp/nqr_phase3`; the checked-in
report contains the trial summaries, exact assumptions, gate checks and source
hashes. `--quick` checks one point-packet schedule and does not overwrite the
checked-in full report or claim full gate closure. Raw voltages are volts,
times seconds, temperatures kelvin and energy joules. Plotting is presentation
only; the gate operates on the stored numerical records.

Next stages should use these records for detector calibration and held-out ROC
analysis, while expanding geometry/velocity, waveform and loading sweeps before
selecting hardware. Installed-site spectra and reference pickup belong in the
experimental validation loop, not fabricated replacements for missing data.
