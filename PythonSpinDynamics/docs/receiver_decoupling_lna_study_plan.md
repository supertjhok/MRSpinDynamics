# Receiver Decoupling and LNA Architecture Study

**Status:** Part 1 and the active-LNA core are implemented; robustness, cable, and noise-parameter studies are next

This note defines the study inserted between the coupled receiver-network work
and Phase 5 birdcage coils. It separates two questions that are often mixed:

1. how a passive network cancels mutual inductive coupling between receive
   elements; and
2. how the input impedance, noise properties, and placement of the first LNA
   change decoupling and end-to-end SNR.

The existing complex reciprocity maps and `ReceiverNetwork` remain the common
forward model.

## Architectures to compare

The study uses three front-end controls:

- a directly matched 50-ohm LNA;
- a conventional MRI isolation preamplifier, where a low input impedance and
  a narrowband transformation network present high impedance to the coil;
- a high-input-impedance LNA located at the coil, optionally with a broadband
  transformer.

LNA placement is a separate factor. Each architecture is evaluated both
on-coil and behind a passive cable where meaningful. This prevents pre-LNA
cable loss from being incorrectly attributed to input impedance alone.

## Part 1: resonant cancellation of mutual inductance

For two loop meshes, write the mutual impedance as

\[
  Z_{21}=R_{21}+j\omega M+Z_d.
\]

A shared capacitor contributes

\[
  Z_d=\frac{1}{j\omega C_d},
  \qquad
  C_d=\frac{1}{\omega_0^2|M|}
\]

with its mesh orientation selected so that the capacitive term opposes
\(j\omega M\). At the design frequency this cancels the mutual reactance, but
not the mutual resistance \(R_{21}\). Correlated sample noise may therefore
remain even when reactive crosstalk is well cancelled.

Part 1 delivers:

- a physical shared-branch mesh impedance;
- passive frequency-sweep diagnostics;
- induced-current and loaded-transfer isolation metrics;
- residual mutual impedance and noise-correlation metrics;
- tolerance, detuning, and loading sweeps;
- a runnable two-coil example and analytic regression tests.

## Part 2: robustness and decoupling strategies

Compare:

- no decoupling;
- geometric overlap;
- resonant capacitive cancellation;
- conventional preamplifier decoupling;
- combined passive and preamplifier decoupling.

Sweep coil spacing and orientation, sample loading, capacitor tolerance,
frequency detuning, cable electrical length, and preamplifier input impedance.
Report \(Z_{21}\), induced-current coupling, an S21-equivalent isolation
metric, mode splitting, isolation bandwidth, loaded Q, sensitivity mixing, and
noise correlation.

## Part 3: active LNA models

The passive `ReceiverNetwork` treats every dissipative termination at the
stated physical temperature. An active LNA must instead separate its input
impedance from its noise sources to avoid double-counting thermal noise.

Add:

- complex, frequency-dependent input impedance including input capacitance;
- input-referred voltage- and current-noise spectra;
- their complex cross spectrum;
- an alternative standard noise-parameter form
  \(F_\min, R_n, Z_\mathrm{opt}\);
- gain, output impedance, and downstream receiver noise;
- passive cable and transformer loss before or after the first gain stage;
- optional stability, input-voltage, and dynamic-range diagnostics.

The implemented active-LNA foundation now provides complex R-X||C input
impedance, input-referred voltage/current noise and their complex cross
spectrum, gain, downstream noise, separated covariance contributions, noise
figure, and covariance-optimal array SNR. The matched-50-ohm versus on-coil
high-Z example uses identical PEEC coils and reciprocal maps so loading and
noise trade-offs remain visible.

Still pending in this part are standard noise-parameter conversion, cable and
transformer loss, output impedance, stability, compression, and dynamic-range
diagnostics. The principal comparison remains the Pareto surface between noise
figure, decoupling, bandwidth, and robustness rather than a single nominal SNR.

## Part 4: end-to-end array imaging

Feed each front end into the Phase 4 array workflow and compare:

- loaded complex sensitivity maps;
- channel covariance and correlation;
- Roemer-combined SNR;
- SENSE conditioning, rank, and g-factor;
- sensitivity to coil displacement, loading, and component tolerances.

## Validation gates

- \(C_d=1/(\omega_0^2|M|)\) cancels mutual reactance analytically.
- A shared passive branch remains reciprocal and positive semidefinite in its
  dissipative part.
- Mutual resistance remains after reactive cancellation.
- Frequency-sweep results reduce to the single-frequency
  `ReceiverNetwork` solution.
- Passive noise obeys fluctuation-dissipation and active LNA noise is not
  counted as a thermal termination.
- Noise-parameter and voltage/current-noise forms agree at the same source
  impedance.
- Cable loss follows Friis limits before and after the first gain stage.
- Imaging results preserve the established complex-map, covariance, and SENSE
  conventions.

## Primary references

- Wang et al., “Trade-off between preamplifier noise figure and decoupling in
  MRI detectors,” *Magnetic Resonance in Medicine* 89 (2023), 859–871,
  <https://doi.org/10.1002/mrm.29489>.
- Sun et al., “Wideband receive-coil array design using high-impedance
  amplifiers for broadband decoupling,” *Magnetic Resonance in Medicine* 90
  (2023), 2198–2210, <https://doi.org/10.1002/mrm.29755>.
- Avdievich et al., “Capacitive versus overlap decoupling of adjacent radio
  frequency phased array coil elements,”
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8640609/>.

## Boundary with Phase 5

Phase 4.5 starts with a two-loop mesh branch and a terminal frequency sweep.
The eventual arbitrary node/branch graph remains a Phase 5 dependency for
birdcage rungs, end-ring sections, and distributed capacitor placement. The
noise-source separation and sweep diagnostics developed here should be reused
by that graph solver.
