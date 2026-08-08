# Receiver Decoupling and LNA Architecture Study

**Status:** Passive cancellation, active-LNA loading/noise, preamplifier-decoupling/interconnect robustness, and the Phase 5 birdcage handoff are implemented; standard noise-parameter conversion, measured front ends, and active-network end-to-end imaging remain extensions

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

The first implemented robustness slice compares no cancellation, resonant
capacitive cancellation, a low-Z LNA behind a quarter-wave line, and combined
passive plus preamplifier decoupling. It sweeps frequency, cable electrical
length, cable loss, and preamplifier input resistance. The reusable result
reports the transformed coil-port load, voltage transfer, induced-current
coupling/isolation, separated passive-front-end noise, and noise figure.

The nominal cable-only 2-ohm example deliberately produces strong isolation but
poor noise figure. This is a diagnostic, not a recommended circuit: a complete
MRI front end must co-design the coil matching network, transformed preamplifier
impedance, and LNA optimum noise impedance. Remaining robustness axes are coil
spacing/orientation, sample loading, geometric overlap, explicit matching
networks, component Monte Carlo, mode splitting, loaded Q, and measured network
comparison.

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

The passive-front-end layer now supports reciprocal ABCD cascades, series and
shunt elements, ideal transformers, and uniform lossy transmission lines. It
propagates coupled-source signal transfer and derives pre-LNA thermal noise from
fluctuation-dissipation balance. Lossless networks add exactly zero noise, and
a matched lossy line recovers the Friis limit in which noise figure equals
insertion loss.

Still pending in this part are standard noise-parameter conversion, measured
S-parameter import, output impedance, cross-channel amplifier noise, stability,
compression, and dynamic-range diagnostics. The principal comparison remains
the Pareto surface between noise figure, decoupling, bandwidth, and robustness
rather than a single nominal SNR.

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

Phase 4.5 uses two-loop mesh branches and terminal frequency sweeps. Phase 5
now provides a dedicated birdcage rung/end-ring topology with distributed
capacitors, full branch mutual and loss matrices, physical ports, matching,
reciprocal B1- maps, and passive noise covariance. It reuses the same
noise-source separation and covariance-aware reconstruction conventions. A
fully generic arbitrary circuit graph remains a later extension rather than a
prerequisite for the implemented birdcage model.
