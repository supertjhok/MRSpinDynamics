# Realistic relaxation in NQR--NMR powder simulations

This note records the modeling and validation procedure needed to obtain a
physical relaxation curve from a quadrupolar powder. It exists because several
shortcuts can produce smooth, plausible-looking echo trains that are not what a
coil would measure. The central rule is simple:

> Propagate every crystallite with its own Hamiltonian and relaxation generator,
> form the complex acquired voltage for every crystallite, coherently average
> those voltages, and only then apply the receiver and estimate an echo amplitude.

The ordering is part of the physical model. Operations such as magnitude,
root-mean-square averaging, echo-center sampling, filtering, and normalization
do not generally commute with the powder sum.

## 1. What must be averaged

For orientation $k$, let `rho_k(t)` be the density matrix, `M_k` the complex
receive operator, and `w_k` the normalized orientation weight. The ideal coil
voltage is

\[
    s(t) = \sum_k w_k\,\mathrm{Tr}[M_k\rho_k(t)].
\]

The sum is complex and signed. It represents phase cancellation between
crystallites. In particular,

\[
    \left|\sum_k w_k s_k\right|
    \ne \sum_k w_k |s_k|
    \ne \sqrt{\sum_k w_k |s_k|^2}.
\]

The latter two expressions are useful diagnostics of local excitation, but they
are not inductive signal voltages. Using either as the reported powder signal
usually suppresses the destructive interference that dominates a broadened
powder echo and can create an artificially smooth or slowly decaying train.

At exact zero field, all crystallites have the same transition frequencies and
the echo center can be a useful diagnostic. Once Zeeman broadening is present,
different crystallites refocus at different phases and with different local
waveform shapes. A single value at the nominal echo center is then not a stable
experimental observable. The complete acquired echo waveform must be averaged.

## 2. Orientation measure and laboratory geometry

### Zero field

At $B_0=0$, there is no physical static-field axis. For one linear coil, the
only required orientation is the RF direction in the EFG principal-axis system.
The correct integral is therefore over the sphere, implemented by
`powder_average_grid`. Introducing an arbitrary `B0` frame and integrating its
redundant rotation angle adds quadrature noise and prevents the zero-field limit
from reducing cleanly to the established NQR powder average.

### Nonzero field

At nonzero field, the crystallite sees correlated laboratory `B0` and `B1`
directions. For a transverse linear coil, the angle between them is fixed in the
laboratory, but the complete frame can have any orientation relative to the EFG
tensor. The integral is therefore over SO(3), including the rotation `chi` of
`B1` around `B0`. `b0_b1_powder_average_grid` supplies a product quadrature;
`b0_b1_powder_average_halton` supplies a low-discrepancy, prefix-nested SO(3)
sequence.

Use the Halton sequence for expensive convergence studies. The first `N`
samples are exactly the prefix of the first `2N`, so their difference measures
quadrature refinement without comparing unrelated grids. Preserve sample order
when parallelizing or taking prefixes.

For circular or multiple orthogonal coils, one direction is insufficient. Use a
full `OrientationFrame`/SO(3) representation so all lab coil axes rotate together.

## 3. One static problem and one relaxation problem per crystallite

For each nonzero-field orientation, construct the complete Hamiltonian

\[
    H_k = H_Q - 2\pi\gamma\,\mathbf B_{0,k}\cdot\mathbf I
\]

and diagonalize it without switching perturbative formulas between regimes. The
same matrix covers zero-field NQR, Zeeman-perturbed NQR, the intermediate
crossover, and quadrupolar NMR. The equilibrium state must be computed from this
same Hamiltonian:

\[
    \rho_{G,k} = \frac{\exp[-h H_k/(2\pi k_B T)]}
                       {\mathrm{Tr}\{\exp[-h H_k/(2\pi k_B T)]\}}.
\]

The relaxation superoperator must also be evaluated for `H_k`; reusing a
single-crystal generator for every powder orientation is incorrect when energy
gaps and eigenvectors vary with orientation.

PythonSpinDynamics provides three useful levels:

- `FieldDependentRelaxationModel` is a robust, completely positive Gibbs-reset
  plus energy-manifold dephasing model. It is the safest phenomenological choice
  through exact degeneracies, but its rates are inputs rather than microscopic
  predictions.
- `FieldDependentDaviesRelaxationModel` constructs magnetic rank-1 and EFG
  rank-2 thermal jump channels, imposes detailed balance, and evaluates a
  Lorentzian spectral density at each Bohr frequency. It is appropriate when
  distinct transition-frequency groups are secularly resolved.
- `FieldDependentNonsecularRelaxationModel` groups unresolved Bohr frequencies
  into common jump operators. This retains coherence-transfer cross terms near
  degeneracies while remaining completely positive. Its finite cluster width
  makes exact Gibbs stationarity approximate away from exactly degenerate
  groups; always inspect `gibbs_stationarity_error` and vary the cluster width.

No one of these models predicts material-specific rates from the quadrupole
coupling alone. Channel amplitudes, correlation times, activation laws, and
temperature dependence must be fitted or independently justified for the
compound under study. Rates must not be transferred between NaNO2 and NaClO3
merely because both are quadrupolar solids. The current NaNO2 crossover example
anchors its overall rate scale to the measured room-temperature 332 +/- 23 ms
SLSE lifetime; its magnetic/EFG split and correlation time remain explicit model
assumptions.

## 4. Pulse propagation and a single global carrier

A physical experiment has one transmitter carrier, not a separately optimized
carrier for each crystallite. Choose one intensity-weighted powder carrier (or
use the experimental carrier) and hold it fixed for the entire ensemble.

Finite-sideband Floquet propagation is the reference for monochromatic linear
RF when counter-rotating terms or multiband coupling matter. A phase-consistent
RWA backend can be used after comparison with Floquet over representative
orientations, offsets, and pulse strengths. Pulse phases must include the
absolute carrier phase at each pulse start; resetting the rotating-frame phase
for every crystallite or refocusing pulse destroys coherent powder cancellation.

The current crossover SLSE implementation includes relaxation during free
evolution and treats pulses as hard relative to the relaxation timescale. If
pulse durations become comparable to relaxation times, the relaxation generator
must be included during the driven interval as well.

## 5. Spectral slicing without losing normalization

At intermediate and high fields, only a small fraction of a broad powder may
fall inside the excitation and receiver band. Propagating the entire SO(3)
ensemble is wasteful. `select_powder_frequency_slice` keeps crystallites with an
RF-active transition inside a chosen carrier-centered interval and returns:

1. the selected orientations, renormalized within the slice; and
2. the retained weight of that slice in the full powder.

The conditional, renormalized ensemble is convenient for comparing echo-train
shape. For an absolute full-powder voltage, multiply by the retained full-powder
weight and retain the same physical sample/coil normalization. Never report a
frequency-sliced amplitude as an absolute powder signal without this factor.

The slice should cover both receiver and excitation support. A practical initial
choice is

\[
    \Delta\nu_{\mathrm{slice}}
      \gtrsim \tfrac12\mathrm{BW}_{\mathrm{receiver}} + 2\nu_1,
\]

but the final result must be repeated with a wider interval. Also report the
retained weight; a converged conditional shape can still represent a very small
fraction of the sample.

## 6. Acquire, average, filter, then estimate

For every echo and crystallite, sample the complex waveform on the same offsets
around the nominal echo center. Then perform these operations in order:

1. coherently sum the local waveforms using powder weights;
2. apply the complex receiver transfer function to the summed waveform;
3. estimate each echo amplitude from the filtered waveform.

`matched_filter_echo_waveforms` uses a Gaussian frequency response and projects
every filtered echo onto the filtered first-echo template:

\[
    a_n = \frac{\langle s_1,s_n\rangle}{\langle s_1,s_1\rangle}.
\]

This gives a reproducible receiver-derived amplitude when the echo shape evolves
moderately. If the shape changes strongly, inspect the waveforms directly and
compare alternative estimators or a measured instrument template.

The acquisition grid must support the receiver bandwidth. For sample spacing
`dt`, the implementation rejects `receiver_bandwidth_hz > 1/dt`; in practice,
use margin and test at least a doubled point count. The acquisition window must
contain the filtered echo and fit between adjacent RF pulses. Truncating the
echo changes the matched amplitude even if the time step is fine.

The first failed crossover calculation used 65 points over 800 microseconds
while claiming a 200 kHz receiver. The trace looked structured but was aliased.
The validated NaNO2 setup uses 129 points over 100 microseconds; doubling to 257
points changes the matched train by less than 3%.

## 7. Required convergence and physics checks

Do not accept a powder relaxation curve after varying only the orientation
count. Converge independent numerical and physical controls:

### Numerical controls

- **Orientation ensemble:** compare nested `N` and `2N` SO(3) prefixes using a
  norm over the complete matched echo train, not one selected echo.
- **Waveform sampling:** double the acquisition points at fixed window.
- **Acquisition window:** widen it while preserving adequate sampling rate.
- **Receiver bandwidth:** bracket the intended value with narrower and wider
  responses.
- **Frequency slice:** widen the slice and report retained full-powder weight.
- **RF propagation:** increase Floquet sidebands, or compare RWA with Floquet on
  representative active orientations.
- **Relaxation resolution:** vary the nonsecular cluster width and verify the
  Gibbs-stationarity residual.

### Structural checks

- trace and Hermiticity are preserved for every local state;
- minimum density-matrix eigenvalues remain nonnegative within tolerance;
- zero field reduces to the ordinary spherical powder integral and is invariant
  to any fictitious static-field frame;
- zero-field NaNO2 gives a smooth, nearly exponential decay with the measured
  lifetime when the model is calibrated to that experiment;
- serial, threaded, and process backends agree within floating-point tolerance;
- powder results are invariant to harmless changes in sample chunking/order
  apart from the documented quadrature sequence;
- the receiver output is formed from the coherent complex average, never from
  averaged local magnitudes.

The validated 30-echo NaNO2 calculation obtained nested-prefix errors of 1.5%
in perturbed NQR, 1.9% in the intermediate regime, and 0.81% in quadrupolar NMR.
The zero-field fitted lifetime was 328 ms, consistent with the experimental
332 +/- 23 ms value. These are validation results for that parameter set, not
universal tolerances or predictions for other compounds.

## 8. Parallel execution without changing the answer

Crystallites are independent until the coherent reduction, so orientation-level
parallelism is natural. `simulate_crossover_slse_powder` supports deterministic
thread and process backends. Use processes for long CPU-bound powder jobs; use
threads when the linear-algebra library releases the GIL efficiently and data
transfer dominates.

Production runs should use `retain_local_results=False`. Workers then return only
the local echo/waveform arrays needed for the final reduction rather than every
density-matrix trajectory. Work is split into multiple chunks per worker for load
balance, and executor mapping preserves deterministic order. Avoid nested BLAS
thread oversubscription when also using process workers.

Prefix convergence is computed from the already propagated ordered ensemble, so
it does not require a second calculation. This is valid only for a genuinely
nested orientation sequence such as the Halton prefixes.

## 9. Recommended build-and-validate recipe

1. Validate the exact `H_Q + H_Z` spectrum and RF-active transitions for several
   single-crystal orientations.
2. Choose a relaxation level and verify its equilibrium fixed point or report its
   Gibbs-stationarity residual.
3. Validate RWA pulses against Floquet, or use Floquet directly.
4. Run zero field with the spherical powder grid and verify the known smooth
   decay before introducing `B0`.
5. At nonzero field, generate a nested SO(3) candidate ensemble, choose one global
   carrier, and select/report the active spectral slice.
6. Acquire full complex echo waveforms, coherently average them, apply the
   receiver, and use a documented amplitude estimator.
7. Perform the independent convergence checks above. Withhold a plotted regime
   when the chosen tolerance is not met.
8. Only then compare field regimes, compounds, or relaxation mechanisms.

The executable reference is `examples/plot_nano2_crossover_slse.py --powder`.
The implementation and validation tests are in
`spin_dynamics.nqr.crossover_sequences` and
`tests/test_nqr_field_relaxation.py`, respectively.

## 10. Known limitations

- The present pulse path neglects relaxation during RF pulses.
- The Gaussian receiver is an explicit but idealized model; quantitative
  comparison should use the measured complex transfer function when available.
- Frequency slicing is based on active transition frequencies, not a fully
  adaptive resonance-manifold quadrature.
- The nonsecular unified-GKLS model is a controlled phenomenology for unresolved
  frequency clusters, not a complete microscopic solid-state bath theory.
- Absolute signal prediction additionally requires sample amount, filling
  factor, coil sensitivity, gain, and noise calibration.

These limitations should be stated rather than hidden by normalizing every
curve to its first echo.
