# Known Gaps and Capability Roadmap

This page records what the package does **not** yet model or validate well
enough for routine use. It is organized by architectural and scientific risk,
not by release promise. A listed gap may mean missing physics, limited scaling,
incomplete integration, insufficient experimental evidence, or an API that is
still too specialized.

Read this page before treating a new model as quantitatively predictive. For
the exact supported claims, use the [Validation Evidence
Matrix](../validation_matrix.md); for stable public signatures, use the
[API Reference](api_reference.md).

## 2026-07 Capability Audit

PythonSpinDynamics now has about 65,000 source lines in 191 Python modules,
25,000 test lines, and 165 example scripts. Its breadth has moved the
main architectural risk from missing individual models to disconnected feature
islands. The next stabilization phase should emphasize composition,
validation, reproducibility, and API maturity.

The highest-leverage gaps identified in the July 2026 audit are listed below.
The first composition milestone is now complete: `spin_dynamics.composition`
provides shared typed adapters for spatial fields, thermal states, flow fields,
hardware responses, and sequence timelines, including explicit SI boundaries
and spatial/time resampling. Backend-specific execution adapters remain in the
items below.

1. **General sequence composition.** The backend-neutral `SequenceIR`, Pulseq
   1.4/1.5 import/export, compiler, plotting layer, motion adapter, and facade
   execution route are implemented. Specialized `run_*` workflows deliberately
   remain the clearest entry points for experiments with domain-specific
   preparation, detection, or analysis. The remaining composition work is
   probe-aware general execution and adapters for physics engines beyond moving
   isochromats.
2. **Facade coverage.** The recommended `spin_dynamics.experiment` facade has
   31 registered routes spanning CPMG/IR, imaging, deterministic and walker
   PGSE, general sequence IR, NQR, ESR, ESEEM, HYSCORE, and ENDOR. Planning,
   config round trips, provenance, and seeded reproducibility are shared.
   Nonuniform/pipe flow, thermal, non-inductive detection, exchange, optimal
   control, and field/circuit design still use direct APIs.
3. **Validation depth.** MATLAB fixtures strongly anchor the original
   CPMG/probe workflows, while many newer modules rely on analytic limiting
   cases, cross-backend parity, or synthetic data. Capabilities should carry a
   documented validation level: measured benchmark, literature reproduction,
   independent solver parity, analytic/convergence validation, or internal
   consistency only. The measured-data program now begins with weak-field
   sodium-chlorate relaxation, transition-resolved bismuth NQR, and joint
   NMR/NQR RDX dynamics; see
   [Microscopic Relaxation Validation](../relaxation_validation.md).
4. **Reproducible results.** Saved experiments now carry canonical experiment
   and result hashes, callable/module/package-tree identities, Git revision and
   dirty state, numerical build/thread environment, input identities,
   seeded/unseeded randomness classification, archive-integrity checks, and an
   exact rerun verifier. Remaining depth is backend device metadata, solver
   tolerances/convergence diagnostics, and serialization for result fields that
   are still intentionally omitted.
5. **Backend coverage and numerical scaling.** Numba/JAX acceleration exists
   for selected kernels, but motion/walkers, powder averaging, large coupled
   systems, field solves, and large inverse problems remain uneven. Sparse
   Krylov propagation, matrix-free 2-D inversion, adaptive high-Q transients,
   and out-of-core result handling remain important options.
6. **Distribution and API maturity.** CI covers Linux/Windows and Python
   3.10--3.12; wheel/sdist installation, the installed console entry point,
   an initial typed surface, branch-coverage threshold, and deprecation policy
   are now gated. Remaining maturity work is expanding static typing and the
   coverage floor and publishing artifacts to a package index. Optional
   Numba/JAX lanes and host-normalized benchmark regression gates are now in CI.
7. **Discoverability.** The 165 examples are a major asset and the catalog is
   grouped by capability, but many scripts still assume readers already know
   the physics and package conventions. The current documentation overhaul is
   adding stronger introductions, decision-point comments, and beginner-first
   journeys; a registry-generated capability matrix remains useful follow-on.

Longer-term scientific opportunities include a sparse general coupled-spin
engine, solid-state/MAS interactions, non-Cartesian and parallel MRI,
experimental-data fitting with uncertainty quantification, richer transport
models, and feedback between temperature, circuits, fields, relaxation, and
flow. The most distinctive unifying target is a virtual instrument pipeline:

`geometry -> fields -> circuits -> realized sequence -> spin/motion/thermal
evolution -> detector/noise -> reconstruction -> inference`.

For image-guided magnetic therapy, dynamic inversion now removes the prior
assumption that every target entry is immobilized. The current mechanism model
is still two-dimensional and uses inferred inverse-power gradient sources.
Particle imaging now compares finite-pulse spin echo, Cartesian GRE, paired
GRE phase-gradient mapping, and radial center-out short-TE acquisition under a
common EPM B0 and equivalent-sphere dipole model. It is not yet a calibrated
relaxometry model: the short-TE path preserves rapidly decaying signal but does
not add an inferred positive T1 effect or steady-state preparation. Particle
shape anisotropy, measured field-dependent r1/r2/r2-star,
aggregation-dependent magnetization, water exchange, slice selection,
multicoil reception, and experimental B0/B1 maps remain absent. Optional SNR
is relative per acquisition rather than an absolute bandwidth-, sampling-, and
scan-time-matched noise model. Susceptibility contrast blooms and is not a
linear particle count, so target contrast fraction is not interchangeable with
particle occupancy.
Quantitative hardware conclusions require measured coil and EPM field maps,
particle remanence and Brownian/internal relaxation data, calibrated per-pulse
energy, 3-D focusing, and vascular/tissue interactions. EPM-only switching is
therefore classified as waveform-limited by default; the hybrid conclusion is
an architecture assessment rather than a synthesized 72-channel hardware
design.

The main MATLAB-to-Python porting phase is now largely complete for the
canonical Version 3 workflows. The remaining gaps are mostly validation depth,
specialized variants, packaging polish, and performance backends rather than
missing primary workflows.

Ported and validated:

- ideal CPMG asymptotic magnetization and echo construction;
- ideal FID acquisition and time-domain trace construction;
- `sim_spin_dynamics_arb10` and the FID-compatible `arb7` path;
- tuned-probe CPMG original/reference asymptotic path;
- untuned-probe CPMG original/reference asymptotic path;
- matched-probe CPMG original/reference asymptotic path;
- ideal-probe finite acquisition with relaxation through
  `calc_macq_ideal_probe_relax4`;
- tuned, untuned, and matched finite-acquisition receiver wrappers;
- isochromat-grid rephasing analysis, warning/raise behavior, and optional
  finite-train grid refinement;
- chunked multicore isochromat propagation for the `arb10` finite-acquisition
  path;
- public finite ideal CPMG echo-train workflow;
- public finite tuned-probe CPMG echo-train workflow;
- public finite untuned- and matched-probe CPMG echo-train workflows;
- ideal, tuned, untuned, and matched CPMG inversion-recovery finite trains over
  `tauvect`;
- tuned and matched Q/mistuning sweep workflows;
- Python-native finite-train Q/mistuning sweeps for tuned, untuned, and
  matched probes;
- first matched-probe diffusion CPMG workflow and compact diffusion Q sweep;
- ideal PGSE and PGSE-prepared CPMG workflows with a deterministic
  gradient-moment backend, explicit random-walker backend, Stejskal-Tanner
  validation, and a PGSE D-T2 inverse-Laplace plotting example;
- fixture-validated ideal, tuned, and matched CPMG imaging, k-space
  reconstruction, arbitrary B0/B1 field-map loading helpers, and tuned
  receive-weighted imaging mode;
- ideal inversion-recovery T1-prepared phase-encoded CPMG imaging with
  selected-echo, echo-summed, fitted-rho, and fitted-T2 image formation;
- frequency-encoded spin-warp and RARE imaging, slice-selective excitation with
  gradient-shaped (windowed-sinc) pulses, and true-3D slice-selective multi-slice
  imaging in spatially varying `(B0, B1)` fields (`run_multislice_imaging`, with a
  fast separable approximation), all sharing a dimension-agnostic 1D/2D/3D
  field-map layer (`spin_dynamics.fields`) and an n-D moving-isochromat engine;
- analytic magnet field sources (`spin_dynamics.fields.magnetostatics`):
  charged-sheet B0 of permanent-magnet bars with a soft-iron return yoke (method
  of images) and Biot-Savart coil B1, and a single-sided NMR-MOUSE
  (`spin_dynamics.workflows.single_sided`) depth-resolved relaxation/diffusion
  simulation that diffuses walkers through the magnet's real static gradient,
  with the engine's constant-gradient diffusion validated against the exact
  Carr-Purcell law;
- moving-isochromat sequence driver primitives, including explicit sequence
  intervals, RF/free-precession substeps, receive samples, and a rectangular
  CPMG runner for static-gradient diffusion/advection studies;
- 1D and separable 2D inverse Laplace transform helpers for T1, T2, T1-T2,
  D-T2, and T2-T2 synthetic analysis, with adjustable Tikhonov regularization
  and SNR-informed automatic strength selection plus SciPy-backed non-negative
  solves;
- Bloch-McConnell site/chemical exchange (`spin_dynamics.exchange`):
  multi-site kinetic generators with detailed-balance checks, transverse
  lineshape coalescence, longitudinal mixing propagators, and encode-mix-detect
  T2-T2 relaxation exchange (REXSY) data that inverts to an exchange map;
- susceptibility-induced internal gradients (`spin_dynamics.susceptibility`):
  analytic 2D cylindrical-grain off-resonance maps that feed the moving-
  isochromat pipeline, plus pore-space internal-gradient distributions for
  diffusion-in-internal-gradient studies;
- bipolar 13-interval PGSTE (`spin_dynamics.workflows.bipolar`): the Cotts
  alternating PGSTE (Bruker `diff_stebp`) with the 16-step `diff_stebp` phase
  cycle, toggling-frame moment integration, an explicit moving-walker runner
  (`run_cotts_thirteen_interval_walkers`) validated against `exp(-b D)`, and
  background-gradient cross-term suppression versus a monopolar stimulated echo;
- fixture-validated pulse-shape utilities for JMR rectangular pulse responses,
  phase quantization, and untuned segment adjustment;
- WURST pulse construction, matched-probe frequency-swept transmit response,
  ideal WURST inversion, matched WURST inversion, and matched WURST-CPMG
  workflows;
- SPA refocusing pulse catalog and normalized SNR/FOM metric bookkeeping;
- tuned, untuned, and matched fixed-refocusing-pulse SPA/OCT evaluation wrappers;
- tuned, untuned, and matched SPA summary workflows over rectangular and
  catalog pulses;
- lightweight discrete phase-program search scaffold for optimizer experiments;
- ideal no-probe v0crit, excitation-aware v0crit, and time-varying-field
  refocusing evaluation, bounded optimizers, and multi-start drivers;
- tuned, untuned, and matched bounded refocusing phase optimizer wrappers
  around the fixed-amplitude SNR evaluators;
- tuned excitation-pulse evaluation and bounded phase optimizer wrappers for
  supplied refocusing axes;
- tuned inverse-excitation evaluation and bounded phase optimizer wrappers for
  target received spectra, with compact MATLAB optimizer-result and residual
  spectrum validation;
- array-returning multi-start optimization driver scaffolds for repeated
  refocusing, tuned excitation, and phase-flipped tuned inverse-excitation
  searches;
- plotting-free optimization pipeline handoff from selected refocusing result
  to tuned excitation and inverse-excitation searches;
- MATLAB-style optimization result-cell conversion plus `.npz` archive
  load/export, script-aware `plot_opt_*_results.m` layout analysis,
  selected score/program/metadata inspection, tuned original/inverse
  result-pair comparison, compact tuned/untuned/matched refocusing result
  fixtures, and optional SciPy-backed `.mat` import/export;
- optional SciPy continuous bounded optimizer backend for phase optimization,
  with NumPy pattern search as the minimal-install fallback;
- matched-probe z-magnetization Q sweep workflow;
- ideal, tuned, untuned, and matched time-varying-field CPMG final-echo and
  amplitude-sweep workflows;
- public CPMG workflow runners returning `CPMGResult`.
- initial single-electron ESR helpers with scalar/anisotropic `g` tensors,
  dense Zeeman diagonalization, orientation grids, frequency spectra, and field
  sweeps, Gaussian/Lorentzian derivative display, diagonal `g` strain and
  applied-field offset distributions, plus rectangular-pulse FID and Hahn-echo
  simulations with Liouville-space T1/T2 relaxation, and isotropic
  electron-nuclear hyperfine doublet simulations;
- four-pulse DEER/PELDOR pulsed dipolar ESR (`spin_dynamics.esr.dipolar`,
  `spin_dynamics.esr.deer`): point-dipole electron-electron coupling derived
  from fundamental constants (canonical 52.04 MHz nm^3), distance/frequency
  conversions and the `(1 - 3 cos^2 theta)` angular dependence, single-pair and
  powder-averaged DEER kernels, a forward model from a distance distribution
  P(r), the dipolar (Pake) spectrum, Tikhonov-regularized P(r) recovery reusing
  the inverse-problem machinery in `analysis.regularization`, and an independent
  two-electron density-matrix simulation of the full four-pulse sequence that
  reproduces the analytic kernel to floating-point precision.
- electron-nuclear correlation spectroscopy for an S=1/2 electron coupled to a
  nucleus of spin I=1/2, 1, or 3/2 (`spin_dynamics.esr.eseem`, `esr.hyscore`,
  `esr.endor`): the secular plus pseudosecular `HyperfineCoupling` with an
  optional nuclear quadrupole interaction (`quadrupole_hz`, `eta`, reusing the
  NQR quadrupole Hamiltonian), per-manifold nuclear frequencies by
  diagonalization (`manifold_frequencies`), the spin-1/2 closed-form nuclear
  frequencies and modulation depth, analytic two- and three-pulse ESEEM with a
  density-matrix engine (electron coherence-pathway selection plus an explicit
  phase-cycled variant shown to match it), 2D HYSCORE with cross-peaks at the
  nuclear frequencies, and Davies/Mims ENDOR including the Mims `sin^2(pi nu
  tau)` blind spots. The spin-1/2 density-matrix traces reproduce the analytic
  results to floating-point precision, and the I=1 `14N` exact-cancellation
  regime reproduces the pure-quadrupole NQR lines.

Remaining gaps:

- newer tuned-probe helper variants outside the original/reference and JMR
  rectangular-pulse paths;
- newer untuned-probe helper variants outside the original/reference and JMR
  rectangular-pulse paths;
- newer matched-probe helper variants outside the original/reference and JMR
  rectangular-pulse paths;
- probe-shaped T1-prepared imaging for tuned or matched inversion pulses;
- general phase cycling now supports arbitrary labeled `SequenceIR` programs:
  `PhaseCycle.apply_to_sequence()` produces independently executable branches
  with receiver weights, including repeated logical pulse roles. Direct
  workflow helpers do not all expose a `phase_cycle=` argument, and NQR/ESR
  pathway selection remains workflow-specific. See
  [Phase Cycling Findings](phase_cycling.md);
- broad diffusion sweeps, Q>2000 matched-diffusion validation, trapezoidal
  (ramped) diffusion-gradient shapes for the bipolar 13-interval walker runner
  (it currently uses rectangular lobes), and probe-shaped PGSE pulses;
- probe-shaped (tuned/matched) pulses in the moving-isochromat imaging
  workflows, and genuinely Fourier-encoded 3D imaging (slab-select plus a second
  phase encode reconstructed with `ifftn`); the current 3D capability is
  slice-selective multi-slice, and the in-plane readout workflows are ideal-probe;
- exact MATLAB WURST fixture parity beyond finite-output and physical sanity
  tests, because the MATLAB WURST scripts are exploratory and include
  placeholder or plotting-oriented branches;
- full historical MATLAB `.mat` result-file parity and broad `fmincon` parity
  validation. Inverse-excitation workflows now report integrated and peak
  residual ratios, inverse coherence, and optional SNR balance, but strong
  cancellation parity still requires broader authoritative result archives;
- broader compiled/GPU backend coverage beyond the current selected Numba/JAX
  kernels. Dedicated CI jobs now install both optional extras and compare the
  supported CPU paths with NumPy; GPU runners remain unavailable;
- received-signal noise-model validation: the probe-noise variance carries a
  `/ dx**2` rescaling onto a user-supplied `sample_axis` (a no-op for the
  default unit grid) whose convention is not independently validated for
  physical sample spacings. The matched-probe amplifier noise-figure term uses
  the available-power `k*T*Rin*(F-1)` basis, consistent with the matched coil
  term (the factor-of-4 difference from the open-circuit `4*k*T*R` form is the
  matched 1/2-voltage / 1/4-power transfer); absolute SNR magnitudes still
  benefit from validation against a measured noise figure. The physical
  spin-noise models (`spin_dynamics.sample`, `spin_dynamics.spin_noise`, and
  the `sample=` keyword on the probe noise densities; see
  `docs/spin_noise.md`) are validated against internal identities and analytic
  linear-response results but not yet against measured spin-noise spectra, and
  the matched-probe variant treats the sample impedance as a series
  perturbation without re-matching (quantitative for `R_n0 << Rc`);
- ESEEM/HYSCORE/ENDOR beyond a single I<=3/2 nucleus with a field-collinear
  quadrupole tensor: spin I>3/2, anisotropic hyperfine `A` tensors with powder
  averaging, tilted (Euler-rotated) quadrupole tensors, and multiple coupled
  nuclei;
- finite/shaped DEER pump pulses and explicit excitation-bandwidth selection (the
  DEER and ESEEM/HYSCORE models use ideal spin-selective pulses); anisotropic
  hyperfine and exchange couplings in the CW/echo helpers; higher-spin zero-field
  splitting; temperature-dependent equilibrium magnetization; and ESR saturation
  or resonator models;
- coupled thermal modeling (`spin_dynamics.thermal`, see
  `docs/thermal_modeling.md`) currently provides the lumped RC network, the
  quasi-static coupling loop, and 1D/axisymmetric conduction with Pennes
  perfusion, plus electrothermal electromagnet B0 sources with voltage,
  temperature, current, and field-lock control. These are validated
  analytically and selected conduction cases against FEMM. The stretch 3D
  finite-difference conduction backend on the `fields.domain` grids (Phase 5)
  is not yet implemented, convection enters only through film coefficients (no
  flow solving), the sample-SAR source inherits the quasistatic solver's Born
  limits (no skin-depth shielding), and thermo-mechanical expansion,
  permanent-magnet drift, and nonlinear ferromagnetic-core behavior remain out
  of scope;
- electropermanent AlNiCo B0 sources (`spin_dynamics.fields.electropermanent`,
  see `docs/electropermanent_magnets.md`) model finite rods and bundles, field
  maps and composition adapters, capacitor/H-bridge/RLC programming pulses,
  passive recovery, lumped pulse heating, and protocol-scoped empirical state
  transitions. Weighted play operators now provide return-point memory, and a
  quasistatic fixed-point solve couples self/neighbor fields, state-dependent
  inductance, and retained state. A multi-winding transient solve now adds
  mutual inductance, induced neighbor currents/voltages, closed recovery paths,
  winding leakage, and pulse-driven disturbance of every return-point state.
  The remaining gaps are measured demagnetization/minor-loop and array-coupling
  matrices, repeated-pulse retention, selectable open/clamped inactive-channel
  topologies, switching-loss temperature impulses, exact hybrid-array CAD,
  experimentally calibrated particle contrast and state estimation,
  vascular/tissue transport physics, concentration-dependent particle interactions, calibrated
  multi-channel programming energy, and conductive-shield eddy loss. The
  specified 18-sub-unit/72-control hierarchy, illustrative hybrid geometry,
  cached field bases, and bounded imaging/off/affine-transport synthesis are
  implemented, as are dense nonlinear encoding, nonnegative regularized
  reconstruction, acquisition noise, a simple tissue-phantom example, and
  superparamagnetic aggregate force/trajectory modeling with Langevin
  saturation, Stokes drag, Brownian diffusion, background flow, boundaries,
  target capture, particle-sensitive reconstruction, and an alternating
  image-estimate-program-transport-verify controller with explicit mode timing
  and image-derived capture-goal stopping. The concentration channel assumes a
  calibrated signal per particle and synthetic noise; experimental contrast,
  the boundary-capture partial-volume correction, identity-free tracking, and
  estimator uncertainty remain to be validated experimentally;
- flowing-sample modeling (`spin_dynamics.flow`, see `docs/flow_modeling.md`)
  covers plug and laminar pipe flow -- washout during acquisition, transit-time
  inflow polarization, and thermal advection (lumped `flow_conductance` plus a
  1D upwind advection term in `Conduction1D`). Turbulent/well-mixed (CSTR)
  flow, graded coil-sensitivity washout, analyte Taylor dispersion, and a
  probe-shaped moving-isochromat inflow/outflow boundary are not yet modeled;
- stable distribution artifacts, typed-package metadata, API lifecycle policy,
  and a published documentation site (the Markdown API inventory is already
  autogenerated and checked by CI).

The next natural work is therefore a stabilization phase: broader PGSE and
constant-gradient diffusion validation, stronger inverse-excitation
cancellation workflows, exact historical MATLAB result-file parity where those
files are still authoritative, and packaging/API-reference polish.
