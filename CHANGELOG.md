# Changelog

All notable changes to the **MRSpinDynamics** workspace are recorded here. The
repository is released as a single citable unit (see [`CITATION.cff`](CITATION.cff)
and [`docs/release_process.md`](docs/release_process.md)); one version covers all
subprojects. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Completed the PythonSpinDynamics image-guided therapy feedback loop with a
  calibrated particle-sensitive forward image, nonlinear noisy reconstruction,
  image-derived particle count, representative positions, centroids, and target
  occupancy, plus post-transport verification imaging. The controller now aims
  and stops from reconstructed particle state; simulator truth is retained only
  for estimator-error diagnostics.
- Added a dynamic-inversion magnetic-particle trap to PythonSpinDynamics with
  rigid sphere/finite-cylinder hydrodynamics, coupled body and moment rotation,
  Brownian motion, finite internal relaxation, non-sticky stability metrics,
  the published Nacev polarize-delay-gradient timing, and coil/EPM/hybrid
  timing and switching-burden assessments.
- Started stream-function MRI gradient-coil design in PythonSpinDynamics with
  cylindrical thin-wire source meshes, chunked per-segment Biot-Savart
  sensitivities, exact axial KCL constraints, Tikhonov-regularized NumPy/SciPy
  solvers, discrete stream-function recovery, L-curve regularization paths,
  seam-aware oriented 3-D winding contours, plotting examples, tests, and user
  manual documentation. Active shielding and PEEC integration remain planned.
- Ported the multi-axis NQR SLSE optimizer from the retired feature branch,
  including degeneracy-safe JAX/NumPy multi-coil propagation, bounded
  amplitude/phase/timing optimization, powder-averaged validation tests, and a
  runnable one-/two-/three-axis comparison example.

### Fixed

- Hardened PEEC impedance extraction against platform-dependent numerical
  failures: unstable distant-filament closed forms are repaired by fixed
  Gauss-Legendre integration, while terminal reduction retries an equilibrated
  system and can fall back to partial-pivoted Gaussian elimination if NumPy's
  direct and SVD-based solvers both fail.

### Changed

- Rewrote the PythonSpinDynamics web documentation openings around reader
  questions, prerequisites, model boundaries, and neighboring guides; added a
  documentation map and maintainer writing standard; clarified that the
  magnetic-therapy controller reconstructs its tissue target while particle
  feedback still comes from simulation state; and converted stale planning
  language in implemented thermal, coil, NQR, and workflow material into
  current guidance.
- Reorganized the PythonSpinDynamics print and web manuals around user tasks
  instead of one catch-all workflow chapter. Coupled-spin and hyperpolarization
  material now shares one physical-model chapter; imaging, fields, EPM
  hardware, magnetic therapy, transport, detection, analysis, and optimization
  have distinct chapters; and MkDocs exposes image-guided therapy as a
  dedicated application guide under Fields and Hardware.
- Expanded the PythonSpinDynamics user manual with thirteen reproducible
  example-result figures spanning CPMG probe response, BPP relaxation, ESR
  powders, the NQR--NMR crossover, multinuclear ZULF coupling, sequence timing,
  bSSFP imaging, solenoid and single-sided magnet design, restricted diffusion,
  radiation damping, exchange inversion, and SQUID detection. Figures now
  default to top-of-page placement in the print edition, and
  `docs/generate_manual_figures.py` regenerates the assets.
- Parallelized the branch-coverage suite with pytest-xdist and pytest-cov while
  preserving combined branch measurement and the existing coverage floor.
- Added a strict MkDocs Material documentation build and GitHub Pages deployment
  with client-side search across the web manual, generated API reference,
  examples, model documentation, and validation evidence; the paginated PDF
  remains available as the print edition.

## [0.3.0] - 2026-07-12

This workplace-readiness release combines the scientific capabilities developed
since v0.2.0 with hardened packaging, documentation, performance guardrails, and
release automation.

### v0.3.0 highlights

- Completed a regime-independent NQR-to-quadrupolar-NMR workflow spanning exact
  static spectra, laboratory/Floquet RF references, field-dependent equilibrium
  and secular/nonsecular relaxation, receiver-correct powder echo trains,
  parallel powder execution, and history-dependent field sweeps.
- Added a backend-neutral Pulseq-capable sequence IR, arbitrary phase-cycle
  branches, general facade execution, deterministic saved-result provenance,
  and typed cross-component adapters for fields, thermal states, flow,
  hardware responses, and compiled timelines.
- Added validated 3-D scalar-potential magnetostatics, arbitrary-geometry PEEC
  coil impedance/capacitance/loss models, physical spin-noise sources, coupled
  thermal modeling, and flowing-sample transport workflows.
- Delivered a book-calibrated portable Halbach MRI design workflow and robust
  finite-pulse q-space imaging studies with committed validation evidence.
- Expanded measured-data relaxation validation, PHIP/long-lived-singlet
  workflows, Bloch-Siegert modeling, the experiment facade, cross-project
  uncertainty propagation, and beginner-oriented documentation.
- Added change-aware tests, isolated wheel/sdist checks, optional-backend parity,
  branch coverage, scoped typing, reproducible-result verification, and a
  host-normalized performance-regression gate.
- Hardened release packaging with license-bearing distributions, public package
  and CLI version reporting, checked project metadata, a shared release
  preflight, checksummed workspace artifacts, and workplace installation notes.
- Corrected the authoritative full and impacted test paths to use pytest so both
  unittest classes and pytest-style functions execute; added deterministic BLAS
  thread limits, minimum-dependency coverage, and a full Windows lane.
- Extended the performance gate with a compact powder-waveform workload and
  forward-compatible benchmark schemas that preserve old coverage while new
  cases establish a baseline.

### Compatibility

- No intentional public API removals are planned for v0.3.0. Newly deprecated
  APIs follow the documented two-minor-release compatibility window.
- Python 3.10--3.12 remains the supported CI range. Core PythonSpinDynamics
  continues to require only NumPy; SciPy, plotting, JAX, Numba, and benchmark
  dependencies remain optional extras.
- The release remains one versioned workspace and is not a public PyPI release.
  Install from the workspace or the checked GitHub Release artifacts.

### PythonSpinDynamics

- Added a regime-independent exact static NQR-to-NMR crossover spectrum for
  `H_Q + H_Z`, with Boltzmann population weighting, complex transmit/receive
  polarization geometry, all-transition spectra, and basis-invariant Kramers
  manifold intensities at exact degeneracies.
- Added overlap-tracked NQR-to-NMR field sweeps and a repository-parameterized
  NaNO2 `14N` / NaClO3 `35Cl` energy-level and transition-intensity example.
- Added optional powder-averaged crossover field-frequency maps and a general
  piecewise-constant laboratory-frame RF reference with exact unitary
  propagation and no rotating-wave approximation.
- Added quantitative single-band RWA validity maps and finite-sideband Floquet
  RF propagation, checked against direct laboratory-frame dynamics throughout
  the crossover.
- Added exact field-dependent Gibbs equilibrium, a degeneracy-safe
  Gibbs-reset/dephasing model, thermal Davies magnetic/EFG relaxation, and a
  completely positive unified-GKLS model retaining unresolved nonsecular
  coherence-transfer terms.
- Added SLSE dynamics across zero field, Zeeman-perturbed NQR, the intermediate
  regime, and quadrupolar NMR using one complete `H_Q + H_Z` description.
- Corrected nonzero-field powder relaxation observables by acquiring complete
  complex echo waveforms, coherently averaging crystallites before receiver
  filtering, and estimating the echo train with a matched receiver instead of
  averaging local magnitudes or sampling only echo centers.
- Added nested low-discrepancy SO(3) convergence, reported frequency-slice
  weights, finite-bandwidth receiver checks, and deterministic thread/process
  powder execution with a reduce-only production mode.
- Added piecewise-linear vector-field histories, including reversals and
  rotations, with coherent or relaxing propagation and diagnostics for
  populations, coherences, polarization, energy, Gibbs deviation, and density
  positivity.
- Documented the full physical and numerical procedure for realistic powder
  relaxation so averaging order, receiver sampling, slicing, convergence, and
  parallelization are not reinvented in later work.
- Added a declarative change-aware local test selector and restored the smoke
  tier to constant-time behavior by leaving catalog-wide example CLI checks in
  the example/full tiers.
- Added a 3-D reduced-scalar-potential magnetostatic solver with sparse
  direct/AMG backends and integrated it into single-sided NMR-MOUSE field,
  sensitivity, and imaging workflows.
- Added arbitrary-geometry PEEC coil extraction with filament mutual
  inductance, proximity/surface loss, capacitance and grounded-shield models,
  radiation resistance, differential/ground-mode analysis, and independent
  FastHenry/FasterCap comparisons. Stabilized parallel-terminal reduction by
  validating the direct complex solve and falling back to SVD least squares
  when a numerical backend returns non-finite currents.
- Added physical sample-impedance and stochastic spin-noise models rather than
  treating all receiver noise as an externally supplied scalar.
- Added coupled thermal materials, duty-cycled sources, lumped RC networks,
  quasi-static electro-thermal iteration, 1-D/axisymmetric conduction with
  perfusion, and NMR-MOUSE thermal examples.
- Added pipe-flow kinematics, velocity-profile washout, transit-time inflow
  polarization, and thermal advection workflows, including a fix for spurious
  unused-tail overflow warnings at zero time.
- Added complete dense two-spin SLIC long-lived-state preparation,
  phenomenological measured-`T_S` storage, symmetry purge, and readout, plus
  pairwise-yield-aware hydrogenative PHIP mapping, high-field PASADENA,
  explicit-trajectory ALTADENA, selected-spin hard pulses/acquisition, tests,
  examples, and user-manual coverage.
- Added the first PHIP/long-lived-singlet foundation: exact embedded
  singlet/triplet projectors, pair-swap symmetry, normalized singlet-order
  observables, physical and deviation parahydrogen states versus para fraction,
  and a compact enrichment example. Corrected the SLIC convention so the
  spin-lock amplitude matches `|J|` while chemical-shift inequivalence controls
  the transfer time, and documented the staged PHIP/SABRE implementation plan.
- Added a book-calibrated end-to-end portable Halbach MRI workflow coupling
  finite-magnet and book-dimensioned RF/gradient-coil field maps, PEEC-predicted
  receive impedance and SNR, field-integrated ferrite RF loss, duty-cycled I2R
  heating, ferrite resonance drift,
  tuned-receiver loss, predicted complex noise, sparse/TV
  compressed sensing, and a held-out k-space quality metric that automatically
  stops acquisition when improvement plateaus.
- Expanded the portable Halbach workflow into a designer-facing capstone with
  RF power/SNR/active-volume pulse sweeps, RF and gradient coil tables, peak
  gradient current and voltage, ADC gain budgeting, effective slice thickness,
  system mass accounting, and two generated user-manual figures.
- Replaced ideal rectangular-current and scalar-Q shortcuts in the capstone
  with the scanner's actual inverse-PCMCD series-tuned Tx chain and actively
  feedback-damped parallel-tuned Rx chain, including finite envelope bandwidth,
  ring-up, echo filtering, and filtered receiver-noise bandwidth.
- Corrected the capstone's compressed-sensing comparison by using an incoherent
  variable-density mask, withholding validation data only during stopping,
  folding all acquired data into the delivered image, and using finite-difference
  TV for the final reconstruction. A regression gate now requires CS NRMSE to
  beat zero-fill by at least 15% on the 64×64 capstone.

- Added a host-normalized CI performance regression gate that benchmarks the
  Git base and candidate commits on the same runner. Compact core, workflow,
  compiler, and spatial-sampling workloads fail on large individual or broad
  aggregate slowdowns, with JSON comparisons retained as workflow artifacts.

- Made arbitrary phase cycles first-class for labeled `SequenceIR` programs,
  including reusable branch generation, receiver weights, repeated logical RF
  roles, and compile-level validation. Added quantitative inverse-excitation
  cancellation diagnostics and dedicated Numba/JAX CPU parity jobs in CI.

- Added `spin_dynamics.composition`, a typed interchange layer for named SI
  spatial grids, absolute time axes, field solutions, thermal states, flow
  fields, hardware responses, and compiled sequence timelines. The adapters
  validate units and axes, perform spatial/time resampling, and bridge existing
  magnetostatic, thermal, flow, optimal-control, and sequence result types.

- Added the installed `spin-dynamics` experiment CLI, typed-wheel marker,
  isolated wheel/sdist installation checks, scoped MyPy and branch-coverage CI
  gates, and an explicit two-minor-release deprecation policy with reusable
  warning utilities.

- Suppressed spurious floating-point warnings from the unused laminar-washout
  tail branch at zero time and added an extreme-residence-time regression test.
- Began the backend-neutral sequence IR and compiler: sequential blocks with
  concurrent RF, three-axis gradient, and ADC events; piecewise-constant
  timeline compilation; a moving-isochromat adapter; and core text import for
  the open Pulseq 1.4/1.5 `.seq` format, including compressed shapes,
  trapezoids, ADC phase shapes, and PPM offsets. Required Pulseq extensions are
  rejected until their semantics are implemented.
- Added an aligned sequence timeline visualizer for native and Pulseq-imported
  sequences, with RF I/Q, Gx/Gy/Gz, ADC, block-boundary lanes, `.plot()` hooks,
  and a `plot_sequence_timeline.py` example that also accepts external `.seq`
  files.
- Added strict Pulseq 1.5.0 export with MD5 signatures, RF-center and gradient-
  boundary preservation, raster and gradient-continuity validation, PyPulseq
  interoperability checking, and export support in the timeline example.
- Expanded the declarative experiment facade to deterministic PGSE diffusion,
  including sample diffusion coefficients, plan-time timing validation, cost
  estimates, config/JSON/NPZ round-trips, and direct-workflow parity tests.
- Added a seeded random-walker PGSE facade route with explicit 2-D transport
  domains, boundary policies, uniform flow/advection, stochastic cost estimates,
  friendly configs, and exact direct-workflow parity.
- Added finite-pulse walker-to-q-space acquisition with arbitrary in-plane PGSE
  gradient directions, zero-q normalization, elliptical-pore reconstruction
  validation, and a runnable walker-to-pore-imaging example.
- Completed q-space imaging robustness studies for ellipse, slit, and connected
  pores across finite intensity SNR, radial q-window truncation, and random
  missing samples; added mask-aware phase retrieval, explicit noise-floor
  gating, ambiguity-invariant shape metrics, and committed trial-level results.
- Deepened microscopic relaxation validation with measured sodium-chlorate,
  transition-resolved bismuth, and joint RDX NMR/NQR benchmarks; added a
  read-only NQRDatabase relaxation audit, non-diagonal zero-field EFG Redfield
  model, Arrhenius fitting, transition-resolved quadrupolar relaxation and
  field-cycling/NMRD workflows, runnable examples, and expanded manual guidance.

### Integration

- Added a structure-to-validation target survey that joins the DFT structure
  inventory, curated EFG summaries, and measured NQR lines; automatically runs
  all comparison-ready summaries; and exposes the next five material DFT
  targets plus missing database metadata.
- Added correlated `(C_Q, eta)` uncertainty propagation through both independent
  Hamiltonian implementations, producing per-line prediction intervals and
  measured-line coverage reports.

### QuadrupolarDFT

- Rewrote the user manual around a beginner-first path from crystal structure
  to a validated NQR prediction, with explicit checkpoints, clearer links to
  the companion Beginner's Guide, and a two-tier ABINIT/Elk strategy explaining
  the accuracy-versus-compute-time tradeoff.

## [0.2.0] - 2026-07-05

A large feature release built on the v0.1.0 foundation. The headline addition is
a unified, declarative experiment facade for PythonSpinDynamics; alongside it,
this release adds optimal-control pulse design, non-inductive detection, pulsed
dipolar and hyperfine ESR spectroscopy, higher-spin NQR, several new field
solvers, and an all-electron DFT EFG cross-check that resolves the
asymmetry-parameter accuracy gap. All changes are additive; there are no
breaking API changes.

### PythonSpinDynamics — Python simulation package

- **Unified experiment facade and CLI.** New `spin_dynamics.experiment` package:
  declarative frozen-dataclass specs (sample, hardware, sequence, acquisition)
  wrapping the validated `run_*` / `simulate_*` workflows, so a default
  experiment reproduces the direct call bit for bit. A `plan()` stage resolves
  the workflow, runs compatibility rules (isochromat-grid rephasing pre-check,
  noise-spec validation, coil-field wiring, and reduced-vs-full NQR model
  selection), and reports an advisory host-calibrated runtime/memory estimate;
  `run()` delegates and captures provenance; results save to NPZ with JSON
  provenance and full spec round-trip. It covers the CPMG family, phase-encoded
  imaging (with automatic Biot-Savart transmit-coil B1 solving on the phantom
  grid), pulsed NQR (SLSE/SORC), and pulsed ESR (FID/Hahn echo). A config-driven
  CLI (`python -m spin_dynamics.experiment` with `plan` / `run` / `show` /
  `convert`) reads human-friendly TOML or JSON configs. See
  `docs/python_api/experiment_workflow.md`.
- **Optimal control (GRAPE).** New `spin_dynamics.optimal_control` package:
  gradient-ascent pulse design for RF amplitude/phase with state-transfer,
  propagator-fidelity, and robust/ensemble objectives; a gradient-waveform
  control channel; NQR/quadrupolar targets; and a hardware-response layer
  (probe, gradient-driver, and receiver LTI filters) with PGSE diffusion
  objectives. Includes a CPMG SNR-per-time / AMEX example.
- **Non-inductive detection.** New `spin_dynamics.detection` subpackage: a
  field-referred `Detector` layer (transfer `H(f)` and field-noise `N²(f)`) with
  an inductive-coil adapter, `SQUIDMagnetometer` and `OPMMagnetometer` models
  with a Faraday crossover baseline, gradiometer pickup geometry with spatial
  detected SNR, and a detector-aware GRAPE objective for flux readouts.
- **Pulsed ESR spectroscopy.** Added dipolar ESR (DEER/PELDOR) with distance
  recovery, and ESEEM/HYSCORE/ENDOR pulsed-ESR spectroscopy generalized to
  quadrupolar nuclei (I = 1, 3/2), reusing the package phase-cycling machinery.
- **NQR.** Support for quadrupolar spins > 3/2 (5/2, 7/2, 9/2) in the `(2I+1)`
  density-matrix engine; circular and multi-axis RF excitation for powder ¹⁴N;
  a corrected SLSE-vs-SORC sensitivity-per-time analysis; and a glycine
  piezoelectric NQR drive workflow.
- **Nonresonant field-reversal (CSAR).** New `spin_dynamics.nonresonant`
  subpackage: RF-free spin manipulation by sudden field switching and adiabatic
  rotations, reproducing Brill 2002.
- **Radio-frequency interference.** New `spin_dynamics.interference` toolkit:
  robust and active-feedforward cancellation, a frequency-domain Wiener
  canceller, a reference-free Kalman harmonic tracker, a sparse-decomposition
  canceller for impulsive pickup, a measured-record loader, Numba-accelerated
  adaptive LMS/RLS, and a pulsed NQR RFI simulation.
- **Field solvers.** Quasistatic E-field and eddy-current solvers, a nonlinear
  magnetostatic solver for soft-magnetic and permanent-magnet materials, and
  coil sample loading (reflected impedance) added to `spin_dynamics.fields`.
- **Low-field and relaxation workflows.** Zero/ultra-low-field J-coupled spin
  networks with ¹⁴N quadrupolar relaxation derived from `C_Q` and `τ_c`; a
  prepolarized T1-rho spin-lock sequence with molecular-size dispersion; and a
  balanced SSFP (bSSFP) steady-state imaging sequence.
- **Examples.** A Jasper-Jackson multinuclear CPMG well-logging workflow (with
  an optimized ferrite-focused RF coil and honest loss-limited SNR), an
  accelerated porous-rock walker challenge, an inhomogeneous-`(B0, B1)` ESR echo
  study, and higher-spin NQR surveys.

### QuadrupolarDFT — ab initio EFG and quadrupolar coupling

- A structure-relaxation stage and harmonic finite-temperature EFG averaging for
  the NaNO₂ ¹⁴N workflow, with a CIF-based EFG validation example.
- An EFG convergence-study tool and an asymmetry-parameter (η) accuracy
  investigation.
- An all-electron Elk EFG cross-check backend that resolves the η accuracy gap:
  ABINIT PAW under-predicts η (≈ 0.11 vs experiment 0.38), while all-electron
  Elk PBE at the same geometry and functional gives η ≈ 0.33, identifying the
  gap as a PAW pseudopotential artifact rather than missing physics.
- A DFT beginner's guide and an expanded user-manual η section.

### Workspace

- Zenodo archival with a concept DOI (10.5281/zenodo.21016177) wired into
  `CITATION.cff` and the README.
- Per-subproject `LICENSE` copies and a CC-BY-4.0 data license for the
  NQRDatabase curated data.
- A scientific-frontiers roadmap and updated development-environment
  documentation for OneDrive/WSL `$HOME` virtual-environment layouts.

## [0.1.0] - 2026-06-28

First tagged release of the workspace. This consolidates the work tracked in
[`docs/roadmap.md`](docs/roadmap.md) into one citable snapshot. Subprojects are
at deliberately different maturity levels (see each subproject's README and the
roadmap); the shared version marks the state of the workspace as a whole, not a
uniform stability guarantee.

### PythonSpinDynamics — production-grade simulation package

- NMR workflows: ideal/tuned/untuned/matched-probe CPMG, finite echo trains,
  inversion recovery, Q/mistuning sweeps, diffusion CPMG, PGSE/PGSTE, WURST,
  prepolarization, radiation damping, and moving-isochromat motion.
- Imaging: spin-warp, RARE, slice-selective and true-3D multi-slice imaging in
  spatially varying `(B0, B1)`; analytic magnet field sources including a
  single-sided NMR-MOUSE simulation.
- NQR: spin-1 reduced two-level and spin-3/2 full density-matrix models, powder
  averaging, weak-B0 spectra, SLSE, and population transfer.
- ESR/EPR: single-electron spectra, anisotropic g tensors, powder grids, pulsed
  FID/Hahn echo, hyperfine doublets.
- Relaxation: shared Liouville helpers, phenomenological models, and an opt-in
  Redfield/dipolar model with rigid-solid and isotropic-liquid averaging.
- Analysis: 1D/2D inverse-Laplace for T1, T2, T1-T2, D-T2; coupling and exchange
  models; q-space pore-imaging inversion and phase retrieval.
- Optional Numba and JAX isochromat backends. Core runtime depends only on NumPy.
- Validated against the MATLAB reference; see
  [`PythonSpinDynamics/docs/python_api/validation.md`](PythonSpinDynamics/docs/python_api/validation.md)
  and remaining limitations in
  [`PythonSpinDynamics/docs/python_api/known_gaps.md`](PythonSpinDynamics/docs/python_api/known_gaps.md).

### QuadrupolarDFT — ab initio EFG and quadrupolar coupling

- EFG -> C_Q, eta, and predicted NQR frequencies from first-principles outputs.
- Harmonic finite-temperature vibrational averaging of the EFG tensor with a
  Bayer single-libration fit; full three-stage ABINIT DFPT workflow
  (phonon -> displace -> collect), validated against real `.abo` EFG output.

### NQRDatabase — curated measured spectra

- 184 compounds with measured NQR frequencies, provenance, and citations.
- SQLite and JSONL exports; a review workflow for OCR-derived Landolt-Bornstein
  tables; an explorer web app with per-site consistency badges.

### integration (mr_integration) — predict-simulate-validate loop

- Validated C_Q <-> nu_Q conversion bridging the DFT and simulator conventions.
- Cross-validation that runs DFT-derived parameters through the simulator and
  checks self-consistency against the curated database.
- Simulator-driven database self-consistency and Landolt review-queue flagging,
  surfaced in the NQR explorer.

### Workspace

- Single-version release process, `CITATION.cff`, `.zenodo.json`, and
  [`docs/release_process.md`](docs/release_process.md) for GitHub-Release-driven
  Zenodo DOI minting.

[Unreleased]: https://github.com/supertjhok/MRSpinDynamics/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/supertjhok/MRSpinDynamics/releases/tag/v0.3.0
[0.2.0]: https://github.com/supertjhok/MRSpinDynamics/releases/tag/v0.2.0
[0.1.0]: https://github.com/supertjhok/MRSpinDynamics/releases/tag/v0.1.0
