# Changelog

- Stabilize PEEC parallel-terminal reductions by validating the direct complex solve and
  falling back to an SVD least-squares solve when a numerical backend returns non-finite
  currents.

All notable changes to the **MRSpinDynamics** workspace are recorded here. The
repository is released as a single citable unit (see [`CITATION.cff`](CITATION.cff)
and [`docs/release_process.md`](docs/release_process.md)); one version covers all
subprojects. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### PythonSpinDynamics

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

[Unreleased]: https://github.com/supertjhok/MRSpinDynamics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/supertjhok/MRSpinDynamics/releases/tag/v0.1.0
