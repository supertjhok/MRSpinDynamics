# Changelog

All notable changes to the **MRSpinDynamics** workspace are recorded here. The
repository is released as a single citable unit (see [`CITATION.cff`](CITATION.cff)
and [`docs/release_process.md`](docs/release_process.md)); one version covers all
subprojects. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### PythonSpinDynamics

- New `spin_dynamics.experiment` facade (unified workflow PR-1): declarative
  frozen-dataclass experiment specs (`Experiment`, `Sample`, `Hardware`,
  `Acquisition`, `CPMG`/`CPMGTrain`/`CPMGIRTrain`), a workflow registry
  covering the CPMG family (ideal/tuned/untuned/matched × asymptotic, finite
  train, inversion-recovery train), a `plan()` stage that resolves the
  workflow and warns about spec fields it would ignore, `run()` delegation
  that reproduces the direct `run_*` calls bit for bit, and NPZ save/load
  with JSON provenance and spec round-trip. Design and milestones in
  `PythonSpinDynamics/docs/unified_workflow_plan.md`.
- Experiment facade PR-2: `plan()` now runs a declarative compatibility rule
  engine (`experiment/rules.py`). A front-loaded rephasing pre-check reports,
  before `run()` executes, whether the isochromat grid is fine enough for the
  sequence (with the recommended `numpts`), respecting
  `acquisition.rephase_action` (`ignore`/`warn`/`raise`); a noise-spec rule
  rejects malformed or unsupported noise at plan time. Findings are exposed
  structurally on the plan (`ExperimentPlan.findings`) and merged into
  `errors`/`warnings`. The per-workflow `max_time` timing formulas were
  extracted into shared helpers so the plan-time verdict matches the
  workflow's own run-time check exactly.

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
