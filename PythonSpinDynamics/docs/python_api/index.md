# Python API Documentation

This directory documents the Python port as it exists today. The MATLAB
implementation under `../MATLABSpinDynamics/SpinDynamicsUpdated/Version_2/code`
remains the reference implementation, but the major Version 2 workflow port is
now mostly complete; remaining work is mainly validation depth, specialized
variants, packaging, and performance.

## Start Here

- [Installation](installation.md)
- [Concepts and Units](concepts.md)
- [Examples](examples.md)
- [Parameters](parameters.md)
- [Core Numerical Functions](core.md)
- [Workflows](workflows.md)
- [Analysis](analysis.md)
- [Chemical / Site Exchange](exchange.md)
- [Internal / Susceptibility Gradients](internal_gradients.md)
- [J-Coupling Models](j_coupling.md)
- [NQR Models](nqr.md)
- [ESR Models](esr.md)
- [Phase Cycling Findings](phase_cycling.md)
- [API Reference](api_reference.md)
- [Performance](performance.md)
- [Validation](validation.md)
- [Known Gaps](known_gaps.md)

## Current Supported Surface

The validated Python API currently covers:

- ideal CPMG asymptotic magnetization and echo construction;
- public ideal, tuned, untuned, and matched CPMG runners returning a common
  `CPMGResult`;
- public finite ideal CPMG acquisition returning `CPMGTrainResult`;
- finite CPMG train rephasing checks, optional grid refinement, and chunked
  multicore isochromat propagation;
- ideal, tuned, untuned, and matched CPMG inversion-recovery finite trains over
  `tauvect`;
- Python-native finite-train Q/mistuning sweeps for tuned, untuned, and
  matched probes;
- first matched-probe diffusion CPMG workflow and compact diffusion Q sweep;
- rectangular PGSE and PGSE-prepared CPMG workflows with deterministic
  gradient-moment and explicit random-walker backends;
- fixture-validated ideal, tuned, and matched CPMG imaging, k-space
  reconstruction, and arbitrary B0/B1 field-map loading helpers;
- fixture-validated pulse-shape utilities for JMR rectangular pulse responses,
  phase quantization, and untuned segment adjustment;
- tuned and matched CPMG Q/mistuning sweep workflows;
- matched-probe z-magnetization Q sweep workflow;
- ideal, tuned, untuned, and matched time-varying-field CPMG final-echo and
  amplitude-sweep workflows;
- WURST pulse construction, matched-probe WURST transmit response, ideal and
  matched WURST inversion, and matched WURST-CPMG workflows;
- ideal FID acquisition and time-domain trace construction;
- 1D and separable 2D inverse Laplace analysis helpers for T1, T2, T1-T2, and
  D-T2 kernels with manual or SNR-selected Tikhonov regularization;
- opt-in moving-isochromat motion helpers for B0/B1 field-map sampling,
  deterministic advection, seeded diffusion, and receive summation;
- ideal-probe finite acquisition with relaxation through
  `calc_macq_ideal_probe_relax4`;
- SPA refocusing pulse catalog, normalized SNR/FOM metric bookkeeping,
  tuned/untuned/matched fixed-refocusing evaluators, lightweight discrete
  phase-search scaffold, ideal v0crit, excitation-aware v0crit, and
  time-varying-field refocusing evaluation/search, bounded refocusing phase
  optimizers, and tuned excitation-pulse evaluation/phase search for supplied
  refocusing axes, diagnostic tuned inverse-excitation search for target
  spectra, with optional SciPy continuous optimization backend;
- array-returning multi-start optimization driver scaffolds for repeated
  ideal/tuned/untuned/matched refocusing, tuned excitation, and phase-flipped
  tuned inverse-excitation searches;
- plotting-free optimization pipeline handoff from selected refocusing result
  to tuned excitation and inverse-excitation searches;
- MATLAB-style optimization result-cell conversion, `.npz` archive loading and
  export, script-aware `plot_opt_*_results.m` layout analysis, selected
  score/program/metadata inspection, tuned original/inverse result-pair
  comparison, and optional SciPy-backed `.mat` import/export for multi-start
  results;
- plotting examples for CPMG comparisons, finite trains, parameter sweeps,
  diffusion, PGSE D-T2 inverse Laplace, time-varying fields, imaging, motion,
  WURST, inverse Laplace, and compact optimization workflows;
- low-level rotation matrix and effective-axis helpers;
- first pulsed NQR helpers for spin-1 quadrupolar sites, selective
  transition pulses, single-crystal and powder orientations, SLSE echo trains,
  and perturbation-plus-detection population-transfer experiments;
- first ESR helpers for single-electron spin-1/2 systems, scalar/anisotropic
  `g` tensors, single-crystal and powder orientation grids, and fixed-field or
  fixed-frequency spectra, CW derivative/lineshape display, static
  strain/disorder sampling, rectangular-pulse FID and Hahn-echo helpers, and
  Liouville-space pulsed T1/T2 relaxation, plus isotropic electron-nuclear
  hyperfine doublet simulations;
- the current `sim_spin_dynamics_arb10` kernel;
- the legacy-compatible `sim_spin_dynamics_arb7` path needed by ideal FID;
- original/reference tuned, untuned, and matched probe CPMG paths.

Full historical MATLAB `.mat` file parity, strong inverse excitation
cancellation parity, and broad `fmincon` parity validation are still
reference-only beyond the compact optimization fixtures.
