# Python API Documentation

This directory documents the Python port as it exists today. The MATLAB
implementation under `../SpinDynamicsUpdated/Version_2/code` remains the
reference implementation during migration.

## Start Here

- [Installation](installation.md)
- [Concepts and Units](concepts.md)
- [Examples](examples.md)
- [Parameters](parameters.md)
- [Core Numerical Functions](core.md)
- [Workflows](workflows.md)
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
- matched-probe CPMG inversion-recovery finite trains over `tauvect`;
- Python-native finite-train Q/mistuning sweeps for tuned, untuned, and
  matched probes;
- first matched-probe diffusion CPMG workflow and compact diffusion Q sweep;
- fixture-validated ideal, tuned, and matched CPMG imaging, k-space
  reconstruction, and arbitrary B0/B1 field-map loading helpers;
- fixture-validated pulse-shape utilities for JMR rectangular pulse responses,
  phase quantization, and untuned segment adjustment;
- tuned and matched CPMG Q/mistuning sweep workflows;
- matched-probe z-magnetization Q sweep workflow;
- ideal time-varying-field CPMG final-echo and amplitude-sweep workflows;
- ideal FID acquisition and time-domain trace construction;
- ideal-probe finite acquisition with relaxation through
  `calc_macq_ideal_probe_relax4`;
- SPA refocusing pulse catalog, normalized SNR/FOM metric bookkeeping,
  tuned/untuned/matched fixed-refocusing evaluators, lightweight discrete
  phase-search scaffold, bounded refocusing phase optimizers, and tuned
  excitation-pulse evaluation/phase search for supplied refocusing axes,
  diagnostic tuned inverse-excitation search for target spectra, with optional
  SciPy continuous optimization backend;
- array-returning multi-start optimization driver scaffolds for repeated
  refocusing, tuned excitation, and phase-flipped tuned inverse-excitation
  searches;
- plotting examples for CPMG comparisons, finite trains, parameter sweeps,
  diffusion, time-varying fields, imaging, and compact optimization workflows;
- low-level rotation matrix and effective-axis helpers;
- the current `sim_spin_dynamics_arb10` kernel;
- the legacy-compatible `sim_spin_dynamics_arb7` path needed by ideal FID;
- original/reference tuned, untuned, and matched probe CPMG paths.

MATLAB `.mat` result-file compatibility, strong inverse excitation
cancellation parity, and broad `fmincon` parity validation are still
reference-only beyond the compact optimization fixtures.
