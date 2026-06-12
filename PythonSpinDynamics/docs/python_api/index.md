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
- fixture-validated ideal, tuned, and matched CPMG imaging and k-space
  reconstruction;
- fixture-validated pulse-shape utilities for JMR rectangular pulse responses,
  phase quantization, and untuned segment adjustment;
- tuned and matched CPMG Q/mistuning sweep workflows;
- matched-probe z-magnetization Q sweep workflow;
- ideal time-varying-field CPMG final-echo and amplitude-sweep workflows;
- ideal FID acquisition and time-domain trace construction;
- ideal-probe finite acquisition with relaxation through
  `calc_macq_ideal_probe_relax4`;
- low-level rotation matrix and effective-axis helpers;
- the current `sim_spin_dynamics_arb10` kernel;
- the legacy-compatible `sim_spin_dynamics_arb7` path needed by ideal FID;
- original/reference tuned, untuned, and matched probe CPMG paths.

OCT/SPA optimization workflows are still MATLAB reference-only.
