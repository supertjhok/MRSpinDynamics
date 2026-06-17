# Version and Documentation Guide

This repository contains several generations of MATLAB spin-dynamics code. The
current documentation policy is:

1. Treat `SpinDynamicsUpdated/Version_2` as the active MATLAB implementation.
2. Treat `SpinDynamicsUpdated/Version_1` and `SpinDynamics` as legacy/reference
   implementations unless a specific historical comparison is needed.
3. Document the active function families first, then document legacy copies only
   where they explain an important algorithmic change.

This guide is an initial audit. It is meant to help organize future
documentation, Python conversion, compiled numerical kernels, and GUI work.

For a practical first run, see [`QUICK_START.md`](QUICK_START.md). For a map of
active `Version_2` workflows, see
[`VERSION_2_WORKFLOWS.md`](VERSION_2_WORKFLOWS.md). For a first pass at speed
and modernization targets, see [`SPEED_AUDIT.md`](SPEED_AUDIT.md).

## Repository Generations

### `SpinDynamics`

Original/historical code. It contains:

- `code/basic`: early general spin-dynamics routines and plotting scripts.
- `code/matched_probe`: matched-probe simulation routines, examples, and image
  simulation data.
- `code/circsim_new`: tuned-probe circuit simulation and OCT-related work.
- `code/diffusion`: older diffusion/coherence calculations.
- `reports`: DOCX/PDF reports that explain probe and OCT theory.
- `references`: literature PDFs used by the models.

Use this tree mainly as background material.

### `SpinDynamicsUpdated/Version_1`

Intermediate reorganization. Many files from `SpinDynamics` were copied into a
flatter `code` tree and partially grouped into folders such as `calc_macq`,
`calc_masy`, `calc_rot`, `Sim_CPMG`, and `circuit_simulation`.

Use this tree mainly for understanding how the current folder organization
evolved.

### `SpinDynamicsUpdated/Version_2`

Recommended active MATLAB implementation. The root `README.md` already points
users here. This version adds or expands:

- untuned, tuned, and matched probe parameter sets;
- OCT and SPA pulse optimization examples;
- WURST inversion routines;
- diffusion examples and diffusion-aware acquisition calculations;
- time-varying-field simulations;
- MEX build scripts and generated MEX artifacts;
- a more structured folder layout for current code.

## Active `Version_2` Code Map

The most important active folders under `SpinDynamicsUpdated/Version_2/code`
are:

| Folder | Role |
| --- | --- |
| `Params` | Parameter constructors. These create the `sp` and `pp` structures used by examples and simulation routines. |
| `sim_spin_dynamics_arb` | Core arbitrary-pulse spin-dynamics simulators. These are candidates for Python conversion and compilation. |
| `sim_spin_dynamics_asymp` | Asymptotic magnetization and excitation simulation routines. |
| `calc_rot` | Effective rotation-axis and rotation-matrix calculations. |
| `calc_masy` | Asymptotic magnetization calculations for ideal, tuned, matched, untuned, and WURST cases. |
| `calc_macq` | Acquisition magnetization calculations. |
| `calc_echo` | Time-domain echo calculations. |
| `circuit_simulation` | Probe circuit models for matched, tuned, and untuned probes. |
| `Sim_CPMG` | CPMG simulation drivers, including image simulations. |
| `Sim_CPMG_IR` | Inversion-recovery CPMG simulation drivers. |
| `Sim_FID` | FID simulation drivers. |
| `Sim_Diffusion` and `calc_macq_diff` | Diffusion-aware simulations and acquisition calculations. |
| `CPMG_Asymp_Examples` | Simple examples comparing ideal, untuned, tuned, and matched probe effects. |
| `FID_Example` | FID example scripts. |
| `Imaging_demo` | Imaging examples for ideal, tuned, and matched probe cases. |
| `OCT_Pulse_Examples` and `opt_pulse` | OCT/SPA pulse construction, optimization, and plotting scripts. |
| `Wurst_Inversion` | WURST pulse and inversion simulation routines. |
| `mex` | MATLAB Coder/MEX build scripts and generated MEX artifacts. |
| `time_varying_field` | Time-varying-field CPMG simulations and comparisons. |

## Repeated Function Names

Many function names appear in multiple locations. In the initial audit, repeated
names were found across the historical trees, and raw file hashes show that
same-named copies are often not byte-identical.

This means documentation should avoid saying "these are duplicates" unless the
files have been compared. The safer terms are:

- `active`: the `Version_2` implementation to document and use;
- `legacy`: older copy retained for reference;
- `variant`: same conceptual function, but with a different model, parameter
  convention, or performance tradeoff;
- `generated`: MEX/codegen output or build artifact.

High-frequency repeated names include:

| Function | Copies Found | Initial interpretation |
| --- | ---: | --- |
| `calc_time_domain_echo.m` | 6 | Echo calculation migrated across historical trees. Active copy is in `Version_2/code/calc_echo`. |
| `sim_spin_dynamics_asymp_mag3.m` | 6 | Core asymptotic simulator used across older and current examples. Active copy is in `Version_2/code/sim_spin_dynamics_asymp`. |
| `calc_rot_axis_arba3.m` | 6 | Rotation-axis helper used broadly by CPMG/OCT workflows. Active copy is in `Version_2/code/calc_rot`. |
| `sim_matched_probe_coil_Q.m` | 6 | Q-comparison/mistuning study variants. Active usage is split between `CompareQ` and `CompareMistuned`. |
| `sim_spin_dynamics_arb6.m` | 5 | Earlier arbitrary-pulse simulator. Newer active numerical kernel is likely `sim_spin_dynamics_arb10.m`. |
| `set_params_matched.m` | 4 | Matched-probe parameter setup evolved across versions. Active copy is in `Version_2/code/Params`. |

## Core Numerical Routine Lineage

The `sim_spin_dynamics_arb*` family is the most important starting point for
future compiled Python kernels.

Observed progression from comments in the code:

- `sim_spin_dynamics_arb6.m`: arbitrary pulse amplitudes, normalized `w1`,
  vectorized magnetization propagation, direct pulse phase/amplitude inputs.
- `sim_spin_dynamics_arb9.m`: adds precomputed pulse rotation matrices, gradient
  offsets through `del_wg`, relaxation maps, and removes acquisition-window
  convolution for speed.
- `sim_spin_dynamics_arb10.m`: keeps the `arb9` model but rewrites internals to
  eliminate nested structures so MATLAB Coder/MEX compilation can work.

Recommended documentation stance:

- document `sim_spin_dynamics_arb10.m` as the current core arbitrary-pulse
  numerical kernel;
- document `sim_spin_dynamics_arb9.m` as the immediate predecessor;
- document `sim_spin_dynamics_arb6.m` only when explaining lineage or validating
  older results.

## Parameter Set Naming

`Version_2/code/Params` contains the active parameter constructors:

- `set_params_ideal.m`
- `set_params_ideal_FID.m`
- `set_params_ideal_tv.m`
- `set_params_ideal_tv_exc.m`
- `set_params_untuned_Orig.m`
- `set_params_untuned_JMR.m`
- `set_params_untuned_OCT.m`
- `set_params_untuned_SPA.m`
- `set_params_tuned_Orig.m`
- `set_params_tuned_JMR.m`
- `set_params_tuned_OCT.m`
- `set_params_tuned_SPA.m`
- `set_params_matched.m`
- `set_params_matched_Orig.m`
- `set_params_matched_JMR.m`
- `set_params_matched_OCT.m`
- `set_params_matched_SPA.m`

Suggested documentation convention:

- `ideal`: no probe dynamics;
- `untuned`, `tuned`, `matched`: probe model family;
- `Orig`: original/reference model for that family;
- `JMR`: settings used for JMR paper workflows;
- `OCT`: settings for OCT pulse optimization workflows;
- `SPA`: settings for SPA pulse workflows;
- `FID`, `tv`, `tv_exc`: specialized workflows.

These meanings should be confirmed against function bodies before being treated
as final API documentation.

## Documentation Plan

Recommended documentation order:

1. Expand the root `README.md` with a quick-start path that adds the active
   `Version_2/code` folders to MATLAB's path and runs one minimal example.
2. Add a `docs/version-2-map.md` page describing the active folder structure and
   the main workflows.
3. Add one page per workflow:
   - CPMG asymptotic examples;
   - FID examples;
   - imaging examples;
   - diffusion examples;
   - OCT/SPA pulse optimization;
   - WURST inversion.
4. Add function-family pages for:
   - core numerical kernels;
   - parameter constructors;
   - probe circuit models;
   - echo/acquisition calculations.
5. Add a legacy index that points to older folders but does not attempt to fully
   document every historical copy.

Use the script/function help format in
[`HELP_COMMENT_STANDARD.md`](HELP_COMMENT_STANDARD.md) when adding MATLAB help
blocks.

## Open Questions

- Which `Version_2` examples should be considered canonical smoke tests?
- Should generated files such as `.mexw64`, `.mldatx`, `.asv`, `~`, `.fig`, and
  large `.mat` result artifacts remain in the repository?
- Should legacy functions be moved under an explicit `legacy/` tree in a future
  cleanup, or only documented as legacy for now?
- Which routines must preserve MATLAB output exactly before Python conversion?
