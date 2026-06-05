# Version 2 Workflow Map

This page maps the active MATLAB implementation under
`SpinDynamicsUpdated/Version_2/code`. It is intended as the first place to look
after the quick start.

## Setup

From the repository root, add the active code tree to the MATLAB path:

```matlab
addpath(genpath(fullfile(pwd,'SpinDynamicsUpdated','Version_2','code')));
```

## Workflow Summary

| Workflow | Start Here | Main Parameter Constructors | Main Simulation Functions | Notes |
| --- | --- | --- | --- | --- |
| Ideal CPMG asymptotic echo | `CPMG_Asymp_Examples/noProbeEffects_CPMG_Asymp.m` | `set_params_ideal` | `calc_masy_ideal`, `sim_spin_dynamics_asymp_mag3`, `calc_time_domain_echo` | Best first smoke test. |
| Untuned-probe CPMG | `CPMG_Asymp_Examples/UntunedProbeEffects_CPMG_Asymp.m` | `set_params_untuned_Orig` | `calc_masy_untuned_probe_lp`, `untuned_probe_lp`, `untuned_probe_rx` | Includes untuned transmitter/receiver probe effects. |
| Tuned-probe CPMG | `CPMG_Asymp_Examples/TunedProbeEffects_CPMG_Asymp.m` | `set_params_tuned_Orig` | `calc_masy_tuned_probe_lp_Orig`, `tuned_probe_lp`, `tuned_probe_rx` | Includes tuned transmitter/receiver probe effects. |
| Matched-probe CPMG | `CPMG_Asymp_Examples/MatchedProbeEffects_CPMG_Asymp.m` | `set_params_matched_Orig` | `calc_masy_matched_probe_Orig`, `find_coil_current`, `matched_probe_rx` | Includes matching-network design and receiver filtering. |
| FID | `FID_Example/noProbeEffects_FID.m` | `set_params_ideal_FID` | `simFID_ideal` | Minimal ideal FID workflow. |
| Imaging | `Imaging_demo/imaging_example_ideal.m` | Script-local `params` | `sim_cpmg_ideal_probe_img` | Uses `Images/flower.png`; tuned and matched variants are also available. |
| Diffusion | `DIffusion_Example/Diff_Echo_Q.m` | Script-local parameters | `sim_dif_matched_CPMG_noRx` | Compact Q sweep for diffusion-aware CPMG. |
| Q comparison | `CompareQ/matchedCompareQ.m`, `CompareQ/tunedCompareQ.m` | `set_params_matched_Orig`, `set_params_tuned_Orig` | `calc_masy_*`, `calc_time_domain_echo` | Uses `parfor`; serial/export variants also exist. |
| Mistuning comparison | `CompareMistuned/.../sim_*_mistuned.m` | `set_params_matched_Orig`, `set_params_tuned_Orig` | `calc_masy_*`, `calc_time_domain_echo` | Sweeps probe frequency error in units of `fin/Q`. |
| OCT pulse workflows | `OCT_Pulse_Examples/TunedProbe_OCT.m` | Tuned/OCT parameter sets | `opt_ref_pulse_*`, `plot_opt_ref_results_*`, `opt_exc_pulse_*` | Can be slow and may overwrite result files if names are not changed. |
| SPA pulse workflows | `OCT_Pulse_Examples/SPA_optimization_*.m`, `SPA_pulse_list.m` | `set_params_*_SPA` | SPA optimization and plotting helpers | Intended for SPA pulse searches and summary plots. |
| WURST inversion | `Wurst_Inversion/MatchedWurstInversion.m` | `set_params_matched` | `sim_inv_matched_probe_WURST` | Exploratory; review placeholder `params` before treating as canonical. |
| Time-varying fields | `time_varying_field/cpmg_ideal_tv_example.m` | `set_params_ideal_tv` | `sim_cpmg_ideal_tv`, `sim_cpmg_ideal_tv_final` | Demonstrates CPMG under a user-edited B0 waveform. |
| MEX/codegen | `mex/build_mex_*.m` | Script-local `params` examples | `sim_spin_dynamics_arb10` | Requires MATLAB Coder. |

## Function Families

### Parameter Constructors

Folder: `Params`

These functions construct the `sp`, `pp`, and sometimes `params` structures used
throughout the examples.

- `sp`: system, sample, circuit, simulation-grid, noise, and plotting
  parameters.
- `pp`: pulse-sequence parameters such as pulse lengths, phases, amplitudes,
  echo spacing, acquisition timing, and numerical thresholds.
- `params`: compact pulse/circuit structure used by some tuned and untuned
  probe helper routines.

Common naming:

- `ideal`: no probe dynamics.
- `untuned`, `tuned`, `matched`: probe model family.
- `Orig`: original/reference workflow settings.
- `JMR`: settings used for JMR-paper workflows.
- `OCT`: settings for OCT pulse workflows.
- `SPA`: settings for SPA pulse workflows.
- `FID`, `tv`, `tv_exc`: specialized FID or time-varying-field workflows.

### Core Numerical Kernels

Folder: `sim_spin_dynamics_arb`

- `sim_spin_dynamics_arb10`: current arbitrary-pulse numerical kernel and main
  candidate for compiled/Python translation.
- `sim_spin_dynamics_arb9`: immediate predecessor to `arb10`.
- `sim_spin_dynamics_arb6`: older arbitrary-pulse simulator retained for
  lineage and validation.

Folder: `sim_spin_dynamics_asymp`

- `sim_spin_dynamics_asymp_mag3`: core asymptotic magnetization simulator used
  by many CPMG workflows.

### Rotation and Echo Helpers

Folder: `calc_rot`

- `calc_rot_axis_arba3`: effective rotation axis for arbitrary-amplitude cycles.
- `calc_rot_axis_arba4`: effective rotation axis plus net rotation angle.
- Probe-specific helpers calculate rotation axes with circuit effects.

Folder: `calc_echo`

- `calc_time_domain_echo`: converts offset-domain spectra into time-domain
  echoes using zero-filled inverse FFT.

### Asymptotic and Acquisition Helpers

Folder: `calc_masy`

These functions compute asymptotic magnetization and often apply excitation
pulse and probe models before receiver filtering.

Folder: `calc_macq`

These functions compute acquired magnetization for pulse-sequence simulations.
Some variants include relaxation, gradients, or probe-specific behavior.

Folder: `calc_macq_diff`

Diffusion-aware acquisition calculations.

### Probe Circuit Models

Folder: `circuit_simulation`

- `matched_probe`: matching-network design, coil-current calculations, and
  matched-probe receiver filtering.
- `tuned_probe`: tuned-probe transmit and receive models.
- `untuned_probe`: untuned-probe transmit and receive models.

## Suggested Reading Order

1. `docs/QUICK_START.md`
2. `docs/VERSION_GUIDE.md`
3. This workflow map
4. Script-level help comments for the workflow you want to run
5. Function-level help comments for the helpers called by that workflow

## Current Documentation Caveats

- Many same-named functions exist in older folders. Prefer the active
  `Version_2` copy unless a legacy comparison is explicitly needed.
- Some scripts are exploratory and still contain hard-coded export paths.
- MEX build scripts and generated artifacts need a separate cleanup/encoding
  pass before they can be documented consistently.
