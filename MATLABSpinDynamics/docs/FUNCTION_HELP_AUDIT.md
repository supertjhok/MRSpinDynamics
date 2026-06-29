# Function Help Audit

This audit tracks standardized MATLAB help comments for key function files in
the active implementation tree, `Version_3/code`.

The standard format is defined in
[`HELP_COMMENT_STANDARD.md`](HELP_COMMENT_STANDARD.md).

## Initial Key Function Coverage

The first two passes plus the imaging dependency update document 26 key function
files:

| Function | File | Role |
| --- | --- | --- |
| `sim_spin_dynamics_arb10` | `Version_3/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m` | Current core arbitrary-pulse numerical kernel; MEX/codegen candidate. |
| `sim_spin_dynamics_arb9` | `Version_3/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb9.m` | Immediate predecessor to `arb10`. |
| `sim_spin_dynamics_asymp_mag3` | `Version_3/code/sim_spin_dynamics_asymp/sim_spin_dynamics_asymp_mag3.m` | Core asymptotic magnetization simulator. |
| `calc_time_domain_echo` | `Version_3/code/calc_echo/calc_time_domain_echo.m` | Offset-spectrum to time-domain echo conversion. |
| `calc_rot_axis_arba3` | `Version_3/code/calc_rot/calc_rot_axis_arba3.m` | Effective rotation axis for arbitrary-amplitude cycles. |
| `calc_rot_axis_arba4` | `Version_3/code/calc_rot/calc_rot_axis_arba4.m` | Effective rotation axis plus net rotation angle. |
| `calc_masy_ideal` | `Version_3/code/calc_masy/calc_masy_ideal.m` | Ideal-probe CPMG asymptotic magnetization. |
| `calc_masy_matched_probe` | `Version_3/code/calc_masy/calc_masy_matched_probe.m` | Matched-probe CPMG asymptotic/received magnetization. |
| `calc_masy_tuned_probe_lp` | `Version_3/code/calc_masy/calc_masy_tuned_probe_lp.m` | Tuned-probe CPMG asymptotic/received magnetization. |
| `calc_masy_untuned_probe_lp` | `Version_3/code/calc_masy/calc_masy_untuned_probe_lp.m` | Untuned-probe CPMG asymptotic/received magnetization. |
| `set_params_ideal` | `Version_3/code/Params/set_params_ideal.m` | Ideal-probe parameter constructor. |
| `set_params_matched` | `Version_3/code/Params/set_params_matched.m` | Matched-probe parameter constructor. |
| `set_params_ideal_FID` | `Version_3/code/Params/set_params_ideal_FID.m` | Ideal-probe FID parameter constructor. |
| `set_params_ideal_tv` | `Version_3/code/Params/set_params_ideal_tv.m` | Ideal-probe time-varying-field parameter constructor. |
| `set_params_ideal_tv_exc` | `Version_3/code/Params/set_params_ideal_tv_exc.m` | Ideal-probe time-varying-field excitation parameter constructor. |
| `set_params_matched_Orig` | `Version_3/code/Params/set_params_matched_Orig.m` | Matched-probe original/reference parameter constructor. |
| `set_params_matched_JMR` | `Version_3/code/Params/set_params_matched_JMR.m` | Matched-probe JMR parameter constructor. |
| `set_params_matched_OCT` | `Version_3/code/Params/set_params_matched_OCT.m` | Matched-probe OCT parameter constructor. |
| `set_params_matched_SPA` | `Version_3/code/Params/set_params_matched_SPA.m` | Matched-probe SPA parameter constructor. |
| `set_params_tuned_JMR` | `Version_3/code/Params/set_params_tuned_JMR.m` | Tuned-probe JMR parameter constructor. |
| `set_params_tuned_Orig` | `Version_3/code/Params/set_params_tuned_Orig.m` | Tuned-probe original/reference parameter constructor. |
| `set_params_untuned_Orig` | `Version_3/code/Params/set_params_untuned_Orig.m` | Untuned-probe original/reference parameter constructor. |
| `matched_probe_rx` | `Version_3/code/circuit_simulation/matched_probe/matched_probe_rx.m` | Matched-probe receiver filtering and SNR estimate. |
| `sim_cpmg_ideal_probe_img` | `Version_3/code/Sim_CPMG/sim_cpmg_ideal_probe_img.m` | Ideal-probe CPMG image simulator; uses `parfor`. |
| `sim_cpmg_matched_probe_img` | `Version_3/code/Sim_CPMG/sim_cpmg_matched_probe_img.m` | Matched-probe CPMG image simulator; uses `parfor`. |
| `sim_cpmg_tuned_probe_img` | `Version_3/code/Sim_CPMG/sim_cpmg_tuned_probe_img.m` | Tuned-probe CPMG image simulator; uses `parfor`. |

## Recommended Next Batches

1. Remaining tuned/untuned parameter constructors in `Params`, especially
   `set_params_tuned_OCT`, `set_params_tuned_SPA`,
   `set_params_untuned_JMR`, `set_params_untuned_OCT`, and
   `set_params_untuned_SPA`.
2. Probe-circuit functions under `circuit_simulation/tuned_probe` and
   `circuit_simulation/untuned_probe`.
3. Remaining `calc_rot`, `calc_masy`, `calc_macq`, and `calc_echo` function
   families.
4. Simulation drivers in `Sim_CPMG`, `Sim_CPMG_IR`, `Sim_Diffusion`, `Sim_FID`,
   and `Wurst_Inversion`.

## Notes

- Structure-valued inputs are documented by expected fields when those fields
  are needed to call the function safely.
- The first pass intentionally documents active/current functions before legacy
  or exploratory variants.
