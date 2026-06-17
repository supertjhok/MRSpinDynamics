# Script Help Audit

This audit tracks standardized MATLAB help comments for true scripts in the
active implementation tree, `SpinDynamicsUpdated/Version_2/code`.

The standard format is defined in
[`HELP_COMMENT_STANDARD.md`](HELP_COMMENT_STANDARD.md).

## Current Coverage

As of this audit:

- Active `Version_2` scripts found: 36
- Scripts with standardized help blocks: 18
- Scripts still needing standardized help blocks: 18

## Completed

| Script | Notes |
| --- | --- |
| `SpinDynamicsUpdated/Version_2/code/CPMG_Asymp_Examples/MatchedProbeEffects_CPMG_Asymp.m` | Matched-probe CPMG asymptotic example. |
| `SpinDynamicsUpdated/Version_2/code/CPMG_Asymp_Examples/noProbeEffects_CPMG_Asymp.m` | Ideal-probe CPMG asymptotic example. |
| `SpinDynamicsUpdated/Version_2/code/CPMG_Asymp_Examples/TunedProbeEffects_CPMG_Asymp.m` | Tuned-probe CPMG asymptotic example. |
| `SpinDynamicsUpdated/Version_2/code/CPMG_Asymp_Examples/UntunedProbeEffects_CPMG_Asymp.m` | Untuned-probe CPMG asymptotic example. |
| `SpinDynamicsUpdated/Version_2/code/DIffusion_Example/Diff_Echo_Q.m` | Diffusion/Q sweep example. |
| `SpinDynamicsUpdated/Version_2/code/FID_Example/noProbeEffects_FID.m` | Ideal-probe FID example. |
| `SpinDynamicsUpdated/Version_2/code/Imaging_demo/imaging_example_ideal.m` | Ideal-probe imaging example. |
| `SpinDynamicsUpdated/Version_2/code/Imaging_demo/imaging_example_matched.m` | Matched-probe imaging example. |
| `SpinDynamicsUpdated/Version_2/code/Imaging_demo/Imaging_example_tuned.m` | Tuned-probe imaging example. |
| `SpinDynamicsUpdated/Version_2/code/OCT_Pulse_Examples/TunedProbe_OCT.m` | Tuned-probe OCT workflow. |
| `SpinDynamicsUpdated/Version_2/code/time_varying_field/cpmg_ideal_tv_example.m` | Time-varying-field CPMG example. |
| `SpinDynamicsUpdated/Version_2/code/Wurst_Inversion/MatchedWurstInversion.m` | Matched-probe WURST inversion example. |
| `SpinDynamicsUpdated/Version_2/code/CompareMistuned/matched_probe/sim_matched_probe_mistuned.m` | Matched-probe mistuning comparison workflow. |
| `SpinDynamicsUpdated/Version_2/code/CompareMistuned/tuned_probe/sim_tuned_probe_mistuned.m` | Tuned-probe mistuning comparison workflow. |
| `SpinDynamicsUpdated/Version_2/code/CompareQ/matchedCompareQ.m` | Parallel matched-probe Q comparison workflow. |
| `SpinDynamicsUpdated/Version_2/code/CompareQ/sim_matched_probe_coil_Q.m` | Serial matched-probe Q sweep/export workflow. |
| `SpinDynamicsUpdated/Version_2/code/CompareQ/sim_tuned_probe_coil_Q.m` | Serial tuned-probe Q sweep/export workflow. |
| `SpinDynamicsUpdated/Version_2/code/CompareQ/tunedCompareQ.m` | Parallel tuned-probe Q comparison workflow. |

## Remaining

| Script | Suggested documentation category |
| --- | --- |
| `SpinDynamicsUpdated/Version_2/code/chirp.m` | Utility/demo script; confirm role before documenting. |
| `SpinDynamicsUpdated/Version_2/code/CompareMistuned/matched_probe/sim_matched_probe_coil_Q.m` | Probe comparison/mistuning workflow. |
| `SpinDynamicsUpdated/Version_2/code/JMR Paper/FIDMatched.m` | JMR-paper workflow; confirm canonical status. |
| `SpinDynamicsUpdated/Version_2/code/JMR Paper/simFID.m` | JMR-paper workflow; confirm canonical status. |
| `SpinDynamicsUpdated/Version_2/code/JMR Paper/txPulse.m` | JMR-paper workflow; confirm canonical status. |
| `SpinDynamicsUpdated/Version_2/code/mex/build_lib_fixed.m` | MEX/library build script. |
| `SpinDynamicsUpdated/Version_2/code/mex/build_lib_variable.m` | MEX/library build script. |
| `SpinDynamicsUpdated/Version_2/code/mex/build_mex_fixed.m` | MEX build script. |
| `SpinDynamicsUpdated/Version_2/code/mex/build_mex_variable_arb10.m` | MEX build script. |
| `SpinDynamicsUpdated/Version_2/code/mex/build_mex_variable.m` | MEX build script. |
| `SpinDynamicsUpdated/Version_2/code/mex/test_2d.m` | MEX test script. |
| `SpinDynamicsUpdated/Version_2/code/mex/test.m` | MEX test script. |
| `SpinDynamicsUpdated/Version_2/code/opt_pulse/plot_oct_summary.m` | OCT result plotting workflow. |
| `SpinDynamicsUpdated/Version_2/code/opt_pulse/sensitivityParameter.m` | Utility/demo script; confirm role before documenting. |
| `SpinDynamicsUpdated/Version_2/code/Sim_CPMG_IR/sim_cpmg_ir_matched_probe_compare.m` | CPMG inversion-recovery comparison script. |
| `SpinDynamicsUpdated/Version_2/code/Sim_CPMG/sim_cpmg_phase_gradxy.m` | CPMG phase/gradient script. |
| `SpinDynamicsUpdated/Version_2/code/z_mag/z_Mag_Q.m` | Z-magnetization/Q workflow. |
| `SpinDynamicsUpdated/Version_2/code/Z_Magnetization_Single_Sided/calc_z_freq.m` | Single-sided Z-magnetization frequency script. |

## Notes

- The audit only covers true scripts. Function files should use compatible help
  section names, but their help placement should be checked against MATLAB's
  function-help behavior.
- Some remaining scripts appear exploratory or build-related. Their help blocks
  should avoid presenting them as canonical examples until their role is
  confirmed.
