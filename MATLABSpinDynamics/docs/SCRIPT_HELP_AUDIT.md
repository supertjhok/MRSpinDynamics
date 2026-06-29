# Script Help Audit

This audit tracks standardized MATLAB help comments for true scripts in the
active implementation tree, `Version_3/code`.

The standard format is defined in
[`HELP_COMMENT_STANDARD.md`](HELP_COMMENT_STANDARD.md).

## Current Coverage

As of this release cleanup:

- Active `Version_3` scripts found: 36
- Scripts with standardized help blocks: 36
- Scripts still needing standardized help blocks: 0

## Completed

| Script | Notes |
| --- | --- |
| `Version_3/code/CPMG_Asymp_Examples/MatchedProbeEffects_CPMG_Asymp.m` | Matched-probe CPMG asymptotic example. |
| `Version_3/code/CPMG_Asymp_Examples/noProbeEffects_CPMG_Asymp.m` | Ideal-probe CPMG asymptotic example. |
| `Version_3/code/CPMG_Asymp_Examples/TunedProbeEffects_CPMG_Asymp.m` | Tuned-probe CPMG asymptotic example. |
| `Version_3/code/CPMG_Asymp_Examples/UntunedProbeEffects_CPMG_Asymp.m` | Untuned-probe CPMG asymptotic example. |
| `Version_3/code/DIffusion_Example/Diff_Echo_Q.m` | Diffusion/Q sweep example. |
| `Version_3/code/FID_Example/noProbeEffects_FID.m` | Ideal-probe FID example. |
| `Version_3/code/Imaging_demo/imaging_example_ideal.m` | Ideal-probe imaging example. |
| `Version_3/code/Imaging_demo/imaging_example_matched.m` | Matched-probe imaging example. |
| `Version_3/code/Imaging_demo/Imaging_example_tuned.m` | Tuned-probe imaging example. |
| `Version_3/code/OCT_Pulse_Examples/TunedProbe_OCT.m` | Tuned-probe OCT workflow. |
| `Version_3/code/time_varying_field/cpmg_ideal_tv_example.m` | Time-varying-field CPMG example. |
| `Version_3/code/Wurst_Inversion/MatchedWurstInversion.m` | Matched-probe WURST inversion example. |
| `Version_3/code/CompareMistuned/matched_probe/sim_matched_probe_mistuned.m` | Matched-probe mistuning comparison workflow. |
| `Version_3/code/CompareMistuned/tuned_probe/sim_tuned_probe_mistuned.m` | Tuned-probe mistuning comparison workflow. |
| `Version_3/code/CompareQ/matchedCompareQ.m` | Parallel matched-probe Q comparison workflow. |
| `Version_3/code/CompareQ/sim_matched_probe_coil_Q.m` | Serial matched-probe Q sweep/export workflow. |
| `Version_3/code/CompareQ/sim_tuned_probe_coil_Q.m` | Serial tuned-probe Q sweep/export workflow. |
| `Version_3/code/CompareQ/tunedCompareQ.m` | Parallel tuned-probe Q comparison workflow. |

## Remaining

None for the active `Version_3` script set.

## Notes

- The audit only covers true scripts. Function files should use compatible help
  section names, but their help placement should be checked against MATLAB's
  function-help behavior.
- Some remaining scripts appear exploratory or build-related. Their help blocks
  should avoid presenting them as canonical examples until their role is
  confirmed.
