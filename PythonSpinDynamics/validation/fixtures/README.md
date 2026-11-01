# Validation fixtures

Compact reference arrays consumed by `tests/octave_fixtures/`. Most files are
written by the Octave/MATLAB generators in `validation/octave/` (CSV written
with `dlmwrite(..., 'precision', '%.17g')`).

## Tuned-probe fixtures regenerated after the Version 3 transient correction

`MATLABSpinDynamics/Version_3/code/circuit_simulation/tuned_probe/tuned_probe_lp.m`
and `tuned_probe_lp_Orig.m` carried an extra `1/(2*alpha)` on the first
homogeneous term of the driven-response transient (`Icd1`, `Icd2`, `dIcd1`,
`dIcd2`). Versions 1 and 2 had the correct zero-initial-condition
coefficients. The extra factor made the coil current non-zero at every pulse
start and propagated wrong derivative initial conditions into later segments
(about 50 % transient error against a direct ODE solve; see
`tests/test_tuned_probe_transient.py`). Both the MATLAB Version 3 files and
`spin_dynamics.probes.tuned` were corrected.

Octave was not available when the correction was made, so the affected arrays
below were regenerated from the corrected Python implementation, keeping each
file's existing layout and inputs:

- `tuned_probe_lp_orig.csv`, `calc_masy_tuned_probe_lp_orig.csv`
  (`generate_basic_fixtures.m`)
- `pulse_tuned_rectangular.csv` (`generate_pulse_fixtures.m`)
- `run_tuned_cpmg_train_mrx.csv`, `run_tuned_cpmg_train_echo.csv`,
  `run_tuned_cpmg_train_integrals.csv`, `run_tuned_cpmg_imaging_kspace.csv`
- `optimization_tuned_excitation_phase_shift.csv`
  (`generate_optimization_fixtures.m`)
- `optimization_tuned_excitation_result.csv`,
  `optimization_tuned_inverse_result.csv`,
  `optimization_tuned_inverse_spectra.csv`: the pulse phases are the
  committed MATLAB optimizer outputs and were kept unchanged; only the
  objective values (SNR, mismatch, residual ratio, inverse SNR) and spectra
  were re-evaluated, mirroring the generator's no-`fmincon` fallback path.
- `optimization_tuned_excitation_results.mat`,
  `optimization_tuned_excitation_results_inv.mat`,
  `optimization_tuned_refocusing_results.mat`: only the stored score element
  of each result cell was replaced with the re-evaluated value; the phases and
  parameter structs are unchanged.

Rerunning the Octave generators against the corrected Version 3 MATLAB code
should reproduce these values to the usual fixture tolerances.
