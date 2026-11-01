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

## Imaging and tuned-train fixtures regenerated for review items M-12 and M-44

Two workflow corrections changed arrays that these fixtures had locked in.
Octave was again unavailable, so the files below were regenerated from the
corrected Python with each file's existing layout and inputs (the inputs used by
`tests/octave_fixtures/support.py`):

- `run_ideal_cpmg_imaging_kspace.csv`, `run_tuned_cpmg_imaging_kspace.csv`,
  `run_matched_cpmg_imaging_kspace.csv` (M-12). The phase-encoded imagers
  used `linspace(-1, 1, n)` for both the phase-encode gradients and the default
  `del_wx`/`del_wz` position maps. For even `n` (these fixtures are 2x2) that
  grid has no k=0 sample and a half-integer centre, while the reconstruction
  (`fftshift(ifft2(ifftshift(.)))`) and the SENSE folding assume index `n // 2`
  is the origin, so a point voxel was reconstructed half a pixel off and spread
  over a 2x2 block. Both grids are now `(i - n // 2) / ((n - 1) / 2)`, which is
  identical to `linspace(-1, 1, n)` for odd `n` and keeps the same step for even
  `n`. The MATLAB side now uses the same grid (L-54):
  `MATLABSpinDynamics/Version_3/code/Sim_CPMG/sim_cpmg_{ideal,tuned,matched}_probe_img.m`
  build `gradx`/`gradz` as `((0:n-1) - floor(n/2)) / max((n-1)/2, 1)`,
  `create_fields/create_fields_single_sided.m` builds `del_wgx`/`del_wgz` the
  same way, and so does the matched-probe helper inside
  `validation/octave/generate_imaging_fixtures.m`. Rerunning that generator
  should therefore reproduce these regenerated arrays (not yet run: Octave
  was unavailable).
- `run_tuned_cpmg_train_mrx.csv`, `run_tuned_cpmg_train_echo.csv`,
  `run_tuned_cpmg_train_integrals.csv` (M-44). Tuned-probe pulse shapes were
  min-max (peak) normalized. With the corrected gate-on transient (H-01) the
  peak overshoot depends on the absolute RF phase (up to about 1.7x the steady
  current), so the nominal pulse amplitude moved with phase. Shapes are now
  normalized to the analytic steady-state coil current of the drive, so a long
  pulse plateaus at amplitude 1 for any gate-on phase; the echo integrals grow
  by about 22 % (first echo 411.3 to 501.8). MATLAB's `calc_masy_tuned_probe_lp.m`
  also normalizes to a mid-pulse plateau, while `sim_cpmg_tuned_probe_img.m`
  uses the peak; the tuned imaging path keeps its MATLAB-as-executed shape.
