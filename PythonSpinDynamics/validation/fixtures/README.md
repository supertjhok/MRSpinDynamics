# Validation fixtures

Compact reference arrays consumed by `tests/octave_fixtures/`. Most files are
written by the Octave/MATLAB generators in `validation/octave/` (CSV written
with `dlmwrite(..., 'precision', '%.17g')`).

## Current provenance: MATLAB R2025a, 2026-11-02

All five generators were run in MATLAB R2025a (with the Optimization, Control
System and Statistics toolboxes), so every array here, including the
matched-probe and optimizer-result fixtures, again comes from the MATLAB
Version 3 code rather than from the Python regenerations described in the
sections below. `generate_optimization_result_fixtures_main` re-ran the
`fmincon` optimizations, so the stored optimizer phases are new. All 158
fixture tests pass against these files. The Octave parity CI job regenerates
the Octave-compatible subset on every relevant change.

## Regenerating with Octave

The generators are function files, so call them by name from
`PythonSpinDynamics/`:

```bash
octave --no-gui --no-window-system --eval "addpath('validation/octave'); generate_basic_fixtures();"
```

and likewise `generate_pulse_fixtures`, `generate_imaging_fixtures` and
`generate_optimization_fixtures`. Plain Octave lacks `fmincon` and `mvnpdf`;
the pulse and imaging generators then add the bound-only and exact fallbacks
in `validation/octave/compat/`. The matched-probe arrays
(`pulse_matched_rectangular.csv`, `run_matched_cpmg_imaging_kspace.csv`) need
`tf` from the Control System Toolbox and are skipped without it, and
`generate_optimization_result_fixtures` needs the Optimization Toolbox, so
those files keep their committed MATLAB (or documented Python) values. The
`Octave parity` CI workflow runs the four generators and then
`tests/test_basic_octave_fixtures.py` against their output.

On 2026-11-02 the four generators were run with Octave 10.3, which reproduced
the July MATLAB imaging fixture to about 1e-15 relative error when given the
July MATLAB code. Their 28 output files replaced the Python-regenerated values
described below; all 158 fixture tests pass against them. Getting there fixed
three problems the earlier Python-only regeneration could not see:

- The generators were scripts with local functions after the main body,
  which MATLAB accepts and Octave does not; they are now function files.
- The tuned-train section of `generate_basic_fixtures.m` still peak-normalized
  the pulse shapes (M-44); it now normalizes to the steady-state coil current
  like `spin_dynamics.workflows.cpmg`.
- The centred-grid edits for L-54 in `create_fields_single_sided.m`, the three
  `sim_cpmg_*_probe_img.m` scripts and the imaging generator divided by
  `max((n-1)/2, 1)`, which is `1` instead of `0.5` for `n = 2`, so the 2x2
  imaging fixtures disagreed with Python. The guard is now
  `max((n-1)/2, 0.5)`, which only matters for `n = 1`.

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

## Damping-resistor noise in the tuned/untuned receivers (L-32)

`tuned_probe_rx` and `untuned_probe_rx` now include the thermal current noise
`4 k T / Rd` of the damping resistor, seen at the receiver input through `Zs`.
The same receiver model (`tuned_probe_receiver_noise`,
`untuned_probe_receiver_noise`) now also backs
`spin_dynamics.noise.{tuned,untuned}_probe_output_noise_density`, so the probe
SNR, the injected probe noise and the field-referred detectors use one formula
(at the default `Rd = 1 MOhm` that density rises by 0.02-0.8 %, and by about
3 % for the untuned SPA set). Octave was unavailable, so these SNR or score
values were regenerated from Python (signal spectra are unchanged):

- `calc_masy_tuned_probe_lp_orig.csv`, `calc_masy_untuned_probe_lp.csv`: SNR
  column only (at the default `Rd = 1 MOhm`: tuned 0.88349 to 0.88212, -0.16 %;
  untuned 0.66021 to 0.65844, -0.27 %).
- `optimization_tuned_excitation_phase_shift.csv`,
  `optimization_tuned_excitation_result.csv`,
  `optimization_tuned_inverse_result.csv`,
  `optimization_tuned_inverse_spectra.csv`: SNR, mismatch and residual-ratio
  values re-evaluated with the committed phases.
- `optimization_tuned_excitation_results.mat`,
  `optimization_tuned_excitation_results_inv.mat`,
  `optimization_tuned_refocusing_results.mat`,
  `optimization_untuned_refocusing_results.mat`: the stored score element only
  (untuned refocusing 0.93154 to 0.91805 and 0.86146 to 0.84900).

The MATLAB Version 3 `circuit_simulation/tuned_probe/tuned_probe_rx.m` and
`circuit_simulation/untuned_probe/untuned_probe_rx.m` now add the same term
(`vnd2 = 4*k*T/Rd*abs(Zs).^2`), so rerunning the Octave generators should
reproduce these regenerated values (not yet run: Octave was unavailable). The
older `untuned_probe_rx2.m` / `untuned_probe_rx3.m` variants, which the
generators do not call, are unchanged.
