# Validation Results

This document records Python-vs-MATLAB/Octave checks against the active MATLAB
source tree. Octave is useful for reproducible low-friction fixture generation;
MATLAB is used for matched-probe fixtures that depend on toolbox behavior not
available in a stock Octave install.

## Environment

- Octave: GNU Octave 11.3.0
- Python: bundled Codex Python runtime
- NumPy: 2.3.5
- MATLAB reference tree: `../SpinDynamicsUpdated/Version_2/code`

## Fixture Generation

The full fixture suite is generated with MATLAB by:

```powershell
matlab -batch "run('validation/octave/generate_basic_fixtures.m')"
matlab -batch "run('validation/octave/generate_imaging_fixtures.m')"
matlab -batch "run('validation/octave/generate_pulse_fixtures.m')"
```

Most fixtures can also be generated with Octave by:

```powershell
& 'C:\Program Files\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe' --quiet validation\octave\generate_basic_fixtures.m
```

The script writes CSV fixtures under `validation/fixtures`. Octave generation
skips matched-probe files when `fmincon` is unavailable.

## Initial Functions

| Python function | MATLAB/Octave reference | Fixture | Status |
| --- | --- | --- | --- |
| `spin_dynamics.core.echo.calc_time_domain_echo` | `calc_echo/calc_time_domain_echo.m` | `calc_time_domain_echo.csv` | Passed |
| `spin_dynamics.core.echo.calc_time_domain_echo_arb` | `calc_echo/calc_time_domain_echo_arb.m` | `calc_time_domain_echo_arb.csv` | Passed |
| `spin_dynamics.core.rotations.sim_spin_dynamics_asymp_mag3` | `sim_spin_dynamics_asymp/sim_spin_dynamics_asymp_mag3.m` | `sim_spin_dynamics_asymp_mag3.csv` | Passed |
| `spin_dynamics.core.rotations.calc_rot_axis_arba3` | `calc_rot/calc_rot_axis_arba3.m` | `calc_rot_axis_arba.csv` | Passed |
| `spin_dynamics.core.rotations.calc_rot_axis_arba4` | `calc_rot/calc_rot_axis_arba4.m` | `calc_rot_axis_arba.csv` | Passed |
| `spin_dynamics.workflows.cpmg.calc_masy_ideal` | `calc_masy/calc_masy_ideal.m` | `calc_masy_ideal.csv` | Passed |
| `spin_dynamics.parameters.set_params_ideal` | `Params/set_params_ideal.m` | `set_params_ideal.csv` | Passed |
| `spin_dynamics.parameters.set_params_ideal_fid` | `Params/set_params_ideal_FID.m` | `set_params_ideal_fid.csv` | Passed |
| `spin_dynamics.parameters.set_params_tuned_orig` | `Params/set_params_tuned_Orig.m` | `set_params_tuned_orig.csv` | Passed |
| `spin_dynamics.parameters.set_params_untuned_orig` | `Params/set_params_untuned_Orig.m` | `set_params_untuned_orig.csv` | Passed |
| `spin_dynamics.parameters.set_params_matched_orig` | `Params/set_params_matched_Orig.m` | `set_params_matched_orig.csv` | Passed |
| `spin_dynamics.core.rotations.calc_rotation_matrix` | `calc_rot/calc_rotation_matrix.m` | `calc_rotation_matrix.csv` | Passed |
| `spin_dynamics.core.kernels.sim_spin_dynamics_arb10` | `sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m` | `sim_spin_dynamics_arb10.csv` | Passed |
| `spin_dynamics.core.kernels.sim_spin_dynamics_arb10_chunked` | Serial Python `arb10` kernel | direct serial/chunked equality test | Passed |
| `spin_dynamics.core.isochromats.check_rephasing` | Python extension beyond MATLAB warning-only checks | direct warning and recommendation test | Passed |
| `spin_dynamics.workflows.acquisition.calc_macq_ideal_probe_relax4` | `calc_macq/calc_macq_ideal_probe_relax4.m` | `calc_macq_ideal_probe_relax4.csv` | Passed |
| `spin_dynamics.workflows.acquisition.calc_macq_tuned_probe_relax4` | `calc_macq/calc_macq_tuned_probe_relax4.m` | `calc_macq_tuned_probe_relax4.csv` | Passed |
| `spin_dynamics.workflows.acquisition.calc_macq_matched_probe_relax4` | `calc_macq/calc_macq_matched_probe_relax4.m` | `calc_macq_matched_probe_relax4.csv` | Passed |
| `spin_dynamics.workflows.acquisition.calc_macq_untuned_probe_relax4` | Python analogue of tuned `relax4` receiver-map contract | direct receiver contract test | Passed |
| `spin_dynamics.workflows.run_ideal_cpmg_train` | `time_varying_field/sim_cpmg_ideal_tv.m` assembly pattern | `run_ideal_cpmg_train_*.csv` | Passed |
| `spin_dynamics.workflows.run_tuned_cpmg_train` | `Sim_CPMG/sim_cpmg_tuned_probe_img.m` assembly pattern without phase encoding | `run_tuned_cpmg_train_*.csv` | Passed |
| `spin_dynamics.workflows.run_untuned_cpmg_train` | Python analogue of tuned finite-train assembly with untuned pulse and receiver models | `run_untuned_cpmg_train_*.csv` | Passed |
| `spin_dynamics.workflows.run_matched_cpmg_train` | `Sim_CPMG/sim_cpmg_matched_probe_img.m` assembly pattern without phase encoding | `run_matched_cpmg_train_*.csv` | Passed |
| `spin_dynamics.workflows.run_matched_cpmg_ir_train` | `Sim_CPMG_IR/sim_cpmg_ir_matched_probe_relax4.m` | workflow shape, finite-output, and tau-parallel equality smoke tests | Passed |
| `spin_dynamics.workflows.run_*_finite_q_sweep` | Python-native wrappers around finite train runners | workflow shape and finite-output smoke tests | Passed |
| `spin_dynamics.workflows.run_*_finite_mistuning_sweep` | Python-native wrappers around finite train runners | workflow shape, finite-output, and sweep-parallel equality smoke tests | Passed |
| `spin_dynamics.core.kernels.sim_spin_dynamics_arb10_diffusion` | `sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m` design, modernized to `arb10` structure | zero-diffusion equality with `arb10` and chunked equality tests | Passed |
| `spin_dynamics.workflows.run_matched_diffusion_cpmg` | `Sim_Diffusion/sim_dif_matched_CPMG_noRx.m` | workflow shape and finite-output smoke test | Passed |
| `spin_dynamics.workflows.run_matched_diffusion_q_sweep` | `DIffusion_Example/Diff_Echo_Q.m` | workflow shape and sweep-parallel equality smoke test | Passed |
| `spin_dynamics.workflows.run_ideal_cpmg_imaging` | `Imaging_demo/imaging_example_ideal.m`, `Sim_CPMG/sim_cpmg_ideal_probe_img.m` | `run_ideal_cpmg_imaging_kspace.csv`, workflow shape, and phase-parallel equality tests | Passed |
| `spin_dynamics.workflows.run_tuned_cpmg_imaging` | `Sim_CPMG/sim_cpmg_tuned_probe_img.m` | `run_tuned_cpmg_imaging_kspace.csv`, workflow shape, and phase-parallel equality tests | Passed |
| `spin_dynamics.workflows.run_matched_cpmg_imaging` | `Sim_CPMG/sim_cpmg_matched_probe_img.m` | `run_matched_cpmg_imaging_kspace.csv` and workflow shape tests | Passed |
| `spin_dynamics.pulses.tuned_rectangular_pulse_response` | `Pulse Shape/tunedPulse.m` | `pulse_tuned_rectangular.csv` | Passed |
| `spin_dynamics.pulses.untuned_rectangular_pulse_response` | `Pulse Shape/untunedPulse.m` | `pulse_untuned_rectangular.csv` | Passed |
| `spin_dynamics.pulses.matched_rectangular_pulse_response` | `Pulse Shape/matchedPulse.m` | `pulse_matched_rectangular.csv` | Passed |
| `spin_dynamics.pulses.quantize_phase` | `opt_pulse/quantize_phase.m` | `pulse_quantize_phase.csv` | Passed |
| `spin_dynamics.pulses.adjust_untuned_segment_lengths` | `opt_pulse/untuned_pulse_adjust.m` | `pulse_untuned_segment_adjust*.csv` | Passed |
| `spin_dynamics.optimization.spa_pulse_list` | `OCT_Pulse_Examples/SPA_pulse_list.m` | direct catalog check | Passed |
| `spin_dynamics.optimization.evaluate_spa_metrics` | `OCT_Pulse_Examples/SPA_optimization_*.m` metric formulas | direct normalization check | Passed |
| `spin_dynamics.optimization.evaluate_tuned_refocusing_pulse` | `opt_pulse/plot_masy_arbref_tuned.m` | lower-level tuned asymptotic equivalence and SPA pulse smoke tests | Passed |
| `spin_dynamics.workflows.run_tuned_q_sweep` | `CompareQ/sim_tuned_probe_coil_Q.m` | workflow shape and finite-output smoke test | Passed |
| `spin_dynamics.workflows.run_matched_q_sweep` | `CompareQ/sim_matched_probe_coil_Q.m` | workflow shape and finite-output smoke test | Passed |
| `spin_dynamics.workflows.run_tuned_mistuning_sweep` | `CompareMistuned/tuned_probe/sim_tuned_probe_mistuned.m` | workflow shape and finite-output smoke test | Passed |
| `spin_dynamics.workflows.run_matched_mistuning_sweep` | `CompareMistuned/matched_probe/sim_matched_probe_mistuned.m` | workflow shape and finite-output smoke test | Passed |
| `spin_dynamics.probes.matched.calc_masy_matched_nut` | `calc_masy/calc_masy_matched_nut.m` | exercised through z-magnetization sweep smoke test | Passed |
| `spin_dynamics.workflows.run_matched_z_magnetization_q_sweep` | `z_mag/z_Mag_Q.m` | workflow shape and finite-output smoke test | Passed |
| `spin_dynamics.workflows.run_ideal_time_varying_cpmg_final` | `time_varying_field/sim_cpmg_ideal_tv_final.m` | workflow shape and finite-output smoke test | Passed |
| `spin_dynamics.workflows.run_ideal_time_varying_amplitude_sweep` | `time_varying_field/compare_cpmg_results_ideal_tv.m` | serial/parallel equality and finite-output smoke test | Passed |
| `spin_dynamics.workflows.fid.sim_fid_ideal` | `Sim_FID/simFID_ideal.m` via `calc_macq_fid`/`calc_FID_time_domain` | `sim_fid_ideal_macq.csv`, `sim_fid_ideal_echo.csv` | Passed |
| `spin_dynamics.probes.tuned.tuned_probe_lp_orig` | `circuit_simulation/tuned_probe/tuned_probe_lp_Orig.m` | `tuned_probe_lp_orig.csv` | Passed |
| `spin_dynamics.probes.tuned.calc_masy_tuned_probe_lp_orig` | `calc_masy/calc_masy_tuned_probe_lp_Orig.m` | `calc_masy_tuned_probe_lp_orig.csv` | Passed |
| `spin_dynamics.probes.untuned.untuned_probe_lp` | `circuit_simulation/untuned_probe/untuned_probe_lp.m` | `untuned_probe_lp.csv` | Passed |
| `spin_dynamics.probes.untuned.calc_masy_untuned_probe_lp` | `calc_masy/calc_masy_untuned_probe_lp.m` | `calc_masy_untuned_probe_lp.csv` | Passed |
| `spin_dynamics.probes.matched.find_coil_current` | `circuit_simulation/matched_probe/find_coil_current.m` | `find_coil_current_matched.csv` | Passed |
| `spin_dynamics.probes.matched.calc_masy_matched_probe_orig` | `calc_masy/calc_masy_matched_probe_Orig.m` | `calc_masy_matched_probe_orig.csv` | Passed |

## Workflow API Smoke Tests

The public CPMG runners are also tested for result-container shape and metadata:

- `run_ideal_cpmg`
- `run_tuned_cpmg`
- `run_tuned_cpmg_train`
- `run_untuned_cpmg`
- `run_untuned_cpmg_train`
- `run_matched_cpmg`
- `run_matched_cpmg_train`
- `run_matched_cpmg_ir_train`
- `run_tuned_finite_q_sweep`
- `run_untuned_finite_q_sweep`
- `run_matched_finite_q_sweep`
- `run_tuned_finite_mistuning_sweep`
- `run_untuned_finite_mistuning_sweep`
- `run_matched_finite_mistuning_sweep`
- `run_matched_diffusion_cpmg`
- `run_matched_diffusion_q_sweep`
- `run_ideal_cpmg_imaging`
- `run_tuned_cpmg_imaging`
- `run_matched_cpmg_imaging`
- `run_tuned_q_sweep`
- `run_matched_q_sweep`
- `run_tuned_mistuning_sweep`
- `run_matched_mistuning_sweep`
- `run_matched_z_magnetization_q_sweep`
- `run_ideal_time_varying_cpmg_final`
- `run_ideal_time_varying_amplitude_sweep`

The finite ideal train is also tested for `auto_refine_grid=True`.

## Run Log

2026-06-07:

```text
Generated fixtures with Octave 11.3.0.
Ran 8 Python unittest comparisons.
Result: OK
```

2026-06-07:

```text
Generated fixtures with MATLAB and Octave 11.3.0.
Ran 20 Python unittest comparisons and workflow smoke tests.
Result: OK
```

2026-06-08:

```text
Ran 30 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Ran Python compile checks over src, tests, and examples.
Result: OK
```

2026-06-08:

```text
Generated tuned finite CPMG train fixtures with Octave 11.3.0.
Ran 31 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-08:

```text
Generated untuned finite CPMG train fixtures with Octave 11.3.0.
Generated matched finite CPMG train fixtures with MATLAB R2025b.
Ran 33 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-08:

```text
Added isochromat-grid rephasing checks, optional finite-train grid refinement,
and chunked multicore `arb10` propagation.
Ran 35 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-08:

```text
Benchmarked long finite ideal CPMG trains across isochromat vector sizes and
worker counts on a 24-logical-CPU Windows host.
Best measured speedup was 1.88x for 64,001 isochromats, 256 echoes, and
4 workers versus 1 worker.
Detailed results are in benchmarks/README.md and benchmarks/results/.
```

2026-06-08:

```text
Added tuned/matched Q and mistuning sweep workflows plus a compact example.
Added a Matplotlib sweep plotting example with CLI smoke coverage.
Ran 37 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-09:

```text
Added matched-probe z-magnetization Q sweep from z_mag/z_Mag_Q.m.
Ran 38 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-09:

```text
Added ideal time-varying-field CPMG final-echo and amplitude-sweep workflows
from time_varying_field/sim_cpmg_ideal_tv_final.m and comparison scripts.
Ran 40 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-09:

```text
Added matched-probe CPMG-IR finite train over tauvect from
Sim_CPMG_IR/sim_cpmg_ir_matched_probe_relax4.m.
Ran 42 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-09:

```text
Added Python-native finite-train Q and mistuning sweep wrappers around the
tuned, untuned, and matched finite CPMG train workflows.
Ran 44 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-09:

```text
Added the first matched-probe diffusion CPMG path with an arb10-style
diffusion kernel, compact Q sweep, and non-plot example.
Ran 47 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-09:

```text
Added matched diffusion high-Q validation benchmark.
For a 17-offset, 2-echo smoke case, Q values 20-100 remained finite and
Q >= 200 produced non-finite transient outputs with RuntimeWarnings.
Added a least-squares fallback for singular matching-network Newton steps so
extreme-Q cases are recorded as transient failures rather than design failures.
The public matched-diffusion workflow exposes the finite-through-Q=100
solver-validation boundary through a warning/raise helper.
Ran 48 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
Detailed results are in benchmarks/README.md and benchmarks/results/.
```

2026-06-09:

```text
Added ideal CPMG imaging workflow and flower-phantom plotting example.
Ran 50 Python unittest comparisons, workflow smoke tests, and example smoke tests.
Result: OK
```

2026-06-10:

```text
Added compact tuned- and matched-probe CPMG imaging workflows and exposed them
through the flower-phantom plotting example.
Focused imaging and example smoke tests passed.
```

2026-06-10:

```text
Added MATLAB-generated ideal, tuned, and matched CPMG imaging k-space fixtures.
Validated all three Python imaging workflows against the fixture k-space arrays.
The matched-probe network design now uses the analytic positive-capacitance
match solution, and the matched transient solver substeps stiff RF intervals.
Ran 51 Python unittest comparisons and workflow smoke tests.
Result: OK
```

2026-06-11:

```text
Added fixture-validated pulse-shape utilities for JMR tuned, untuned, and
matched rectangular pulse responses, phase quantization, and untuned pulse
segment-length adjustment. The MATLAB pulse fixture generator writes compact
sampled response arrays and timing metadata.
Ran 61 Python unittest comparisons and workflow smoke tests.
Result: OK
```

Matched-probe fixtures are generated by MATLAB. Octave skips them when
`fmincon` is unavailable.

The `sim_spin_dynamics_arb10` fixture covers:

- two precomputed RF pulse matrices;
- free-precession segments;
- nonzero gradient scaling through `del_wg`;
- vector-valued `T1n`, `T2n`, `m0`, and `mth`;
- three acquired `M-` spectra.

## Examples

`examples/ideal_cpmg.py` runs the validated ideal CPMG path:

```powershell
$env:PYTHONPATH='src'
& '<python.exe>' examples\ideal_cpmg.py --numpts 101
```

Latest output summary:

```text
num offsets: 101
masy shape: (101,)
echo shape: (404,)
peak echo time: 0.0389775763473
peak echo value: (0.12325886877816501+1.09923071745065e-18j)
sum |masy|: 13.439665452
sum |echo|: 4.25940404244
```

`examples/ideal_fid.py` runs the validated ideal FID path:

```powershell
$env:PYTHONPATH='src'
& '<python.exe>' examples\ideal_fid.py --numpts 101
```

Octave printed `error: ignoring const execution_exception& while preparing to
exit` after fixture generation, but returned exit code 0 and wrote the expected
fixture files. The Python validation passed against those files.

## Notes

- The first tests intentionally avoid plotting, toolboxes, MEX, and `.mat`
  parsing.
- CSV stores complex arrays as real and imaginary columns.
- Most fixture comparisons use tight floating-point tolerances around
  `rtol=1e-13` and `atol=1e-13`.
- Matched-probe comparisons use practical tolerances because the Python port
  deliberately uses an independent NumPy-only nonlinear solve and RK4 response
  calculation instead of MATLAB's optimization and ODE solver stack.
