# PythonSpinDynamics

Python port workspace for the MATLAB spin-dynamics simulation package.

The original MATLAB implementation remains unchanged and should be treated as
the reference implementation during the port:

```text
../SpinDynamicsUpdated/Version_2/code
```

The active MATLAB tree contains several generations of implementation details.
For Python work, start from the current `Version_2` routines documented in the
repository-level `docs` folder, especially:

- `docs/QUICK_START.md`
- `docs/VERSION_GUIDE.md`
- `docs/VERSION_2_WORKFLOWS.md`
- `docs/SPEED_AUDIT.md`

## Porting Strategy

1. Preserve MATLAB behavior with small, reproducible reference cases.
2. Port the simple numerical helpers first: rotation matrices, free precession,
   asymptotic magnetization, and echo construction.
3. Port parameter constructors into typed Python dataclasses.
4. Port the current arbitrary-pulse kernel,
   `sim_spin_dynamics_arb10.m`, with NumPy first.
5. Add optimized backends only after the NumPy implementation is validated.

See `docs/migration_plan.md` and `docs/matlab_mapping.md` for the current
conversion status and remaining roadmap.

The Python API documentation starts at `docs/python_api/index.md`.

Application code should prefer the public CPMG workflow runners:

```python
from spin_dynamics.workflows import run_tuned_cpmg

result = run_tuned_cpmg(numpts=101, maxoffs=10)
```

The currently validated public runners are `run_ideal_cpmg`,
`run_ideal_cpmg_train`, `run_tuned_cpmg`, `run_tuned_cpmg_train`,
`run_untuned_cpmg`, `run_untuned_cpmg_train`, `run_matched_cpmg`, and
`run_matched_cpmg_train`. Probe-parameter sweep runners are available as
`run_tuned_q_sweep`, `run_matched_q_sweep`, `run_tuned_mistuning_sweep`, and
`run_matched_mistuning_sweep`. The matched-probe excitation/nutation sweep
`run_matched_z_magnetization_q_sweep` mirrors the MATLAB `z_Mag_Q` workflow.
The ideal time-varying-field workflow is available as
`run_ideal_time_varying_cpmg_final`, with
`run_ideal_time_varying_amplitude_sweep` for compact fluctuation-amplitude
studies. The matched-probe inversion-recovery finite train
`run_matched_cpmg_ir_train` extends the matched finite CPMG path over an
inversion-delay vector. Python-native finite-train sweep wrappers are available
as `run_tuned_finite_q_sweep`, `run_untuned_finite_q_sweep`,
`run_matched_finite_q_sweep`, and their `*_finite_mistuning_sweep` variants.
The first diffusion-aware matched CPMG workflow is available as
`run_matched_diffusion_cpmg`, with `run_matched_diffusion_q_sweep` for compact
Q studies. Compact ideal, tuned, and matched CPMG imaging workflows are
available as `run_ideal_cpmg_imaging`, `run_tuned_cpmg_imaging`, and
`run_matched_cpmg_imaging`.

The validated lower-level workflow surface also includes
`calc_macq_ideal_probe_relax4`, `calc_macq_tuned_probe_relax4`,
`calc_macq_untuned_probe_relax4`, and `calc_macq_matched_probe_relax4` for
assembled arbitrary sequences with relaxation during free-precession intervals.
Finite CPMG train runners can warn about isochromat-grid rephasing, refine the
offset grid with `auto_refine_grid=True`, and split isochromat propagation
across CPU cores with `num_workers`.

## Validation Status

| Area | Status | Notes |
| --- | --- | --- |
| Core rotations, echo conversion, FID, and `arb10` kernel | Fixture validated | Tight MATLAB/Octave CSV comparisons. |
| Ideal, tuned, untuned, and matched reference CPMG | Fixture validated | Matched-probe paths use practical tolerances because Python uses an independent NumPy solver. |
| Finite CPMG trains, finite Q/mistuning sweeps, and matched CPMG-IR | Validated plus smoke-tested | Includes serial/parallel equality checks where applicable. |
| Matched diffusion CPMG | Compact validation and smoke-tested | Very high-Q diffusion cases remain a known stiffness target. |
| Ideal, tuned, and matched CPMG imaging | Fixture validated | MATLAB-generated k-space fixtures plus visual plotting examples. |
| Pulse-shape helpers | Fixture validated | JMR rectangular pulse responses, phase quantization, and untuned segment adjustment. |
| OCT/SPA optimization | Not yet ported | Planned after the validated workflow kernels stay stable. |

See `docs/validation_results.md` for fixture details, run logs, and tolerance
notes.

## Examples

Run a small ideal CPMG workflow from this directory:

```powershell
python examples\ideal_cpmg.py --numpts 101
```

The example scripts also work when run directly from the `examples` directory.
For normal package development, you can install the source tree into your active
environment:

```powershell
python -m pip install -e .
```

Use the bundled Codex Python executable explicitly if `python` is not on PATH.

## Tests

Run the fast smoke tier during normal edit loops:

```powershell
python -m unittest tests.smoke_tests
```

Run the full validation suite before committing numerical or workflow changes:

```powershell
python -m unittest discover -s tests
```

On Codex desktop workspaces where `python` resolves to the Microsoft Store
shim, use the bundled Python runtime path reported by the workspace dependency
tool instead.

Run a small ideal FID workflow similarly:

```powershell
python examples\ideal_fid.py --numpts 101
```

Run a finite ideal CPMG echo train:

```powershell
python examples\ideal_cpmg_train.py --numpts 101 --num-echoes 8
```

Run an ideal CPMG final-echo sweep with time-varying B0 offsets:

```powershell
python examples\ideal_time_varying_cpmg.py --numpts 101 --num-echoes 16
```

Run a finite tuned-probe CPMG echo train:

```powershell
python examples\tuned_cpmg_train.py --numpts 101 --num-echoes 8
```

Run finite untuned- and matched-probe CPMG echo trains:

```powershell
python examples\untuned_cpmg_train.py --numpts 101 --num-echoes 8
python examples\matched_cpmg_train.py --numpts 101 --num-echoes 8
```

Run a matched-probe CPMG inversion-recovery finite train:

```powershell
python examples\matched_cpmg_ir_train.py --numpts 21 --num-echoes 4 --num-tau 4
```

Run compact finite-train probe parameter sweeps:

```powershell
python examples\finite_probe_train_sweeps.py --numpts 21 --num-echoes 3
```

Run a compact matched-probe diffusion CPMG Q sweep:

```powershell
python examples\matched_diffusion_cpmg.py --numpts 21 --num-echoes 3
```

Compare the currently validated workflows:

```powershell
python examples\compare_cpmg_fid.py --numpts 101
```

Export compact `.npz` arrays for notebooks or quick inspection:

```powershell
python examples\export_validation_arrays.py results\validation_arrays.npz --numpts 101
```

Plot the ideal workflows if Matplotlib is installed:

```powershell
python examples\plot_ideal_workflows.py --numpts 201 --output results\ideal_workflows.png
```

The plotting example uses a narrower FID offset range by default for readability.
Use `--fid-maxoffs 10 --raw-fid-scale` to show the raw MATLAB-style FID defaults.

Plot an ideal CPMG image reconstruction from the flower phantom:

```powershell
python examples\plot_ideal_imaging.py --pixels 6 --ny 7 --output results\ideal_imaging.png
```

Use `--probe tuned` or `--probe matched` to run the probe-aware imaging paths.

Run the original/reference tuned-probe CPMG comparison:

```powershell
python examples\tuned_probe_cpmg.py --numpts 101
```

Compare ideal, tuned, untuned, and matched CPMG:

```powershell
python examples\probe_cpmg_compare.py --numpts 101
```

Run compact tuned and matched Q/mistuning sweeps, including matched-probe
Z-magnetization versus Q:

```powershell
python examples\probe_parameter_sweeps.py --numpts 101
```

Plot a tuned or matched Q/mistuning sweep if Matplotlib is installed:

```powershell
python examples\plot_probe_parameter_sweep.py --probe tuned --sweep q --output results\tuned_q_sweep.png
```

Plot the same comparison if Matplotlib is installed:

```powershell
python examples\plot_probe_cpmg.py --numpts 101 --output results\probe_cpmg.png
```

The probe comparison plot shows asymptotic magnetization magnitude by default;
use `--masy-component real`, `imag`, or `phase` to inspect phase-sensitive
components.

## Pulse Evaluation

The SPA/OCT bridge currently includes the fixed SPA phase catalog, normalized
SNR/FOM bookkeeping, and tuned/untuned/matched non-plotting refocusing-pulse
evaluators:

```python
from spin_dynamics.optimization import (
    evaluate_matched_refocusing_pulse,
    evaluate_tuned_refocusing_pulse,
    evaluate_untuned_refocusing_pulse,
    summarize_tuned_spa_refocusing,
    spa_pulse_list,
)

pulse = spa_pulse_list()[0]
tuned = evaluate_tuned_refocusing_pulse(pulse.phases, numpts=101)
untuned = evaluate_untuned_refocusing_pulse(pulse.phases, numpts=101)
matched = evaluate_matched_refocusing_pulse(pulse.phases, numpts=9)
summary = summarize_tuned_spa_refocusing(numpts=101)
```

The optimization module also includes `summarize_tuned_spa_refocusing`,
`summarize_untuned_spa_refocusing`, and `summarize_matched_spa_refocusing`,
which return MATLAB-style normalized SNR and FOM arrays for rectangular and SPA
catalog pulses. A lightweight `optimize_spa_phase_program` discrete search
scaffold is available for small phase-state experiments.

Full continuous OCT/SPA optimizer loops are still deferred.
The matched evaluator uses the matched-network transient solver and is much
slower than the tuned and untuned evaluators, so start with small offset grids.
