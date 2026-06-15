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
studies. CPMG inversion-recovery finite trains are available as
`run_ideal_cpmg_ir_train`, `run_tuned_cpmg_ir_train`,
`run_untuned_cpmg_ir_train`, and `run_matched_cpmg_ir_train`; these sweep an
inversion-delay vector and return tau-by-echo arrays useful for T1-T2
simulation grids. Python-native finite-train sweep wrappers are available as
`run_tuned_finite_q_sweep`, `run_untuned_finite_q_sweep`,
`run_matched_finite_q_sweep`, and their `*_finite_mistuning_sweep` variants.
The first diffusion-aware matched CPMG workflow is available as
`run_matched_diffusion_cpmg`, with `run_matched_diffusion_q_sweep` for compact
Q studies. Compact ideal, tuned, and matched CPMG imaging workflows are
available as `run_ideal_phase_encoded_cpmg_imaging`,
`run_tuned_phase_encoded_cpmg_imaging`, and
`run_matched_phase_encoded_cpmg_imaging`. The older
`run_*_cpmg_imaging` names remain compatibility aliases. Imaging runs can also
consume arbitrary two-dimensional B0, transmit-B1, and receive-B1 maps through
`make_imaging_field_maps` or `load_imaging_field_maps_npz`.

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
| Ideal, tuned, and matched CPMG imaging | Fixture validated | MATLAB-generated k-space fixtures, arbitrary B0/B1 map helpers, and visual plotting examples. |
| Pulse-shape helpers | Fixture validated | JMR rectangular pulse responses, phase quantization, and untuned segment adjustment. |
| OCT/SPA pulse evaluation | Partly ported and validated | Fixed SPA catalog, SNR/FOM summaries, tuned/untuned/matched refocusing evaluators, lightweight search scaffolds, and bounded phase optimizers are available. MATLAB-equivalent optimizer loops remain deferred. |

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

Run a CPMG inversion-recovery finite train:

```powershell
python examples\matched_cpmg_ir_train.py --probe ideal --numpts 21 --num-echoes 4 --num-tau 4
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
To inspect custom B0, transmit-B1, and receive-B1 maps in the same workflow,
run:

```powershell
python examples\plot_custom_imaging_fields.py --pixels 8 --ny 7 --output results\custom_imaging_fields.png
```

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

Plot compact tuned-probe optimization workflows if Matplotlib is installed:

```powershell
python examples\plot_optimization_workflows.py --numpts 11 --segments 2 --starts 2 --inverse-starts 4 --output results\optimization_workflows.png
```

The optimization plot builds an excitation target first, then probes the
inverse-excitation objective with a MATLAB-style multi-start search that starts
from the phase-flipped target and refines around the best inverse found so far.
Treat that inverse panel as a diagnostic until MATLAB parity is validated for a
stronger inverse-cancellation workflow.

Compare optimization backends without plotting:

```powershell
python examples\diagnose_optimization_backends.py --backend all --numpts 21 --segments 3
```

On WSL2, use a virtual environment and install SciPy through the optional
optimization extra first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[opt]"
python examples/diagnose_optimization_backends.py --backend all --numpts 21 --segments 3
```

Plot finite echo trains, diffusion sweeps, and time-varying-field sweeps:

```powershell
python examples\plot_finite_train_workflows.py --numpts 17 --num-echoes 4 --output results\finite_trains.png
python examples\plot_diffusion_sweep.py --numpts 17 --num-echoes 3 --output results\diffusion_sweep.png
python examples\plot_time_varying_sweep.py --numpts 51 --num-echoes 12 --output results\time_varying_sweep.png
```

## Pulse Evaluation

The SPA/OCT bridge currently includes the fixed SPA phase catalog, normalized
SNR/FOM bookkeeping, and tuned/untuned/matched non-plotting refocusing-pulse
evaluators. Fixed-amplitude refocusing phases can also be optimized with small
bounded pattern-search wrappers. Tuned excitation and inverse-excitation pulse
evaluation/search are available when a refocusing axis and, for inverse pulses,
a target received spectrum are supplied. Multi-start driver scaffolds can run
repeated seeded starts and return ranked results:

```python
import numpy as np

from spin_dynamics.core.rotations import calc_rot_axis_arba3
from spin_dynamics.optimization import (
    evaluate_matched_refocusing_pulse,
    evaluate_tuned_refocusing_pulse,
    evaluate_untuned_refocusing_pulse,
    optimize_tuned_refocusing_phases,
    run_tuned_excitation_multistart,
    run_tuned_inverse_excitation_multistart,
    run_tuned_refocusing_multistart,
    summarize_tuned_spa_refocusing,
    spa_pulse_list,
)

pulse = spa_pulse_list()[0]
tuned = evaluate_tuned_refocusing_pulse(pulse.phases, numpts=101)
untuned = evaluate_untuned_refocusing_pulse(pulse.phases, numpts=101)
matched = evaluate_matched_refocusing_pulse(pulse.phases, numpts=9)
summary = summarize_tuned_spa_refocusing(numpts=101)
optimum = optimize_tuned_refocusing_phases(pulse.phases[:6], numpts=21)
repeated = run_tuned_refocusing_multistart(6, num_starts=4, seed=123, numpts=21)
del_w = np.linspace(-10.0, 10.0, 21)
neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
target = run_tuned_excitation_multistart(3, neff, num_starts=4, seed=123, numpts=21)
inverse = run_tuned_inverse_excitation_multistart(
    3,
    neff,
    target.best_result.best_evaluation.mrx,
    target.best_result.best_evaluation.snr,
    target.best_result.best_phases,
    num_starts=4,
    seed=123,
    numpts=21,
)
```

The optimization module also includes `summarize_tuned_spa_refocusing`,
`summarize_untuned_spa_refocusing`, and `summarize_matched_spa_refocusing`,
which return MATLAB-style normalized SNR and FOM arrays for rectangular and SPA
catalog pulses. A lightweight `optimize_spa_phase_program` discrete search
scaffold is available for small phase-state experiments. The phase optimizers
accept `optimizer="auto"`, `"pattern"`, or `"scipy"`: `auto` uses SciPy's
bounded continuous optimizer when the optional `opt` extra is installed and
falls back to the dependency-light NumPy pattern search otherwise. The
default phase bounds and random starts match the MATLAB `0` to `2*pi`
convention. The multi-start drivers are array-returning Python scaffolds rather
than MATLAB `.mat` result-file writers.

MATLAB `.mat` result-file compatibility and broad `fmincon` parity are still
deferred beyond the compact optimization fixtures.
The matched evaluator uses the matched-network transient solver and is much
slower than the tuned and untuned evaluators, so start with small offset grids.
