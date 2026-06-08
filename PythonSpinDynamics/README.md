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
`run_ideal_cpmg_train`, `run_tuned_cpmg`, `run_untuned_cpmg`, and
`run_matched_cpmg`.

The validated lower-level workflow surface also includes
`calc_macq_ideal_probe_relax4`, `calc_macq_tuned_probe_relax4`,
`calc_macq_untuned_probe_relax4`, and `calc_macq_matched_probe_relax4` for
assembled arbitrary sequences with relaxation during free-precession intervals.

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

Run a small ideal FID workflow similarly:

```powershell
python examples\ideal_fid.py --numpts 101
```

Run a finite ideal CPMG echo train:

```powershell
python examples\ideal_cpmg_train.py --numpts 101 --num-echoes 8
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

Run the original/reference tuned-probe CPMG comparison:

```powershell
python examples\tuned_probe_cpmg.py --numpts 101
```

Compare ideal, tuned, untuned, and matched CPMG:

```powershell
python examples\probe_cpmg_compare.py --numpts 101
```

Plot the same comparison if Matplotlib is installed:

```powershell
python examples\plot_probe_cpmg.py --numpts 101 --output results\probe_cpmg.png
```

The probe comparison plot shows asymptotic magnetization magnitude by default;
use `--masy-component real`, `imag`, or `phase` to inspect phase-sensitive
components.
