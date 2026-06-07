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

See `docs/migration_plan.md` and `docs/matlab_mapping.md` for the initial
conversion map.

## First Example

Run a small ideal CPMG workflow from this directory:

```powershell
$env:PYTHONPATH='src'
python examples\ideal_cpmg.py --numpts 101
```

Use the bundled Codex Python executable explicitly if `python` is not on PATH.
