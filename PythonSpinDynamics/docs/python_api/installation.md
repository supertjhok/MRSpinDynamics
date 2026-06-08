# Installation

The Python package is currently a source-tree workspace. The cleanest setup is
an editable install from `PythonSpinDynamics`:

```powershell
python -m pip install -e .
```

You can also run examples directly from the source tree. The scripts in
`examples/` add `../src` to `sys.path` automatically, so this works from either
`PythonSpinDynamics` or `PythonSpinDynamics/examples`:

```powershell
python examples\ideal_cpmg.py --numpts 101
```

Tests can also be run directly from `PythonSpinDynamics`:

```powershell
python -m unittest discover -s tests
```

If the system `python` does not have NumPy installed, use an environment that
does. In Codex, the bundled Python runtime has NumPy available.

## Dependencies

Required:

- Python 3.10 or newer
- NumPy

Optional:

- Matplotlib, only for `examples/plot_ideal_workflows.py` and
  `examples/plot_probe_cpmg.py`

The package metadata is in `pyproject.toml`. The port is not yet published as a
wheel or conda package.

## NumPy Compatibility

The package metadata currently requires NumPy 1.24 or newer. Avoid calling
newer NumPy-only aliases directly in ported code unless they are wrapped by a
local compatibility helper. For example, use `spin_dynamics.core.numerics` for
trapezoidal integration so both older Anaconda NumPy and newer NumPy 2.x
environments work.
