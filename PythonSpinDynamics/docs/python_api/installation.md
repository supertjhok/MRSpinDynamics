# Installation

The Python package is currently a source-tree workspace. The cleanest setup is
an editable install from `PythonSpinDynamics`:

```powershell
python -m pip install -e .
```

On Windows, avoid creating virtual environments inside a OneDrive-synced
checkout. Put the environment in a local, unsynced directory and install the
package from the source tree:

```powershell
cd "C:\Users\smandal\OneDrive - Brookhaven National Laboratory\Codex\NMR\MATLABSpinDynamics\PythonSpinDynamics"
conda create -p "C:\Users\smandal\codex-envs\python-spin-dynamics" python=3.11 numpy scipy matplotlib -y
conda run -p "C:\Users\smandal\codex-envs\python-spin-dynamics" python -m pip install -e .
```

Then run tests or examples with:

```powershell
& "C:\Users\smandal\codex-envs\python-spin-dynamics\python.exe" -m unittest tests.smoke_tests
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

If an older `.venv` or `.conda-env` was created inside the checkout, verify the
external environment first, then remove the in-tree environment to avoid
OneDrive sync and file-lock overhead.

## Dependencies

Required:

- Python 3.10 or newer
- NumPy

Optional:

- Matplotlib and Pillow, for plotting and image-phantom examples. Install with
  `python -m pip install -e .[plot]`.
- SciPy, for `optimizer="scipy"` in pulse-optimization helpers. Install with:

```powershell
python -m pip install -e .[opt]
```

The package metadata is in `pyproject.toml`. The port is not yet published as a
wheel or conda package.

## NumPy Compatibility

The package metadata currently requires NumPy 1.24 or newer. Avoid calling
newer NumPy-only aliases directly in ported code unless they are wrapped by a
local compatibility helper. For example, use `spin_dynamics.core.numerics` for
trapezoidal integration so both older Anaconda NumPy and newer NumPy 2.x
environments work.
