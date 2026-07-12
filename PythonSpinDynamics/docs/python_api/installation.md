# Installation

PythonSpinDynamics builds standard wheels and source distributions. Until a
package index release is published, install from a checkout or from an artifact
produced by `python -m build`. The recommended setup for development, examples,
tests, plotting, and benchmarks is a persistent virtual environment managed by
the repository scripts.

From `PythonSpinDynamics` on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev_env.ps1
& ".\.venv-win\Scripts\Activate.ps1"
python scripts\verify_dev_env.py --strict
```

From `PythonSpinDynamics` on WSL/Ubuntu:

```bash
bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict
```

For CUDA-enabled JAX in WSL:

```bash
JAX_CUDA=13 bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict --require-jax-gpu
```

The scripts create or update `.venv-win` on Windows or `.venv-wsl` on WSL,
install the package in editable mode, and install the standard development
extras:

```text
.[dev,opt,plot,perf,bench]
```

See [Development Environment](../development_environment.md) for WSL
`Ubuntu-24.04` commands, external virtual-environment paths, smoke checks, and
benchmarking notes.

## Minimal Install

For a runtime-only editable install, use:

```powershell
python -m pip install -e .
```

After installation, the experiment facade is available as a console command:

```powershell
spin-dynamics plan examples\experiment_config_cpmg.toml
spin-dynamics run examples\experiment_config_cpmg.toml -o run.npz
spin-dynamics verify run.npz
```

`python -m spin_dynamics.experiment` remains an equivalent, supported entry
point. To build and inspect release artifacts locally:

```powershell
python -m build
python -m twine check dist\*
python -m pip install dist\python_spin_dynamics-*.whl
```

## Workplace Release Bundle

The GitHub Release is the supported workplace distribution route while the
workspace is not published to a public package index. Download the wheel files
and `SHA256SUMS` into one directory, verify the checksums, and install the
integration distribution. Pip will select both workspace dependencies from the
same directory:

```powershell
Get-FileHash -Algorithm SHA256 *.whl,*.tar.gz,*.pdf
python -m pip install --find-links . "mr-integration==0.3.0"
spin-dynamics --version
spin-dynamics --help
```

On Linux, `sha256sum -c SHA256SUMS` verifies the complete published bundle.
`--find-links` makes the three MRSpinDynamics distributions local, but their
third-party dependencies (notably NumPy) must already be installed or available
from an approved workplace package index/cache. For a fully disconnected
installation, mirror those dependencies into the directory first and add
`--no-index` only after confirming that the directory is complete.

The integration wheel depends on the exact matching versions of
`python-spin-dynamics` and `quadrupolar-dft`; do not mix artifacts from different
workspace releases. Each installed run records the PythonSpinDynamics version
in its provenance metadata.

The scripts in `examples/` also add `../src` to `sys.path` automatically, so
simple examples can run from either `PythonSpinDynamics` or
`PythonSpinDynamics/examples` while developing:

```powershell
python examples\ideal_cpmg.py --numpts 101
```

## Dependencies

Required:

- Python 3.10 or newer
- NumPy

Optional extras:

- `opt`: SciPy-backed optimization and inverse-Laplace workflows.
- `plot`: Matplotlib and Pillow for plotting and image-phantom examples.
- `dev`: test and lint tooling.
- `perf`: Numba and JAX acceleration backends.
- `bench`: benchmark tooling.
- `jax-cuda13` / `jax-cuda12`: CUDA-enabled JAX wheels for Linux/WSL GPU
  environments.

Install a custom subset only when deliberately building a smaller environment:

```powershell
python -m pip install -e ".[opt,plot]"
```

The package metadata is in `pyproject.toml`. Wheels include the `py.typed`
marker and GPL license text. The project is not yet published to a public
package index or conda channel.

## NumPy Compatibility

The package metadata currently requires NumPy 1.24 or newer. Avoid calling
newer NumPy-only aliases directly in ported code unless they are wrapped by a
local compatibility helper. For example, use `spin_dynamics.core.numerics` for
trapezoidal integration so both older Anaconda NumPy and newer NumPy 2.x
environments work.
