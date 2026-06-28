# Installation

PythonSpinDynamics is currently a source-tree package. The recommended setup
for development, examples, tests, plotting, and benchmarks is a persistent
virtual environment managed by the repository scripts.

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

The package metadata is in `pyproject.toml`. The port is not yet published as a
wheel or conda package.

## NumPy Compatibility

The package metadata currently requires NumPy 1.24 or newer. Avoid calling
newer NumPy-only aliases directly in ported code unless they are wrapped by a
local compatibility helper. For example, use `spin_dynamics.core.numerics` for
trapezoidal integration so both older Anaconda NumPy and newer NumPy 2.x
environments work.
