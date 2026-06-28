# Development Environment

Use a persistent virtual environment for PythonSpinDynamics development,
testing, plotting, and performance work. The setup scripts below create or
update the environment, install the package in editable mode, and verify the
optional numerical stack.

The default Windows environment path is `PythonSpinDynamics/.venv-win`; the
default WSL environment path is `PythonSpinDynamics/.venv-wsl`. They are
separate because Windows and Linux virtual environments are not
interchangeable. Both paths are ignored by git. If OneDrive file locking or WSL
`/mnt/c` performance becomes a problem, use the same scripts with an external
path.

## Windows PowerShell

From `PythonSpinDynamics`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev_env.ps1
& ".\.venv-win\Scripts\Activate.ps1"
python scripts\verify_dev_env.py --strict
```

Use an external unsynced environment when needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev_env.ps1 -Venv "$env:USERPROFILE\venvs\python-spin-dynamics"
& "$env:USERPROFILE\venvs\python-spin-dynamics\Scripts\Activate.ps1"
```

## WSL / Ubuntu

For plotting, SciPy/NNLS checks, and Numba/JAX performance benchmarking on this
repository, use the `Ubuntu-24.04` WSL distribution.

From Windows PowerShell:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '/mnt/c/path/to/PythonSpinDynamics' && bash scripts/setup_dev_env_wsl.sh"
```

From an interactive WSL shell already inside `PythonSpinDynamics`:

```bash
bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict
```

For NVIDIA GPU acceleration through JAX, use the CUDA-enabled JAX wheels from
inside WSL. CUDA 13 is the default recommendation for current NVIDIA drivers:

```bash
JAX_CUDA=13 bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict --require-jax-gpu
```

Use `JAX_CUDA=12` only when driver or deployment constraints require the CUDA
12 wheel. Native Windows JAX can run on CPU, but NVIDIA GPU wheels are Linux
wheels; use WSL for CUDA benchmarking.

On small GPUs or shared workstations, disable JAX's default large memory
preallocation for smoke tests:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/verify_dev_env.py --strict --require-jax-gpu
```

For a faster Linux-filesystem environment, keep the source tree on `/mnt/c` but
put the virtual environment under `$HOME`:

```bash
VENV="$HOME/.venvs/python-spin-dynamics" bash scripts/setup_dev_env_wsl.sh
source "$HOME/.venvs/python-spin-dynamics/bin/activate"
```

## Installed Extras

The setup scripts install:

```text
.[dev,opt,plot,perf,bench]
```

- `dev` provides Ruff and test tooling.
- `opt` provides SciPy for optimization and inverse-Laplace workflows.
- `plot` provides Matplotlib and Pillow for examples.
- `perf` provides Numba and JAX for accelerated numerical backends.
- `bench` provides benchmark tooling.

The CUDA-enabled JAX wheel is installed separately by the WSL setup script when
`JAX_CUDA=13` or `JAX_CUDA=12` is set, because those wheels are platform- and
driver-specific.

Override the extra set only for deliberately small environments:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev_env.ps1 -Extras "dev,opt,plot"
```

```bash
EXTRAS="dev,opt,plot" bash scripts/setup_dev_env_wsl.sh
```

## Verification

Run the verifier whenever the active Python interpreter is in doubt:

```powershell
python scripts\verify_dev_env.py --strict
```

The report prints the Python executable, package import status, NumPy, SciPy,
Matplotlib, Numba, JAX, `jaxlib`, Ruff, benchmark tooling, and visible JAX
devices. A JAX CPU-only report is valid for CPU benchmarking. GPU benchmarking
requires a CUDA-enabled `jaxlib`; use `--require-jax-gpu` when a benchmark is
intended to run on the GPU.

## Smoke Checks

Before committing numerical or public-workflow changes, run the same checks as
the GitHub smoke job:

```powershell
python -m unittest tests.smoke_tests
python -m ruff check src tests examples
python docs\generate_api_reference.py
git diff --exit-code docs\python_api\api_reference.md
```

Use focused tests during edit loops:

```powershell
python -m unittest tests.test_motion tests.test_motion_sequence
python -m unittest tests.example_tests
```

## Benchmarking

Run benchmarks from an activated persistent environment, not from the system
Python:

```bash
python examples/porous_rock_cpmg_walkers.py --grid 24 --z-cells 32 --pores 90 --walkers-per-voxel 2 --num-echoes 6 --substeps 2 --benchmark-backends --plot-output .tmp/porous_dt2.png
python -B benchmarks/forward_kernel.py --group rawkernel --backend numba --sizes 4001,16001,64001 --num-echoes 256
```

Keep timing comparisons on the same machine, operating system, Python version,
dependency versions, and BLAS/JAX device settings. Save the command line and
the verifier output beside any curated performance result.

## Rebuilding

To refresh an existing environment after dependency metadata changes, rerun the
setup script. To rebuild from scratch, remove only the environment directory and
then rerun the script. Do not remove source files or generated validation
fixtures as part of environment cleanup.
