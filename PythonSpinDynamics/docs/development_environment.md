# Development Environment

This page is for contributors and reproducibility work. If you only need to run
the package, the shorter [Installation](python_api/installation.md) page is
enough. Continue here when modifying source, rebuilding documentation, running
the complete test matrix, plotting in WSL, or benchmarking optional backends.

Use a persistent virtual environment for that work. The setup scripts below
create or update the environment, install the package in editable mode, and
verify the optional numerical stack.

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

If the repository lives on a OneDrive-synced path (for example
`C:\Users\<you>\OneDrive\...`), do not use the default `.venv-wsl` at the
repository root: it is placed under OneDrive and will be synced, which is slow
and can corrupt the environment. Use the `$HOME` virtual-environment layout in
[Environment on the Linux filesystem](#environment-on-the-linux-filesystem)
instead.

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

### Environment on the Linux filesystem

Keep the source tree on `/mnt/c` but put the virtual environment under `$HOME`,
on the native Linux filesystem. This is the recommended WSL layout, and it is
required when the repository is on a OneDrive-synced path:

```bash
VENV="$HOME/.venvs/python-spin-dynamics" bash scripts/setup_dev_env_wsl.sh
source "$HOME/.venvs/python-spin-dynamics/bin/activate"
```

The `VENV` variable accepts any absolute path; the setup script uses it as-is
and only falls back to `.venv-wsl` at the repository root when it is unset.
Placing the environment under `$HOME` keeps it off the OneDrive tree, so it is
never synced, and on `ext4` rather than `/mnt/c`, so installs and imports are
much faster. The editable install still points back at the source under
`/mnt/c`, so edits take effect immediately.

### Checkout on the Linux filesystem

Tests also run faster when the source tree itself is on the Linux filesystem,
because every import and fixture read from `/mnt/c` crosses the Windows/WSL
file-system boundary. Measured on 2026-11-02 with the same environment and
single-threaded BLAS, the full suite took 8.3 minutes from `~/MRSpinDynamics`
and 12.2 minutes from `/mnt/c` (smoke tests: 12 s against 17 s). A second
checkout for test runs is cheap with a blob-less clone (about 21 MiB of git
data):

```bash
git clone --filter=blob:none https://github.com/supertjhok/MRSpinDynamics.git ~/MRSpinDynamics
cd ~/MRSpinDynamics/PythonSpinDynamics
VENV="$HOME/.venvs/python-spin-dynamics-native" bash scripts/setup_dev_env_wsl.sh
source "$HOME/.venvs/python-spin-dynamics-native/bin/activate"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests
```

To test uncommitted Windows-side work there, push it to a branch or fetch from
the Windows checkout (`git fetch /mnt/c/path/to/MRSpinDynamics <branch>`).
Editors on Windows can open the Linux checkout through `\\wsl$\Ubuntu-24.04\`.
Leave the BLAS thread variables set unless a test needs threading: without
them each test process may start one BLAS thread per core.

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
python -m mypy
python -m coverage run -m unittest discover -s tests
python -m coverage report
python docs\generate_api_reference.py
python docs\generate_validation_matrix.py --check
git diff --exit-code docs\python_api\api_reference.md
python docs\check_doc_snippets.py
python docs\check_doc_snippets.py --run
```

`check_doc_snippets.py` parses every Python snippet in the user manual and the
web documentation and checks its `spin_dynamics` imports; `--run` (a separate
CI job) also executes them, each document top to bottom in one namespace.
Mark a snippet that cannot run in a scratch directory with
`% snippet: skip -- reason` (LaTeX) or `<!-- snippet: skip -- reason -->`
(Markdown) on the line above it.

`tests/test_physics_invariants.py` draws its inputs with Hypothesis. The
default profile is derandomized so CI is reproducible; for a longer randomized
search run it with `HYPOTHESIS_PROFILE=thorough`.

The static part of that list also runs automatically on each commit through
the repository's pre-commit hooks (ruff, API-reference and validation-matrix
regeneration, and the static snippet check). Set them up once per clone from the repository root:

```powershell
python -m pip install pre-commit
pre-commit install
```

A generator hook that rewrites its output fails the commit; stage the
regenerated file and commit again.

The coverage command enforces the floor configured in `pyproject.toml`. MyPy
analyses the whole package against the shrink-only type-debt list in
`pyproject.toml`. CI also
builds both wheel and source distributions, checks their metadata, installs
each into a fresh environment, and executes the installed `spin-dynamics`
entry point.

Use focused tests during edit loops:

```powershell
python scripts\run_impacted_tests.py
python -m unittest tests.test_motion tests.test_motion_sequence
python -m unittest tests.example_tests
```

`scripts/run_impacted_tests.py` reads `tests/test_groups.json`, adds directly
changed test modules, and performs CLI checks only for changed examples. See
[`testing_strategy.md`](testing_strategy.md) for the local, smoke, focused, and
full-validation tiers.

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
