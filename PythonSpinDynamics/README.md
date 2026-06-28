# PythonSpinDynamics

PythonSpinDynamics is the Python simulation package in the MRSpinDynamics
repository. It models magnetic-resonance experiments in which spin ensembles
evolve under magnetic fields, radio-frequency pulses, relaxation, motion,
diffusion, exchange, and probe imperfections.

The package started as a Python port of the MATLAB NMR code in
`../MATLABSpinDynamics`. That MATLAB implementation remains the numerical
reference for the validated nuclear magnetic resonance (NMR) workflows. The
Python package now also includes newer nuclear quadrupole resonance (NQR),
electron spin resonance/electron paramagnetic resonance (ESR/EPR), exchange,
diffusion, imaging, and analysis tools.

This workspace contains the installable Python package, examples, tests,
MATLAB/Octave validation fixtures, and user documentation.

## What It Is For

Use PythonSpinDynamics when you want to:

- simulate Carr-Purcell-Meiboom-Gill (CPMG) echo trains, free-induction decays,
  inversion-recovery trains, and related NMR pulse workflows;
- compare ideal pulses with tuned, untuned, and matched radio-frequency probe
  models;
- study how non-uniform static fields, transmit/receive fields, diffusion,
  flow, motion, or susceptibility gradients affect measured signals;
- simulate magnetic-resonance imaging examples, including spin-warp, RARE,
  slice-selective, and single-sided-field workflows;
- run inverse-Laplace analyses for T1, T2, T1-T2, D-T2, and exchange maps;
- explore small scalar-coupled spin-1/2 systems, including J-editing and
  simple SLIC/TANGO-style filters;
- model pulsed NQR responses for quadrupolar nuclei, including powder
  averaging, weak static-field splitting, SLSE echo trains, and
  population-transfer examples;
- model single-electron ESR/EPR spectra, anisotropic g tensors, hyperfine
  doublets, and pulsed FID/Hahn-echo responses.

The package is not intended to be a general-purpose arbitrary quantum
pulse-sequence simulator. The original MATLAB-compatible NMR workflows mostly
use baths of uncoupled spin-1/2 nuclei, with spin-spin effects represented
through relaxation, field maps, exchange models, or explicit small-system
extensions.

## Main Areas

- `spin_dynamics.workflows` contains high-level NMR workflows such as ideal,
  tuned, untuned, and matched-probe CPMG simulations, finite echo trains,
  diffusion workflows, imaging workflows, time-varying-field examples, WURST
  pulses, radiation damping, motion, and prepolarization.
- `spin_dynamics.core`, `fields`, `probes`, `sequences`, and `parameters`
  provide lower-level numerical pieces used by the workflows.
- `spin_dynamics.analysis` contains inverse-Laplace and regularization helpers
  for relaxation, diffusion, and exchange-map analysis.
- `spin_dynamics.coupling` contains explicit small-system scalar-coupling
  models for spin-1/2 nuclei.
- `spin_dynamics.nqr` contains quadrupolar NQR models. Embedded two-level
  selective-pulse workflows are spin-1; full spin-3/2 chlorine-style FID, echo,
  and SLSE helpers use a `(2I+1)`-level density-matrix model.
- `spin_dynamics.esr` contains single-electron ESR/EPR spectrum and pulse
  response helpers.
- `spin_dynamics.exchange` and `spin_dynamics.susceptibility` add
  Bloch-McConnell exchange and internal-gradient field models.

The most stable high-level imports are listed in
`spin_dynamics.workflows.STABLE_WORKFLOW_API`. Advanced workflows may be better
imported from their specific submodules.

## Installation

Python 3.10 or newer is required. The core package depends on NumPy and does
not require MATLAB at runtime. MATLAB is only needed when regenerating the full
MATLAB reference fixture set.

For development, examples, plotting, and benchmarking, use the repo-owned setup
scripts from this directory. They create a persistent OS-specific virtual
environment, install the package in editable mode, and verify the optional
numerical stack:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev_env.ps1
& ".\.venv-win\Scripts\Activate.ps1"
python scripts\verify_dev_env.py --strict
```

On WSL/Ubuntu:

```bash
bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict
```

For NVIDIA GPU JAX benchmarking in WSL:

```bash
JAX_CUDA=13 bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict --require-jax-gpu
```

The setup scripts install `.[dev,opt,plot,perf,bench]` by default:

- `opt` installs SciPy-backed optimization and inverse-Laplace tools.
- `plot` installs Matplotlib and Pillow for plotting examples.
- `dev` installs test and lint tooling.
- `perf` installs Numba and JAX for accelerated numerical backends.
- `bench` installs benchmark tooling.

CUDA-enabled JAX is installed separately by the WSL setup script when
`JAX_CUDA=13` or `JAX_CUDA=12` is set, because those `jaxlib` wheels are
Linux/driver-specific.

For a minimal runtime-only editable install:

```powershell
python -m pip install -e .
```

If OneDrive file locking or WSL `/mnt/c` performance becomes a problem, keep
the source tree here and pass an external virtual-environment path to the same
setup scripts. See `docs/development_environment.md` for the full workflow.

## Quick Start

Run a simple tuned-probe CPMG simulation:

```python
from spin_dynamics.workflows import run_tuned_cpmg

result = run_tuned_cpmg(numpts=101, maxoffs=10)
print(result.echo.shape, result.snr)
```

Run a finite echo train:

```python
from spin_dynamics.workflows import run_matched_cpmg_train

train = run_matched_cpmg_train(
    numpts=51,
    num_echoes=8,
    auto_refine_grid=True,
)
print(train.echo.shape)
```

Run a pulsed NQR SLSE example:

```python
from spin_dynamics.nqr import QuadrupolarSite, simulate_slse, slse_sequence

site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
sequence = slse_sequence(
    "x",
    pulse_duration_seconds=25e-6,
    nutation_hz=10e3,
    echo_spacing_seconds=1e-3,
    num_echoes=8,
)

slse = simulate_slse(site, sequence, orientations="powder", t2e_seconds=20e-3)
print(slse.echo_amplitudes.shape)
```

Run an ESR/EPR powder spectrum:

```python
from spin_dynamics.esr import ESRSpinSystem, simulate_field_sweep

system = ESRSpinSystem(g_tensor=[2.00, 2.08, 2.24])
spectrum = simulate_field_sweep(
    system,
    microwave_frequency_hz=9.5e9,
    orientations="powder",
    detection_mode="derivative",
)
print(spectrum.fields_tesla.shape)
```

## Examples

Examples live in `examples/` and can be run from this directory. A few useful
entry points are:

```powershell
python examples\ideal_cpmg.py --numpts 101
python examples\ideal_fid.py --numpts 101
python examples\plot_ideal_workflows.py --numpts 201 --output results\ideal_workflows.png
python examples\plot_inverse_laplace.py --output results\inverse_laplace.png
python examples\plot_pgse_d_t2.py --output results\pgse_d_t2.png
python examples\porous_rock_cpmg_walkers.py --estimate-only
python examples\porous_rock_cpmg_walkers.py --backend jax --plot-output results\porous_rock_challenge.png --output results\porous_rock_challenge.npz
python examples\plot_nqr_powder_nutation.py --output results\nqr_powder_nutation.png
python examples\plot_nqr_population_transfer.py --output results\nqr_population_transfer.png
python examples\plot_esr_powder_spectrum.py --output results\esr_powder_spectrum.png
python examples\plot_esr_pulsed_echo.py --output results\esr_pulsed_echo.png
```

The full example catalog is documented in `docs/python_api/examples.md`.

## Documentation

- `docs/user_manual.tex` is the LaTeX user manual with model equations,
  examples, validation notes, and an API reference.
- `docs/python_api/index.md` is the Markdown documentation index.
- `docs/python_api/api_reference.md` is generated from public functions,
  classes, and docstrings.
- `docs/python_api/concepts.md` describes units and conventions.
- `docs/python_api/workflows.md`, `nqr.md`, `esr.md`, `j_coupling.md`,
  `exchange.md`, and `internal_gradients.md` describe major feature areas.
- `docs/matlab_mapping.md`, `docs/migration_plan.md`, and
  `docs/validation_results.md` document the MATLAB-to-Python port and fixture
  parity checks.

Build the manual from this directory with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs docs\user_manual.tex
```

Refresh the generated Markdown API inventory after changing public functions,
classes, or docstrings:

```powershell
python docs\generate_api_reference.py
```

## Tests And Validation

Run the fast smoke tier during normal edit loops:

```powershell
python -m unittest tests.smoke_tests
```

Run focused tiers when touching reference parity or examples:

```powershell
python -m unittest tests.fixture_tests
python -m unittest tests.example_tests
```

Run the broader validation suite before committing numerical or public-workflow
changes:

```powershell
python -m unittest discover -s tests
python -m ruff check src tests examples
```

Fixture generation scripts are in `validation/octave/`. MATLAB is required for
the complete matched-probe fixture set; Octave can regenerate the smaller
dependency-light fixtures.
