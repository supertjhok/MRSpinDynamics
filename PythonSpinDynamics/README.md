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
diffusion, imaging, analysis, optimal-control pulse design, radio-frequency
interference modeling, and zero/ultra-low-field J-coupled spectroscopy tools.

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
- explore scalar-coupled spin systems, from spin-1/2 J-editing and
  SLIC/TANGO-style filters to multinuclear mixed-spin (e.g. 1H/19F/14N)
  zero/ultra-low-field and Earth's-field J-coupled spectra, including
  quadrupolar (e.g. 14N) relaxation broadening and self-decoupling;
- study rotating-frame (T1-rho) spin-lock relaxation dispersion and
  prepolarized low-field workflows;
- model pulsed NQR responses for quadrupolar nuclei, including powder
  averaging, weak static-field splitting, SLSE echo trains, and
  population-transfer examples;
- model single-electron ESR/EPR spectra, anisotropic g tensors, hyperfine
  doublets, and pulsed FID/Hahn-echo responses;
- model diamond NV-minus and 4H-SiC PL6 defect sensors, including
  ZFS-plus-Zeeman ODMR, point-dipole coupling, phase-aware Ramsey/Hahn/CPMG/XY
  control, ideal and finite-pulse filter functions, addressed-qubit
  propagation, optical photon-count readout, analytic layers and half-spaces,
  voxel isotope densities, multi-isotope statistical spectra, and Gaussian
  filter-overlap coherence; clocked Qdyne, ENDOR-QDyne, and synchronized I/Q
  readout, separately budgeted sensor/sample/diffusion/memory coherence, effective
  sensor-memory correlation, coherent thermal chemical-shift/J spectra, and
  optional explicit DNP build-up/decay; plus unified statistical-spectrum and
  Qdyne experiment specs, nested TOML, result archives, and adaptive Qdyne;
- design radio-frequency and gradient pulses with gradient-ascent optimal
  control (GRAPE), including robust/ensemble and NQR/quadrupolar targets;
- design cylindrical MRI gradient-coil current distributions and stream
  functions with a KCL-constrained, Tikhonov-regularized Biot-Savart inverse
  solve, L-curve regularization selection, and periodic 3-D winding-contour
  extraction;
- model radio-frequency interference (RFI) pickup and cancellation for
  low-field NQR detection.

The package is not intended to be a general-purpose arbitrary quantum
pulse-sequence simulator. The original MATLAB-compatible NMR workflows mostly
use baths of uncoupled spin-1/2 nuclei, with spin-spin effects represented
through relaxation, field maps, exchange models, or explicit small-system
extensions.

## Main Areas

- `spin_dynamics.composition` provides typed boundaries between advanced
  component APIs: named spatial grids in metres, absolute time axes in seconds,
  explicit channel units, spatial/time resampling, and adapters for field
  solutions, thermal states, flow fields, hardware responses, and compiled
  sequence timelines. See
  [`docs/python_api/composition.md`](docs/python_api/composition.md).
- `spin_dynamics.experiment` is the unified, declarative facade and the
  recommended entry point for new code. You describe an experiment with
  frozen-dataclass specs (sample, hardware, sequence, acquisition), front-load
  validation with `plan()`, run it with `run()`, and save the result with
  provenance. It delegates to the validated workflows below, so a default
  experiment reproduces the direct call exactly, and adds compatibility
  checking, runtime/memory estimation, automatic coil-geometry-to-B1 field
  wiring for imaging, reduced-vs-full NQR engine dispatch, NPZ/JSON save-load,
  and a config-driven CLI (`spin-dynamics`, with
  `python -m spin_dynamics.experiment` as the equivalent module form) across the
  NMR, imaging, NQR, and ESR engines. See
  [`docs/python_api/experiment_workflow.md`](docs/python_api/experiment_workflow.md).
- `spin_dynamics.workflows` contains the high-level NMR workflows the facade
  delegates to, such as ideal, tuned, untuned, and matched-probe CPMG
  simulations, finite echo trains, diffusion workflows, imaging workflows,
  time-varying-field examples, WURST pulses, radiation damping, motion, and
  prepolarization.
- `spin_dynamics.sequences` contains timing helpers plus a backend-neutral
  block/event sequence IR. An explicit transmit/receive hardware-effects policy
  lets a backend distinguish ideal execution from required probe realization
  without baking filtered waveforms into the sequence. It can import core
  Pulseq 1.4/1.5 text `.seq` files,
  export raster-validated Pulseq 1.5.0 sequences,
  compile concurrent RF/gradient/ADC events to a piecewise-constant timeline,
  visualize aligned RF/gradient/ADC lanes, and adapt that timeline to the
  moving-isochromat engine. General native or Pulseq-imported IR can be planned,
  executed, archived, and reproduced through `spin_dynamics.experiment`. See
  [`docs/python_api/sequence_ir.md`](docs/python_api/sequence_ir.md).
- `spin_dynamics.core`, `fields`, `probes`, `sequences`, and `parameters`
  provide lower-level numerical pieces used by the workflows. `fields` also
  includes magnetostatic, quasistatic eddy-current, and induced-electric-field
  solvers for coils and gradient drivers, cylindrical stream-function gradient
  coil design with active shielding, winding engineering metrics, and
  PEEC/eddy/imaging adapters, including complete shared-layer XYZ sets and
  conducting-cylinder time-response prediction, plus a nonlinear magnetostatic solver
  (`nonlinear_magnetostatics`) for flux-shaping materials -- high-permeability RF
  ferrites and saturable iron -- in planar and axisymmetric geometries, and a 3D
  reduced-scalar-potential solver (`scalar_potential_3d`) for asymmetric magnet
  and soft-magnetic geometries.
- `spin_dynamics.thermal` couples electrical losses to lumped and spatial
  thermal models. It includes electromagnets as time-dependent B0 sources,
  predicting RL ramping, Joule warm-up, resistance and field drift, supply
  limits, and temperature/current/field-lock stabilization, with adapters to
  the existing hardware and imaging field-map workflows.
- `spin_dynamics.analysis` contains inverse-Laplace and regularization helpers
  for relaxation, diffusion, and exchange-map analysis.
- `spin_dynamics.relaxation` contains microscopic relaxation models: BPP scalar
  rates, dipolar and quadrupolar rate/Redfield models, and wall-collision
  relaxation.
- `spin_dynamics.coupling` contains explicit small-system scalar-coupling
  models: spin-1/2 J-editing, SLIC, and TANGO filters, plus multinuclear
  mixed-spin (spin-1/2 with quadrupolar, e.g. 1H/19F/14N) zero/ultra-low-field
  J-coupled spectra with a per-spin relaxation superoperator.
- `spin_dynamics.optimal_control` contains gradient-ascent (GRAPE)
  optimal-control pulse design for RF amplitude/phase and gradient waveforms,
  with state-transfer, propagator-fidelity, and robust/ensemble objectives.
- `spin_dynamics.interference` contains radio-frequency-interference source,
  pickup, and cancellation models (adaptive, frequency-domain, robust, and
  reference-free tracker cancellers) for low-field detection.
- `spin_dynamics.nqr` contains quadrupolar NQR models. Embedded two-level
  selective-pulse workflows are spin-1; full spin-3/2 and higher-spin
  (5/2, 7/2, 9/2) FID, echo, and SLSE helpers use a `(2I+1)`-level
  density-matrix model.
- `spin_dynamics.esr` contains single-electron ESR/EPR spectrum and pulse
  response helpers.
- `spin_dynamics.nano_mr` contains the defect-spin nanoscale-MR foundation:
  explicit coordinate frames and surfaces, arbitrary-spin ZFS and Zeeman
  Hamiltonians, ODMR transitions, point nuclei, diamond NV-minus and 4H-SiC
  PL6 presets, dynamical-decoupling sequences and filter functions,
  addressed-qubit control propagation, optical photon-count readout, explicit
  statistical/thermal/fixed-polarization nuclear baths, analytic layers and
  half-spaces, voxel densities, multi-isotope spectra, and Gaussian
  filter-overlap coherence; exact small clusters with chemical shift, scalar
  and nuclear dipolar coupling, anisotropic sensor-target coupling, nuclear RF,
  CW transitions, and two-block 2-D correlation spectroscopy; plus strict
  conversion of compatible spin-1/2 zero-ZFS models to and from the pure ESR
  module; and seeded Brownian/advection trajectories with reflecting,
  periodic, or clipping confinement, precessing nuclear dipolar field records,
  autocorrelations, Welch spectra, and displacement diagnostics; plus raster,
  arbitrary-path, and sensor-array nano-MRI scans, coherent-field and
  statistical-variance dipolar forward operators, analytic depth profiling,
  nonnegative regularized density reconstruction, bounded sparse point-source
  localization, and local uncertainty estimates; plus correlated
  target/sensor field processes, five-level-plus-charge optical cycling,
  shot-resolved emission and SPAD transfer, and covariance-aware generalized
  least-squares reconstruction. Calibration fitting, additional detector
  types, high-resolution protocol refinements, and an experiment-level facade
  remain staged in the linked implementation plan.
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

On a OneDrive-synced checkout, do not use the default `.venv-wsl` at the
repository root; it would be synced by OneDrive. Put the environment under
`$HOME` instead (see the note below).

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

Keep the source tree here and pass an external virtual-environment path to the
same setup scripts to avoid OneDrive sync and slow `/mnt/c` access. On WSL, put
the environment under `$HOME` on the native Linux filesystem:

```bash
VENV="$HOME/.venvs/python-spin-dynamics" bash scripts/setup_dev_env_wsl.sh
source "$HOME/.venvs/python-spin-dynamics/bin/activate"
```

See `docs/development_environment.md` for the full workflow.

## Quick Start

The recommended entry point is the `spin_dynamics.experiment` facade. Describe
the experiment, inspect the plan (resolved workflow, compatibility checks, and a
runtime/memory estimate) before running, then run and save with provenance:

```python
from spin_dynamics.experiment import Experiment, CPMGTrain, Sample, Hardware, Acquisition

study = Experiment(
    sequence=CPMGTrain(num_echoes=8),
    sample=Sample(t1_seconds=2.0, t2_seconds=2.0),
    hardware=Hardware(probe="tuned"),
    acquisition=Acquisition(numpts=501, maxoffs=10.0),
)

print(study.plan().report())     # resolved workflow, warnings, cost estimate
record = study.run()             # delegates to run_tuned_cpmg_train
record.save("run1.npz")          # arrays + JSON provenance + spec round-trip
```

Saved runs carry canonical SHA-256 identities for the complete experiment and
native result, callable/module/package-tree hashes, Git revision/dirty state,
NumPy/SciPy/build/thread environment, and seeded/unseeded randomness status.
`load_run("run1.npz").verify_reproduction()` reruns the stored experiment and
reports whether the exact result identity was reproduced; the CLI equivalent is
`spin-dynamics verify run1.npz`.

The same interface drives imaging (with automatic transmit-coil B1 solving),
NQR (`NQRFID`, `NQRPopulationTransfer`, `NQRSLSE`, and `NQRSORC`) and ESR
(`ESRCWSweep`, `ESRDEER`, pulsed FID/Hahn, ESEEM, 2-D HYSCORE, and
Davies/Mims ENDOR). See
[`docs/python_api/experiment_workflow.md`](docs/python_api/experiment_workflow.md)
for the full guide.

You can also drive it from a TOML or JSON config file with the CLI:

```powershell
spin-dynamics plan examples\experiment_config_cpmg.toml
spin-dynamics run  examples\experiment_config_cpmg.toml -o run.npz
spin-dynamics show run.npz
```

The console command is installed by the package; `python -m
spin_dynamics.experiment` remains equivalent.

The subcommands are `plan` (resolve and validate; non-zero exit on plan
errors), `run` (plan then run, refusing an erroring plan, with an optional
`-o` result archive), `show` (print a saved run's spec and provenance), and
`convert` (rewrite a config between `.toml` and `.json`). Configs are the
human-friendly form where each spec is a table tagged by its `type`; the
`sample`, `hardware`, and `acquisition` sections imply their class and may be
omitted to accept defaults.

### Working with the underlying workflows directly

If you already know which workflow you want, the validated runners remain fully
supported. Run a simple tuned-probe CPMG simulation:

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

Run an Earth's-field multinuclear J-coupled spectrum with 14N quadrupolar
relaxation derived from the quadrupole coupling and rotational correlation time:

```python
import numpy as np
from spin_dynamics.coupling import (
    multinuclear_system,
    multinuclear_quadrupolar_rates,
    simulate_zulf_spectrum,
)

couplings = np.zeros((3, 3))
couplings[0, 1] = couplings[1, 0] = 8.0    # J(1H, 19F)
couplings[1, 2] = couplings[2, 1] = 37.0   # J(19F, 14N)
system = multinuclear_system(["1H", "19F", "14N"], couplings, b0_tesla=50e-6)

r1, r2 = multinuclear_quadrupolar_rates(
    system,
    correlation_time_seconds=3e-12,
    quadrupole_coupling_hz=4.0e6,
    asymmetry=0.4,
)
spectrum = simulate_zulf_spectrum(
    system, r1_per_second=r1, r2_per_second=r2,
    dwell_seconds=2e-4, n_points=32768,
    detect_indices=system.indices_for_isotope("19F"),
)
print(spectrum.frequencies_hz.shape, spectrum.spectrum.shape)
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
python examples\experiment_sequence_ir.py --timeline-output results\sequence_ir.png --output results\sequence_ir.npz
python examples\plot_shim_a_ring_magnet.py --output results\shim_a_ring.png
python examples\plot_logging_ferrite_b1_focusing.py --output results\logging_ferrite_b1.png
python examples\plot_zulf_quadrupolar_jcoupling.py --output results\zulf_jcoupling.png
python examples\plot_zulf_quadrupolar_relaxation.py --output results\zulf_quadrupolar_relaxation.png
python examples\plot_t1rho_prepolarized_dispersion.py --output results\t1rho_dispersion.png
python examples\plot_portable_halbach_adaptive_mri.py --output results\portable_halbach_adaptive_mri.png
```

The full example catalog is documented in `docs/python_api/examples.md`.

## Documentation

- The searchable documentation site is published at
  <https://supertjhok.github.io/MRSpinDynamics/>. It includes the web manual,
  generated API reference, examples guide, validation evidence, and a
  downloadable PDF print edition.
- `docs/user_manual.tex` is the LaTeX user manual with model equations,
  examples, validation notes, and an API reference.
- `docs/python_api/index.md` is the Markdown documentation index.
- `docs/python_api/experiment_workflow.md` is the getting-started guide for the
  unified experiment facade (the recommended entry point) and its CLI.
- `docs/python_api/api_reference.md` is generated from public functions,
  classes, and docstrings.
- `docs/validation_matrix.md` is generated from
  `validation/evidence.json` and records claim-level evidence, ranges,
  tolerances, reproducers, and limitations.
- `docs/nqr_powder_relaxation_methods.md` records the averaging order, receiver
  model, convergence checks, and parallel workflow required for realistic
  NQR--NMR powder relaxation calculations.
- `docs/python_api/concepts.md` describes units and conventions.
- `docs/python_api/workflows.md`, `nqr.md`, `esr.md`, `j_coupling.md`,
  `exchange.md`, and `internal_gradients.md` describe major feature areas.
- `docs/matlab_mapping.md`, `docs/migration_plan.md`, and
  `docs/validation_results.md` document the MATLAB-to-Python port, fixture
  parity checks, and historical run log.

Build the manual from its documentation directory so relative asset paths
resolve correctly:

```powershell
cd docs
pdflatex -interaction=nonstopmode -halt-on-error user_manual.tex
```

The example-result figures embedded in the manual are reproducible from the
same plotting scripts documented in their captions. Regenerate all of them, or
one named figure, with:

```powershell
python docs\generate_manual_figures.py
python docs\generate_manual_figures.py nqr-nmr-crossover
```

Refresh the generated Markdown API inventory after changing public functions,
classes, or docstrings:

```powershell
python docs\generate_api_reference.py
python docs\generate_validation_matrix.py
```

## Tests And Validation

Use the change-aware selector for normal edit loops. It combines the short
smoke tier with test groups inferred from changed paths and checks only example
scripts that changed:

```powershell
python scripts\run_impacted_tests.py
python scripts\run_impacted_tests.py --list
python scripts\run_impacted_tests.py --group nqr
```

See [`docs/testing_strategy.md`](docs/testing_strategy.md) for the tiering and
impact-map maintenance policy.

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
python docs\generate_validation_matrix.py --check
```

Fixture generation scripts are in `validation/octave/`. MATLAB is required for
the complete matched-probe fixture set; Octave can regenerate the smaller
dependency-light fixtures.
