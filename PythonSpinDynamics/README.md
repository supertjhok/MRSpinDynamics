# PythonSpinDynamics

PythonSpinDynamics is the Python port of the MATLAB spin-dynamics simulation
package in `../MATLABSpinDynamics`. The MATLAB Version 2 code remains the
numerical reference; this workspace contains the Python package, examples,
validation fixtures, tests, and user documentation.

The main Version 2 workflow port is mostly complete. The package includes
ideal, tuned, untuned, and matched-probe CPMG workflows; finite echo trains;
FID, imaging, diffusion, time-varying-field, WURST, radiation-damping, motion,
noise, inverse-Laplace analysis, and pulse-optimization helpers.

Most MATLAB-compatible Bloch workflows assume a bath of uncoupled spin-1/2
nuclei in a possibly non-uniform and time-varying \(B_0\) field; spin-spin
effects enter those workflows indirectly through effective relaxation or
field-map inputs. The `spin_dynamics.coupling` namespace is the explicit
extension for small scalar-coupled spin-1/2 systems, including low-field
J-editing, ideal TANGO-B filtering, dense Hamiltonian propagation, B0/B1
isochromat ensembles, and initial SLIC models. The `spin_dynamics.nqr`
namespace is the early quadrupolar extension for selective pulsed NQR, SLSE,
powder averaging, and two-frequency population-transfer experiments. The
package still does not attempt chemical exchange or arbitrary nonselective
multi-quantum pulse-sequence simulation.

## Documentation

- `docs/user_manual.tex` is the main LaTeX user manual, with model equations,
  workflow examples, validation notes, and an API reference.
- `docs/python_api/index.md` is the lightweight Markdown API index, including
  the generated `docs/python_api/api_reference.md` inventory.
- `docs/python_api/j_coupling.md` describes the scalar-coupled spin-1/2
  extension layer.
- `docs/python_api/nqr.md` describes the pulsed NQR extension layer.
- `docs/nqr_module_plan.md` tracks planned NQR milestones.
- `docs/matlab_mapping.md` and `docs/migration_plan.md` track MATLAB-to-Python
  mapping and remaining porting work.
- `docs/validation_results.md` records fixture comparisons and tolerance notes.
- `docs/radiation_damping.md` contains focused details for the radiation
  damping model.

Build the manual from this directory with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs docs\user_manual.tex
```

Refresh the Markdown API inventory after changing public functions, classes, or
docstrings:

```powershell
python docs\generate_api_reference.py
```

## Installation

PythonSpinDynamics requires Python 3.10 or newer. The core package depends on
NumPy 1.24 or newer and does not require MATLAB at runtime. MATLAB is only
needed when regenerating the complete MATLAB reference fixture set.

Create or activate a Python environment, then install the package in editable
mode from this directory. On Windows, prefer an environment outside a
OneDrive-synced checkout to avoid file-lock and sync overhead:

```powershell
python -m pip install -e .
```

Optional extras:

```powershell
python -m pip install -e ".[opt,plot,dev]"
```

Use `opt` for SciPy-backed optimization and inverse-Laplace solves, `plot` for
Matplotlib examples, and `dev` for test/lint tooling.

For development and validation, use the editable install with all common extras:

```powershell
python -m pip install -e ".[dev,opt,plot,bench]"
python -m unittest tests.smoke_tests
python -m unittest tests.fixture_tests
python -m unittest tests.example_tests
python -m unittest discover -s tests
python -m ruff check src tests examples
```

If `python` resolves to an interpreter without NumPy, activate the intended
environment first or invoke that environment's full `python.exe` path.

## Quick Start

```python
from spin_dynamics.workflows import run_tuned_cpmg

result = run_tuned_cpmg(numpts=101, maxoffs=10)
print(result.echo.shape, result.snr)
```

The most stable high-level imports are listed in
`spin_dynamics.workflows.STABLE_WORKFLOW_API`. More specialized imaging,
diffusion, WURST, time-varying-field, and sweep helpers remain available from
`spin_dynamics.workflows`, but new code may prefer direct submodule imports
when using those advanced workflows.

Finite train example:

```python
from spin_dynamics.workflows import run_matched_cpmg_train

train = run_matched_cpmg_train(
    numpts=51,
    num_echoes=8,
    auto_refine_grid=True,
)
```

Radiation-damping FID example:

```python
from spin_dynamics.workflows import run_radiation_damping_fid

fid = run_radiation_damping_fid(
    probe="matched",
    fill_factor=0.7,
    equilibrium_magnetization=0.8,
    flip_angle=1.0,
)
```

Pulsed NQR example:

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

## Examples

Examples live in `examples/` and can be run from this directory:

```powershell
python examples\ideal_cpmg.py --numpts 101
python examples\ideal_cpmg_train.py --numpts 101 --num-echoes 8
python examples\probe_cpmg_compare.py --numpts 101
python examples\matched_diffusion_cpmg.py --numpts 21 --num-echoes 3
python examples\radiation_damping_fid.py --probe matched --points 401
python examples\plot_inverse_laplace.py --output results\inverse_laplace.png
python examples\plot_nqr_powder_nutation.py --output results\nqr_powder_nutation.png
python examples\plot_nqr_population_transfer.py --output results\nqr_population_transfer.png
```

See the user manual and `docs/python_api/examples.md` for the full example
catalog.

## Tests

Run the fast smoke tier during normal edit loops:

```powershell
python -m unittest tests.smoke_tests
```

Run focused tiers when touching reference parity or examples:

```powershell
python -m unittest tests.fixture_tests
python -m unittest tests.example_tests
```

Run the full validation suite before committing numerical or public-workflow
changes:

```powershell
python -m unittest discover -s tests
```

Fixture generation scripts are in `validation/octave/`. MATLAB is required for
the complete matched-probe fixture set; Octave can regenerate the smaller
dependency-light fixtures.
