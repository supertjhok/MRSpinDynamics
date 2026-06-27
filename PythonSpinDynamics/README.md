# PythonSpinDynamics

PythonSpinDynamics is the Python package workspace within MRSpinDynamics. It
began as the Python port of the MATLAB NMR spin-dynamics package in
`../MATLABSpinDynamics`, and now also hosts Python-native quadrupolar NQR
models and single-electron ESR/EPR helpers. The MATLAB Version 2 code remains
the numerical reference for the
validated NMR Bloch workflows; this workspace contains the Python package,
examples, validation fixtures, tests, and user documentation.

The main Version 2 workflow port is mostly complete. The package includes
ideal, tuned, untuned, and matched-probe CPMG workflows; finite echo trains;
FID, imaging, diffusion, time-varying-field, WURST, radiation-damping, motion,
prepolarization, BPP-style relaxation, noise, inverse-Laplace analysis, PGSE
D-T2 examples, and pulse-optimization helpers.

Most MATLAB-compatible Bloch workflows still assume a bath of uncoupled
spin-1/2 nuclei in a possibly non-uniform and time-varying \(B_0\) field;
spin-spin effects enter those workflows indirectly through effective relaxation
or field-map inputs. The `spin_dynamics.coupling` namespace is the explicit
extension for small scalar-coupled spin-1/2 systems, including low-field
J-editing, ideal TANGO-B filtering, dense Hamiltonian propagation, B0/B1
isochromat ensembles, and initial SLIC models. The `spin_dynamics.nqr`
namespace is the quadrupolar extension for selective pulsed NQR, SLSE, powder
averaging, EFG inhomogeneity, weak-B0 spectra, two-frequency
population-transfer experiments, and polarization-enhanced NQR transport with
CIF-based proton-coupling estimates. The embedded two-level selective-pulse
workflows are spin-1; spin-3/2 (chlorine-style) FID, echo, and SLSE -- whose
single zero-field line connects two degenerate Kramers doublets -- run on the
full `(2I+1)`-level density-matrix model (`simulate_full_fid`,
`simulate_full_echo`, `simulate_full_slse`), with powder averaging and weak
Zeeman perturbations. The `spin_dynamics.esr` namespace is the single-electron
ESR/EPR extension for anisotropic g-tensor spectra, CW derivative display,
static disorder, pulsed FID/Hahn echo simulations with T1/T2 relaxation, and
first isotropic electron-nuclear hyperfine doublets. The
`spin_dynamics.exchange` namespace adds Bloch-McConnell site/chemical exchange:
multi-site kinetic magnetization transfer with per-site `T1`/`T2` and offset,
lineshape coalescence, and encode-mix-detect `T2`-`T2` relaxation exchange
(REXSY) data that inverts to an exchange map through the existing 2D
inverse-Laplace solver. The `spin_dynamics.susceptibility` namespace generates
the internal field from magnetic-susceptibility contrast in porous media:
analytic 2D cylindrical-grain off-resonance maps that drop into the moving-
isochromat pipeline, plus pore-space internal-gradient distributions for
diffusion-in-internal-gradient studies. The bipolar 13-interval PGSTE workflow
(`spin_dynamics.workflows.bipolar`, with the Bruker `diff_stebp` 16-step phase
cycle) suppresses the background-gradient cross-term that would otherwise bias
those diffusion measurements. The package still does not attempt arbitrary
nonselective multi-quantum pulse-sequence simulation.

## Documentation

- `docs/user_manual.tex` is the main LaTeX user manual, with model equations,
  workflow examples, validation notes, and an API reference.
- `docs/python_api/index.md` is the lightweight Markdown API index, including
  the generated `docs/python_api/api_reference.md` inventory.
- `docs/python_api/j_coupling.md` describes the scalar-coupled spin-1/2
  extension layer.
- `docs/python_api/nqr.md` describes the pulsed NQR extension layer.
- `docs/python_api/esr.md` describes the ESR/EPR extension layer.
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

Spin-3/2 nuclei such as `35Cl` and `37Cl` are supported at the Hamiltonian and
transition-frequency metadata level. Full spin-3/2 pulsed SLSE response still
requires a degenerate-doublet RF manifold model, so the bundled pulsed NQR
examples state that they are spin-1 only.
Weak static B0 line splitting is available for both spin-1 and spin-3/2 through
`simulate_weak_b0_spectrum`, with an explicit `|gamma B0| / nu_ref` weak-field
regime check.

ESR example:

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

Examples live in `examples/` and can be run from this directory:

```powershell
python examples\ideal_cpmg.py --numpts 101
python examples\ideal_cpmg_train.py --numpts 101 --num-echoes 8
python examples\probe_cpmg_compare.py --numpts 101
python examples\matched_diffusion_cpmg.py --numpts 21 --num-echoes 3
python examples\radiation_damping_fid.py --probe matched --points 401
python examples\plot_inverse_laplace.py --output results\inverse_laplace.png
python examples\plot_pgse_d_t2.py --output results\pgse_d_t2.png
python examples\plot_dexsy_exchange.py --output results\dexsy_exchange.png
python examples\plot_t2_t2_exchange.py --output results\t2_t2_exchange.png
python examples\plot_internal_gradients.py --output results\internal_gradients.png
python examples\plot_bipolar_pgste.py --output results\bipolar_pgste.png
python examples\plot_cpmg_pipe_flow.py --output results\cpmg_pipe_flow.png
python examples\plot_bpp_relaxation_temperature.py --output results\bpp_relaxation_temperature.png
python examples\plot_t1rho_prepolarized_dispersion.py --output results\t1rho_prepolarized_dispersion.png
python examples\plot_earth_field_prepolarized_nmr.py --output results\earth_field_prepolarized_nmr.png
python examples\plot_nqr_powder_nutation.py --output results\nqr_powder_nutation.png
python examples\plot_nqr_population_transfer.py --output results\nqr_population_transfer.png
python examples\plot_nqr_slse_offset.py --output results\nqr_slse_offset.png
python examples\plot_nqr_slse_spacing.py --output results\nqr_slse_spacing.png
python examples\plot_nqr_efg_broadening.py --output results\nqr_efg_broadening.png
python examples\plot_nqr_temperature_broadening.py --output results\nqr_temperature_broadening.png
python examples\plot_nqr_slse_efg_broadening.py --output results\nqr_slse_efg_broadening.png
python examples\plot_nqr_weak_b0_spectrum.py --output results\nqr_weak_b0_spectrum.png
python examples\plot_nqr_spin32_slse.py --output results\nqr_spin32_slse.png
python examples\plot_halbach_dipole_field.py --output results\halbach_dipole.png
python examples\plot_nqr_polarization_enhancement.py --output results\nqr_polarization_enhancement.png
python examples\plot_esr_powder_spectrum.py --output results\esr_powder_spectrum.png
python examples\plot_esr_pulsed_echo.py --output results\esr_pulsed_echo.png
python examples\plot_esr_relaxation.py --output results\esr_relaxation.png
python examples\plot_esr_hyperfine_doublet.py --output results\esr_hyperfine_doublet.png
```

The SLSE EFG broadening plot forms its spectrum from a finite acquired echo
window. Add `--noise-snr 20 --deconvolve` to inspect time-domain receiver noise
and regularized finite-window deconvolution.

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
