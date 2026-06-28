# QuadrupolarDFT

QuadrupolarDFT is the ab initio electric-field-gradient workspace within
MRSpinDynamics. It starts with an ABINIT-centered static EFG workflow and keeps
the analysis layer backend-neutral so Quantum ESPRESSO/GIPAW, Elk, CP2K, or
snapshot-averaging pipelines can be added later without changing tensor
conventions downstream.

The first practical target is:

1. Generate or document ABINIT PAW inputs with `nucefg 2` and isotope-specific
   `quadmom` values.
2. Parse ABINIT EFG output into structured records.
3. Convert EFG tensors to principal components, asymmetry `eta`, quadrupolar
   coupling constants `C_Q`, and zero-field NQR transition frequencies.
4. Preserve enough provenance to support convergence studies and later
   finite-temperature tensor averaging.

ABINIT is the first backend because its PAW EFG route is native and direct:
the `nucefg` keyword reports EFG tensors, principal values and axes, and, with
`quadmom`, the quadrupolar coupling in MHz. Norm-conserving pseudopotentials are
not appropriate for this calculation; use PAW datasets and converge the EFG
components, not only the total energy.

## Documentation

The full user manual — physical scope and conventions, the finite-temperature
theory, the three-stage DFPT workflow, the NaNO2 worked example, and the API
reference — is in [`docs/user_manual.pdf`](docs/user_manual.pdf), with the LaTeX
source tracked beside it. This README is a quick entry point.

## Installation

From this directory:

```powershell
python -m pip install -e .
```

For development checks:

```powershell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
python -m ruff check src tests examples
```

## Quick Start

```python
from quadrupolar_dft import EFGTensor, coupling_constant_hz, nqr_frequencies_hz

efg = EFGTensor.from_components([
    [-0.236543, 0.864057, 0.0],
    [0.864057, -0.236543, 0.0],
    [0.0, 0.0, 0.473085],
])

vzz_si = efg.principal_components_si[2]
cq_hz = coupling_constant_hz(vzz_si, quadrupole_moment_barns=-0.02558)
print(cq_hz / 1e6)
print(nqr_frequencies_hz(spin=2.5, cq_hz=cq_hz, eta=efg.eta))
```

The tensor constructor assumes ABINIT atomic units by default. Pass
`unit="si"` when the tensor is already in V/m^2.

## Finite-Temperature EFG

A static DFT EFG is a 0 K, fixed-geometry quantity, but NQR frequencies are
measured at finite temperature and are strongly temperature dependent. Two
distinct corrections bridge the gap: the *static* effect of thermal expansion
(use a temperature-appropriate structure) and the *dynamic* vibrational
averaging of the tensor over phonon motion (including zero-point motion, present
even at 0 K). For molecular crystals the dynamic term usually dominates --
low-frequency librations reorient the EFG principal axes and lower the line
frequency as temperature rises (the Bayer law).

`quadrupolar_dft.vibrational` implements the harmonic vibrational average,

```
<V_ij>(T) = V_ij^eq + (1/2) sum_k (d^2 V_ij / dQ_k^2) <Q_k^2>(T)
```

The tensor is averaged in the crystal frame and only then diagonalized, so the
asymmetry `eta` -- not just `V_zz` -- shifts with temperature. The thermal
physics (`quadrupolar_dft.thermal`) is unit-tested without running DFT, and
`efg_temperature_sweep` returns `C_Q(T)`, `eta(T)`, and the lines at each
temperature.

Those per-mode curvatures come from a **finite-displacement driver**
(`quadrupolar_dft.finite_displacement`) that separates the slow DFT step from
the cheap bookkeeping:

1. `parse_abinit_structure` reads the equilibrium cell and positions from a
   converged ABINIT input; `generate_displacement_jobs` builds one `+/-`
   displacement per phonon mode (correct mass-weighted normal-coordinate
   physics); `write_jobs` emits one displaced `.abi` per job plus a
   `manifest.json` (every other setting -- `nucefg`, `quadmom`, cutoffs,
   k-points -- is preserved verbatim).
2. You run those inputs with ABINIT locally.
3. `vibrational_modes_from_efg` central-differences the per-job target-nucleus
   EFGs into `VibrationalMode` curvatures for the sweep.

### Running the full workflow with ABINIT

`examples/abinit/efg_temperature.py` drives the three stages against real ABINIT
runs (no synthetic data). Phonon eigenvectors come from an ABINIT DFPT + anaddb
calculation:

```bash
# 1. stage and run the phonon calculation (ABINIT DFPT + anaddb)
python examples/abinit/efg_temperature.py phonon \
    --base examples/abinit/nano2_efg.abi --out runs/nano2_ph
bash examples/abinit/run_phonon_wsl.sh runs/nano2_ph
#    -> writes runs/nano2_ph/anaddb.out with the phonon eigenvectors

# 2. stage displaced EFG inputs from the phonon eigenvectors
python examples/abinit/efg_temperature.py displace \
    --base examples/abinit/nano2_efg.abi --anaddb runs/nano2_ph/anaddb.out \
    --target 2 --max-modes 6 --out runs/nano2_disp
#    run ABINIT EFG on every runs/nano2_disp/*.abi:
bash examples/abinit/run_finite_displacement_wsl.sh runs/nano2_disp

# 3. collect the real EFG outputs into C_Q(T), eta(T), nu(T), dnu/dT
python examples/abinit/efg_temperature.py collect \
    --workdir runs/nano2_disp --temperatures 0,77,150,300 --quadmom 0.02044
```

The whole chain has been run against real ABINIT 9.10.4 output for NaNO2 14N
(the parser, collect path, and temperature sweep all validated on real `.abo`
and `anaddb.out` files). Two notes: the generated DFPT input is a starting
template (verify the tolerances/k-mesh for your cell), and if your anaddb
eigenvector layout differs, pass `--modes modes.json`
(`{"wavenumbers_cm_inv": [...], "eigendisplacements": [[[x,y,z],...]], ...}`,
or build it with `modes_from_arrays`) to drive the displacements directly.
`examples/finite_displacement_workflow.py` is an in-process plumbing smoke test
(no ABINIT) for CI.

Where per-mode EFG curvatures are not available, `fit_bayer_single_mode` fits
the analytic single-libration limit `nu(T) = nu0 (1 - a coth(hbar omega/2kT))`
directly to a measured line series. The worked example validates this against
the NaNO2 14N temperature data from the NQR database:

```powershell
python examples/nano2_temperature_efg.py
```

It recovers a physical ~210 cm^-1 librational frequency and reproduces the sign
and rough magnitude of the measured `dnu/dT` (the residual gap is NaNO2's
ferroelectric-transition softening, which a single harmonic mode cannot capture
-- the case for escalating to AIMD/PIMD averaging).

## Technical Note

For the derivation and practical DFT context behind this workspace, see the
self-authored note
[Ab Initio Electric-Field Gradients and Quadrupolar Resonances in Crystals](../References/efg_quadrupolar_technical_note.pdf).
It summarizes EFG tensor conventions, quadrupolar coupling constants,
asymmetry parameters, NQR transition frequencies, and backend considerations
for ABINIT, Quantum ESPRESSO/GIPAW, Elk, CP2K, and related workflows. The LaTeX
source is tracked at
`../References/efg_quadrupolar_technical_note.tex`.

## Layout

- `src/quadrupolar_dft/tensors.py` implements EFG tensor conventions,
  principal-axis sorting, asymmetry, and tensor averaging.
- `src/quadrupolar_dft/quadrupolar.py` implements `C_Q` and NQR transition
  calculations.
- `src/quadrupolar_dft/abinit.py` contains the first ABINIT parser and input
  block helper.
- `src/quadrupolar_dft/thermal.py` implements harmonic phonon occupation and
  vibrational amplitudes (`coth(hbar omega / 2kT)`).
- `src/quadrupolar_dft/vibrational.py` implements harmonic EFG tensor averaging,
  finite-displacement curvatures, the temperature sweep, and the analytic Bayer
  fit.
- `src/quadrupolar_dft/finite_displacement.py` implements the displaced-structure
  generator, ABINIT structure parse/write, manifest, and EFG-to-curvature
  collection (including parsing real ABINIT `.abo` outputs).
- `src/quadrupolar_dft/abinit_phonon.py` generates the DFPT phonon and anaddb
  inputs and parses phonon eigenvectors into modes.
- `examples/abinit/efg_temperature.py` is the three-stage CLI (phonon ->
  displace -> collect); `examples/abinit/run_phonon_wsl.sh` runs the DFPT phonon
  calculation plus anaddb, and `examples/abinit/run_finite_displacement_wsl.sh`
  runs ABINIT EFG over the displaced inputs.
- `examples/parse_abinit_efg.py` shows how to parse an ABINIT output file;
  `examples/nano2_temperature_efg.py` validates finite-temperature averaging
  against measured NaNO2 14N data; `examples/finite_displacement_workflow.py`
  runs the generate -> collect -> sweep chain.
- `tests/` contains focused unit tests that do not require ABINIT.

## Near-Term Milestones

- Add CIF-to-ABINIT structure generation.
- Add convergence-study manifests for cutoff, PAW fine grid, and k-point mesh.
- Add backend-neutral result files with structure, pseudopotential, functional,
  and convergence provenance.
- Finite-temperature tensor averaging is done end to end and validated on real
  ABINIT 9.10.4 output: the harmonic average, the finite-displacement driver, and
  the three-stage DFPT workflow (phonon -> displace -> collect). Next, in order
  of impact: a structure-relaxation stage (so the geometry is at an energy
  minimum -- removing imaginary modes and correcting `eta`), then a netCDF/phonopy
  phonon reader, and AIMD/PIMD snapshot averaging for strongly anharmonic cases
  such as NaNO2 near its ferroelectric transition.
