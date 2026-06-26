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

## Layout

- `src/quadrupolar_dft/tensors.py` implements EFG tensor conventions,
  principal-axis sorting, asymmetry, and tensor averaging.
- `src/quadrupolar_dft/quadrupolar.py` implements `C_Q` and NQR transition
  calculations.
- `src/quadrupolar_dft/abinit.py` contains the first ABINIT parser and input
  block helper.
- `examples/parse_abinit_efg.py` shows how to parse an ABINIT output file.
- `tests/` contains focused unit tests that do not require ABINIT.

## Near-Term Milestones

- Add CIF-to-ABINIT structure generation.
- Add convergence-study manifests for cutoff, PAW fine grid, and k-point mesh.
- Add backend-neutral result files with structure, pseudopotential, functional,
  and convergence provenance.
- Add finite-temperature tensor averaging over phonon or AIMD snapshots.
