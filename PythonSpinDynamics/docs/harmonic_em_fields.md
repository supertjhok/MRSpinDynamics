# Harmonic electromagnetic field interchange

`HarmonicEMSolution` is the solver-neutral boundary between full-wave
electromagnetic tools and PythonSpinDynamics. It stores sampled complex fields,
not solver geometry or mesh commands, so the same downstream API can accept
openEMS, Palace, commercial-solver, or reference data.

## Contract

A solution records:

- a one-, two-, or three-dimensional rectilinear `SpatialDomain` in metres;
- positive frequency and an explicit `exp(+iwt)` or `exp(-iwt)` phasor
  convention;
- complex electric field E and at least one of magnetic flux density B or
  magnetic field H;
- optional relative-permeability data for converting an H-only result to B;
- absolute, per-ampere, per-volt, or per-square-root-watt normalization;
- port voltage, current, accepted-power, and backend-specific metadata;
- material identities and electromagnetic properties; and
- backend/version/model provenance plus numerical convergence information.

Fields use SI units before the declared normalization denominator. For example,
a per-ampere B field is in T/A and a per-square-root-watt E field is in
V/m/sqrt(W). Arrays have shape `domain.shape + (3,)`, with Cartesian vector
components on the final axis.

The object converts imported fields to the package's canonical `exp(+iwt)`
phasor before deriving circular components. For B0 along +z,
`b1_plus()` returns `(Bx + 1j*By)/2` and `b1_minus()` returns
`(Bx - 1j*By)/2`.

## Example

```python
import numpy as np

from spin_dynamics.fields import (
    HarmonicEMSolution,
    HarmonicFieldNormalization,
    HarmonicSolverProvenance,
    SpatialDomain,
    save_harmonic_em,
)

domain = SpatialDomain(
    (
        np.linspace(-0.05, 0.05, 41),
        np.linspace(-0.05, 0.05, 41),
        np.linspace(-0.10, 0.10, 81),
    )
)
shape = domain.shape + (3,)

solution = HarmonicEMSolution(
    domain=domain,
    frequency_hz=128e6,
    phasor_convention="exp(+iwt)",
    electric_field_v_per_m=np.zeros(shape, dtype=complex),
    magnetic_flux_density_t=np.zeros(shape, dtype=complex),
    normalization=HarmonicFieldNormalization("per_ampere", port_index=0),
    provenance=HarmonicSolverProvenance(
        backend="external-reference",
        backend_version="1.0",
    ),
)

b1_plus = solution.b1_plus()
save_harmonic_em("coil_fields.npz", solution)
```

## Existing spatial-map adapter

`to_spatial_field_maps()` combines the harmonic B1 result with spin density,
relaxation, off-resonance, and optional diffusion maps. The existing
`SpatialFieldMaps` contract stores nonnegative scalar transmit/receive
sensitivities, so this adapter uses `abs(B1+)` and `abs(B1-)`. The original
complex phase is retained in the harmonic solution and remains available from
`b1_plus()` and `b1_minus()`.

Use `tx_scale` and `rx_scale` to apply the drive implied by the normalization.
For example, a T/A result with `tx_scale=2` produces the B1 magnitude for a 2 A
drive. PythonSpinDynamics does not guess this conversion.

## NPZ and HDF5 schema

`save_harmonic_em()` and `load_harmonic_em()` dispatch by `.npz`, `.h5`, or
`.hdf5` extension. Both representations carry the schema identifier
`python-spin-dynamics.harmonic-em/v1`, coordinate axes, field arrays, and JSON
metadata. Loaders reject missing fields and unknown schema versions rather than
silently interpreting incompatible data.

NPZ support only requires NumPy and loads with pickling disabled. HDF5 support
is optional:

```bash
python -m pip install "python-spin-dynamics[fullwave]"
```

The format is intentionally independent of a solver. A backend adapter should
resample staggered or unstructured fields onto a common rectilinear domain,
record the original mesh and interpolation details in provenance metadata, and
set convergence fields before constructing `HarmonicEMSolution`.

## Scope and safeguards

This interchange layer is not a Maxwell solver and does not make an imported
result trustworthy by itself. Backend validation must still cover mesh
convergence, accepted-power/loss balance, low-frequency agreement with
Biot--Savart, transmit/receive reciprocity, and an independent benchmark. The
[full-wave strategy](full_wave_field_solver_strategy.md) describes that staged
validation sequence.
