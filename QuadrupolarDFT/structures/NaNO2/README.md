# NaNO2 Structure Inputs

This folder stores crystallographic source data for sodium nitrite EFG and NQR
calculations.

## Intended Files

- `experimental_rt.cif` - preferred room-temperature ferroelectric NaNO2 CIF.
- `source.md` - bibliographic/source notes for the chosen structure.
- `generated/` - ABINIT inputs generated from the chosen CIF or structure file.

## Current Status

No trusted CIF has been added yet. The earlier
`examples/abinit/nano2_efg.abi` file used a hand-entered starter geometry and
should be treated only as a workflow smoke test.

Preferred next step: add an experimental room-temperature `Im2m` NaNO2
structure, ideally from Kay, Ferroelectrics 4, 235 (1972), the later
ferroelectric sodium-nitrite refinement, ICSD, or another traceable
crystallographic source.
