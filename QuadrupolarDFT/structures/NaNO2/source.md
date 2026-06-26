# NaNO2 Structure Source Notes

## Target Structure

- Compound: sodium nitrite, NaNO2
- Phase: ferroelectric room-temperature phase
- Space group: Im2m, conventional orthorhombic setting
- Goal: experimental structure suitable for ABINIT PAW EFG calculations and
  comparison with room-temperature 14N NQR data.

## Candidate Sources

- ICSD 82857, `EntryWithCollCode82857.cif`: preferred room-temperature
  ferroelectric candidate. Gohda, Ichikawa, Gustafsson, and Olovsson, "The
  refinement of the structure of ferroelectric sodium nitride" [sic in CIF],
  Journal of the Korean Physical Society 29, 551-554 (1996). Space group
  `I m 2 m`, a = 3.5653(8) A, b = 5.5728(7) A, c = 5.3846(13) A, Z = 2.
- M. I. Kay, "Ferroelectrics" 4, 235 (1972). Cited as the bulk NaNO2 structure
  reference by Fokin et al., arXiv:cond-mat/0205303.
- "The Refinement of the Structure of Ferroelectric Sodium Nitrite", Journal of
  the Korean Physical Society (1996), listed in public sodium-nitrite structure
  summaries.
- ICSD or another institutional crystallographic database entry for NaNO2.

## Uploaded ICSD Entries

The uploaded CIFs include low-temperature/ferroelectric `I m 2 m` structures,
high-temperature/paraelectric `I m m m` structures, and temperature-series
entries near the transition.

Recommended first EFG run:

- `EntryWithCollCode82857.cif` - best room-temperature/standard ferroelectric
  candidate among the uploaded files. It has the lattice constants also quoted
  in public sodium-nitrite summaries and includes anisotropic displacement
  parameters.

Useful comparison entries:

- `EntryWithCollCode15400.cif` - Kay and Frazer, Acta Crystallographica 14,
  56-57 (1961), neutron refinement of the low-temperature phase. Coordinates
  are close to ICSD 82857.
- `EntryWithCollCode68707.cif` - 120 K electron-density study, useful as a
  low-temperature structural comparison.
- `EntryWithCollCode9265.cif` - Kay, Ferroelectrics 4, 235-243 (1972), 423 K
  ferroelectric member of the temperature series.
- `EntryWithCollCode9266.cif` and `EntryWithCollCode9267.cif` - Kay
  high-temperature `I m m m` structures at 458 K and 498 K.

## Notes

The public sodium-nitrite summary lists an orthorhombic Im2m phase with
approximately:

- a = 3.5653 A
- b = 5.5728 A
- c = 5.3846 A
- Z = 2

These lattice constants are useful sanity checks, but the EFG calculation needs
trusted internal fractional coordinates for Na, N, and O.
