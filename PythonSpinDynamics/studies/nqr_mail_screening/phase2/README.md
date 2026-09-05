# Phase 2: parcel and aperture model

**In progress; Gate 2 is open.** Gate 1 passed its declared single-line reference
checks before this phase began. All dimensions here are candidate study inputs.

`aperture.py` implements:

- Regions with material identity, dimensions, mass, crystalline fraction,
  temperature, permittivity and conductivity; mass-conserving voxels and proper
  rotation/translation transforms.
- An enclosing six-turn helix, an open split rectangular pair, and a series axial
  gradiometer. These are connected wire paths, including the gradiometer's external
  bridge. Remaining feed/return wiring and shielding are not included.
- Built-in Biot-Savart Tx and reciprocal same-port Rx maps, plus material/pose
  minimum, median and mass-weighted coupling metrics. The squared-field metric is
  a small-tip comparison proxy, not a finite-pulse SNR or an optimized array result.
- Built-in surface-impedance PEEC calculations at 3.1024 and 3.3043 MHz and a
  self-capacitance estimate. The eight-perimeter chain solve is coarse: it is not
  converged resistance or a full proximity-current solution.
- Direct-field comparisons against trilinear maps and refined wire paths. Initial
  interpolation normalized RMS errors are roughly 5.4%, 6.2%, and 9.3%; wire-path
  refinements are roughly 0.9%, negligible for straight split-loop edges, and 1.7%.
- A lumped added-R/C sensitivity sweep showing resonance, Q, ringdown, delivered
  current per drive volt and Johnson noise. It is explicitly **not** a spatial
  dielectric/conductive loading solution.

The two example regions are a 1 g target packet and a 20 g paper surrogate.
Their electrical loading values are assumptions, not measured material data.
Whole-parcel translations sample entry, centre and exit poses; individual voxels
may be outside the active coil length during passage. Frequency/temperature and
finite-pulse spin dynamics remain in the Phase 1/3 consumers of these maps.

Run from `PythonSpinDynamics`:

```bash
OPENBLAS_NUM_THREADS=1 python studies/nqr_mail_screening/phase2/aperture.py
python -m unittest tests.test_nqr_mail_screening_phase2
```

Outputs are `.tmp/nqr_phase2/aperture_report.json` and per-candidate NPZ files
containing grid Tx/Rx vectors, posed voxel positions, masses and coupling vectors.
`aperture_report.json` here is a snapshot. No candidate is selected as optimal.

To close Gate 2: refine voxels/poses and interpolation grids; converge PEEC path
and surface-current resolution; couple material-dependent spatial loading to the
network; then validate surrogate errors across that declared domain. Changes in
magnet or coil placement must also pass the Phase 1 Zeeman-spacing screen before
being used in a system comparison.
