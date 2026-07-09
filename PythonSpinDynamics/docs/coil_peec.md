# PEEC Coil Solver: Arbitrary-Geometry L, R, C, Q, and Self-Resonance

Design/progress note for `spin_dynamics.fields.coil_peec`, the field-based coil-property
solver for arbitrary wire geometries.

## Motivation

`fields.coil_properties` (the QOIL port) extracts L/R/C/Q/f_res but only for **single-layer
air-core solenoids**. Real RF coils (multi-layer, saddle, birdcage, surface) and gradient
coils have arbitrary wire paths that model cannot express. This module computes the same
lumped properties for **any single wire path** from first principles, resolving the skin
and proximity effects rather than tabulating them.

## Method (PEEC)

The partial-element equivalent-circuit method, mesh-light like FastHenry/FastCap. A
conductor is a polyline centreline (`path_points`) plus a round cross-section. To resolve
the current distribution the cross-section is tiled into `K = n_radial * n_angular` parallel
sub-filaments, each following the whole path.

Because every sub-filament is a simple series chain from one terminal to the other, the
branch system reduces **exactly** to a `K x K` chain-impedance system: sub-filament `k` is
the path swept to cross-cell `k`, and

- `L_chain[k, k'] = mutual_inductance(filament_k, filament_k')` — reusing the exact Neumann
  kernel (`quasistatic.vector_potential`), vectorized here as `_pair_mutual`; the diagonal
  adds the analytic straight-filament `self_partial_inductance`.
- `R_chain[k] = rho(T) * total_length / area_k` — with `rho` from
  `coil_properties.ConductorMaterial.resistivity_at(T)`, so the temperature model carries
  over.

At each frequency the parallel sub-filaments reduce to the terminal impedance
`Z(w) = 1 / (1^T (R + jwL)^{-1} 1)`, giving `L(w) = Im Z / w` and `R(w) = Re Z`. The current
`I_k ~ (Z^{-1} 1)_k` redistributes as `w` rises — crowding to the surface (skin) and
toward/away from neighbouring turns/legs (proximity). A dual thin-wire **electrostatic**
solve on the same geometry (`self_capacitance`, potential-coefficient energy method with a
linear winding potential) gives the self-capacitance and hence `self_resonant_frequency =
1/(2 pi sqrt(L_dc C))`.

The DC inductance is the resistance-weighted limit `L_dc = i0^T L i0` with
`i0 = R^{-1}1 / (1^T R^{-1}1)` (uniform-ish current), **not** the minimum-energy
`1/(1^T L^{-1} 1)` (which is the high-frequency limit).

## API

```python
from spin_dynamics.fields.coil_peec import helical_solenoid, coil_properties_peec

coil = helical_solenoid(diameter=20e-3, length=30e-3, turns=10, wire_radius=0.5e-3,
                        n_per_turn=12, n_radial=4, n_angular=8)
p = coil_properties_peec(coil, frequency=5e6)
p.inductance, p.ac_resistance, p.q_factor, p.self_capacitance, p.self_resonant_frequency
p.to_probe_params()            # {"L","R","C"} -> same probe-noise / matching pipeline
```

Build any coil from its wire path with `Conductor(path_points, wire_radius, ...)` or
`conductor_from_segments(...)`; `extract_impedance(coil, freqs)` sweeps `L(w)`/`R(w)`;
`current_distribution(coil, f)` returns the per-cell current for the crowding picture.

`PEECCoilProperties` mirrors the universal subset of `coil_properties.CoilProperties`
(same field names) so an arbitrary coil drops into the same SNR/matching workflow.

## Validation

`tests/test_coil_peec.py`:

- **Skin effect** — an isolated straight wire's `R_ac(w)/R_dc` matches the exact
  Kelvin-function ratio (`scipy.special`) at moderate `a/delta` (< 15% with a few hundred
  filaments; converges with refinement).
- **Kernels** — `_pair_mutual` reproduces the Neumann `mutual_inductance`;
  `self_partial_inductance` matches the analytic asymptote.
- **DC inductance** — a helical solenoid's `L_dc` matches QOIL's current-sheet `L_s`
  (< 10%) and, more loosely, the stacked-loop `coil_inductance`.
- **Solenoid vs QOIL** — `L(w)` within 8%, and the PEEC self-resonance within 10% of
  QOIL's `f_res`; the self-capacitance within 30% of the Medhurst empirical formula. Note
  QOIL's `C_p` is a *lumped-equivalent fitting* capacitance, distinct from the physical
  self-resonance capacitance the PEEC (and Medhurst) compute.

The example `examples/plot_peec_coil_solver.py` shows a two-layer solenoid (`L`, `R`, `Q`,
`f_res`, and the cross-section current-crowding map) and a gradient Maxwell pair (`L`, `R`,
`dB_z/dz` efficiency via Biot-Savart, and shield eddy modes via `eddy_modes`) — the same
solver spanning RF and gradient coils.

## Resolution ceiling (state it honestly)

The cross-section must resolve the skin depth `delta`. At `a/delta` up to ~5 a few hundred
cells give sub-percent AC-resistance accuracy; deeper skin (high-frequency RF,
`a/delta >~ 15`) is under-resolved unless the cell count is raised, so `R` is
**under-predicted** there (and `Q` correspondingly over-predicted). **Inductance,
capacitance and self-resonance are geometry-dominated and accurate at coarse cross-section
resolution.** The cost is `O(K^2 M^2)` (K cells, M path segments), so validation/example
geometries keep both modest; for a single-layer solenoid in the deep-skin regime prefer the
analytic `coil_properties.solenoid_properties`.

## Deferred follow-ons

- A **surface-impedance backend** for the deep-skin RF resistance regime.
- **Magnetic core/shield** coupling (via `fields.nonlinear_magnetostatics` /
  `fields.eddy_modes`).
- Rectangular/tube/litz cross-sections; multi-conductor (multi-port) networks.
- **Acceleration** (FMM / H-matrix) to lift the `O(K^2 M^2)` ceiling.
- Surface-panel MoM capacitance (vs the present line-charge energy method).
