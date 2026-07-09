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

### Cross-validation against FastHenry and FastCap

`fields.fasthenry_interop` exports a `Conductor` to a FastHenry `.inp` deck
(`to_fasthenry_inp`) and, on Windows with FastHenry2 installed (its COM automation server),
runs it and returns `L(f)`/`R(f)` (`run_fasthenry`, `compare_with_fasthenry`). A round wire
is exported as an equal-area square bar; use `cross_section="rect"` for an exact match to
FastHenry's rectangular conductors. `examples/validate_peec_vs_fasthenry.py` runs the
comparison; `tests/test_coil_peec_fasthenry.py` checks the export format everywhere and runs
a live comparison when the FastHenry COM server is present (skipped otherwise, so CI stays
green).

Measured agreement (matched square cross-section, 3×3 filaments vs FastHenry nwinc=nhinc=3):

| geometry | f | L FastHenry | L PEEC | R FastHenry | R PEEC |
|---|---|---|---|---|---|
| 1×1 mm bar, 100 mm | 10 kHz | 102.1 nH | 102.2 nH (**+0.1%**) | 1.744 mΩ | 1.744 mΩ (0.0%) |
| 1×1 mm bar, 100 mm | 1 MHz | 97.9 nH | 98.3 nH (+0.4%) | 8.31 mΩ | 6.51 mΩ (−22%*) |
| 6-turn solenoid, D=20 mm | 100 kHz | 400.4 nH | 402.7 nH (**+0.6%**) | 9.87 mΩ | 9.09 mΩ (−7.9%) |
| 6-turn solenoid, D=20 mm | 1 MHz | 393.8 nH | 397.4 nH (+0.9%) | 15.63 mΩ | 13.86 mΩ (−11%) |

**Inductance matches FastHenry to <1%** for both straight bars and helical solenoids. The
mutual inductance between segment pairs uses the exact closed form (`_mutualfil_matrix`, a
NumPy port of FastHenry's Grover-method `mutualfil`) rather than a numerical quadrature: a
quadrature over-couples adjacent path segments that share a vertex and inflates coil
inductance without bound as the path is refined (an earlier quadrature version drifted to
2× for a finely-segmented helix). The self-capacitance and self-resonance are unaffected and
remain within ~1% / a few percent of FasterCap and QOIL.

Resistance matches FastHenry to ~1% at low frequency and converges to the exact Kelvin value
with cell count (1 MHz: 6.5 → 7.8 → 8.1 mΩ at 8²/16²/24² cells; Kelvin ≈ 7.8). *The −22% at
1 MHz above is the coarse 3×3/8×8 mesh under-resolving the skin depth — FastHenry
under-resolves the same way at the same filament count; raise the cell count (or use the
planned surface-impedance backend) for deep-skin resistance.

**Capacitance vs FasterCap.** `fields.fastercap_interop` panelizes a conductor's wire
surface (`to_fastercap_panels`) and runs FasterCap (the FastFieldSolvers successor to
FastCap) through its COM server (`run_fastercap`, `compare_capacitance_with_fastercap`).
`capacitance_to_ground` (`= 1ᵀ P⁻¹ 1`, the isolated-conductor self-capacitance) matches the
FasterCap boundary-element solve to **~1%** for a straight wire (e.g. 1 m, 1 mm wire:
PEEC 8.59 pF vs FasterCap 8.50 pF) — both ~10% below the crude analytic
`2π ε0 L/(ln(L/a)−1)`, confirming the thin-wire electrostatic kernel against the reference
rather than the approximation. The coil self-resonance capacitance (`self_capacitance`) is
additionally validated against the Medhurst formula and reproduces QOIL's `f_res`.
`tests/test_coil_peec_fastercap.py` checks the panel format everywhere and runs the live
comparison when FasterCap is present.

**Self-resonant frequency vs the Hamwaves (QOIL) model.** PEEC's SRF
(`self_resonant_frequency`, `= 1/(2π√(L·C))`) agrees with QOIL's independent sheath-helix
prediction (`βℓ = π/2`) to **~1–3%** across designs (e.g. 20 mm/10-turn: 177 vs 180 MHz;
15 mm/25-turn: 111 vs 113 MHz), and stays robust where QOIL's transcendental dispersion
solve fails (short/fat coils return `NaN` from QOIL but a finite value from PEEC). Two
fully independent methods landing within a few percent is a strong cross-check on both.

> Registration note: FasterCap's COM server resolves method names only if its type library
> is registered. If `Run`/`GetCapacitance` raise "Library not registered", register it once
> from an **elevated 32-bit** PowerShell (`C:\Windows\SysWOW64\WindowsPowerShell\v1.0\
> powershell.exe`) with `LoadTypeLibEx("...\IFasterCap.tlb", REGKIND_REGISTER)`. The call
> registers the typelib even though it then throws a harmless variant-marshaling error.

## Resolution ceiling (state it honestly)

The cross-section must resolve the skin depth `delta`. At `a/delta` up to ~5 a few hundred
cells give sub-percent AC-resistance accuracy; deeper skin (high-frequency RF,
`a/delta >~ 15`) is under-resolved unless the cell count is raised, so `R` is
**under-predicted** there (and `Q` correspondingly over-predicted) — but it **converges
monotonically to the exact Kelvin value as the cell count rises**, and FastHenry
under-resolves the same way at the same filament count (`examples/validate_peec_vs_fasthenry.py`
prints the convergence table; `tests/test_coil_peec.py` asserts it). **Inductance,
capacitance and self-resonance are geometry-dominated and accurate at coarse cross-section
resolution.** The round cross-section is tiled on a Cartesian grid masked to the disc (cells
stay square, areas rescaled to `pi a^2`) so the resistance converges; an equal-area *polar*
tiling diverges as its wedge cells elongate. The cost is `O(K^2 M^2)` (K cells, M path
segments), so validation/example geometries keep both modest.

## Deferred follow-ons

- A **surface-impedance backend** for the deep-skin RF resistance regime.
- **Magnetic core/shield** coupling (via `fields.nonlinear_magnetostatics` /
  `fields.eddy_modes`).
- Rectangular/tube/litz cross-sections; multi-conductor (multi-port) networks.
- **Acceleration** (FMM / H-matrix) to lift the `O(K^2 M^2)` ceiling.
- Surface-panel MoM capacitance (vs the present line-charge energy method).
