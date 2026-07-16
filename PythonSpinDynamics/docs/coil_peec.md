# PEEC Coil Solver: Arbitrary-Geometry L, R, C, Q, and Self-Resonance

Real RF and gradient coils often cannot be reduced to a single-layer solenoid.
Their turns, legs, crossovers, shields, and dielectric formers create distributed
inductance, resistance, and capacitance that depend on the actual wire path.

Use this page when the geometry falls outside the specialized solenoid model.
The partial-element equivalent-circuit (PEEC) solver converts an arbitrary
polyline conductor into terminal L, R, C, Q, and self-resonance estimates. It is
more general and more expensive than the solenoid extractor, and its accuracy
depends on conductor discretization and loss formulation.

> **Status (audited 2026-07-11): implemented and stabilized.** The arbitrary
> wire-path solver, terminal reduction, RF resistance, capacitance, and
> validation examples are available.

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
`I_k ~ (Z^{-1} 1)_k` redistributes as `w` rises — crowding to the surface (skin) and, in a
path-averaged way, toward/away from neighbouring turns/legs. Because the chain reduction
forces each sub-filament's current to be constant along the path, it captures only part of
the turn-to-turn **proximity** loss; `formulation="full"` removes the reduction (one branch
per segment × sub-filament, FastHenry's actual system — see "Loss modelling" below). A dual
thin-wire **electrostatic** solve on the same geometry (`self_capacitance`,
potential-coefficient energy method with a linear winding potential) gives the
self-capacitance and hence `self_resonant_frequency = 1/(2 pi sqrt(L_dc C))`.

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

**Mesh-matching gotcha (found reading FastHenry's source):** FastHenry silently grades its
`nwinc × nhinc` filaments toward the conductor surface with adjacent-size ratio 2 unless
told otherwise — `readGeom.c` initializes `DEFAULTS` with `rw = rh = 2.0`. Early
comparisons at "matched" filament counts were therefore graded-vs-uniform and diverged in
the deep-skin regime. `to_fasthenry_inp(..., rw=1.0, rh=1.0)` pins FastHenry to the uniform
tiling for apples-to-apples comparisons.

Measured agreement (matched square cross-section, uniform filaments both sides;
solenoid = 6 turns, D=20 mm, l=30 mm, 1 mm wire; PEEC `formulation="full"` for the coil):

| geometry | f | L FastHenry | L PEEC | R FastHenry | R PEEC |
|---|---|---|---|---|---|
| 1×1 mm bar, 100 mm (8×8) | 10 kHz | 102.09 nH | 102.15 nH (+0.06%) | 1.7441 mΩ | 1.7441 mΩ (0.0%) |
| 1×1 mm bar, 100 mm (8×8) | 1 MHz | 98.22 nH | 98.30 nH (+0.08%) | 6.585 mΩ | 6.514 mΩ (−1.1%) |
| 1×1 mm bar, 100 mm (8×8) | 10 MHz | 97.89 nH | 97.98 nH (+0.09%) | 8.280 mΩ | 8.121 mΩ (−1.9%) |
| 6-turn solenoid (3×3, full) | 100 kHz | 401.7 nH | 399.6 nH (−0.5%) | 9.10 mΩ | 9.82 mΩ (+7.9%) |
| 6-turn solenoid (3×3, full) | 1 MHz | 396.1 nH | 393.7 nH (−0.6%) | 14.38 mΩ | 14.89 mΩ (+3.5%) |

**Inductance matches FastHenry to <1%** for both straight bars and helical solenoids. The
mutual inductance between segment pairs uses the exact closed form (`_mutualfil_matrix`, a
NumPy port of FastHenry's Grover-method `mutualfil`) rather than a numerical quadrature: a
quadrature over-couples adjacent path segments that share a vertex and inflates coil
inductance without bound as the path is refined (an earlier quadrature version drifted to
2× for a finely-segmented helix). The self-capacitance and self-resonance are unaffected and
remain within ~1% / a few percent of FasterCap and QOIL.

Straight-bar resistance matches FastHenry to ~2% at every frequency once the meshes really
match, and converges to the exact Kelvin value with cell count. The few-% solenoid-R
residual is the near-field kernel: for close parallel filament pairs FastHenry integrates
the exact rectangular cross-sections (`parallel_fils`/`exact_mutual`, or its `fourfil`
5-filament approximation), while this solver uses curved thin filaments with GMD self-terms
— the two discretizations converge with refinement but approach from different sides. DC
resistance agrees to 0.07%. On two straight antiparallel legs (a hairpin, where no curvature
is involved) the two solvers agree to 0.5%.

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

## Performance: surface-impedance backend and graded meshing

The volume solver's cost is `O(K^2)` in the cross-section cell count, and deep skin needs
`K ~ (a/delta)^2` cells to resolve — prohibitive at RF. Two accelerations fix this
(`examples/plot_peec_performance.py` benchmarks them):

- **Surface-impedance backend** — `extract_impedance_surface(conductor, freqs, n_perimeter)`
  places a ring of filaments around the cross-section boundary, one half skin depth inside,
  each with the analytic surface impedance `Z_s = (1+j) rho/delta * (length/width)` (real
  part = skin resistance, imaginary = internal inductance); the external inductance is the
  exact ring-filament mutual. Cost is independent of `a/delta` and the accuracy *improves*
  with frequency — the opposite of the volume solver. Validated against Kelvin: −3.3% at
  `a/delta = 15` and −0.3% at `a/delta = 151`, with `n_perimeter = 48`. For a deep-skin
  solenoid it is ~10–100× faster than a volume solve **and** more accurate.
- **Graded meshing** — `Conductor(..., grading=g)` (`g > 1`) concentrates volume cells
  toward the surface, so the moderate-skin regime converges with far fewer cells (e.g. a
  1 MHz square bar reaches ~5% at 64 cells vs a uniform mesh needing ~250).

Rule of thumb: uniform/graded volume for low–moderate skin (`a/delta <~ 10`), the
surface-impedance backend for deep skin. Both accept `formulation="full"` (per-segment
currents, resolves turn-to-turn proximity — see "Loss modelling") at cost `O((MK)^3)` per
frequency; for the SIBC backend at deep skin this is *the* way to get a tight coil's R.

## Shielding and dielectric formers

A grounded metal enclosure and a dielectric coil former both change the coil's
electrostatics (and hence its self-resonance), and the enclosure adds loss. `self_capacitance`
and `capacitance_to_ground` accept:

- `shield=GroundedBox(center, half_extents)` — a grounded rectangular enclosure entered
  through its 3-D image-charge lattice (walls at zero potential, consistent with a grounded
  coil end). It raises the coil capacitance and lowers the SRF; a box receding to infinity
  recovers the free-space value, and `n_orders=1` is normally converged.
- `shield=GroundPlane(point, normal)` — a single grounded plane (a half-space wall, not a
  closed cavity): the PCB/chassis ground plane below a planar coil, entered as the single
  method-of-images mirror charge. Matches the analytic wire-over-plane
  `C/ℓ = 2π ε₀ / acosh(h/a)` to a few percent.
- `relative_permittivity` — a dielectric former as an effective medium (e.g. a PTFE former
  with partial fill, `1 + (eps_r - 1) * fill`), which scales the capacitance.

**A metal shield is also a good conductor magnetically, so it lowers L too.** The same
`GroundedBox` or `GroundPlane` passed to `extract_impedance(..., ground_plane=...)` or
`extract_impedance_surface(..., shield=...)` adds the walls' flux-exclusion image
(`magnetic_image_filaments`: reflect each sub-filament and reverse its current). For a box the
six primary wall reflections are summed (the leading lattice term; higher-order edge/corner
images add ~1% and are omitted for speed). So a metal enclosure has **two** competing effects
on the self-resonance: it raises C (electric image, lowers SRF) *and* lowers L (magnetic
image, raises SRF), which partly offset — the shielded-NQR example prints both (box: C
1.5 → 3.9 pF, L 4.8 → 4.2 µH, net SRF 59 → 39 MHz). Model only the capacitance and the SRF
comes out too low. The image is lossless (a PEC mirror); the shield's *resistive* wall loss is
the separate surface-resistance integral below.

**Wall eddy-loss factor.** For a good conductor the tangential field at the wall is *doubled*
by the image (surface current `K = 2·H_tan,inc`), so the series loss is
`R_wall = 4·R_s·∮|H_tan,inc/I|² dA` with `R_s = √(πfμ₀/σ)` (from `P = ½R_s|K|²` equated to
`½I²R`). Both shielded examples use this `4·R_s` form (the box-eddy and the PCB ground-eddy
integrals); dropping the image doubling underestimates the loss by 2×.

## Single-ended vs differential Q over a ground plane

`self_capacitance` takes a `potential` profile — either an `(N,)` array or a callable of the
arc-length fraction `s ∈ [0,1]` — defaulting to the **single-ended** linear ramp `0→1` (one
terminal grounded). Passing `potential=lambda s: s-0.5` gives the **differential** (balanced)
drive, antisymmetric about a virtual ground at the winding centre. Over a ground plane the two
couple to it very differently: the coil-to-ground capacitance `C_g = C_eff(with plane) −
C_eff(free space)` is several times smaller differentially, because the two winding halves'
displacement currents to ground cancel in common mode. The lossy board (`tan δ`) turns `C_g`
into an equivalent series resistance `R_diel = tan δ · ω³ L² · C_g` (first-order lumped model,
below self-resonance), and **that capacitive difference is the whole single-ended/differential
Q split** — because the magnetic behaviour is mode-independent:

**Inductance is mode-independent (magnetic image).** `extract_impedance(..., ground_plane=
GroundPlane(...))` adds the good-conductor plane's flux-exclusion image (reflect each
sub-filament across the plane and reverse its current — anti-parallel for a coil lying parallel
to the plane), lowering L by the standard `L = L_free − M(2h)` (validated against the analytic
loop-over-plane coaxial mutual to <1%, and recovering `L_free` as the plane recedes). This image
is set by the *winding current*, which is identical in both drive modes (a series winding carries
the same I everywhere regardless of the terminal reference), so **the ground plane's effect on L
— and its resistive eddy loss, and the skin/proximity R — are all mode-independent.** The one
genuinely mode-specific magnetic term would be a single-ended *galvanic* return routed through
the plane to a remote source-ground bond; for the natural local ground bond that current
coincides with the flux-exclusion eddy (no separate term), so it is layout-dependent, not an
intrinsic coil property, and is not modelled. The magnetic image is the exact dual of the
electrostatic `GroundPlane`.

`examples/plot_pcb_coil_ground_mode_q.py` works a 4-turn PCB spiral on 0.8 mm FR4 over a copper
ground plane. The plane lowers L (≈ 619 → 216 nH at this 0.8 mm spacing, −65%, *both modes*) and
raises the single-ended ground coupling ~7× over the differential (2.9 vs 0.38 pF), so the
dielectric loss is ~8× lower differentially (19 vs 145 mΩ at 60 MHz) and the differential Q runs
higher, the gap widening with frequency. Because dielectric loss scales as `L²`, the ground
plane's L-collapse also *suppresses* the mode gain at very close spacing (the coil goes
copper/eddy-limited) — so the differential advantage is largest when the coil is still a good
inductor. This balanced-drive trick is why NMR/MRI surface and PCB coils are built symmetric.

**Why ~7× and not ~2×.** The single-ended ramp is a uniform common-mode offset plus the
differential profile, `v = ½ + (s − ½)`, and the differential drive carries ~zero net charge, so
the ground capacitance splits cleanly (verified by the example to ~1%):

```
C_g(single-ended) = ¼·C_g(uniform) + C_g(differential),   ratio = 1 + ¼·C_g(uniform)/C_g(differential)
```

Two effects push past the naive 2×. First, the common-mode offset carries *three quarters* of
the ramp's mean-square voltage — `⟨v²⟩ = 1/3`, of which the offset is `1/4` and the zero-mean
part only `1/12` — so going differential already cuts the energy 4×, not 2×. Second, `C_g(uniform)`
(the whole spiral held at one potential — a capacitor *plate*, i.e. a net-charge monopole) couples
far more strongly to the plane than the differential mode (a zero-net-charge multipole that closes
locally and cancels in the image): here `C_g(uniform)/C_g(differential) ≈ 26`, so the ratio is
`1 + ¼·26 ≈ 7.5`. The same monopole-vs-multipole asymmetry makes the ratio *grow* as the plane
recedes (6× at 0.4 mm → 17× at 3.2 mm: the monopole coupling decays slowly, the multipole fast).

The example `examples/plot_shielded_nqr_coil.py` designs a ¹⁴N NQR probe (2–3 MHz, coil
ID ≈ 1.5″) on a Teflon former in a grounded aluminium box, one coil end and the box at
ground, and plots the properties with and without the box: L(f) and R(f) from the
surface-impedance backend with `formulation="full"` (the wire is deep-skin at these
frequencies; the full formulation adds the turn-to-turn proximity loss with no empirical
factor), the self-capacitance and SRF for free / +former / +box (e.g. 1.5 → 2.2 → 3.9 pF,
SRF 60 → 50 → 37 MHz — kept above
the 10 MHz target), and the Q reduction from box-wall eddy loss (computed as a
surface-resistance integral `R_box = 2 R_s(f) ∮|H_tan/I|² dA`, `R_s = √(π f μ₀/σ)` ∝ √f —
the correct skin-limited scaling for a solid wall, unlike the full-penetration first-order
eddy model). A loose enclosure mainly shifts the SRF and lowers Q only modestly.

## Loss modelling: proximity and eddy currents (read before trusting a Q)

Getting the *inductance* right is easy; getting the *resistance* (hence Q) right needs the
correct loss model for the situation. Two common trip-ups:

### Proximity effect between turns: the `full` formulation

The AC field of neighbouring turns crowds the current in each wire, raising R above the
isolated-wire skin value (proximity factor Φ, typically 1.3–2 for a closely-wound
solenoid).

**How FastHenry does it (verified in its source):** it never reduces the system. Every
segment keeps its own `nwinc × nhinc` filaments as independent circuit branches
(`assignFil`, `induct.c`), and the mesh analysis adds, *for every segment*, `K − 1`
"mini-meshes" pairing adjacent filaments (`fillM.c`), on top of the loop meshes from the
node graph. The solved system is `Z_m = M (R + jωL) Mᵀ` with per-filament diagonal `R` and
the dense partial-inductance `L` — proximity emerges from the inductive coupling alone.
There is **no empirical factor anywhere** in FastHenry (no "proximity", no "Medhurst" in
the source); the per-segment cross-section redistribution freedom *is* the proximity model.

**This solver now has the same system:** `formulation="full"` on `extract_impedance`,
`extract_impedance_surface` and `coil_properties_peec` keeps one branch per
(segment, sub-filament) and constrains only the total current at the path nodes. For a
series path this reduces to the M×M block system `A V = I·1` with
`A[m,m'] = 1ᵀ (Z_b⁻¹)[m-block, m'-block] 1` and `Z_term = 1ᵀ A⁻¹ 1` (which collapses to the
chain formula `1/(1ᵀ Z_b⁻¹ 1)` when M = 1 — a straight bar gives bit-identical results in
both formulations). Cost is `O((MK)³)` per frequency (guardrailed at MK ≤ 6000), against
`O(K³)` for the reduced `chain` formulation — which forces every sub-filament to carry a
path-constant current and therefore misses the part of the proximity loss that varies along
the wire. `current_distribution(..., formulation="full", segment=...)` shows the actual
asymmetric crowding at any one segment.

**Validation of the full formulation** (10-turn solenoid, D=20 mm, 1 mm round wire, pitch
1.5 mm — tightly wound, Φ = R_coil/R_straight-wire):

| reference | Φ @ 0.5 MHz (a/δ=5.3) | Φ @ 2 MHz (a/δ=10.7) |
|---|---|---|
| FEMM 4.2, axisymmetric continuum FEM (10-ring stack, 25/12.5 µm mesh) | **1.720** | **1.803** |
| this solver, volume `full` (round, ~52 cells) | **1.718** (+0.1%) | — (mesh-limited) |
| this solver, SIBC `full` (n_perimeter=28) | 1.54 (−10%; a/δ too shallow for SIBC) | 1.69 (−6%) |
| FastHenry, graded 8×8 (its converged mesh) | 1.43 (−17%) | 1.47 (−18%) |
| Medhurst measured table (HF asymptote) | 2.27 (+25% vs the a/δ→∞ trend ≈ 1.85–1.9) | |

Independent checks: on two antiparallel straight round wires at s/d = 1.5 the exact HF
proximity factor is `1/√(1−(d/s)²)` = 1.3416; the SIBC-full solve reaches it to ~3% at
a/δ = 24 (and FastHenry agrees with the volume-full solve to 0.5% on the square-equivalent
hairpin — the solvers only diverge on *curved multi-turn* geometries, where FastHenry reads
low against the continuum FEM and Medhurst's empirical table reads high; Medhurst measured
real 1947 coils whose Q included losses beyond eddy-current proximity). Use `full` and trust
it; keep Medhurst's Φ (`coil_properties.medhurst_proximity_factor`) as a conservative
cross-reference, not a correction.

### Radiation resistance (first-order)

`radiation_resistance(conductor, f)` adds the magnetic-dipole radiation loss:
`R_rad = η₀ k⁴ |A_net|² / 6π`, with `A_net = ½ Σ rᵢ × rᵢ₊₁` the net vector area of the wire
path closed end-to-end (translation-invariant; the closure stands in for the feed). It
reduces to the textbook small-loop `20π²(C/λ)⁴` (validated to <1%), scales as `N²` for an
N-turn solenoid, and vanishes for opposed-loop geometries (a Maxwell pair has `A_net ≈ 0`
— gradient coils don't dipole-radiate). `coil_properties_peec` includes it by default:
`ac_resistance` stays purely ohmic, the new `radiation_resistance` field reports it, and
`q_factor` / `to_probe_params()` use the total. With the `k⁴` scaling it is utterly
negligible at NMR/NQR frequencies for ordinary coils (nΩ at 2.5 MHz for the shielded-NQR
example) and becomes the Q ceiling for large loops at VHF.

**Shielding suppresses it.** For radiation a `GroundedBox` is a *cavity*, not the
image-charge lattice used for capacitance: below the box's lowest resonant mode
(`GroundedBox.fundamental_resonance()`, the rectangular-cavity `TE₁₀₁`) a closed conductor
supports no propagating external field, so the far-field radiation is evanescent — the
near-field energy that would have radiated is instead stored reactively and dissipated in
the walls (the wall eddy loss, modelled separately). Passing `shield=` to
`radiation_resistance` (or `coil_properties_peec`) therefore returns 0 below cutoff,
consistent with how the same box enters `self_capacitance`, and warns as the frequency
approaches the cavity resonance where the closed-cavity approximation breaks down. So the
free-space `R_rad` is an *upper bound*: a shielded coil radiates less (the NQR example prints
both — 52 nΩ free-space → 0 inside its 2.6 GHz box, negligible at 2.5 MHz either way).

First-order limits: magnetic dipole only — a `UserWarning` fires when the coil is not small
vs the wavelength (higher multipoles + electric-dipole term missing), and common-mode
"antenna-effect" radiation via the feed leads is a property of the installation, not the coil.

**And even the computed R is a theoretical lower bound on loss:** a measured coil is usually
another ~1.3–2× lossier still (dielectric-former `tan δ`, solder/lead resistance, surface
oxidation, common-mode lead radiation), so a lab Q well below the computed one is expected,
not a bug.

### Which eddy-loss model for a nearby conductor (shield, sample, former)

| regime | model | scaling | use for |
|---|---|---|---|
| skin depth `δ` ≪ conductor thickness (solid metal at RF) | **surface resistance** `R = 2 R_s ∮\|H_tan/I\|² dA`, `R_s = √(πfμ₀/σ)` | `R ∝ √f` | a shield can, a metal box, thick copper at MHz |
| `δ` ≫ thickness, or low-σ (thin foil, saline sample, weak conductor) | **first-order (Born) volume** `fields.quasistatic.reflected_resistance` / `eddy_currents` | `R ∝ f²` | thin shells, conductive samples, gradient-coil eddy shields |

The trap: applying the **first-order volume** model to a **solid metal box at MHz** assumes
the whole wall thickness conducts (no skin shielding) and over-predicts the loss by orders
of magnitude (the skin depth in aluminium at 3 MHz is ~47 µm, far thinner than the wall).
The shielded-NQR example therefore uses the surface-resistance model for the box; the
first-order volume model is right only when the conductor is thin compared with `δ`.

## Deferred follow-ons

- **FMM / H-matrix acceleration** of the `O(K^2)` chain matrix — the benchmark shows clean
  `O(K^2)` scaling (K = 16/64/144/256 → 0.3/5/26/76 s), but the surface-impedance backend
  and graded meshing keep `K` small in the regimes that used to force large meshes, so this
  is the lever only for very large single conductors or big multi-conductor arrays.
- **Magnetic core/shield** coupling (via `fields.nonlinear_magnetostatics` /
  `fields.eddy_modes`).
- Tube/litz cross-sections; multi-conductor (multi-port) networks.
- Surface-panel MoM capacitance (vs the present line-charge energy method).
