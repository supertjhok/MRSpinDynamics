# 3D Nonlinear Magnetostatics via the Reduced Scalar Potential

This note is the design/assessment document for extending the structured-grid
nonlinear magnetostatic solver from 2D (`fields/nonlinear_magnetostatics.py`:
`PlanarMagnetostatics`, `AxisymmetricMagnetostatics`) to full 3D. It records
*why* the reduced scalar potential (RSP) was chosen over the alternatives, the
formulation, and — with numbers — the errors and limits an RSP solver carries so
that a user knows when to trust it and when to reach for something heavier.

Implementation: `fields/scalar_potential_3d.py`
(`ReducedScalarPotential3D`, `ScalarPotentialSolution`).

## Why the 2D solvers do not extend directly

Both 2D solvers exploit symmetry to collapse the magnetic vector potential `A`
to a **single scalar component** (`A_z(x,y)` planar, `A_φ(r,z)` axisymmetric).
That is what makes them cheap: one unknown per node, a symmetric positive
definite (SPD) 5-point Laplacian, and a sparse LU direct solve.

In general 3D there is no symmetry to exploit and the governing equation is the
full curl-curl form

```
∇ × (ν ∇ × A) = J + ∇ × H_c ,     A = (A_x, A_y, A_z).
```

The curl-curl operator has a huge null space (any `∇φ` maps to zero), so a naive
nodal discretization is singular and spurious-mode ridden. The "correct" fixes —
Nédélec/Whitney **edge elements** or a **gauged nodal A** block system — are a
rewrite of the operator/solver layer, need an AMG-preconditioned solve to scale,
and share almost no code with the existing 5-point machinery.

## Chosen route: Reduced Scalar Potential (RSP)

For magnetostatics dominated by permanent magnets and soft-magnetic flux shaping
— the intended use (shim iron, magnet arrays, ferrite cores) — the magnetic
field splits as

```
H = H_s − ∇ψ ,
```

where `H_s` is the **source field of the coil currents in free space**
(Biot–Savart, so `∇ × H_s = J` exactly) and `ψ` is the *reduced scalar
potential*. Because `∇ × (∇ψ) = 0`, Ampère's law is satisfied for **any** `ψ`.
The remaining equation is Gauss's law `∇ · B = 0` with the constitutive law
`B = μ(|B|) (H_s − ∇ψ) + B_r` (`B_r` = remanence, 0 for a non-magnet):

```
∇ · (μ ∇ψ) = ∇ · (μ H_s + B_r) .          (governing PDE)
```

This is exactly the same **variable-coefficient SPD Poisson problem** the 2D
solver already discretizes — only now the coefficient is the permeability `μ`
(not the reluctivity `ν`), the unknown is a scalar `ψ` (so 3D adds only a third
stencil axis, not a vector unknown), and the right-hand side is the divergence of
a magnetic "source flux" concentrated at material interfaces (bound charges).

What this buys us — the reuse the assessment promised:

- **`BrauerBH` + `MagneticMaterial`** — the whole material model is reused
  unchanged (`μ = 1/ν(|B|²)`).
- **`_anderson_picard`** — the tuned nonlinear driver (Anderson mixing, load
  stepping, residual-stall detection) is reused; it was generalized with an
  optional `rhs_fn` so the source term can be recomputed each iteration (see
  "issue 2" below), preserving the 2D behavior bit-for-bit when `rhs_fn=None`.
- **`_conjugate_gradient`** — already dimension-agnostic; the 3D matrix-free
  fallback uses it directly.
- The 7-point sparse LU (`_sparse_factorize_3d`) is the obvious 3D analog of the
  existing 5-point `_sparse_factorize`.

The degenerate check that ties it back to the existing code: a coil in free space
(no materials) must return `ψ ≡ 0` and `B = μ₀ H_s`, i.e. exactly the package's
existing `biot_savart` field. This is a solver unit test.

## Quantified issues with the RSP solver

RSP is the pragmatic choice, not the universally accurate one. Its limits are
well understood; here they are with numbers so the docstrings and examples can
steer users away from the failure regimes.

### 1. Cancellation error in high-permeability regions — the headline limit

Inside soft iron the *physical* field is small: `H = B/μ ≈ B/(μ_r μ₀)`. But the
solver forms it as the **difference** `H = H_s − ∇ψ` of two O(`H_s`) quantities
that nearly cancel (to one part in `μ_r`). The analytic `H_s` is exact at nodes,
but the numerically differentiated `∇ψ` carries the scheme's truncation error
`~ (h/L)²·|∇ψ| ~ (h/L)²·H_s`. That error does **not** cancel, so the fractional
field error inside iron is amplified by the permeability:

```
δB/B  ~  μ_r · (h/L)² .
```

Concretely, to hold **1% field accuracy inside the iron**:

| `μ_r` | required `(h/L)` | cells across a feature of size `L` |
|------:|-----------------:|-----------------------------------:|
| 10    | 3.2e-2           | ~30    |
| 100   | 1.0e-2           | ~100   |
| 1000  | 3.2e-3           | ~300   |
| 5000  | 1.4e-3           | ~700   |

Structured 3D grids of 300–700 cells per feature are impractical (>10⁷–10⁸
nodes). **Practical verdict: pure RSP is trustworthy for `μ_r ≲ 100–300`.** It is
the correct regime for permanent magnets (recoil `μ_r ≈ 1.05`) and moderate
ferrites, and it degrades — predictably, `∝ μ_r` — for high-`μ_r` linear iron and
for unsaturated soft iron near its small-signal permeability.

**Why the Simkin–Trowbridge total/reduced split does *not* help here (a verified
finding).** The classical fix for reduced-potential cancellation is to switch to
the *total* scalar potential inside the iron (`H = −∇φ`, no `H_s` to cancel),
coupled to the reduced potential outside. That remedy targets **floating-point**
cancellation — the historical concern on 32-bit hardware, where `|H_s| ≫ |H|`
inside iron loses significance. On this **centered nodal finite-volume grid**,
however, the total and reduced formulations are related by an *exact linear
variable shift* (`φ = ψ − χ`, where `∇χ = H_s`), so they produce the **identical
discrete solution**. Solving the same μ_r=1000 sphere both ways confirms it: the
two agree to `|B_red − B_tot|/B ≈ 3×10⁻¹³` and `φ = ψ − H₀z` to `1×10⁻¹³` — machine
precision. Because the discrete solution is the same, the split removes *none* of
the error we actually see: the μ_r=1000 sphere is ~20% off at n=41 in **both**
formulations. In `float64` the finite-precision cancellation (`μ_r·ε_machine ≈
1e-13`) is negligible; the real error is **discretization** (`μ_r (h/L)²` plus the
staircased geometry of issue 5), which a variable change cannot touch.

**The lever that does work: grid refinement — now practical via AMG (issue 3).**
The discretization error falls with `h`, so refining the grid is the honest cure.
For the μ_r=1000 sphere in a uniform field (AMG-solved):

| `n` | unknowns | `R/h` | interior-`B` error | solve time |
|----:|---------:|------:|-------------------:|-----------:|
| 41  | 6.9e4    | 6.7   | 20.7 %             | 21 s (splu) |
| 61  | 2.3e5    | 10.0  | 15.0 %             | 1.5 s (amg) |
| 81  | 5.3e5    | 13.3  | 11.6 %             | 3.6 s (amg) |
| 101 | 1.0e6    | 16.7  | 9.7 %              | 6.9 s (amg) |
| 121 | 1.8e6    | 20.0  | 8.5 %              | 13 s (amg)  |

Convergence is ~`O(h)` here (staircase-limited, not the `O(h²)` of the smooth
interior), so high `μ_r` remains genuinely demanding — but it is a controllable
error, not a wall, and AMG puts million-cell grids within reach. For
`μ_r ≲ 100–300` the same refinement reaches a few-percent accuracy at very
modest grids. (A body-fitted / cut-cell treatment of curved surfaces would lift
the `O(h)` staircase floor — a separate, larger piece of work than the potential
split, and the more promising future direction.)

### 2. `B` is not available from the unknown alone

In the A-formulation `B = ∇ × A` comes straight from the unknown, independent of
the material. In the RSP formulation `B = μ(H_s − ∇ψ) + B_r` needs the
constitutive `μ`, so (a) the field of interest is only as good as the current
`μ` estimate, and (b) the **right-hand side `∇·(μ H_s + B_r)` itself depends on
`μ`** and must be recomputed every nonlinear iteration — unlike the 2D solver,
whose RHS (`J + ∇×H_c`) is fixed. Cost impact: one extra divergence assembly per
Picard step (cheap relative to the linear solve); convergence rate is essentially
unchanged. This is why `_anderson_picard` gained the `rhs_fn` hook.

### 3. Linear-solve scaling — the practical grid ceiling

The 7-point SPD matrix on an `n³` grid has `N = n³` unknowns. Three solve paths
are available via `solve(linear_solver=...)`, with `"auto"` choosing between them:

- **Sparse LU (`"splu"`, SciPy).** 3D nested-dissection LU costs `~O(N²)` work and
  `~O(N^{4/3})` memory — fill-in is severe in 3D (unlike 2D, where LU is
  near-optimal). Empirically the practical ceiling is **~50³ (≈1.3×10⁵
  unknowns)**, tens of seconds and ~1–2 GB; 60³–64³ is the edge. It is exact
  (no tolerance) and best for a single small **linear** solve.
- **AMG-preconditioned CG (`"amg"`, pyamg).** Smoothed-aggregation AMG accelerated
  by CG. Setup is `O(N)` and the V-cycle count is **grid-independent (~14–17)**,
  so it scales where LU cannot: 41³ in 0.3 s (vs 21 s for LU), 121³ (1.8×10⁶
  unknowns) in ~13 s, agreeing with LU to ~`1e-10`. This lifts the ceiling to
  128³–256³ and is the path that makes the issue-1 refinement study feasible.
- **Jacobi-preconditioned CG (`"cg"`, NumPy-only fallback).** Memory is `O(N)`,
  but the iteration count grows like `n·√κ` with `κ ∝ (μ_max/μ_min)(L/h)²` — correct
  but slow at high permeability contrast. Used only when neither SciPy nor pyamg
  is present.

**`"auto"`** prefers AMG whenever pyamg is present and the grid is beyond
trivially small (`~20³` — AMG already wins by ~8× at 21³ and ~50× at 41³, and
nonlinear problems compound this since LU re-factorizes every Picard step),
keeping the exact sparse LU only for tiny grids and as the fallback when pyamg is
absent (then CG below the LU ceiling). pyamg is an optional dependency, not a hard
requirement.

### 4. Open-boundary truncation

`ψ` of a magnetized body decays as a dipole tail `~1/r²` in 3D. The Dirichlet
`ψ = 0` outer boundary truncates that tail; the induced error at the sample scales
like `(a/R_box)³` for feature size `a` and box half-size `R_box`. Putting the
boundary at `R_box ≳ 5a` gives `≲ 1%`; `≳ 8–10a` gives `≲ 0.1%`. (The 3D `1/r²`
decay is actually *faster* than the 2D planar `A_z ~ ln r`, so the same box
margin is a little safer here than in `PlanarMagnetostatics`.)

### 5. Staircased geometry — O(h) corner spikes

Curved surfaces (spheres, cylinders) become voxel staircases on a structured
grid, producing `O(h)` spurious field spikes at re-entrant corners — the same
effect the 2D example already documents. Consequence for users: report **region
means**, not the pointwise `|B|` peak, as the saturation/field metric. The 3D
example follows the 2D convention on this.

## Validation targets (unit tests)

1. **Coil in free space** → `ψ ≡ 0`, `B = μ₀ H_s` matches `biot_savart` to
   machine precision (source term + degenerate-solve sanity).
2. **Uniformly magnetized sphere** (recoil `μ_r = 1`) → interior `B = (2/3) B_r`,
   uniform (demagnetizing factor `1/3`; the 3D analog of the 2D cylinder's `1/2`).
   Tolerance is set by staircasing (issue 5) and box truncation (issue 4).
3. **High-`μ_r` sphere in a uniform applied field** → interior field enhancement
   toward the `3μ_r/(μ_r+2)` limit, at a `μ_r` in the trustworthy band (issue 1).
4. **Linear-below-the-knee** → a saturable core at weak drive matches its
   small-signal linear counterpart (mirrors the 2D nonlinear test).

## Worked application: the NMR-MOUSE

`examples/plot_nmr_mouse_3d.py` simulates the **NMR-MOUSE** (MObile Universal
Surface Explorer; Blümich et al., *Magn. Reson. Imaging* **16**, 479 (1998),
`References/`), a single-sided "inside-out" NMR sensor. It is the motivating class
of problem for this solver: it needs **permanent magnets and a soft-magnetic yoke
solved together**, which analytic dipole superposition cannot do.

**Geometry (paper Fig. 1).** Two permanent magnets of *anti-parallel*
magnetization (left `+y`, right `−y`) sit on a soft-iron yoke with a gap between
them; a surface RF coil in the gap excites a sensitive volume in the stray field a
few mm *above* the device. Frame: `z` across the gap, `y` out of the surface, `x`
along the bars. Dimensions: 26 mm-wide × 32 mm-tall magnets, 13 mm gap, 15 mm
yoke. The field arches over the gap so its dominant component `B0_z` (along the
gap) is roughly orthogonal to the coil's `B1` (`y`) — the geometry that makes the
sensor work.

**What it exercises and reproduces** (NdFeB `B_r = 1.2` T, linear `μ_r = 1000`
yoke, 1.5 mm grid, ~0.8 M nodes, one AMG solve):

| quantity | simulation | paper |
|---|---|---|
| `B0` at the sensor (`y = 2.5` mm) | ~0.43 T → **18 MHz** (¹H) | 0.41 T, 17.5 MHz |
| iron-yoke field boost vs magnets-only | **+10 %** | (the yoke's raison d'être) |
| penetration (depth) gradient `dB0/dy` | ~14 T/m | 10–14 T/m |
| lateral `B0_z(z)` across the gap | near-quadratic | Fig. 2 (quadratic) |

The **yoke boost is the headline**: removing the yoke (a one-line change) drops the
sensor field ~10 %, the low-reluctance return-path effect that only a
magnetostatic solver captures. The yoke runs at mean `|B| ≈ 1.1` T, below
saturation, so a linear `μ_r` matches a saturable `soft_iron()` model here (which
agrees to <1 % but costs ~30× more) — a good illustration of when the nonlinear
path is *not* needed. Two accuracy caveats apply and are noted in the example: the
sensor sits just above the sharp magnet corners, whose `O(h)` staircasing (issue
5) gives the *absolute* field a ~10 % resolution sensitivity (the *shape*, the
yoke effect, and the operating point are robust); and the high-`μ_r` yoke is in
the cancellation-error band (issue 1), but the reported fields are in the air
above it, which is unaffected.

A subtle but instructive point surfaced while building the map panel: `B0_z` must
be *even* in `z` for this anti-parallel geometry (and `B0_y` odd), which the solve
satisfies to machine precision — a free symmetry check that caught a plotting-only
axis-transpose bug. It is a reminder that the solver's exact symmetries are a
useful correctness probe for any new geometry.

### Driving the single-sided workflow with the solved field

The solved field is wired into the higher-level workflow the same way the analytic
magnet is. `ScalarPotentialSolution.to_magnet_field_maps(...)` samples the solved
`B0` (and an optional coil `B1`) onto a `(lateral, depth)` plane and returns the
same `MagnetFieldMaps` container the analytic `sample_magnet_field` produces, so a
solved 3-D field is a drop-in field source. `workflows.SolvedMouseField` wraps a
solution behind the `MouseFieldSource` interface, and `simulate_mouse_cpmg` /
`mouse_depth_profile` / `resonant_depth` now accept either bars (analytic, the
unchanged default) or a field source. The end-to-end example
`plot_nmr_mouse_depth_profile_solved.py` drives the moving-walker depth-profiling
measurement from the 3-D MOUSE solve (magnets **and** the finite iron yoke, rather
than the analytic image-yoke) and recovers a buried density gap as a hole in the
profile. `tests/test_single_sided.py` covers the solved-field path and the
backward-compatible analytic path together.

The example is also careful about the depth physics. The walker signal is the
*intrinsic* slice response, which actually *rises* with depth (the gradient
weakens, so a fixed-bandwidth pulse excites a thicker slice); the engine does not
include the geometric detection sensitivity. The measured signal multiplies by
that sensitivity, `S(d) ~ B0(d)^2 B1(d)^2` (Curie polarization `~ B0`, reciprocity
reception `~ omega_0 B1`, transmit `~ B1`, coil reception `~ B1`), computed from
the solved `B0` and a surface-coil `B1`. `S` falls ~100x over the first cm
(B1-dominated), so the *measured* profile decays with depth as real single-sided
NMR does -- the raw walker profile alone must not be read as the measured signal.

## Status

- [x] Assessment + quantified RSP limits (this document)
- [x] `ReducedScalarPotential3D` solver (`fields/scalar_potential_3d.py`)
- [x] `_anderson_picard` `rhs_fn` generalization + `_sparse_factorize_3d`
- [x] Unit tests (`tests/test_scalar_potential_3d.py`)
- [x] Worked examples: `plot_ferrite_sphere_3d.py` (accuracy vs `μ_r`),
      `plot_high_mu_convergence_3d.py` (AMG scaling + refinement),
      `plot_nmr_mouse_3d.py` (NMR-MOUSE magnet + iron yoke, reproduces the paper),
      `plot_nmr_mouse_depth_profile_solved.py` (solved field driving the workflow)
- [x] Workflow integration: `to_magnet_field_maps` adapter + `SolvedMouseField`
      drive the single-sided depth-profiling workflow from a solved field
- [x] Optional AMG (`pyamg`) preconditioner (`_amg_linsolve_3d`,
      `solve(linear_solver="amg")`) — grid-independent iterations, scales to
      >10⁶ unknowns; `"auto"` dispatch + solver-agreement / refinement tests
- [~] Simkin–Trowbridge total/reduced split — **investigated and shown to be a
      no-op** on this centered nodal-FV scheme (discretely identical to the
      reduced formulation, verified to machine precision; see issue 1). Superseded
      by AMG-enabled grid refinement, which is the effective high-`μ_r` lever.
- [ ] Body-fitted / cut-cell curved boundaries to lift the `O(h)` staircase floor
      (the more promising future direction for high `μ_r`; larger scope)
