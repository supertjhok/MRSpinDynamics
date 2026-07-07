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
for unsaturated soft iron near its small-signal permeability. The classical fix
is the **Simkin–Trowbridge total/reduced split** (total scalar potential inside
iron, reduced outside, coupled at the interface), which removes the cancellation
entirely; it is deliberately *out of scope* for this first solver and is the
documented upgrade path. The floating-point part of the cancellation is
negligible by comparison (`μ_r · ε_machine ≈ 1e-13`); the limit is discretization,
so it improves with grid refinement rather than being a hard wall.

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

The 7-point SPD matrix on an `n³` grid has `N = n³` unknowns. The two solve paths
scale very differently:

- **Sparse LU (`splu`, default when SciPy present).** 3D nested-dissection LU
  costs `~O(N²)` work and `~O(N^{4/3})` memory — fill-in is severe in 3D (unlike
  2D, where LU is near-optimal). Empirically the practical ceiling on a
  workstation is **~50³ (≈1.3×10⁵ unknowns)**, tens of seconds and ~1–2 GB;
  60³–64³ is the edge. This is *fine* for design-scale problems (a magnet + shim
  ring resolved at a few mm) but is a hard memory wall, not a gentle slowdown.
- **Jacobi-preconditioned CG (fallback, no SciPy).** Memory is O(N), but the
  iteration count grows like `n·√κ` with condition number `κ ∝ (μ_max/μ_min)(L/h)²`.
  With a 1000:1 permeability contrast and `n ≈ 50` this is many thousands of
  iterations — correct but slow.

**Verdict: cap structured 3D grids at ~50–64³ with the built-in solvers.** The
scale-up path for both accuracy (issue 1) and speed is the same as flagged in the
original assessment: an **AMG-preconditioned CG** (e.g. `pyamg`), which would push
practical grids to 128³–256³. It is left as an optional dependency rather than a
hard requirement.

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

## Status

- [x] Assessment + quantified RSP limits (this document)
- [x] `ReducedScalarPotential3D` solver (`fields/scalar_potential_3d.py`)
- [x] `_anderson_picard` `rhs_fn` generalization + `_sparse_factorize_3d`
- [x] Unit tests (`tests/test_scalar_potential_3d.py`)
- [x] Worked example (`examples/plot_ferrite_sphere_3d.py`)
- [ ] Simkin–Trowbridge total/reduced split for high-`μ_r` iron (future)
- [ ] Optional AMG (`pyamg`) preconditioner for >64³ grids (future)
