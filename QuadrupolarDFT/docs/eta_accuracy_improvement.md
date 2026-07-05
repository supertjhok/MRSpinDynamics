# Improving the accuracy of asymmetry-parameter (eta) predictions

**Status:** analysis / roadmap. Written 2026-07-05.

## The observation

Static-DFT EFG runs on NaNO2 and glycine reproduce the quadrupolar coupling
constant `C_Q` reasonably well but **systematically under-predict the asymmetry
parameter `eta`**. For NaNO2 14N the experimental target is `eta ~ 0.38`
(`C_Q ~ 5.49 MHz`, ordered phase); our static runs come in low.

This document records *why* eta is the harder quantity, rules out the code as the
cause, and lays out a ranked set of experiments to close the gap. It is a
reference for the eta work, not a changelog.

## Why eta is hard, and what "under-predicted" means

`C_Q` is proportional to `V_zz`, the **largest-magnitude** EFG eigenvalue. It is
dominated by the nearest-neighbour bonding charge and is numerically robust.

`eta = (V_xx - V_yy) / V_zz` is a **normalized difference of the two smaller
eigenvalues**. It lives entirely in the transverse plane and is a second-order
quantity: small changes in geometry, long-range electrostatics, sampling, or
functional wash out of `C_Q` but move `eta` a lot.

A *systematically low* eta means the computed EFG is **too axially symmetric** --
the calculation is missing whatever lifts the degeneracy between the two
transverse principal directions. In a real crystal, the things that break that
degeneracy are mostly long-range (crystal field / packing) or dynamic
(large-amplitude, anharmonic motion) -- precisely what a single static PBE
unit-cell calculation handles worst.

**Key framing:** eta is independent of the nuclear quadrupole moment `Q`, so it
is the *honest* benchmark of the electronic structure. `C_Q` agreement can ride
on the tabulated `Q` and on cancellation in `V_zz`; eta cannot hide behind a
calibration constant. Good `C_Q` with low eta is the classic signature of a real
geometry/electronic deficiency rather than luck.

## What is NOT the cause (already checked)

- **Convention / definition.** `tensors.py` orders eigenvalues by |magnitude|
  (`|V_zz| >= |V_yy| >= |V_xx|`) and computes `eta = |(V_xx - V_yy)/V_zz|`. This
  is the standard convention and is correct.
- **Averaging order.** `vibrational.py::thermally_averaged_efg` averages the
  **full Cartesian tensor** over the thermal ensemble and diagonalizes
  *afterwards* (the right order; averaging eigenvalues would be wrong and would
  itself bias eta). So the harmonic averaging code is not artificially killing
  eta -- the gap is physical/methodological.

## Current baseline (for reference)

- Backend: **ABINIT + PAW**, `nucefg 2`.
- Functional: **PBE**, `Pseudodojo_paw_pbe_standard` datasets. **No dispersion.**
- Convergence (starter `nano2_efg.abi`): `ecut 25`, `pawecutdg 50`,
  `ngkpt 4 4 4` -- explicitly flagged as starter, not converged, values.
- Structure: experimental CIF, optional internal-coordinate relaxation at the
  fixed experimental cell (`relax.py`, `optcell 0`, `ionmov 2`).
- Dynamics: second-order harmonic tensor averaging + Bayer single-mode fit.

## Ranked ideas to improve eta

### Tier 1 -- nearly free, diagnostic (do first)

1. **eta-specific convergence study.** `C_Q` converges long before eta, because
   eta is a small difference amplified by the PAW fine grid (`pawecutdg`) and
   k-sampling. Sweep `pawecutdg` (60/80/100) and `ngkpt` (up to 6^3) holding
   everything else fixed, and plot **eta**, not energy or `C_Q`. It is common for
   eta to climb by 0.05-0.15 as the transverse components converge. Cheap, and it
   tells us how much of the gap is just numerics. *(This is the task implemented
   in `examples/abinit/efg_convergence.py`.)*
2. **Symmetry check.** If ABINIT's `nsym`/site symmetry forces `V_xx = V_yy` at
   the target site, eta is clamped near zero by construction. Confirm the space
   group does not impose axial symmetry and, as a control, run once with
   `nsym 1`. If eta jumps, we were symmetry-locked. (The starter input already
   sets `nsym 1`; CIF-expanded inputs may not.)
3. **Relaxed vs experimental geometry, reported as eta.** GGA relaxation tends to
   over-symmetrize local geometry (straightening angles, equalizing near-neighbour
   distances), draining transverse asymmetry and lowering eta. Compare
   experimental-CIF eta against relaxed eta head-to-head. Expectation: the
   experimental (or lightly relaxed) geometry gives higher, more accurate eta.

### Tier 2 -- structural / electrostatic levers (likely the main story)

4. **Add dispersion (D3 / TS / MBD), even at fixed cell.** Both systems are
   molecular/ionic crystals (glycine especially). The transverse EFG at N is set
   largely by the crystal field from neighbouring molecules (H-bond geometry, ion
   packing). PBE without dispersion misplaces those neighbours; even the fixed-cell
   internal relaxation drifts H-bond angles without vdW. Getting packing right is
   probably the single biggest geometry lever for glycine eta.
5. **Fix the proton positions.** X-ray H positions sit ~0.1 A too short along
   N-H / O-H, and the N-H orientation directly tilts the transverse EFG. Use
   neutron H positions for glycine where available, or relax only H with the
   heavy framework fixed at experimental. Quantum-nuclear (ZPE / PIMD) elongation
   of X-H pushes eta further as an upper-bound test.
6. **Decompose eta into intramolecular vs crystal-field.** Compute the isolated
   NO2- / glycine unit at its crystal geometry, then add coordination shells (or a
   point-charge / embedding field). If the isolated molecule gives low eta and it
   grows shell-by-shell, the deficiency is long-range electrostatics -- which
   validates spending effort on #4/#5 rather than on the functional.

### Tier 3 -- method upgrades

7. **Hybrid or meta-GGA (r2SCAN, then PBE0 as a check).** The EFG is governed by
   the anisotropy of the valence-p population, which GGA delocalizes. Hybrids and
   r2SCAN routinely tighten p-shell anisotropy and improve EFGs; eta, being pure
   p-anisotropy, benefits more than `C_Q`.
8. **PAW dataset hardness / Sternheimer anisotropy.** The EFG is an all-electron
   near-nucleus property. A soft PAW set with few projectors under-represents the
   *anisotropic* core polarization (the directional part of Sternheimer
   antishielding), which feeds the transverse components. Test harder datasets /
   more projectors / smaller core radius and converge eta against projector count.
   A scalar `(1 - gamma_inf)` antishielding scales all components equally and
   **cannot** fix eta; the dataset must carry the anisotropy natively.

### Tier 4 -- dynamics beyond harmonic (the creative frontier)

9. **Snapshot / AIMD averaging -- especially for NaNO2.** The harmonic
   second-order correction expands around a single symmetric equilibrium, so it
   mostly renormalizes `V_zz` (Bayer) and captures little *transverse*
   fluctuation. But NaNO2 is a textbook order-disorder ferroelectric: the nitrite
   librates and flips with large, anharmonic amplitude, sampling off-axis
   configurations that raise the time-averaged eta above the static value. Run
   finite-T AIMD, compute the EFG tensor per snapshot, average tensors (reuse
   `average_tensors`), then diagonalize. This is the physically-honest candidate
   for the residual NaNO2 gap that no static fix will close.
10. **Per-mode eta sensitivity.** Cheaply, within the existing harmonic machinery,
    report each mode's contribution to eta (not just `C_Q`). If a couple of
    low-frequency librations dominate the transverse curvature, that both explains
    the temperature dependence and tells us which anharmonic motions the snapshot
    averaging most needs to capture.

### Tier 5 -- calibration / benchmarking

11. **Build an eta benchmark line.** DFT EFGs famously need linear scaling;
    `C_Q` has well-known regression corrections and eta has an analogous slope < 1.
    Two compounds cannot fit it, but computing ~10-20 compounds with well-known
    eta spanning 0->1 under one fixed protocol lets us (a) quantify the bias,
    (b) fit a slope/offset correction, and (c) -- most usefully -- see whether the
    bias is *constant* (a correctable method problem) or *scattered* (per-system
    geometry/dynamics, not correctable by a global factor).

## Suggested first three experiments

1. eta-specific `pawecutdg` / `ngkpt` convergence + `nsym 1` control -- free and
   clarifying, and it protects against "fixing" a number that was only
   under-converged.
2. Dispersion-corrected relaxation with neutron / relaxed H positions, comparing
   experimental-vs-relaxed eta -- where the glycine gap most likely lives.
3. AIMD snapshot averaging for NaNO2 -- the order-disorder dynamics are the
   physically-honest explanation for that system.
