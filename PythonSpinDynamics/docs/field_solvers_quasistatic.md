# Quasistatic E-field and Eddy-Current Solvers

Design/progress note for adding induced-E-field and eddy-current solvers to the
`spin_dynamics.fields` package, alongside the existing magnetostatic (Biot-Savart)
B0/B1 solvers.

## Motivation

The Biot-Savart solver (`fields.magnetostatics`) predicts the B-field of a coil
but **not the accompanying E-field**. In the magnetoquasistatic (MQS) regime a
time-varying coil current induces `E = -dA/dt - grad(phi)`, and in a conductive
sample this E-field drives **eddy currents** `J = sigma*E`. Those eddy currents
cause (i) resistive **power deposition** in the sample, (ii) a secondary field
that perturbs B1, and (iii) for gradient coils, the classic **eddy-current
field decay** that pre-emphasis must correct. None of this is visible from B
alone. This note adds the E-field and eddy machinery and wires the gradient
eddy time constants into the M5 `GradientDriverResponse`.

## Physics and the first-order (Born) approximation

MQS assumptions: quasistatic (displacement current negligible, `sigma >> omega*eps`),
wavelength >> object. Then `B = curl(A)`, `curl(E) = -dB/dt`, `J = sigma*E`,
`div(J) = 0`.

The vector potential of the drive coil (closed filaments -> Coulomb gauge,
`div(A) = 0`) gives the **inductive** induced field `E_ind = -dA/dt = -A_unit*dI/dt`,
where `A_unit(r)` is the per-unit-current vector potential. In a finite conductor,
charge conservation and the no-normal-current boundary condition `J.n = 0` add a
scalar-potential correction `phi` solving a **Neumann-Laplace** problem
(`lap(phi) = 0` inside for homogeneous sigma, `dphi/dn = E_ind.n` on the
boundary). The eddy current is `J = sigma*(E_ind - grad(phi))` and the
time-averaged power is `P = (1/2) integral sigma |E|^2 dV`.

**This is a first-order / Born model. Stated limitations (important):**

- It uses the **primary** field only; it **ignores the secondary (reaction)
  field** of the eddy currents feeding back on the driving field. It is therefore
  valid only when the eddy currents do not significantly perturb the source --
  i.e. **low conductivity, thin samples, or skin depth >> sample size**
  (`delta = sqrt(2/(mu0 sigma omega)) >> L`).
- Consequently there is **no skin-effect shielding**: at high frequency /
  high conductivity (`delta << L`) the model **over-estimates** the interior
  E-field and the deposited power, because in reality the field is expelled from
  the conductor bulk.
- The `phi`-correction enforces charge conservation in a finite sample but is
  still first order (no reaction field).
- It is **not** a full-wave or self-consistent eddy solver; for that regime a
  volume A-V eigenmode / impedance solve would be required (a possible follow-on).

The **gradient eddy-current dynamics** model below is a *separate* and, within
the filamentary-shield idealization, **self-consistent** calculation: it includes
the eddy loops' mutual inductance, so the L/R decay time constants are physical
(not first order). Its idealization is that the conducting structure is
represented by discrete filamentary loops of a chosen wire cross-section.

## Module organization

The `fields` package is the unified home for all field solvers. Existing:
`magnetostatics` (B0/B1 Biot-Savart), `domain`/`maps`/`interpolate`/`positions`
(grid + sampling). Added here:

```text
spin_dynamics/fields/
  coils.py         # coil geometry generators (segment lists, reusing circular_loop)
  quasistatic.py   # vector potential, induced E-field, inductance, eddy power
  eddy_modes.py    # eddy-current L/R eigenmodes -> (alpha_k, tau_k) -> GradientDriverResponse
```

`fields.__init__` re-exports the field-solver families (B0/B1, E/eddy, coils,
grid) so `spin_dynamics.fields` is a single coherent field-solver namespace.

### `coils.py`

Geometry generators returning the existing `[(start, end), ...]` segment format
(so they feed `biot_savart` and `vector_potential` unchanged):
`solenoid`, `planar_spiral`, `maxwell_pair` (gradient), `saddle` (optional),
`conducting_ring`, `cylindrical_shield` (a stack of coaxial loops for the eddy
model). Reuses `magnetostatics.circular_loop`.

### `quasistatic.py`

- `vector_potential(points, segments, current)` -- closed-form per straight
  segment (`A = mu0 I/4pi * L_hat * ln((R1+R2+L)/(R1+R2-L))`), mirroring
  `biot_savart`'s structure. Returns `(..., 3)` in T*m.
- `induced_efield(points, segments, dIdt)` -- `-A_unit * dIdt`, in V/m.
- `mutual_inductance(loop_a, loop_b)`, `self_inductance_circular(radius,
  wire_radius)` -- from the vector potential / standard formulae.
- `EddyRegion` + `eddy_currents(...)` -- the sample conductivity mask on a
  `SpatialDomain`, the Neumann-Laplace `phi`-solve, and `J`, `power`. Symmetric
  shapes (sphere/cylinder/slab) validate against analytics; arbitrary masks use
  the `phi`-correction.
- **Sample loading (reflected impedance).** By reciprocity the eddy loss couples
  back into the coil as an added series resistance
  `R_reflected(omega) = omega^2 * integral sigma |A_unit|^2 dV`
  (`reflected_resistance`; geometry factor `geometric_loss_integral`). It is
  frequency dependent (`~omega^2` in the first-order regime), degrades the loaded
  `Q = omega L / R` and raises the Johnson-noise floor by `sqrt(R_total/R_coil)`.
  `coil_loading` sweeps it across frequency given `coil_inductance` and
  `R_coil(f)`. Demonstrated on an inside-out NMR well-logging tool (Jasper-Jackson
  geometry: a loop/solenoid coil in a saturated-brine borehole, 0.5-2 MHz) where
  the brine loading dominates the coil resistance and collapses Q.

### `eddy_modes.py`

- `EddyModes(loops, wire_radius, resistivity)` -- builds `R` (diagonal), `L`
  (mutual + self inductance) and, given a drive coil and a sample point, `M`
  (coil<->eddy) and the field-coupling vector; eigen-decomposes `L^{-1}R` into
  decay rates `1/tau_k` and residues `alpha_k` for the field at the sample.
- `.eddy_terms()` -> `((alpha_k, tau_k), ...)` and `.to_gradient_driver(tau_rl)`
  -> a M5 `optimal_control.control_response.GradientDriverResponse`. This closes
  the loop: coil + shield geometry now *predicts* the eddy terms that were
  hand-specified in M5.

## Validation (analytic)

- **Solenoid Faraday E:** inside a long solenoid `B_z` is uniform, so
  `E_phi = -(r/2) dB/dt`. Direct test of `induced_efield`.
- **Vector potential / curl:** finite-difference `curl(A)` reproduces
  `biot_savart` B for a loop.
- **Mutual inductance:** two coaxial loops vs Maxwell's elliptic-integral formula.
- **Ring time constant:** single conducting ring `tau = L/R` with
  `L = mu0 a (ln(8a/r_w) - 2)`, `R = 2 pi a rho/(pi r_w^2)`.
- **Conducting sphere power:** uniform sinusoidal `B0 cos(omega t)` ->
  `P_avg = pi sigma omega^2 B0^2 a^5 / 15` (eddy currents divergence-free, so no
  `phi`-correction needed -- the clean first-order test).

## Wiring and examples

- `eddy_modes` -> `GradientDriverResponse` -> M5 `make_pgse_objective`: the
  gradient example derives `(alpha_k, tau_k)` from a Maxwell-pair + shield and
  feeds them into the GRAPE pre-emphasis, so the optimizer corrects the
  *physically predicted* eddy droop.
- `examples/plot_rf_coil_efield.py` -- solenoid & planar spiral: B1 map (Biot-
  Savart) + induced E-field + power deposition in a conductive sample, with the
  first-order caveat annotated.
- `examples/plot_gradient_eddy_preemphasis.py` -- Maxwell pair + cylindrical
  shield: eddy `(alpha_k, tau_k)`, the field step-response droop, and the
  pre-emphasis (analytic + GRAPE) that corrects it.

## Milestones

- [x] Add this design note.
- [x] `coils.py` (solenoid, planar_spiral, maxwell_pair, conducting_ring,
      cylindrical_shield) + tests.
- [x] `quasistatic.py` (vector_potential, induced_efield, inductance, eddy
      power + phi-correction) + tests (curl(A)=B ~1e-10, solenoid E ~1%, mutual L
      ~1e-3, sphere power ~0.6%, charge-correction screening/tangency).
- [x] `eddy_modes.py` (L/R eigenmodes -> terms -> GradientDriverResponse) +
      single-ring tau=L/R test (machine precision).
- [x] Sample loading: `reflected_resistance`/`coil_loading`/`coil_inductance`
      (reflected impedance ~omega^2 -> loaded Q + noise penalty) + tests
      (omega^2 scaling, sigma-linearity, Q degradation); NMR well-logging example.
- [x] `fields.__init__` re-exports; unified field-solver namespace.
- [x] Examples (`plot_rf_coil_efield.py`, `plot_gradient_eddy_preemphasis.py`)
      + registration; the gradient example feeds the geometry-derived driver into
      the M5 PGSE objective.
- [x] Manual section (SS Quasistatic E-field and eddy currents) + API reference
      regen + PDF (99 pp). 12 field tests green, ruff clean.

## Open questions

- Whether to add a self-consistent (reaction-field) eddy solver later for the
  high-conductivity / skin-effect regime the first-order model cannot reach.
- Whether to fit reduced `(alpha_k, tau_k)` (few dominant modes) vs. exposing the
  full eigenspectrum for the gradient pre-emphasis.
