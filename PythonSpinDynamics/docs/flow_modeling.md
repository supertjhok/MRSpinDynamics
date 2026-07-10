# Flowing-Sample NMR

Design and validation note for `spin_dynamics.flow` and the flow extensions to
`spin_dynamics.thermal`. Many NMR measurements run on a moving sample -- inline
process monitoring in a pipe, stopped-flow kinetics, hyphenated LC-NMR -- where
flow changes both the measured signal and the sample temperature.

## Physics

Two flow regimes are modeled, both for a cylindrical pipe of radius `R` with a
sensitive (detection) region of length `L`:

- **plug** -- uniform velocity `v = Q/A`, single residence time `tau = V/Q`;
- **laminar** -- fully developed Hagen-Poiseuille profile
  `v(r) = 2 v_mean (1 - (r/R)^2)`, a *distribution* of residence times.

with `Q` the volumetric flow rate, `A = pi R^2`, `V = A L`,
`v_mean = Q/A`, `tau = V/Q = L/v_mean`.

### Washout during acquisition (`flow.washout_fraction`)

At `t = 0` the sensitive volume is uniformly excited; thereafter excited spins
leave and fresh unexcited spins enter, so the acquired signal is multiplied by
the fraction `W(t)` of originally-excited spins still present. Averaging the
remaining residence time over axial position and the velocity profile:

```text
plug:     W(t) = 1 - t/tau                       (0 <= t <= tau, then 0)
laminar:  W(t) = 1 - t/tau        (t <= tau/2)
          W(t) = tau / (4 t)      (t >  tau/2)
```

Both regimes share the initial slope `-1/tau`, so the early-time apparent rate
is `1/T2 + 1/tau` in either case (`flow.flow_enhanced_rate`); they diverge only
past `t = tau/2`, where plug flow cuts off linearly and laminar flow develops a
slow `1/t` tail from the near-wall streamlines. The measured signal is
`apply_washout(flow, t, clean_signal)`.

The laminar `W(t)` derivation: a spin at axial position `x`, radius `r` is still
present at time `t` iff `(L - x)/v(r) > t`. Integrating the indicator over the
uniformly-filled volume with `u = (r/R)^2`,
`W(t) = integral_0^1 max(1 - (2t/tau)(1-u), 0) du`, which evaluates to the
piecewise form above (continuous at `t = tau/2`, both branches giving `1/2`).

### Transit-time inflow polarization (`flow.inflow_polarization`)

Spins entering the detector carry the longitudinal magnetization they built up
during transit through the upstream prepolarizer; the detector sees the
flux-weighted average. Reusing `prepolarization.longitudinal_recovery`, plug
flow reduces to the single-transit `prepolarization.prepolarized_flow_state`,
while laminar flow integrates over the parabolic profile with the flux weight
`2(1-u)` and transit time `tau_pre/(2(1-u))`:

```text
Mz_inflow = integral_0^1  2(1-u) * recovery(tau_pre / (2(1-u)))  du
```

This bounded quadrature avoids the singular `1/t^3` exit-age RTD tail (the RTD
itself, `E(t) = tau^2 / (2 t^3)` for `t >= tau/2`, is exposed as
`flow.transit_time_distribution` for inspection). Slower flow -> longer transit
-> more polarization, saturating at the polarizing-field equilibrium.

### Thermal advection

A flowing sample carries heat away. Two entry points:

- **Lumped** (`thermal.flow_conductance`): `G = rho c_p Q`, used as a link from
  the sample node to a bath at the inlet temperature, so removed power is
  `G (T_sample - T_in)` and the steady rise is `P / (rho c_p Q)`. This is
  formally identical to the Pennes perfusion sink -- blood perfusion is a flow
  term.
- **Distributed** (`thermal.Conduction1D(velocity=...)`): first-order upwind
  axial advection `-rho c_p v dT/dx` added to the slab conduction operator,
  with the inlet cell drawing from the Dirichlet inlet temperature. The steady
  profile is the classic advection-diffusion boundary layer
  `T(x) = T0 + (TL - T0)(e^{Pe x/L} - 1)/(e^{Pe} - 1)`, `Pe = rho c_p v L / k`.

## Validation

- **Washout** -- analytic piecewise `W(t)`, shared initial slope, the density
  identity `integral_0^T E dt = 1 - W(T)`, and independent Monte-Carlo
  cross-checks (spins sampled uniformly in the pipe volume with the parabolic
  velocity, counted as they exit) for both regimes.
- **Inflow polarization** -- plug equals `prepolarized_flow_state`; RTD
  normalization and mean `tau`; bounds, monotonicity in flow rate, the
  slow-flow full-equilibrium limit, and a flux-weighted Monte-Carlo
  cross-check.
- **Thermal advection** -- `flow_conductance` lumped closed form
  `T = T_in + P/(rho c_p Q)`; the 1D advection-diffusion profile against the
  analytic `exp(Pe x/L)` solution (<1% of span); zero-velocity reduces to
  linear conduction; high-Peclet boundary layer; reverse-flow mirror symmetry.

FEMM is not used here: its heat-flow solver models conduction plus convection
boundaries but not bulk advection, so the advection term is validated
analytically and by Monte-Carlo instead. See
[thermal_modeling.md](thermal_modeling.md) for the FEMM-validated conduction
core.

## Example

`examples/plot_flowing_pipe_nmr.py` -- inline process monitoring on a flowing
pipe sample: CPMG washout (static vs plug vs laminar), apparent `T2` and
detected amplitude versus flow rate, and the steady sample temperature set by
advective cooling.

## Boundaries and caveats

- Plug and laminar are the two implemented regimes; turbulent/well-mixed flow
  (exponential washout, the CSTR limit that the `exchange` Bloch-McConnell
  layer would model) is not included.
- Washout assumes a sharply-bounded sensitive volume uniformly excited at
  `t = 0`; a graded coil sensitivity profile is not modeled.
- Advection in `Conduction1D` is first-order upwind (some numerical diffusion;
  keep the cell Peclet `rho c_p v dr / k` below ~2) and slab-only.
- Diffusion/Taylor dispersion of the analyte itself is not added to the flow
  washout; it is separable and can be layered on later.
