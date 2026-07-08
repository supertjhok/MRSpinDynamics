# Coil-Property Extraction for Single-Layer RF Solenoids

Design/progress note for `spin_dynamics.fields.coil_properties`, which turns a
single-layer solenoid's geometry into the lumped RF properties a probe designer needs.

## Motivation

The field solvers already produce the B-field (`fields.magnetostatics.biot_savart`), the
induced E-field / eddy loss (`fields.quasistatic`), and a field-based **DC inductance**
(`quasistatic.coil_inductance`, a Neumann self+mutual sum over coaxial turns). But those
fields were never distilled into the numbers a matching network and SNR budget are built
from:

- **series inductance** `L` (and how it rises at RF as the coil becomes electrically long),
- **AC resistance** `R` from the skin and proximity effects,
- **stray capacitance** `C` and the **self-resonant frequency** `f_res`,
- the **unloaded quality factor** `Q = omega L / R`.

Without them, the probe noise/SNR model
(`noise.tuned_probe_output_noise_density`, which consumes `L`, `R`, `C`) and the
matching-network designer (`probes.matched`) had to be fed hand-typed values. This module
closes the loop: **geometry -> `(L, R, C, Q, f_res)` -> probe SNR**.

## Model

The extractor implements the `n = 0` **sheath-helix waveguide** model of Corum, with the
NIST round-wire and field-non-uniformity corrections and the Medhurst proximity data. It
is a faithful port of Serge Y. Stroobandt's [QOIL calculator](https://hamwaves.com/qoil/)
(GPL-3.0), vendored at `References/hamwaves_inductance_calculator/`. The scalar
per-frequency `math` pipeline is re-expressed with `scipy.special` Bessel functions and
`scipy.optimize.brentq` root finds; it reproduces QOIL's own output (its polynomial
Bessel approximations + SLATEC `fzero`) to ~6 significant figures.

Pipeline (all SI):

1. **Proximity factor** `Phi` — bilinear interpolation of the Medhurst `(l/D, p/d)` table
   (`medhurst_proximity_factor`), giving the effective diameter
   `D_eff = D − d(1 − 1/sqrt(Phi))`.
2. **Correction factors** — Lundin field non-uniformity `k_L` (short/long-coil branch at
   `l ≤ D_eff`), Rosa round-wire self-inductance `k_s = 5/4 − ln(2p/d)`, and Knight
   mutual-inductance `k_m` (a series in `1/N`).
3. **AC resistance** — skin depth `delta = sqrt(rho/(pi f mu0 mu_r))` and
   `R = rho·l_w,eff /(pi(d·delta − delta^2))·Phi·(N−1)/N`.
4. **Current-sheet inductance** — the frequency-independent geometric value
   `L_s = mu0[pi(D_eff N)^2/(4l)·k_L − D_eff N(k_s+k_m)/2]`.
5. **Sheath-helix dispersion** at the design frequency — solve
   `K1(τa)I1(τa)/(K0(τa)I0(τa)) = (τ/k0·tan psi)^2` for `τ` (`sheath_helix_dispersion`),
   then `beta = sqrt(k0^2+τ^2)` and `Z_c = 60(beta/k0)I0(τa)K0(τa)`.
6. **Effective (RF) circuit** —
   `L_eff = (Z_c/omega)·tan(beta·l)·k_L − mu0 D_eff N(k_s+k_m)/2`, `X_eff = omega L_eff`,
   `Q = X_eff/R`.
7. **Stray capacitance** `C_p` — reconciling the effective and lumped equivalents at the
   design frequency.
8. **Self-resonance** — the first (quarter-wave) resonance where `beta(f)·l = pi/2`.

Near a self-resonance `tan(beta·l)` diverges and the dispersion solve degenerates; the
effective-circuit fields (`L_eff`, `Q`, `C_p`) are then returned as `nan` while `L_s` and
`R` remain valid — mirroring QOIL's graceful degradation.

## API

```python
from spin_dynamics.fields.coil_properties import solenoid_properties, ANNEALED_COPPER

cp = solenoid_properties(
    diameter=20e-3, length=30e-3, turns=10, wire_diameter=1.0e-3,
    frequency=10e6, material=ANNEALED_COPPER,
)
cp.inductance_effective        # RF series inductance (H)
cp.ac_resistance               # skin+proximity resistance (ohm)
cp.q_factor                    # unloaded Q
cp.self_resonant_frequency     # first self-resonance (Hz)
cp.tuning_capacitance()        # C to resonate the tank at the design f (F)
cp.to_probe_params()           # {"L", "R", "C"} for noise.tuned_probe_output_noise_density
```

`CoilProperties` also carries every intermediate (`proximity_phi`, `effective_diameter`,
`k_L/k_s/k_m`, `skin_depth`, `beta`, `characteristic_impedance`, …).

### Conductor material and temperature

Any metal is supported through `ConductorMaterial`; presets `ANNEALED_COPPER`,
`HARD_DRAWN_COPPER`, `SILVER`, `ALUMINIUM` carry the QOIL resistivities (at 293.15 K)
plus a linear temperature coefficient. Only the AC resistance (hence `Q`) depends on
temperature — the inductance, capacitance and self-resonance are geometric.

```python
# A cryoprobe coil: cooling cuts resistivity, so R falls and unloaded Q rises.
ofhc = ConductorMaterial("OFHC copper", 17.241e-9, mu_r=0.99999044,
                         temp_coefficient=3.93e-3, residual_resistivity_ratio=100.0)
cold = solenoid_properties(diameter=20e-3, length=30e-3, turns=10,
                           wire_diameter=1.0e-3, frequency=10e6,
                           material=ofhc, temperature=77.0)   # Q roughly doubles vs 293 K
```

`ConductorMaterial.resistivity_at(T)` uses two models. Without a
`residual_resistivity_ratio` it applies the linear coefficient
`rho(T) = rho_ref[1 + alpha(T − T_ref)]`, accurate within ~±100 K of room temperature but
unphysical (negative) at cryogenic temperatures. With an RRR it switches to Matthiessen's
rule — a residual floor `rho_res = rho_ref/RRR` plus a linear (high-temperature
Bloch–Grüneisen limit) phonon term — valid from cryogenic up through room temperature. The
linear phonon term over-predicts the ideal resistivity below ~Debye/3 (≈110 K for copper),
so cryogenic `R` is a conservative (upper-bound) estimate; supply a measured `resistivity`
at the operating temperature for a precise value.

## Validation

- **QOIL regression** — `tests/test_coil_properties.py` checks four designs (a normal
  coil, a cubical high-Q coil, an electrically-long coil, a single turn) against reference
  values generated from the vendored QOIL math; agreement is < 0.05 %.
- **Field-based cross-check** — `solenoid_field_inductance` builds the coaxial loops and
  sums the Biot-Savart / Neumann partial inductances via `quasistatic.coil_inductance`, an
  independent solver whose DC inductance agrees with `L_s` to within ~10 %.
- **Physical trends** — Q rises with wire diameter, `f_res` falls with more turns, `R`
  grows ~`sqrt(f)`, and Q peaks near a cubical form factor (`l ≈ D`), as
  `examples/plot_solenoid_coil_properties.py` demonstrates.

## Scope and follow-on (staged)

This module covers **single-layer air-core solenoids** — the geometry with a validated
semi-analytical reference. The planned follow-on adds **field-based** extraction for
arbitrary coil geometries, slotting in alongside `solenoid_properties`:

- **Proximity-effect resistance** from a 2-D eddy-current solve over the wire
  cross-section driven by the neighbour-turn field.
- **Turn-to-turn / stray capacitance** from an electrostatic (charge) solve.

These reuse the existing `fields.quasistatic` machinery and remove the single-layer /
round-wire restriction, at the cost of a mesh and no closed-form reference.

## Attribution

The formulas and the Medhurst table are ported from the GPL-3.0 QOIL calculator by
Serge Y. Stroobandt (ON4AA), <https://hamwaves.com/qoil/>. References: Corum & Corum
(IEEE Microwave Review, 2001); D. W. Knight, G3YNH; R. Lundin (Proc. IEEE, 1985);
R. G. Medhurst (Wireless Engineer, 1947).
