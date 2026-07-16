# PHIP, SABRE, and Long-Lived Singlet States

> **Document role (updated 2026-07-16): partially implemented research
> record.** Singlet/triplet state primitives, SLIC long-lived-state workflows,
> and hydrogenative PASADENA/ALTADENA reference models are implemented. A full
> SABRE catalyst-bound/free-pool exchange model remains future work. Users
> should start with
> [`python_api/hyperpolarization.md`](python_api/hyperpolarization.md).

## Research conclusions and implementation plan

This document records the physics and software conclusions from the initial
design study for adding parahydrogen-induced polarization (PHIP), Signal
Amplification By Reversible Exchange (SABRE), and long-lived nuclear singlet
states to PythonSpinDynamics. It is a design reference: implementation should
continue to state its model boundary and validation level explicitly.

## Executive conclusion

PHIP, SABRE, and long-lived singlet states share singlet/triplet operator
algebra, but they do not share the same kinetics. They should therefore use a
common quantum-state foundation and separate workflow layers:

1. singlet/triplet and parahydrogen state primitives;
2. long-lived-state preparation, storage, and readout;
3. hydrogenative PHIP reaction-state mapping and polarization transfer; and
4. SABRE quantum exchange between catalyst-bound and free-spin manifolds.

A Bloch-vector polarization multiplier is not sufficient. These effects
require density matrices because singlet order, antiphase PHIP states,
zero-quantum coherences, and exchange between entangled subsystems are not
representable by independent magnetization vectors.

## Shared singlet physics

For two spin-1/2 nuclei, the antisymmetric singlet state is

```text
|S0> = (|alpha beta> - |beta alpha>) / sqrt(2).
```

The singlet and total-triplet projectors are

```text
P_S = 1/4 I - I1 . I2
P_T = I - P_S,
```

and a convenient trace-zero singlet-order operator is

```text
Q_S = P_S - P_T / 3.
```

For parahydrogen fraction `f_p`, with the three triplet substates equally
populated,

```text
rho_H2 = f_p P_S + (1 - f_p) P_T / 3
       = I / 4 + (f_p - 1/4) Q_S.
```

This normalization has important limiting cases:

- `f_p = 1/4` is the room-temperature statistical mixture and has no usable
  deviation spin order;
- `f_p = 1` is pure parahydrogen; and
- observable hyperpolarization must scale with para excess above `1/4`, not
  with `f_p` alone.

The parahydrogen spin isomer and a molecular long-lived singlet state use the
same two-spin algebra but are not interchangeable physical models. PHIP maps
the former into inequivalent product sites; long-lived-state sequences prepare
and store a singlet-triplet imbalance in an already coupled molecule.

## Hydrogenative PHIP

Hydrogenative PHIP requires pairwise addition of both parahydrogen atoms to the
same product molecule. The reaction breaks their magnetic equivalence and maps
the source singlet correlation into product spin order. A useful simulation
must include:

- para fraction and pairwise-addition fraction;
- source-to-product spin-site mapping;
- product chemical shifts and scalar couplings;
- hydrogenation field and subsequent field trajectory;
- a reaction-time or residence-time distribution;
- relaxation during reaction, transfer, and detection; and
- optional RF or field-cycling transfer to heteronuclei.

PASADENA and ALTADENA should be distinct workflows. PASADENA forms the product
at high field; ALTADENA forms it at low field and transfers it to the detection
field. PH-INEPT-type sequences can then convert the resulting proton order into
heteronuclear magnetization.

An initial implementation should model the reaction as a typed quantum-state
mapping and should not claim to predict catalyst chemistry or pairwise yield.
Those are explicit inputs until independently modeled or measured.

## SABRE

SABRE leaves the target substrate chemically unchanged. Parahydrogen-derived
hydrides and substrate molecules bind reversibly to a polarization-transfer
catalyst. Coherent evolution through the bound scalar-coupling network converts
hydride singlet order, after which polarized substrate dissociates into the free
pool and fresh parahydrogen replenishes the source.

The minimum physically useful SABRE model therefore needs separate quantum
manifolds for:

- catalyst-bound hydrides and substrate;
- free substrate; and
- fresh parahydrogen.

Exchange channels are quantum maps rather than scalar Bloch-McConnell rates:

- association tensors a free ligand state into a bound complex;
- dissociation partially traces the departing ligand into the free pool;
- hydride refresh replaces the hydride subsystem with the specified
  parahydrogen state; and
- site exchange permutes ligands between inequivalent catalyst positions.

Small systems can use a deterministic block stochastic-Liouville equation.
Discrete-residence-time Monte Carlo is valuable as an independent reference in
the intermediate regime where exchange rates and coherent couplings are
similar. Realistic rebinding and concentration effects ultimately require a
manifold-diagonal DMEx-style model; conventional ensemble-averaged exchange can
be inaccurate or inefficient in this regime.

The initial SABRE workflows should cover low-field proton SABRE,
SABRE-SHEATH, and one RF-driven high-field transfer method. SABRE-Relay,
catalyst activation chemistry, bubbling/mass transport, and large full-catalyst
spin systems should remain later extensions.

## Long-lived singlet states

Exact magnetic equivalence protects singlet symmetry but also makes singlet
order inaccessible to ordinary nonselective NMR. A small symmetry-breaking
interaction permits preparation and readout while also causing leakage.

First-class workflows should include:

- SLIC preparation and reconversion;
- M2S/S2M and generalized M2S;
- optional adiabatic-passage conversion;
- storage under static field, spin lock, or field cycling; and
- preparation yield, singlet population, leakage, storage lifetime, and
  reconversion yield.

For a nearly equivalent two-spin pair, the SLIC spin-lock nutation frequency
matches the scalar coupling `J`. The resonance-offset difference breaks the
symmetry and controls the transfer rate. The RF carrier lies at the mean pair
resonance. This convention must be used consistently in code, tests, and
documentation.

Independent-spin T1/T2 relaxation cannot predict a singlet lifetime: it removes
the correlated protection responsible for the long-lived state. The first
relaxing workflow should accept a measured phenomenological `T_S` through a
symmetry-aware, trace-preserving model. A later microscopic path should include
correlated dipolar fluctuations, differential chemical-shift anisotropy,
passive-spin dipolar couplings, paramagnetic relaxation, and other
symmetry-breaking mechanisms.

## Existing package foundations

The package already provides reusable dense spin-1/2 and heteronuclear
Hamiltonians, isotropic scalar coupling, RF controls, density-matrix and
Liouville propagation, sequence timelines, field schedules, and general
relaxation infrastructure. It also has a small SLIC response simulator.

The current `spin_dynamics.exchange` module is classical Bloch-McConnell
exchange between site magnetizations. It should remain available for lineshape
and REXSY work but should not be extended implicitly into SABRE. SABRE needs a
new quantum-manifold exchange layer with explicit subsystem maps and
normalization.

## Recommended implementation order

1. Correct and strengthen the existing SLIC API and documentation.
2. Add rigorous singlet/triplet and parahydrogen state primitives.
3. Implement complete long-lived-state preparation, storage, and readout.
   **Implemented:** matched two-spin SLIC conversion, optional symmetry purge,
   phenomenological measured `T_S` storage, and SLIC reconversion.
4. Add hydrogenative PHIP, PASADENA/ALTADENA, and heteronuclear transfer.
   **Partially implemented:** pairwise product-state mapping, high-field
   PASADENA, explicit-trajectory ALTADENA, hard pulses, and selected-spin
   acquisition are present. A validated PH-INEPT convenience workflow remains.
5. Add small-system quantum exchange and SABRE.
6. Add event-based and scalable manifold-reduced backends only after the
   reference implementation is validated.

## Validation requirements

The feature should be validated against the following hierarchy.

### Algebraic and analytical invariants

- Singlet/triplet projectors are Hermitian, idempotent, orthogonal, and complete.
- `Q_S` is trace zero and has the expected singlet/triplet eigenvalues.
- A para fraction of `1/4` produces zero deviation density.
- Singlet order is invariant under spin-pair permutation up to the expected
  state-vector sign, which cancels in the projector.
- Common-mode pair noise preserves the singlet; differential noise relaxes it.
- SLIC crosses near `nu1 = |J|`; exact equivalence gives zero transfer, while
  the offset difference controls the transfer time.

### Workflow limits

- PHIP signal is linear in para excess and pairwise-addition fraction.
- PASADENA and ALTADENA signs and antiphase patterns reproduce analytical or
  published examples.
- SABRE gives no free-substrate polarization without exchange or without a
  hydride-to-substrate coupling path.
- Quantum exchange preserves Hermiticity, positivity, total concentration, and
  the appropriate trace convention.
- Fresh parahydrogen produces a steady-state build-up; disabling refresh yields
  depletion rather than an unphysical perpetual source.

### External and experimental evidence

- Published SLIC transfer curves and singlet lifetimes.
- Published PHIP spectra and PH-INEPT transfer efficiencies.
- Small-spin SABRE field-dependence and build-up curves.
- Comparison of deterministic SABRE exchange with an event-based Monte Carlo
  calculation before fitting experimental exchange rates.

## Principal references

- Carravetta and Levitt, *J. Am. Chem. Soc.* 126 (2004) 6228-6229,
  https://doi.org/10.1021/ja0490931
- Ahuja et al., *J. Chem. Phys.* 127 (2007) 134112,
  https://doi.org/10.1063/1.2778429
- DeVience, Walsworth, and Rosen, SLIC preparation of singlet states,
  https://arxiv.org/abs/1307.0832
- Knecht et al., coherent SABRE dynamics,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6344499/
- Lindale et al., quantum evolution with exchange and DMExFR2,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7413723/
- Cavallari et al., hydrogenative PHIP and transfer methods,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7910253/
- Pravdivtsev et al., field-dependent SABRE transfer,
  https://pubmed.ncbi.nlm.nih.gov/26529205/
