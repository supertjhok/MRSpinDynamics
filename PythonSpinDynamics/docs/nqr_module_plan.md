# NQR Module Plan

This note tracks the planned `spin_dynamics.nqr` extension for pulsed nuclear
quadrupole resonance. It is based on the local references in `../References`,
especially the 2D NQR population-transfer paper and the pulsed nitrogen-14 NQR
fundamentals chapter.

## Scope

The first target is pulsed, mostly zero-field NQR for solid samples. The key
assumption for the initial implementation is selective RF excitation: ordinary
narrowband pulses address only one transition of the `(2I + 1)` quadrupolar
manifold, so the RF action can be treated as an embedded two-level rotation.

The module should support:

- spin operators for arbitrary integer or half-integer `I`;
- quadrupolar Hamiltonians in the EFG principal-axis system;
- default zero-field simulations and optional weak Zeeman perturbations;
- fixed-orientation single-crystal simulations;
- powder averaging over local EFG orientations relative to the lab RF field;
- classic spin-lock spin-echo (SLSE) detection;
- multi-frequency perturbation plus SLSE detection for 2D NQR-style population
  transfer experiments.

## Proposed Package Layout

```text
spin_dynamics/nqr/
  __init__.py
  operators.py
  systems.py
  hamiltonians.py
  orientations.py
  pulses.py
  sequences.py
  simulation.py
  workflows.py
```

This should remain separate from `spin_dynamics.coupling`, which is currently
scoped to small scalar-coupled spin-1/2 systems.

## Milestones

- [x] Add this design/progress document.
- [x] Add dense arbitrary-spin operators.
- [x] Add validated quadrupolar site and transition metadata.
- [x] Add orientation grids for single crystals and powders.
- [x] Add selective embedded two-level RF pulse propagation.
- [x] Add zero-field SLSE workflow.
- [x] Add two-frequency population-transfer workflow.
- [x] Add weak-B0 Zeeman-perturbed transition calculation.
- [ ] Add probe/circuit integration where useful.
- [x] Add initial documentation and generated API inventory.
- [x] Add diagnostic plotting examples.
- [ ] Add broader user-manual coverage.

## Validation Targets

- Spin matrices satisfy standard angular-momentum commutators.
- Spin-1 quadrupole transition frequencies match the `x`, `y`, and `z`
  convention used in the 2D NQR paper.
- A selective transition pulse matches the expected two-level population
  exchange and leaves spectator levels unchanged.
- Powder orientation weights integrate to unity.
- SLSE echo amplitudes decay with the requested `T2e`.
- A perturbation pulse on one transition changes a later detection transition
  through shared level populations.

## Deliberate Initial Limits

- Dense matrices only.
- Selective pulses only; full nonselective RF Hamiltonian propagation can be
  added later.
- Relaxation is initially phenomenological (`T1`, `T2e`) rather than a full
  Redfield/Lindblad superoperator.
- Multi-site samples are initially handled by summing independent site signals.
