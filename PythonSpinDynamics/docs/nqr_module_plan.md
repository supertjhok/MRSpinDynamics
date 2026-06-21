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
That embedded two-level pulse model is appropriate for the current spin-1
examples; spin-3/2 nuclei require a separate degenerate-doublet RF manifold
model before pulsed chlorine SLSE/FID simulations are considered supported.

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
- [x] Add first Liouville-space relaxation/effective-SLSE-decay path.
- [x] Add SLSE offset and pulse-period sweep helpers with plotting examples.
- [x] Add static EFG-distribution isochromat model with FID/spectrum examples.
- [x] Add SLSE finite-acquisition spectrum example with static EFG broadening.
- [x] Add spin-3/2 Hamiltonian/transition-frequency metadata with the
  chlorine-style eta-zero line convention and zero-frequency Kramers-doublet
  filtering.
- [x] Add weak-static-B0 transition spectra for spin-1 and spin-3/2 using
  exact `H_Q + H_Z` diagonalization plus `|gamma B0| / nu_ref` regime checks.
- [ ] Add spin-3/2 selective-pulse dynamics using a degenerate-doublet RF
  manifold model for chlorine-style SLSE and FID simulations.
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
- Liouville relaxation preserves trace, damps coherences with the requested
  `T2`, and reports a cycle-derived effective SLSE decay time.
- Static EFG distributions normalize weights, produce complex FID dephasing,
  return centered FFT spectra, and reduce to the single-site model at zero
  distribution width.
- Spin-3/2 transition metadata reports the chlorine-style
  `nu_Q * sqrt(1 + eta^2 / 3)` zero-field line and excludes zero-frequency
  Kramers-doublet transitions; current pulsed workflows raise a clear error
  until the degenerate-manifold RF model is implemented.
- Weak-static-B0 spectra work for spin-1 and spin-3/2, report the perturbation
  ratio, and warn or raise when the Zeeman frequency is no longer small
  compared with the selected NQR line.
- EFG frequency grids warn when their spacing can cause artificial rephasing
  within the simulated duration.
- SLSE broadening examples plot the FFT of a finite acquired echo window rather
  than FFTs of the echo train, avoiding sequence-modulation artifacts in
  zero-width cases; optional time-domain noise and regularized acquisition-
  window deconvolution are covered by tests across SNR values.
- A perturbation pulse on one transition changes a later detection transition
  through shared level populations.

## Deliberate Initial Limits

- Dense matrices only.
- Selective pulses only; full nonselective RF Hamiltonian propagation can be
  added later. Spin-3/2 selective pulses are also pending because degenerate
  doublets need a manifold RF treatment, not the current embedded two-level
  pulse.
- Relaxation is phenomenological (`T1`, `T2`, `T2e`) rather than a microscopic
  Redfield/dipolar parameterization.
- Multi-site samples are initially handled by summing independent site signals.
