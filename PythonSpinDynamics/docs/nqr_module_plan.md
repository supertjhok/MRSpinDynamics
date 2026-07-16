# NQR Module Implementation Record

> **Status (audited 2026-07-11): core plan complete.** Arbitrary-spin operators,
> reduced and full pulsed models, spin-3/2 dynamics, weak-field spectra,
> inhomogeneity, relaxation, polarization enhancement, RFI workflows, examples,
> facade routes, and user-manual coverage are implemented. The unchecked items
> below are optional integrations, not blockers for supported NQR use.

This note records the design of the `spin_dynamics.nqr` extension for pulsed nuclear
quadrupole resonance. It is based on the local references in `../References`,
especially the 2D NQR population-transfer paper and the pulsed nitrogen-14 NQR
fundamentals chapter.

Users should read [`python_api/nqr.md`](python_api/nqr.md). The material below
is retained to show how the implementation evolved and why some early
milestones use terminology that is narrower than the current supported model.

## Original scope

The first target was pulsed, mostly zero-field NQR for solid samples. The
initial implementation assumed selective RF excitation and represented one
isolated spin-1 transition as an embedded two-level rotation. The current
package retains that efficient reduced model where its isolation test is
satisfied and also provides full density-matrix dynamics for spin-3/2,
weak-field crossover, and multilevel excitation. The bullets below record the
original requirements, not the present capability boundary.

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
- [x] Add spin-3/2 selective-pulse dynamics using the full `(2I+1)`-level
  density-matrix model (`simulate_full_fid`, `simulate_full_echo`,
  `simulate_full_slse`) for chlorine-style SLSE and FID, including powder
  averaging and weak-Zeeman perturbations. The full RWA propagation handles the
  degenerate Kramers doublets of the zero-field line directly, so a separate
  reduced degenerate-doublet manifold model is not required.
- [ ] Add probe/circuit integration where useful.
- [x] Add initial documentation and generated API inventory.
- [x] Add diagnostic plotting examples.
- [x] Add broader user-manual coverage.

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
  Kramers-doublet transitions; the full density-matrix RF path handles the
  degenerate manifold directly.
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
- The embedded two-level *reduced* pulse path is selective-only and spin-1 only.
  Spin-3/2 (and general) pulsed dynamics use the full `(2I+1)`-level model, whose
  single-carrier RWA addresses one transition band -- valid for the spin-3/2
  zero-field and weak-Zeeman regime, but not yet a general multi-band higher-spin
  (spin >= 5/2) solver.
- Relaxation is phenomenological (`T1`, `T2`, `T2e`) rather than a microscopic
  Redfield/dipolar parameterization.
- Multi-site samples are initially handled by summing independent site signals.
