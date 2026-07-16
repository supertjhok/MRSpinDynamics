# Concepts and Units

Before selecting a function, decide what physical object must be propagated.
An ordinary liquid-state echo may need only a three-component magnetization
vector. J-coupled spins, singlet order, ESR, and NQR generally require a
density matrix or Hamiltonian. Imaging and diffusion additionally require
space; hardware studies may require fields, circuits, particles, or
temperature. This page explains the shared conventions that let those layers
connect without silently mixing units or meanings.

New users should read this page before the workflow catalog. The
[documentation map](../documentation_map.md) defines recurring terms such as
workflow, field map, and evidence.

## Model layers at a glance

| Question | Representation | Start with |
|---|---|---|
| How does an ensemble of uncoupled spin-1/2 nuclei respond to RF and offsets? | Bloch magnetization vectors over isochromats | `spin_dynamics.workflows` |
| Do scalar couplings or singlet/triplet symmetry matter? | Small dense spin-1/2 Hamiltonian and density matrix | [J coupling](j_coupling.md) and [hyperpolarization](hyperpolarization.md) |
| Is a quadrupolar nucleus being excited? | Reduced transition model or full `(2I+1)` density matrix | [NQR models](nqr.md) |
| Is an electron spin being detected? | ESR spin system and electron-spin Hamiltonian | [ESR models](esr.md) |
| Do spins move through spatially varying fields? | Positions or walkers plus sampled field maps | [Workflow catalog](workflows.md) |
| Is the device itself the object of study? | Magnet/coil geometry, circuits, fields, thermal states, or detector models | Fields and Hardware navigation |

The original MATLAB-compatible Bloch workflows use uncoupled spin-1/2
isochromats, coherence-space rotations, and complex acquired spectra. Other
namespaces extend that model deliberately; they are not interchangeable
shortcuts for the same state.

## Coherence Ordering

Low-level kernels use MATLAB's coherence ordering:

```text
M0, M-, M+
```

This ordering is documented in the relevant Python dataclasses and kernel
docstrings because it is easy to transpose accidentally when porting MATLAB.

## Complex Spectra and Probe Phase

Spectra and asymptotic magnetization values are complex. Probe circuit response
can rotate most of the signal into the imaginary component, especially in tuned
and untuned probe examples. For comparison plots, magnitude is often the least
misleading first view; real and imaginary components are still useful for
debugging phase conventions.

## Time Normalization

The core MATLAB routines often normalize time to the nominal `w1 = 1`
convention. In this convention a 90-degree pulse has length `pi / 2`, and a
180-degree pulse has length `pi`.

Parameter constructors that expose physical seconds convert to normalized units
inside workflow helpers. Current examples:

- `calc_masy_ideal` converts CPMG pulse timings using `T_90`.
- `calc_macq_fid` converts ideal FID segment timings and relaxation constants.

Keep new Python function names explicit when a value is not dimensionless.
