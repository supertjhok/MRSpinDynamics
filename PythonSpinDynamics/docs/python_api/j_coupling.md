# J-Coupling Models

Scalar J coupling transfers spin information through chemical bonds. Unlike a
chemical shift, its frequency does not collapse with the applied field, so it
can dominate spectra and pulse responses in zero-, ultra-low-, or strongly
inhomogeneous fields.

Use this page when independent Bloch magnetization vectors are no longer
enough because two or more spin-1/2 nuclei evolve coherently together. The
`spin_dynamics.coupling` namespace offers two levels: fast analytical models
for specific J-editing/filter experiments, and small dense Hamiltonians for
general few-spin propagation. It is not intended for large sparse spin
networks.

For singlet order, PHIP, and long-lived-state workflows built on the same
spin-pair algebra, continue to [Hyperpolarization and Long-Lived Singlet
Workflows](hyperpolarization.md).

## Heteronuclear J-Editing

The first supported model family is analytic heteronuclear J-editing. It
captures the modulation curves used for weak and grossly inhomogeneous fields:

```python
from spin_dynamics.coupling import j_modulation_curve

signal = j_modulation_curve(
    encoding_times,
    couplings_hz=[125.0, 160.0],
    amplitudes=[0.85, 0.15],
    cycles=1,
)
```

For carbon-detected groups, use the `cos(...) ** n` model:

```python
from spin_dynamics.coupling import carbon_detected_j_modulation

signal = carbon_detected_j_modulation(
    encoding_times,
    couplings_hz=[125.0, 160.0],
    abundances=[0.85, 0.15],
    proton_counts=[2, 1],
)
```

Known J positions can be fit by linear least squares:

```python
from spin_dynamics.coupling import fit_known_j_spectrum

fit = fit_known_j_spectrum(
    encoding_times,
    signal,
    couplings_hz=[125.0, 160.0],
    include_background=False,
)
```

The ideal TANGO-B selection profile is available through `tango_b_filter`.

## Dense Coupled-Spin Utilities

Small spin-1/2 systems can be built with `coupled_spin_system`, then propagated
with dense Hamiltonians:

```python
from spin_dynamics.coupling import (
    coupled_spin_system,
    isotropic_j_hamiltonian,
    zeeman_hamiltonian,
)

system = coupled_spin_system(
    offsets_hz=[-0.35, 0.35],
    couplings_hz=[[0.0, 7.0], [7.0, 0.0]],
)
hamiltonian = zeeman_hamiltonian(system) + isotropic_j_hamiltonian(system)
```

Hamiltonians are expressed in radians per second and use spin-1/2 operators
with eigenvalues `+/- 1/2`.

## Inhomogeneous B0/B1 Isochromats

The coupled layer can now use the same isochromat idea as the core Bloch
workflows. Each isochromat contains the same scalar-coupled spin network, but
has its own local B0 offset, B1 transmit scale, B1 receive scale, and weight:

```python
from spin_dynamics.coupling import (
    coupled_isochromat_ensemble,
    free_precession_step,
    rf_step,
    simulate_coupled_isochromat_sequence,
)

ensemble = coupled_isochromat_ensemble(
    system,
    b0_offsets_hz=[-2.0, 0.0, 2.0],
    weights=[0.25, 0.5, 0.25],
    b1_tx_scale=[0.9, 1.0, 1.1],
    b1_rx_scale=[1.0, 1.0, 0.8],
)

result = simulate_coupled_isochromat_sequence(
    ensemble,
    [
        rf_step(duration=0.25, nutation_hz=1.0, phase=1.57079632679),
        free_precession_step(duration=0.01),
    ],
    initial_axis="x",
    detect_axis="x",
)
```

For time-varying fields, pass per-step `b0_offsets_hz` or `b1_tx_scale`
overrides to `free_precession_step` or `rf_step`. The returned ensemble signal
is the weighted sum of raw dense-operator expectation values multiplied by the
receive scale.

## SLIC

The first SLIC helper simulates remaining transverse magnetization after a
spin-lock pulse:

```python
from spin_dynamics.coupling import simulate_slic_spectrum

result = simulate_slic_spectrum(
    system,
    nutation_frequencies_hz,
    spin_lock_time=1.0,
)
```

For a nearly equivalent two-spin system, the SLIC matching condition is that
the spin-lock nutation frequency equals the magnitude of the scalar coupling:
`omega_1 / (2 pi) = |J|`. The RF carrier lies at the mean resonance of the pair.
The resonance-offset (chemical-shift) difference `Delta nu` breaks the exchange
symmetry and mixes the singlet `(|ab> - |ba>)/sqrt(2)` with a triplet level, so
it controls the transfer rate. Exact equivalence gives no transfer under an
ideal nonselective spin lock. `two_spin_slic_matching_nutation_hz` returns the
crossing amplitude, while `two_spin_slic_transfer_time` returns the ideal
maximum-transfer time `1 / (sqrt(2) |Delta nu|)`. This dense simulator is
intended for small systems only; larger homonuclear networks will need sparse
or specialized methods.

## Current Limits

- spin-1/2 only;
- dense matrices only;
- no chemical exchange;
- no quadrupolar nuclei;
- no relaxation superoperators yet;
- no paper-level raw-data fixture comparisons yet.
