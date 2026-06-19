# NQR Models

The `spin_dynamics.nqr` namespace contains the first quadrupolar-spin extension
for pulsed NQR. It is separate from the spin-1/2 Bloch and J-coupling layers.

The initial model is intentionally selective: each RF pulse addresses one NQR
transition and is propagated as an embedded two-level rotation inside the full
quadrupolar energy-level basis. This matches the common narrowband-pulse limit
used by spin-lock spin-echo (SLSE) and two-frequency population-transfer NQR
experiments.

## Site and Transitions

```python
from spin_dynamics.nqr import QuadrupolarSite, diagonalize_site

site = QuadrupolarSite(
    spin=1,
    isotope="14N",
    quadrupole_frequency_hz=900e3,
    eta=0.3,
)

eigensystem = diagonalize_site(site)
for transition in eigensystem.transitions:
    print(transition.label, transition.frequency_hz)
```

For spin-1 at zero field, the transitions are labeled by their dominant
principal-axis RF polarization: `x`, `y`, and `z`.

## Orientations

Single-crystal simulations pass one fixed orientation:

```python
from spin_dynamics.nqr import single_crystal_orientation

orientations = single_crystal_orientation(alpha=0.0, beta=1.57079632679)
```

Powder simulations use a normalized spherical quadrature grid:

```python
from spin_dynamics.nqr import powder_average_grid

orientations = powder_average_grid(n_theta=16, n_phi=32)
```

## SLSE

```python
from spin_dynamics.nqr import simulate_slse, slse_sequence

sequence = slse_sequence(
    "x",
    pulse_duration_seconds=25e-6,
    nutation_hz=10e3,
    echo_spacing_seconds=1e-3,
    num_echoes=16,
)

result = simulate_slse(
    site,
    sequence,
    orientations="powder",
    t2e_seconds=20e-3,
)
```

The returned `SLSEResult` includes echo times, averaged echo amplitudes,
per-orientation echo amplitudes, orientation weights, and transition metadata.

## Population Transfer

```python
from spin_dynamics.nqr import SelectivePulse, simulate_population_transfer

transfer = simulate_population_transfer(
    site,
    SelectivePulse("x", duration_seconds=50e-6, nutation_hz=10e3),
    sequence,
    orientations="powder",
)

print(transfer.normalized_difference)
```

This models the perturbation-plus-SLSE detection experiment used in 2D NQR:
a pulse on one transition changes populations shared with another transition,
changing the detected SLSE amplitude.

## Current Limits

- Dense single-site matrices only.
- Selective pulses only.
- Relaxation is phenomenological through `T2e`; full relaxation
  superoperators are not implemented yet.
- Weak-B0 Zeeman perturbations are available through the Hamiltonian and
  orientation path, but broad validation against experiments remains future
  work.
