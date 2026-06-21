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
principal-axis RF polarization: `x`, `y`, and `z`. For spin-3/2 nuclei such as
`35Cl` and `37Cl`, the Hamiltonian uses `quadrupole_frequency_hz` as the
eta-zero NQR line frequency, so the zero-field line is
`quadrupole_frequency_hz * sqrt(1 + eta**2 / 3)`. The transition inventory
omits zero-frequency Kramers-doublet transitions.

```python
chlorine = QuadrupolarSite(
    spin=1.5,
    isotope="35Cl",
    quadrupole_frequency_hz=30e6,
    eta=0.1,
)

for transition in diagonalize_site(chlorine).transitions:
    print(transition.label, transition.frequency_hz)
```

## Weak Static B0

Weak static fields are modeled by diagonalizing `H_Q + H_Z`, where
`H_Z = -gamma B0 . I`, while reporting the perturbation ratio
`|gamma B0| / nu_ref`. This is intended for the NQR regime where the Zeeman
frequency is nonzero but much smaller than the selected NQR line. Powder
weak-field spectra use correlated B0/B1 orientations and transition intensities
weighted by `|B1 . dipole|**2`, so RF-dark branches are not counted:

```python
from spin_dynamics.nqr import simulate_weak_b0_spectrum

weak = simulate_weak_b0_spectrum(
    chlorine,
    b0_tesla=1e-3,
    orientations="powder",
    broadening_hz=200.0,
    weak_ratio_threshold=0.05,
)

print(weak.max_perturbation_ratio)
print(weak.offsets_hz, weak.spectrum)
```

This static-transition machinery supports both spin-1 and spin-3/2 sites. It
does not require the spin-3/2 pulsed manifold model, so it can already be used
to inspect chlorine line splitting and powder broadening in weak fields.

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
The current selective-pulse and SLSE workflows support spin-1 only. Spin-3/2
SLSE needs a degenerate-doublet RF model that drives the full transition
manifold, rather than the embedded two-level pulse used for spin-1 examples.

SLSE can also use a Liouville-space relaxation model instead of only applying a
post-hoc echo envelope:

```python
from spin_dynamics.nqr import NQRRelaxationModel

result = simulate_slse(
    site,
    sequence,
    orientations="powder",
    relaxation=NQRRelaxationModel(t1_seconds=1.0, t2_seconds=20e-3),
)

print(result.local_effective_t2eff_seconds)
```

In this mode, each orientation is propagated through the repeated SLSE cycle
with Hamiltonian plus relaxation superoperators. The result includes the local
cycle eigenvalues and a dominant non-steady effective decay time, which is the
starting point for modeling spin-lock/T1rho-like SLSE decay.

Offset and pulse-period sweeps are available for exploring the modulation
discussed in SLSE detection:

```python
from spin_dynamics.nqr import simulate_slse_offset_sweep

sweep = simulate_slse_offset_sweep(
    site,
    "x",
    offsets_hz=[-2e3, 0.0, 2e3],
    pulse_duration_seconds=25e-6,
    nutation_hz=10e3,
    echo_spacing_seconds=500e-6,
    relaxation=NQRRelaxationModel(t2_seconds=20e-3),
)

print(sweep.selected_echo_amplitudes)
```

## EFG Inhomogeneity

Static EFG disorder is modeled as an isochromat-style ensemble of independent
quadrupolar sites:

```python
import numpy as np

from spin_dynamics.nqr import (
    SelectivePulse,
    gaussian_efg_distribution,
    simulate_fid_efg_distribution,
    simulate_slse_acquisition_spectrum,
)

distribution = gaussian_efg_distribution(
    site,
    quadrupole_std_hz=2e3,
    samples=41,
)

fid = simulate_fid_efg_distribution(
    distribution,
    "x",
    times_seconds=np.linspace(0.0, 20e-3, 512),
    excitation=SelectivePulse("x", duration_seconds=2.5e-6, nutation_hz=100e3),
)
```

The returned result includes the complex time-domain signal and a centered FFT
spectrum. Temperature or impurity gradients can be represented by constructing
an EFG distribution with shifted `quadrupole_frequency_hz` and `eta` values.
The distribution simulators check the EFG frequency grid against the simulated
duration and warn when a coarse discrete grid may rephase artificially; increase
the number of isochromats or pass `rephase_action="ignore"` after checking
convergence.

For SLSE spectra, the experimentally relevant quantity is the Fourier
transform of the averaged echo acquired over a finite receiver window centered
on one echo. The acquisition window must be shorter than the pulse spacing, and
its rectangular truncation broadens the measured spectrum:

```python
slse_spectrum = simulate_slse_acquisition_spectrum(
    distribution,
    sequence,
    acquisition_duration_seconds=200e-6,
    acquisition_points=256,
    echo_index=-1,
    carrier_frequency_hz=sequence.detection.rf_frequency_hz,
    orientations="powder",
    noise={"target_snr": 20.0, "seed": 123, "domain": "time"},
    deconvolution_strength=1e-2,
)

print(slse_spectrum.spectrum_frequencies_hz, slse_spectrum.spectrum)
print(slse_spectrum.deconvolution.deconvolved_spectrum)
```

Noise is added to the acquired complex echo waveform before the FFT. The
optional deconvolution uses regularized inversion of the same finite-window FFT
operator, so the regularization strength should be checked across plausible SNR
values before interpreting sharpened spectra quantitatively.

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
- Relaxation is available as either a scalar `T2e` envelope or a
  phenomenological Liouville-space population/coherence model; microscopic
  Redfield/dipolar parameterization remains future work.
- Weak-B0 Zeeman perturbations are available through the Hamiltonian and
  orientation path, but broad validation against experiments remains future
  work.
