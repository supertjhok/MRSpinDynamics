# NQR Models

The `spin_dynamics.nqr` namespace contains the quadrupolar-spin extension for
pulsed NQR. It is separate from the spin-1/2 Bloch and J-coupling layers.

## Two modeling regimes: reduced two-level vs. full density matrix

A pulsed-NQR simulator has to make one choice before propagating a sequence:
keep every quantum level of the nucleus, or keep only the two levels of one
selected transition. **That choice is set by how many states the RF pulse can
actually connect, not by the number of peaks in the spectrum.** The package
therefore offers two regimes (the reasoning is laid out in the technical note
`References/Pulsed_NQR_Spin_Dynamics_Narrative_Rewrite`):

**Reduced two-level model (`SelectivePulse`, `simulate_slse`,
`simulate_population_transfer`).** Each RF pulse addresses one transition and is
propagated as an embedded fictitious spin-1/2 rotation inside the full energy
basis. This is efficient and honest *only when the selected transition is
isolated*. For spin-1 the validity condition is

```text
Delta_iso = min_{j != t} |omega_t - omega_j|  >>  max(Omega_1, Delta_omega_pulse, Gamma)
```

i.e. the spacing to the nearest other RF-active transition must dominate the RF
nutation rate `Omega_1`, the pulse bandwidth `Delta_omega_pulse ~ 1/t_p`, and
the linewidth `Gamma`. A nonzero asymmetry `eta` is **necessary but not
sufficient**: at `eta = 0` the spin-1 `X<->Z` and `Y<->Z` transitions coincide,
so a generic pulse drives a one-to-two connection and the reduction breaks. The
reduced path is restricted to spin-1 for exactly this reason.

**Full density-matrix model (`spin_dynamics.nqr.full_dynamics`).** Builds the
complete `(2I+1)`-dimensional Hamiltonian `H_Q + H_Z`, diagonalizes it for the
actual EFG and field orientation, transforms the RF and detection operators
into that eigenbasis, and propagates the full `d x d` density matrix. This is
the correct general model and is **required for spin-3/2**: its single zero-field
NQR line connects two Kramers doublets, so "one line" hides four states, and a
weak Zeeman field splits and mixes them into a cluster of nearby pathways that a
collection of independent two-level rotations cannot reproduce. The full model
also captures transition-specific nutation rates
(`|<m-1|I_x|m>| = (1/2) sqrt(I(I+1) - m(m-1))`), multiple-quantum pathways, and
orientation-dependent transition strengths.

The matrices stay small (`3x3`, `4x4`, ...), so the full model is inexpensive;
the cost is orientation averaging and waveform sampling, not the linear algebra.
Use the reduced model as a deliberate, isolation-checked optimization for spin-1
and the full model elsewhere.

The reduced model below is the common narrowband-pulse limit used by spin-lock
spin-echo (SLSE) and two-frequency population-transfer NQR experiments.

### Choosing the model

`select_nqr_model` reads the choice from the actual static Hamiltonian and the
RF matrix elements for the coil polarization, rather than from the spin or the
number of spectral lines:

```python
from spin_dynamics.nqr import QuadrupolarSite, select_nqr_model

site = QuadrupolarSite(spin=1.5, quadrupole_frequency_hz=30e6, eta=0.1)
choice = select_nqr_model(
    site, "x",
    nutation_hz=5e3,                 # bare field nutation gamma*B1/(2*pi)
    pulse_duration_seconds=50e-6,
    b1_direction_pas=(1, 1, 1),
    linewidth_hz=0.0,
)
print(choice.recommended_model)      # "full" -- spin-3/2 Kramers doublets
print(choice.describe())             # full diagnostic report
```

The reduced two-level path is recommended only when the pulse-addressed set is a
single, non-degenerate, RF-active pair *and* the isolation ratio
`Delta_iso / max(Omega_1, 1/t_p, Gamma)` clears `isolation_threshold` (default
5). The returned `NQRModelSelection` reports the target states, nearest
competing RF-active transition, isolation distance and ratio, pulse bandwidth,
the addressed-state set, a degeneracy flag, and human-readable reasons. It
correctly handles the regime-defining cases: spin-1 at `eta=0` (coincident
lines) and unresolved small-`eta` splittings go full via the isolation ratio;
spin-3/2 Kramers doublets go full because the addressed set spans four states; a
strongly Zeeman-resolved spin-3/2 line can return reduced as a verified special
case; and polarization that makes neighboring lines RF-dark keeps an otherwise
crowded spectrum reducible.

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
principal-axis RF polarization: `x`, `y`, and `z`. `quadrupole_frequency_hz` is
the eta-zero NQR line, i.e. `(3/4) * e**2 q Q / h` (the value to which the `x`
and `y` lines both collapse as `eta -> 0`), **not** `e**2 q Q / h` itself. With
`nu_Q = quadrupole_frequency_hz`, the three spin-1 lines are
`nu_+ = nu_Q (1 + eta/3)`, `nu_- = nu_Q (1 - eta/3)`, and `nu_0 = (2/3) nu_Q eta`.
For spin-3/2 nuclei such as `35Cl` and `37Cl`, the Hamiltonian likewise uses
`quadrupole_frequency_hz` as the eta-zero NQR line frequency, so the zero-field
line is `quadrupole_frequency_hz * sqrt(1 + eta**2 / 3)`. The transition
inventory omits zero-frequency Kramers-doublet transitions.

Selective-pulse `nutation_hz` is the *effective two-level Rabi frequency* of the
addressed transition at full RF coupling (`Omega / (2 pi)` for the embedded
two-level system), already including the transition dipole matrix element. It is
**not** `gamma B1 / (2 pi)`: a 90-degree pulse satisfies
`nutation_hz * duration_seconds = 0.25`. The embedded two-level model also
assumes the pulse is spectrally selective, i.e. `nutation_hz` is small compared
with the spacing to neighboring NQR lines.

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

When no `transition_label` is given, the reference frequency `nu_ref` (the
denominator of the perturbation ratio and the zero of the `offsets_hz` axis) is
the *mean* of the included zero-field transitions. For spin-3/2 that mean is the
single physical line, but for spin-1 it is the centroid of three inequivalent
lines (`nu_+`, `nu_-`, `nu_0`) and is not itself a physical transition. Pass an
explicit `transition_label` for spin-1 so the offset axis is referenced to a
real line.

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
The selective-pulse SLSE/population-transfer workflows support spin-1 only,
because they rely on the isolated-transition reduction. Spin-3/2 sequences use
the full density-matrix model described next, which drives the whole
four-state manifold rather than one embedded two-level pair.

## Full Density-Matrix Sequences (spin-3/2 and general)

`spin_dynamics.nqr.full_dynamics` propagates the complete `(2I+1)`-level density
matrix in a rotating frame at the pulse carrier. It is the required model for
spin-3/2 and also runs for spin-1. A single-pulse FID:

```python
import numpy as np
from spin_dynamics.nqr import QuadrupolarSite, simulate_full_fid

site = QuadrupolarSite(spin=1.5, isotope="35Cl", quadrupole_frequency_hz=30e6, eta=0.1)

fid = simulate_full_fid(
    site,
    nutation_hz=20e3,            # bare field nutation gamma*B1/(2*pi), NOT a per-line Rabi rate
    pulse_duration_seconds=10e-6,
    times_seconds=np.linspace(0.0, 200e-6, 1024),
    rf_frequency_hz=None,        # default: the strongest zero-field NQR line
)
print(fid.signal)               # complex baseband (demodulated at the carrier)
```

A two-pulse (Hahn-style) echo is available through `simulate_full_echo`, and an
optional `relaxation=NQRRelaxationModel(...)` adds Liouville-space T1/T2 decay.
The lower-level primitives `pulse_hamiltonian`, `static_hamiltonian_rotating`,
and `detection_operator` are exposed for building custom sequences.

`examples/plot_nqr_full_powder_nutation.py` builds a spin-3/2 powder nutation
curve from these primitives. As a validation anchor it overlays the spin-1
curve, which converges to the classic 119-degree powder maximum in the
weak-pulse limit, while the four-state spin-3/2 curve peaks at a slightly
smaller flip angle (~105 degrees).

Two cautions specific to the full model:

- `nutation_hz` is the **bare field nutation** `gamma*B1/(2*pi)`. The realized
  Rabi rate on a transition `a-b` is `2*nutation_hz*|<a|e1.I|b>|`, so a pulse
  calibrated on one line is a different flip angle on another -- the physical
  effect the full model captures.
- It uses a single-carrier rotating-wave approximation valid when one carrier
  addresses one transition band (the spin-3/2 zero-field and weak-Zeeman
  regime). It is not yet a general multi-band higher-spin solver.

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
starting point for modeling spin-lock/T1rho-like SLSE decay. Here "dominant"
means the *slowest*-decaying cycle mode, which may differ from the mode with the
largest overlap on the detected coherence.

The scalar `t2e_seconds` envelope and the Liouville-space `relaxation` T2 are
**not** alternatives that the code selects between: if both are set their decay
composes multiplicatively, so coherence is damped twice. Leave
`t2e_seconds=inf` (the default) whenever a `relaxation` model is supplied;
`simulate_slse` warns if both are finite.

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
- Two pulse models: the reduced spin-1 selective (embedded two-level) pulse and
  the full `(2I+1)` density-matrix pulse. The full model's rotating-wave
  approximation assumes a single carrier addressing one transition band, so it
  covers spin-3/2 zero-field and weak-Zeeman sequences but is not yet a general
  multi-band higher-spin (spin >= 5/2 with several excited lines) solver.
- The selective-pulse SLSE and population-transfer workflows remain spin-1 only;
  the equivalent spin-3/2 SLSE/2D workflows on top of the full model are future
  work.
- `select_nqr_model` is available as an explicit model-selection check, but the
  reduced workflows do not yet call it automatically (the note's recommended
  model-selection front end); for now it is advisory.
- Relaxation is available as either a scalar `T2e` envelope or a
  phenomenological Liouville-space population/coherence model; microscopic
  Redfield/dipolar parameterization remains future work.
- Weak-B0 Zeeman perturbations are available through the Hamiltonian and
  orientation path, but broad validation against experiments remains future
  work.
