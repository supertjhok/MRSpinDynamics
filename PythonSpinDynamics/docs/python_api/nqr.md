# NQR Models

The `spin_dynamics.nqr` namespace contains the quadrupolar-spin extension for
pulsed NQR. It is separate from the spin-1/2 Bloch and J-coupling layers.

## Two modeling regimes: reduced two-level vs. full density matrix

A pulsed-NQR simulator has to make one choice before propagating a sequence:
keep every quantum level of the nucleus, or keep only the two levels of one
selected transition. **That choice is set by how many states the RF pulse can
actually connect, not by the number of peaks in the spectrum.** The package
therefore offers two regimes. The self-authored technical note
[Modeling Pulsed NQR Dynamics: Spin 1, Spin 3/2, and Higher Spins](../../../References/Pulsed_NQR_Spin_Dynamics_Narrative_Rewrite.pdf)
gives the reasoning in more detail; its LaTeX source is tracked at
`../../../References/Pulsed_NQR_Spin_Dynamics_Narrative_Rewrite.tex`.

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
inventory omits Kramers-doublet transitions (their sub-Hz numerical splitting is
filtered by a level-spread-relative floor).

**Spin > 3/2 (5/2, 7/2, 9/2).** Any integer or half-integer spin above 1/2 is
supported. `quadrupole_frequency_hz` is always the eta-zero *fundamental* (lowest)
line -- the `1/2 <-> 3/2` line for half-integer spin -- and the higher (satellite)
axial lines follow at `2 nu_Q`, `3 nu_Q`, ... So a half-integer spin `I` shows
`I - 1/2` lines (spin-5/2 -> `nu_Q, 2 nu_Q`; spin-7/2 -> `nu_Q, 2 nu_Q, 3 nu_Q`),
and `eta != 0` shifts them and switches on a weak combination line. Internally the
operator is `H/h = C_Q/(4 I (2I-1)) * [3 Iz^2 - I(I+1) + eta(Ix^2 - Iy^2)]` with
`nu_Q = C_Q * d / (4 I (2I-1))` and `d = 3` (integer) or `6` (half-integer); this
reproduces the spin-1 and spin-3/2 values exactly. Build a site for a named
nucleus directly from `C_Q` or `nu_Q` with the isotope presets:

```python
from spin_dynamics.nqr import quadrupolar_site, diagonalize_site

al = quadrupolar_site("27Al", cq_hz=3.0e6, eta=0.1)   # spin-5/2, gamma filled in
for t in diagonalize_site(al).transitions:
    print(t.label, t.frequency_hz)                    # nu_Q and 2 nu_Q (+ combination)
```

`quadrupolar_site(isotope, cq_hz=... | nu_q_hz=..., eta=...)` fills spin and the
gyromagnetic ratio from `QUADRUPOLAR_ISOTOPES` (27Al, 51V, 55Mn, 63/65Cu, 93Nb,
121/123Sb, 209Bi, 17O, ...); `nu_q_from_cq_hz` / `cq_hz_from_nu_q` are the
standalone conversions.

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

This static-transition machinery supports any quadrupolar spin (1, 3/2, 5/2, 7/2,
9/2): it only diagonalizes `H_Q + H_Z`, so it maps the Zeeman splitting of every
zero-field line and its powder broadening without needing the pulsed manifold
model -- useful for chlorine, but equally for the multi-line `27Al` or `51V`
spectrum.

When no `transition_label` is given, the reference frequency `nu_ref` (the
denominator of the perturbation ratio and the zero of the `offsets_hz` axis) is
the *mean* of the included zero-field transitions. For spin-3/2 that mean is the
single physical line, but for spin-1 it is the centroid of three inequivalent
lines (`nu_+`, `nu_-`, `nu_0`) and is not itself a physical transition. Pass an
explicit `transition_label` for spin-1 so the offset axis is referenced to a
real line.

## Exact NQR-to-NMR Crossover Spectrum

`simulate_crossover_spectrum` is the regime-independent static-spectrum path.
It diagonalizes the complete `H_Q + H_Z` Hamiltonian, includes every transition
allowed by the specified transmit/receive geometry, and does not select lines
by proximity to a zero-field NQR frequency. The reported crossover parameter is

```text
r = nu_L / nu_Q = |gamma| B0 / quadrupole_frequency_hz
```

where `quadrupole_frequency_hz` retains the package's fundamental zero-field
line convention. This ratio, rather than `B0/B1`, distinguishes
Zeeman-perturbed NQR (`r << 1`), the non-perturbative crossover (`r ~ 1`), and
quadrupole-perturbed NMR (`r >> 1`). `B1` remains the oscillating RF field.

```python
from spin_dynamics.nqr import (
    CrossoverOrientation,
    QuadrupolarSite,
    simulate_crossover_spectrum,
)

site = QuadrupolarSite(
    spin=1.5,
    isotope="35Cl",
    quadrupole_frequency_hz=1.0e6,
    eta=0.3,
    gamma_hz_per_t=4.0e6,
)
orientation = CrossoverOrientation(
    b0_direction_pas=(1, 1, 1),
    transmit_direction_pas=(1, -1, 0),
    # receive_direction_pas defaults to the reciprocal transmit direction
)
result = simulate_crossover_spectrum(
    site,
    b0_tesla=0.25,             # nu_L / nu_Q = 1
    orientations=[orientation],
    temperature_kelvin=300.0,
    broadening_hz=1.0e3,       # Gaussian sigma (Lorentzian HWHM if selected)
    normalize=True,
)

for line in result.transitions:
    print(line.frequency_hz, line.lower_levels, line.upper_levels, line.intensity)
```

Transition amplitudes contain the Boltzmann population difference and the
transmit/receive matrix-element product. Transmit and receive directions may be
complex vectors, so circular and elliptical polarization are supported. The
returned spectrum is complex to preserve cross-coil phase; a reciprocal coil
produces a real, non-negative absorption spectrum.

At an exact degeneracy, a `CrossoverTransition` joins complete energy
manifolds rather than arbitrary individual eigenvectors. For example, the
zero-field spin-3/2 line connects `(0, 1)` to `(2, 3)`. The summed manifold
response is invariant under unitary rotations inside either Kramers doublet.
Once `B0` resolves the levels, the same machinery automatically returns
ordinary one-level-to-one-level transitions.

Pass `orientations="powder"` for the existing correlated laboratory
`B0`/linear-`B1` powder geometry. For explicit single-crystal or specialized
transmit/receive geometries, pass normalized `CrossoverOrientation` values.
The current exact crossover model includes the nuclear Zeeman and quadrupolar
interactions. Chemical-shielding tensors and field-history-dependent sweeps are
planned extensions, not implicit parts of this static calculation.

For a continuous field sweep, `track_crossover_field_sweep` assigns each new
energy-ordered eigenbasis to the previous field point by maximizing total
squared eigenvector overlap. It phase-aligns adjacent eigenvectors and returns
tracked levels, state-pair frequencies and intensities, the minimum adjacent
overlap, and the expectation value of spin along `B0`:

```python
import numpy as np
from spin_dynamics.nqr import track_crossover_field_sweep

ratios = np.logspace(-3, 1, 160)
fields = ratios * site.quadrupole_frequency_hz / abs(site.gamma_hz_per_t)
sweep = track_crossover_field_sweep(site, fields, orientation=orientation)
print(sweep.levels_hz.shape, sweep.transition_frequencies_hz.shape)
```

Start at a small nonzero field when individual state identity is required:
states inside an exact zero-field degeneracy form a physical manifold and do
not possess unique individual labels. The example
`examples/plot_nqr_nmr_crossover.py` applies the sweep to repository-backed
NaNO2 `14N` (`I=1`) and NaClO3 `35Cl` (`I=3/2`) parameters:

```powershell
python examples\plot_nqr_nmr_crossover.py --output nqr_nmr_crossover.png
```

Add `--powder` to replace the transition sticks with exact powder-averaged
field-frequency maps. `simulate_crossover_powder_sweep` uses a configurable
correlated `B0`/`B1` SO(3) grid and returns all fields on one absolute frequency
axis. It intentionally does not claim common eigenstate branch labels across
different crystallites:

```python
from spin_dynamics.nqr import simulate_crossover_powder_sweep

powder = simulate_crossover_powder_sweep(
    site,
    fields,
    n_theta=4,
    n_phi=8,
    n_chi=4,
    broadening_hz=20e3,
    frequency_points=384,
)
print(powder.spectra.shape, powder.orientation_count)
```

## Laboratory-frame RF reference

`simulate_lab_frame_rf` propagates an arbitrary piecewise-constant
instantaneous RF magnetic field without a rotating-frame transformation or
RWA. It is the correctness reference for crossover pulses, including cases
with multiple nearby transition bands or appreciable counter-rotating effects:

```python
from spin_dynamics.nqr import sample_linear_rf_pulse, simulate_lab_frame_rf

durations, rf_fields = sample_linear_rf_pulse(
    duration_seconds=50e-6,
    time_step_seconds=20e-9,  # must resolve the carrier
    amplitude_tesla=1e-3,
    carrier_hz=site.quadrupole_frequency_hz,
    direction_pas=(1, 0, 0),
)
lab = simulate_lab_frame_rf(
    site,
    b0_vector_tesla_pas=(0, 0, 0),
    segment_durations_seconds=durations,
    rf_fields_tesla_pas=rf_fields,
)
print(lab.times_seconds, lab.signal)
```

The default initial state is the Boltzmann equilibrium of the complete static
`H_Q + H_Z` Hamiltonian. Each segment is propagated by exact dense unitary
evolution. The sample interval is an explicit convergence parameter: it must
resolve both carrier oscillation and envelope changes.

### Measuring the RWA boundary

`compare_rwa_to_lab_frame` applies the same constant-envelope linear pulse to
the existing single-band RWA and to the exact laboratory-frame solver. It
returns absolute density-matrix errors and an error normalized by the RWA
response. `scan_rwa_validity` follows the strongest transition in the
conventional NQR/NMR band and maps those metrics over both relevant ratios:

```python
from spin_dynamics.nqr import scan_rwa_validity

validity = scan_rwa_validity(
    site,
    interaction_ratios=[0.01, 0.1, 1.0, 10.0],  # nu_L / nu_Q
    rf_strength_ratios=[0.001, 0.01, 0.1],       # gamma B1_peak / nu_Q
    b0_direction_pas=(1, 1, 1),
    b1_direction_pas=(1, -1, 0),
)
print(validity.relative_response_error)
```

The nearest distinct active-line spacing is returned alongside the error map.
It is a diagnostic rather than a universal threshold: the numerical maps show
that crossover error can remain finite as `B1` tends to zero because the
single-band winding assignment itself omits relevant frequency components.
`examples/plot_nqr_rwa_validity.py` visualizes this boundary for the NaNO2
spin-1 and NaClO3 spin-3/2 examples.

### Floquet RF propagation

For a constant-envelope monochromatic linear pulse, `simulate_floquet_rf`
provides a multiband alternative that retains co- and counter-rotating Fourier
components. It diagonalizes a finite Sambe-space Hamiltonian and reconstructs
the physical propagator:

```python
from spin_dynamics.nqr import simulate_floquet_rf

floquet = simulate_floquet_rf(
    site,
    b0_vector_tesla_pas=(0.1, 0.1, 0.1),
    nutation_hz=20e3,
    rf_frequency_hz=site.quadrupole_frequency_hz,
    pulse_duration_seconds=20e-6,
    sidebands=4,
)
print(floquet.density_matrix, floquet.unitarity_error)
```

Increase `sidebands` until both the requested observable and
`unitarity_error` converge. Unlike the sampled laboratory-frame reference,
Floquet propagation does not require time steps fine enough to resolve the
carrier, but it currently assumes one constant-amplitude periodic carrier.
Shaped, chirped, or otherwise arbitrary waveforms should continue to use
`simulate_lab_frame_rf`.

## Field-dependent equilibrium and relaxation

`field_dependent_equilibrium` returns the normalized Gibbs density of the full
static `H_Q + H_Z`, expressed in both the PAS and instantaneous energy basis.
It also reports the spin expectation vector. This is a physical trace-one
state, distinct from the older `equilibrium_density(levels_hz)` helper, which
returns a normalized trace-zero high-temperature deviation density for legacy
pulse simulations.

```python
from spin_dynamics.nqr import field_dependent_equilibrium

equilibrium = field_dependent_equilibrium(
    site,
    b0_vector_tesla_pas=(0.1, 0.05, 0.2),
    temperature_kelvin=300.0,
)
print(equilibrium.populations, equilibrium.spin_expectation_pas)
```

Three complementary relaxation models use this equilibrium:

- `FieldDependentRelaxationModel` is a completely-positive Gibbs-reset model
  with optional dephasing between distinct energy manifolds. Its normalized
  fixed point is unique, and its manifold projectors avoid assigning arbitrary
  dephasing inside a Kramers degeneracy. It is the robust phenomenological
  choice near crossings.
- `FieldDependentDaviesRelaxationModel` derives field-dependent rates from
  magnetic rank-1 and EFG rank-2 fluctuation operators in the instantaneous
  eigenbasis. A Lorentzian correlation-time factor supplies spectral filtering,
  upward/downward rates obey Boltzmann detailed balance, and equal Bohr
  frequencies share a jump operator. Its rate amplitudes remain model-dependent
  unless calibrated independently.
- `FieldDependentNonsecularRelaxationModel` implements the unified-GKLS
  frequency-clustering construction for unresolved Bohr frequencies. Nearby
  transitions share jump operators, retaining coherence-transfer cross terms
  without losing complete positivity. It reduces exactly to the Davies model
  when the cluster width equals the numerical secular tolerance. This follows
  the controlled weak-coupling construction of
  [Trushechkin, Phys. Rev. A 103, 062226 (2021)](https://doi.org/10.1103/PhysRevA.103.062226).

```python
from spin_dynamics.nqr import (
    FieldDependentDaviesRelaxationModel,
    FieldDependentNonsecularRelaxationModel,
    FieldDependentRelaxationModel,
    simulate_field_relaxation,
)

robust = FieldDependentRelaxationModel(
    temperature_kelvin=300.0,
    thermalization_time_seconds=0.2,
    dephasing_time_seconds=20e-3,
)
trajectory = simulate_field_relaxation(
    site,
    (0.1, 0.05, 0.2),
    times_seconds=[0.0, 0.01, 0.1, 0.5],
    relaxation=robust,
    initial_density=prepared_density,
)

microscopic = FieldDependentDaviesRelaxationModel(
    spin=site.spin,
    temperature_kelvin=300.0,
    magnetic_rate_per_second=100.0,
    efg_rate_per_second=30.0,
    correlation_time_seconds=0.2e-6,
)

unresolved = FieldDependentNonsecularRelaxationModel(
    spin=site.spin,
    temperature_kelvin=300.0,
    magnetic_rate_per_second=100.0,
    efg_rate_per_second=30.0,
    correlation_time_seconds=0.2e-6,
    frequency_cluster_width_hz=20e3,
)
print(unresolved.gibbs_stationarity_error(static_hamiltonian))
```

The Davies model is secular: it is controlled when separations between
distinct Bohr-frequency groups exceed the corresponding relaxation rates.
The unified model clusters frequencies that are experimentally unresolved.
Its cluster width must be justified from the relaxation linewidth or an
explicit coarse-graining scale; it is not a fitting knob to make a desired
curve. A finite-width cluster uses one representative KMS frequency, so the
exact Gibbs state is only approximately stationary unless the clustered
frequencies are exactly degenerate. `gibbs_stationarity_error` reports the
resulting residual. For very broad or ambiguous clusters, use the Gibbs-reset
model as the conservative thermalizing fallback.

`simulate_lab_frame_rf(..., relaxation=model)` evaluates the dissipator from
the static `H_Q + H_Z` once and holds that thermal reference fixed while the RF
waveform is applied. Do not pass these field-thermal models directly to the
older rotating-frame full-pulse helpers: those helpers currently give a
relaxation model the rotating-frame Hamiltonian, which is not the laboratory
Gibbs reference. The example
`examples/plot_nqr_field_equilibrium_relaxation.py` shows equilibrium
polarization and Davies decay-rate changes across the crossover.

### Exact-pulse crossover SLSE

`simulate_crossover_slse` combines finite-sideband Floquet excitation and
refocusing pulses with laboratory-frame free evolution under a static-field
relaxation generator. The carrier can follow the strongest transition in the
conventional NQR/NMR band independently at each field:

```python
from spin_dynamics.nqr import simulate_crossover_slse

slse = simulate_crossover_slse(
    site,
    b0_vector_tesla_pas=(0.1, 0.1, 0.1),
    nutation_hz=50e3,
    excitation_duration_seconds=3e-6,
    refocus_duration_seconds=6e-6,
    echo_spacing_seconds=150e-6,
    num_echoes=12,
    relaxation=unresolved,
    floquet_sidebands=5,
)
print(slse.rf_frequency_hz, slse.echo_amplitudes)
```

Relaxation is included during the free intervals; the Floquet pulses use a
hard-pulse approximation relative to the relaxation timescale. The result
reports each pulse's finite-sideband unitarity error so this approximation can
be converged explicitly. `examples/plot_nano2_crossover_slse.py` applies the
workflow to repository-backed NaNO2 `14N` at `nu_L/nu_Q = 0, 0.1, 1, 10` and
compares unified-GKLS and fully secular echo trains.

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

## Full Density-Matrix Sequences (spin-3/2 and higher)

`spin_dynamics.nqr.full_dynamics` propagates the complete `(2I+1)`-level density
matrix in a rotating frame at the pulse carrier. It is the required model for
spin-3/2 and runs for any spin -- spin-1 through spin-9/2 (`27Al`, `51V`, `93Nb`,
...). A single-pulse FID:

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

A two-pulse (Hahn-style) echo is available through `simulate_full_echo`, and the
spin-lock spin-echo (SLSE) detection train -- the chlorine-style spin-3/2
measurement -- through `simulate_full_slse`:

```python
from spin_dynamics.nqr import QuadrupolarSite, simulate_full_slse

site = QuadrupolarSite(spin=1.5, isotope="35Cl", quadrupole_frequency_hz=1.0e6, eta=0.0)

slse = simulate_full_slse(
    site,
    nutation_hz=10e3,
    excitation_duration_seconds=25e-6,
    refocus_duration_seconds=50e-6,
    echo_spacing_seconds=400e-6,
    num_echoes=12,
    orientations="powder",     # powder average over crystallite orientations
    b0_tesla=0.0,              # > 0 adds a weak Zeeman perturbation
    t2e_seconds=3e-3,
)
print(slse.echo_amplitudes.shape)   # one complex echo per cycle
```

It returns a `FullNQRSLSEResult` with the orientation-weighted echo train, the
per-orientation trains, and the weights. A weak static field is applied with
`b0_tesla > 0`; the field direction follows each sample's `b0_direction_pas`
(use `b0_powder_average_grid` for a Zeeman powder). An optional
`relaxation=NQRRelaxationModel(...)` adds Liouville-space T1/T2 decay (use it
with `t2e_seconds=inf` to avoid double counting). The lower-level primitives
`pulse_hamiltonian`, `static_hamiltonian_rotating`, and `detection_operator`
remain exposed for building custom sequences.

`examples/plot_nqr_full_powder_nutation.py` builds a spin-3/2 powder nutation
curve from these primitives. As a validation anchor it overlays the spin-1
curve, which converges to the classic 119-degree powder maximum in the
weak-pulse limit, while the four-state spin-3/2 curve peaks at a slightly
smaller flip angle (~105 degrees). `examples/plot_nqr_spin32_slse.py` runs the
`35Cl` powder SLSE train and shows how a weak Zeeman field detunes the
crystallites and reshapes the decay.

Two cautions specific to the full model:

- `nutation_hz` is the **bare field nutation** `gamma*B1/(2*pi)`. The realized
  Rabi rate on a transition `a-b` is `2*nutation_hz*|<a|e1.I|b>|`, so a pulse
  calibrated on one line is a different flip angle on another -- the physical
  effect the full model captures.
- It uses a single-carrier rotating-wave approximation. Analysis of the level and
  transition structure (`diagonalize_site`) is exact for any spin, but the *pulsed*
  model is faithful only near the **fundamental (lowest) NQR line**, where the
  addressed transition maps to a single-quantum coherence. This is the default
  carrier (the strongest line, which is the fundamental for half-integer spin), and
  it is reliable when the pulse is spectrally selective (Rabi rate and detuning
  small compared with the spacing to the nearest other line). Driving a higher /
  satellite line of a spin > 3/2 nucleus (e.g. `2 nu_Q`) causes a winding collision
  that would silently select the wrong transition; the model emits a `RuntimeWarning`
  in that case rather than returning a wrong answer. Correct satellite-line pulsing
  needs exact time-domain propagation (a planned follow-up).

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

### Microscopic Redfield/dipolar relaxation

As a non-default alternative to scalar `T1`/`T2`, the same `relaxation=...`
slot accepts a secular Redfield model built from fluctuating point-dipole
couplings:

```python
from spin_dynamics.relaxation import (
    DipolarRelaxationSource,
    RedfieldDipolarRelaxationModel,
    RigidSolidMotionalAveraging,
)
from spin_dynamics.nqr import (
    simulate_full_slse,
)

relaxation = RedfieldDipolarRelaxationModel.from_dipolar_sources(
    site.spin,
    (
        DipolarRelaxationSource(
            vector_angstrom=(1.05, 0.20, 0.10),
            coupling_hz=1.3e3,        # optional; computed from distance if omitted
        ),
    ),
    motion=RigidSolidMotionalAveraging(correlation_time_seconds=2.0e-6),
)

result = simulate_full_slse(
    site,
    nutation_hz=10e3,
    excitation_duration_seconds=25e-6,
    refocus_duration_seconds=50e-6,
    echo_spacing_seconds=400e-6,
    num_echoes=12,
    relaxation=relaxation,
)
```

Each `DipolarRelaxationSource` represents one stochastic bath spin, normally a
nearby proton, in the NQR principal-axis frame. It contributes the tensor
`2*pi*d*(I - 3 n n^T)`, where
`d = mu0 h gamma_Q gamma_b / (4*pi*r^3)`, and converts the bath spin variance
`S(S+1)/3` into a target-spin covariance. The motion object then decides how
that covariance is averaged before Redfield propagation.

`RigidSolidMotionalAveraging` keeps the fixed-frame dipolar anisotropy and is
the natural starting point for solid NQR. `IsotropicLiquidMotionalAveraging`
rotationally averages the tensor covariance to a scalar isotropic fluctuation,
which is the clean hook for liquid-state NMR-style tumbling:

```python
from spin_dynamics.relaxation import IsotropicLiquidMotionalAveraging

liquid_relaxation = RedfieldDipolarRelaxationModel.from_dipolar_sources(
    site.spin,
    sources,
    motion=IsotropicLiquidMotionalAveraging(correlation_time_seconds=120e-12),
)
```

The Redfield model diagonalizes the motion-averaged covariance into independent
fluctuating fields, decomposes the target spin operators into secular
Bohr-frequency components of the sequence Hamiltonian, and applies the motion
model's spectral density. The built-in regimes currently use the Lorentzian
`J(omega) = 2*tau_c / (1 + omega^2*tau_c^2)`.

This is still a compact Markovian, high-temperature treatment: it models
stochastic dipolar relaxation, not a coherent multi-spin cluster. It is useful
when nearby-spin geometry and a correlation time are more meaningful inputs
than fitted scalar `T1`/`T2`. Leave `relaxation=None` for the historical default,
or use `PhenomenologicalRelaxationModel` for the older scalar Liouville model.
The NQR namespace re-exports these relaxation classes for compatibility, but
their implementation lives in the shared `spin_dynamics.relaxation` module.

Two plotting examples exercise the microscopic model:

- `examples/plot_redfield_nano2_slse.py` reads NaNO2 EFG and CIF geometry from
  the adjacent QuadrupolarDFT workspace, builds a rigid-solid dipolar bath, and
  propagates coherent full-density-matrix 14N SLSE echo trains. The powder case
  uses equal SLSE pulse lengths at the spin-1 powder optimum near 119 degrees
  and the full-model pi/2 refocusing phase.
- `examples/plot_redfield_water_cpmg.py` uses the same shared Redfield model for
  a spin-1/2 proton CPMG decay in isotropically tumbling water. Its public
  options follow the existing CPMG vocabulary (`num_echoes`,
  `echo_spacing_seconds`) while retaining unit-suffixed plotting aliases.

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

## SORC

The strong off-resonance comb (SORC) sequence is represented as
`(tau - phi - tau)^N`, with the signal sampled in the observation window between
pulses. By default, `simulate_sorc` iterates the finite-pulse SORC pathway
recurrence used by Konnai, Odano, and Asaji (2008); as `N` grows, it converges
to the published steady-state denominator and carries the expected
`delta_omega * tau` periodicity. Pass `model="coherent"` only when you want the
closed, finite unitary transient train rather than the dephased pathway
approximation:

```python
from spin_dynamics.nqr import QuadrupolarSite, simulate_sorc, sorc_sequence

site = QuadrupolarSite(spin=1, isotope="14N", quadrupole_frequency_hz=4.2e6, eta=0.3)
sequence = sorc_sequence(
    "x",
    pulse_duration_seconds=20e-6,
    nutation_hz=16.5e3,
    half_spacing_seconds=0.8e-3,
    num_pulses=96,
    rf_frequency_hz=4.60425e6 - 2.05e3,
)

result = simulate_sorc(site, sequence, orientations="powder")
print(result.signal_amplitudes[-1])
```

Two powder comparison helpers are also exposed:
`sorc_powder_pathway_signal` evaluates the same finite-`N` pathway recurrence
directly in Konnai's one-dimensional powder angle, avoiding coarse spherical
grid aliasing in pulse-width sweeps, while
`sorc_powder_theory_signal` evaluates the steady-state powder expression used
for the Konnai SORC offset, spacing, and pulse-width plots, and
`fid_powder_theory_signal` returns the spin-1 powder FID pulse-width response.
The example `examples/plot_nqr_sorc_konnai2008.py` overlays finite-`N`
pathway markers with these infinite-`N` theory curves for the three key paper
sweeps.

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

## Polarization-Enhanced NQR Transport

`spin_dynamics.nqr.polarization_enhancement` models the instrument-level
polarization-enhanced NQR workflow described by Glickstein and Mandal: protons are
pre-polarized in a permanent magnet, the sample is translated through a falling
fringe field, and cross-polarization can occur at level crossings where
`gamma_H * B0 = nu_NQR`. The model is intentionally an adiabatic-transfer
estimator, not a microscopic coupled-spin propagation through every avoided
crossing.

```python
from spin_dynamics.nqr import (
    CylindricalSampleGeometry,
    HalbachPrepolarizationMagnet,
    LinearTransportMotion,
    PolarizationEnhancedNQRSample,
    simulate_adiabatic_polarization_transfer,
)

sample = PolarizationEnhancedNQRSample(
    name="melamine-like",
    line_labels=("nu+", "nu-", "nu0"),
    line_frequencies_hz=(2.766e6, 2.034e6, 0.732e6),
    protons_per_molecule=6,
    nitrogens_per_molecule=6,
    proton_t1_seconds=48.6,
    proton_linewidth_hz=80e3,
    proton_nitrogen_coupling_hz=1.3e3,
)
magnet = HalbachPrepolarizationMagnet(rod_shape="square", rod_width=25.4e-3)
geometry = CylindricalSampleGeometry(length=20e-3, diameter=8e-3)
motion = LinearTransportMotion(0.0, 0.10, velocity=0.1667, axis="z")

result = simulate_adiabatic_polarization_transfer(
    magnet, sample, geometry, motion, prepolarization_time_seconds=100.0
)
print(result.practical_enhancement)
```

The returned `PolarizationTransferResult` reports the ideal spin-1 enhancement
factors, practical enhancement after finite proton build-up and transfer
efficiency, crossing positions, local field gradients, adiabatic ratios, and the
sample-averaged field profile along the transport axis. Use
`examples/plot_nqr_polarization_enhancement.py` to sweep speed, pre-polarization
time, and sample size.

### Estimating the 1H-14N coupling from CIF structures

The main uncertainty in the adiabatic criterion is the effective
`proton_nitrogen_coupling_hz`. `spin_dynamics.nqr.structure_coupling` provides a
lightweight CIF reader and point-dipole estimator. Given a quadrupolar atom label,
it applies CIF symmetry, searches periodic images for nearby protons, computes

```text
d_HQ = (mu0 / 4pi) h gamma_H gamma_Q / r^3
```

and reports individual couplings plus an RMS effective coupling:

```python
from spin_dynamics.nqr import estimate_proton_dipolar_couplings_from_cif

estimate = estimate_proton_dipolar_couplings_from_cif(
    "../QuadrupolarDFT/structures/Melamine/237082.cif",
    "N101",
    proton_radius_angstrom=3.0,
)
print(estimate.effective_rms_hz)
for item in estimate.proton_couplings:
    print(item.proton_label, item.distance_angstrom, item.coupling_hz)
```

For the bundled melamine CIF, the default `N101` ring nitrogen finds four nearby
protons within `3 A` and gives an RMS coupling of about `1.3 kHz`; directly
protonated amine nitrogens have much larger direct N-H couplings. The plotting
example uses this CIF estimate by default, while still accepting a manual
`--nh-coupling-hz` override.

## Current Limits

- Dense single-site matrices only.
- Two pulse models: the reduced spin-1 selective (embedded two-level) pulse and
  the full `(2I+1)` density-matrix pulse. The full model's rotating-wave
  approximation assumes a single carrier addressing one transition band, so it
  covers spin-3/2 zero-field and weak-Zeeman sequences but is not yet a general
  multi-band higher-spin (spin >= 5/2 with several excited lines) solver.
- The *embedded two-level* SLSE and population-transfer workflows remain spin-1
  only. Spin-3/2 SLSE/FID is supported through the full density-matrix model
  (`simulate_full_slse`, `simulate_full_fid`, `simulate_full_echo`). A
  two-frequency 2D population-transfer workflow on top of the full model is
  still future work.
- `select_nqr_model` is available as an explicit model-selection check, but the
  reduced workflows do not yet call it automatically (the note's recommended
  model-selection front end); for now it is advisory.
- Relaxation is available as a scalar `T2e` envelope, a phenomenological
  Liouville-space population/coherence model, or an opt-in microscopic
  Redfield/dipolar model for stochastic nearby-spin baths.
- Weak-B0 Zeeman perturbations are available through the Hamiltonian and
  orientation path, but broad validation against experiments remains future
  work.
