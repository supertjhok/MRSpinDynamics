# Birdcage Coil Reference Model

Phase 5 begins with a prescribed-current reference model for cylindrical
birdcage coils. It is deliberately separate from the later resonant circuit
solver: the reference model states the expected rung currents, completes the
end-ring currents by Kirchhoff's current law, and computes the resulting
complex magnetic field. This gives the circuit implementation an independent
field and mode benchmark.

## Geometry and branch orientation

`BirdcageGeometry` creates:

- \(N\) straight rungs at
  \(\phi_n=2\pi n/N\), directed from the negative to the positive axial end;
- one positive and one negative end ring;
- \(N\) separately addressable arc sections on each end ring, directed toward
  increasing azimuth.

Each arc can be subdivided into several chords for Biot-Savart convergence.
The geometry supports cages along the Cartesian x, y, or z axis and uses a
right-handed transverse basis.

```python
from spin_dynamics.fields import BirdcageGeometry

cage = BirdcageGeometry(
    radius=0.15,
    length=0.30,
    n_rungs=16,
    axis="z",
    ring_segments_per_section=8,
)
```

## Current modes and end-ring KCL

For prescribed rung currents \(I_n\), the positive and negative end-ring
section currents \(J_n^+\) and \(J_n^-\) satisfy

\[
  J_n^+ - J_{n-1}^+ = I_n,\qquad
  J_n^- - J_{n-1}^- = -I_n.
\]

A closed cage requires \(\sum_n I_n=0\). `birdcage_current_mode` chooses the
minimum-norm end-ring solution by removing the otherwise arbitrary common
circulating current. The returned `BirdcageCurrentMode` exposes the complete
complex branch-current phasors and the node-by-node KCL residual.

The ideal sinusoidal modes are

\[
  I_n = I_0\cos(m\phi_n-\alpha).
\]

For the fundamental \(m=1\), \(\alpha=0\) and \(\alpha=\pi/2\) give the
degenerate cosine and sine modes. `birdcage_linear_mode` constructs either
orientation.

## Quadrature and circular components

`birdcage_quadrature_mode` combines the degenerate modes with a 90-degree
temporal phase:

\[
  I_n = I_0 \exp(-i h m\phi_n),\qquad h\in\{-1,+1\}.
\]

With the cage axis and \(B_0\) along +z, `handedness=+1` produces the package's
transmit convention

\[
  B_1^+ = \frac{B_x+iB_y}{2},
\]

while `handedness=-1` produces the opposite rotating component
\(B_1^-=(B_x-iB_y)/2\). The sign convention is explicit because changing the
phasor time convention or reversing \(B_0\) reverses the physical rotation
label.

```python
from spin_dynamics.fields import (
    birdcage_quadrature_mode,
    solve_birdcage_field,
)

mode = birdcage_quadrature_mode(cage, handedness=1)
solution = solve_birdcage_field(cage, mode, points_m)

b1_plus = solution.b1_plus_t
b1_minus = solution.b1_minus_t
```

The field solver sums the finite straight-segment Biot-Savart field of every
rung and end-ring branch, multiplied by its complex current phasor. It returns
the full complex Cartesian field as well as B1+ and B1-.

## Uniformity and circularity metrics

`birdcage_field_metrics` reports within a selected ROI:

- mean magnitude of the requested circular component;
- coefficient of variation;
- peak-to-peak nonuniformity divided by the mean;
- normalized circularity
  \((P_\mathrm{desired}-P_\mathrm{counter})/
  (P_\mathrm{desired}+P_\mathrm{counter})\);
- counter-rotating amplitude ratio and isolation in dB;
- fraction of total RMS field transverse to \(B_0\).

These metrics distinguish three effects that a center-field number alone
cannot:

1. spatial B1 magnitude variation;
2. ellipticity or counter-rotating contamination;
3. unwanted axial field.

Run the reference example with:

```powershell
python examples\plot_birdcage_quadrature.py --rungs 16 --radius-cm 15 --length-cm 30 --output results\birdcage_quadrature.png
```

The figure shows the explicit branches, the cosine/sine rung currents, their
center-field polarization, the midplane B1+ magnitude, percent nonuniformity,
and circular isolation.

## Validation and current boundary

The analytical and regression checks require:

- end-ring KCL to numerical precision;
- equal center-field magnitude for the fundamental cosine and sine modes;
- orthogonal center fields for those modes;
- selection of opposite circular components when quadrature handedness is
  reversed;
- sub-percent central-ROI coefficient of variation and high circular isolation
  for the reference 16-rung cage;
- convergence as each end-ring arc is segmented more finely.

## Resonant circuit layer

`BirdcageCircuit` adds series resistance, inductance, and optional capacitance
to every rung and end-ring section. Component arguments may be scalars for a
uniform cage or length-\(N\) arrays for tolerance and fault studies. End-ring
values describe one section and are applied to the matching section on both
rings.

The conventional architectures are represented directly:

- low-pass: capacitors in the rungs and continuous end rings;
- high-pass: capacitors in the end-ring sections and continuous rungs;
- band-pass: capacitors in both branch families.

For a closed cage, the rung-current vector has zero sum. If \(T\) completes
those currents into the positive end ring, the negative-ring currents are
\(-T I\). A branch quantity \(X\), such as resistance or inductance, therefore
reduces to

\[
  X_\mathrm{eff}
  = \operatorname{diag}(X_\mathrm{rung})
  + 2T^\mathsf{T}\operatorname{diag}(X_\mathrm{ring})T.
\]

The lossless modes follow from

\[
  C_\mathrm{eff}^{-1} I
  = \omega^2 L_\mathrm{eff} I,
\]

solved in an orthonormal zero-sum basis. This produces \(N-1\) rung-current
modes, including the two orientations of each degenerate azimuthal family.
The returned modes include complete KCL-consistent branch currents and the
series-loss estimate

\[
  Q = \omega
  \frac{I^\mathrm{H}L_\mathrm{eff}I}
       {I^\mathrm{H}R_\mathrm{eff}I}.
\]

For uniform branch values and \(s_m=\sin(\pi m/N)\), the implemented analytical
check is

\[
  \omega_m^2 =
  \frac{C_\mathrm{rung}^{-1}
       +(2s_m^2 C_\mathrm{ring})^{-1}}
       {L_\mathrm{rung}+L_\mathrm{ring}/(2s_m^2)},
\]

with absent capacitor terms set to zero. This is also used by the tuned
low-pass and high-pass factories.

```python
from spin_dynamics.fields import tuned_low_pass_birdcage

circuit = tuned_low_pass_birdcage(
    cage,
    63.87e6,
    rung_inductance_h=180e-9,
    end_ring_inductance_h=35e-9,
    rung_resistance_ohm=0.08,
    end_ring_resistance_ohm=0.015,
)
analysis = circuit.modal_analysis()
fundamental_pair = analysis.azimuthal_modes(1)
print(fundamental_pair[0].frequency_hz)
print(fundamental_pair[0].quality_factor)
```

### Tolerances, splitting, and driven quadrature

Nonuniform component arrays break rotational symmetry. The modal solver then
predicts the frequency splitting of the nominal cosine/sine pair:

```python
import numpy as np
from spin_dynamics.fields import BirdcageCircuit

capacitors = np.array(circuit.rung_capacitance_f)
capacitors[0] *= 1.03
perturbed = BirdcageCircuit(
    geometry=cage,
    rung_inductance_h=circuit.rung_inductance_h,
    end_ring_inductance_h=circuit.end_ring_inductance_h,
    rung_capacitance_f=capacitors,
    rung_resistance_ohm=circuit.rung_resistance_ohm,
    end_ring_resistance_ohm=circuit.end_ring_resistance_ohm,
)
print(perturbed.modal_analysis().splitting_hz(1))
```

`solve_drive` solves the complex finite-loss impedance matrix for series
voltage sources inserted into rung branches. The quadrature helper places two
equal sources one quarter-cage apart and drives them 90 degrees apart in RF
phase. Its returned `BirdcageCurrentMode` passes directly to
`solve_birdcage_field`, so capacitor imbalance can be followed all the way to
B1 nonuniformity and counter-rotating contamination.

Run the circuit example with:

```powershell
python examples\plot_birdcage_circuit.py --output results\birdcage_circuit.png
```

The figure compares low- and high-pass modal spectra, shows the split driven
response after one capacitor is shifted, and compares balanced and perturbed
B1 maps and circular isolation.

### Present circuit-model boundary

This layer is a lumped, quasistatic series-branch model. It does not yet
include mutual inductance between physical branches, conductor/sample-derived
loss matrices, shield coupling, explicit matching networks, or full-wave
effects. Its rung sources describe ideal series feeds; matching to a 50-ohm
transmitter is a subsequent port-network layer. Those additions can reuse the
same mode, KCL, drive, and field interfaces introduced here.

## References

- C. E. Hayes, W. A. Edelstein, J. F. Schenck, O. M. Mueller, and M. Eash,
  “An efficient, highly homogeneous radiofrequency coil for whole-body NMR
  imaging at 1.5 T,” *Journal of Magnetic Resonance* **63** (1985), 622–628,
  <https://doi.org/10.1016/0022-2364(85)90257-4>.
- J.-M. Jin, *Electromagnetic Analysis and Design in Magnetic Resonance
  Imaging*, CRC Press, 1999,
  <https://www.routledge.com/Electromagnetic-Analysis-and-Design-in-MagneticResonance-Imaging/Jin/p/book/9780849396939>.
- C.-L. Chin, C. M. Collins, S. Li, B. J. Dardzinski, and M. B. Smith,
  “BirdcageBuilder: Design of specified-geometry birdcage coils with desired
  current pattern and resonant frequency,” *Concepts in Magnetic Resonance*
  **15** (2002), 156–163, <https://doi.org/10.1002/cmr.10030>.
- S. F. Ahmad et al., “Recent progress in birdcage RF coil technology for MRI
  systems,” *Diagnostics* **10** (2020), 1017,
  <https://doi.org/10.3390/diagnostics10121017>.
