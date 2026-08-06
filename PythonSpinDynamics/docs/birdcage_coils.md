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

This first Phase 5 layer does **not** yet predict resonance frequency, capacitor
values, port matching, loaded Q, mode splitting, conductor/sample loss, or
current imbalance. Those require the explicit node/branch circuit solver. The
prescribed-current modes and field metrics are the acceptance reference for
that next layer, not a substitute for it.

## References

- C. E. Hayes, W. A. Edelstein, J. F. Schenck, O. M. Mueller, and M. Eash,
  “An efficient, highly homogeneous radiofrequency coil for whole-body NMR
  imaging at 1.5 T,” *Journal of Magnetic Resonance* **63** (1985), 622–628,
  <https://doi.org/10.1016/0022-2364(85)90257-4>.
- J.-M. Jin, *Electromagnetic Analysis and Design in Magnetic Resonance
  Imaging*, CRC Press, 1999,
  <https://www.routledge.com/Electromagnetic-Analysis-and-Design-in-MagneticResonance-Imaging/Jin/p/book/9780849396939>.