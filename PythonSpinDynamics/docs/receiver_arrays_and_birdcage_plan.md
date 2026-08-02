# Receiver Arrays, Sensitivity Encoding, and Quadrature-Coil Plan

**Status:** foundational phase started 2026-08-02

This note records the repository assessment and staged implementation plan for
multiple receiver coils, sensitivity encoding, mutual coupling, quadrature
operation, and birdcage resonators.

## Existing foundation

The repository already has a valid single-coil receive sensitivity model. An
`RxCoil` is driven with unit current, its Biot-Savart field is sampled on the
phantom, and reciprocity supplies the receive weighting. Existing imaging paths
use the transverse magnitude normalized to a rho-weighted mean of one.

Other reusable pieces are:

- arbitrary segmented-wire Biot-Savart fields;
- pairwise filament-loop mutual inductance;
- single-conductor PEEC L/R/C/Q/self-resonance and current-distribution solves;
- tuned and matched receiver transfer functions and scalar noise models;
- circular-component projection relative to local B0;
- multi-axis NQR excitation and ideal quadrature detection operators.

The important gaps are not a missing reciprocity formula. They are reductions
that currently discard information needed downstream:

1. receive sensitivity is exposed as one real nonnegative normalized map;
2. imaging results have no receiver-channel axis;
3. the public PEEC solve reduces one conductor path to one terminal impedance;
4. general receiver noise has no channel covariance;
5. a birdcage is a branched capacitor-loaded network, not one series wire path.

## Conventions and data model

For B0 along +z, the package convention is

```text
B1+ = (Bx + i By) / 2    transmit, handedness +1
B1- = (Bx - i By) / 2    reciprocal receive, handedness -1
```

For nonuniform B0, a deterministic local transverse basis defines the same
components. All channels at a voxel share that basis, so their relative complex
phase is preserved. Channel-leading arrays are used throughout:

```text
B_vector_per_amp : (n_rx, *spatial_shape, 3)  [T/A]
B1_minus_per_amp: (n_rx, *spatial_shape)      [T/A, complex]
channel_kspace  : (n_rx, *kspace_shape, n_echo)
```

Absolute T/A maps and calibration/normalization must remain separate. The legacy
single-channel map is retained as
`abs(B1-) / rho_weighted_mean(abs(B1-))`.

## Staged implementation

### Phase 1: complex reciprocal sensitivities and channel identity

- Preserve the existing scalar single-coil map and numerical behavior.
- Expose the absolute Cartesian unit-current field and complex B1- map.
- Add an ordered `RxArray` geometry container.
- Return channel-leading `ReceiveSensitivityMaps` with explicit T/A
  normalization factors.
- Define and test handedness, phase, shapes, serialization, and legacy parity.
- Reject accidental use of an array in a workflow that would silently collapse
  its channel axis.

Initial implementation lives in `motion.py` and `experiment/wiring.py`. The
current CPMG forward model remains single-channel until Phase 2.

A minimal channel-resolved solve is:

```python
import numpy as np

from spin_dynamics.experiment import (
    Hardware, ImagingPlane, Phantom, RxArray, RxCoil, SolenoidCoil,
    solve_receive_sensitivities,
)

rho = np.ones((64, 64))
receivers = RxArray((
    RxCoil(SolenoidCoil(0.015, 0.03, 10, axis="x")),
    RxCoil(SolenoidCoil(0.015, 0.03, 10, axis="y")),
))
sensitivity = solve_receive_sensitivities(
    Phantom(rho),
    Hardware(rx_coil=receivers, plane=ImagingPlane(plane="xy")),
)
# sensitivity.b1_minus_t_per_a.shape == (2, *rho.shape)
```

The complete visualization is
`examples/plot_receiver_array_sensitivities.py`:

```bash
python examples/plot_receiver_array_sensitivities.py \
  --output .tmp/receiver_array_sensitivities.png
```

### Phase 2: uncoupled multi-receiver forward model

- Add receiver-channel axes to imaging k-space and image results.
- Propagate spin dynamics once and project the resulting magnetization into all
  receive channels linearly.
- Support independent noise and user-supplied channel covariance.
- Add raw-channel output, sum-of-squares display, sensitivity-weighted complex
  combination, and noise-whitened Roemer combination.
- Preserve single-channel compatibility views.

Validation gates: single-channel parity, channel phase reinforcement and
cancellation, symmetry cases, and sqrt(N) SNR scaling for independent identical
channels.

### Phase 3: sensitivity encoding

Use the explicit forward model

```text
y_c = P F { s_c(r) rho(r) } + n_c
```

where `s_c` is complex B1-, `P` is the sampling mask, and `F` is the spatial
Fourier operator.

- Add Cartesian undersampling masks and a reusable encoding operator.
- Whiten using a supplied noise covariance matrix.
- Implement regularized Cartesian SENSE by alias-set solves.
- Report conditioning and g-factor maps.
- Keep sensitivity estimation separate; start with supplied or simulated maps.
- Treat compressed sensing as an optional regularizer, not a replacement for
  coil encoding.

Validation gates: full-sampling parity, an analytic R=2 unfolding problem,
rank-deficiency detection, and Monte Carlo agreement with predicted g-factor.

### Phase 4: multiport coupling and correlated noise

Generalize the PEEC kernels to several conductors and explicit ports:

```text
Z_coil(omega) = R(omega) + i omega L(omega)
```

- Represent conductors and lumped elements as a node/branch graph.
- Extract reciprocal frequency-dependent port impedance matrices.
- Add tuning, matching, decoupling, cables, preamp input impedances, and loads.
- Transform geometric maps into loaded effective sensitivities with the network
  transfer matrix.
- Generate correlated thermal noise from dissipative elements, consistent with
  `4 k T Re(Z)` at uniform temperature, plus transformed preamp voltage/current
  noise.

A scalar mutual inductance alone is insufficient because coupling changes
resonances, current modes, sensitivities, and noise covariance together.

Validation gates: reciprocal/passive Z, positive-semidefinite noise covariance,
uncoupled-limit parity, two-loop resonance splitting, and FastHenry or analytic
coupled-resonator comparisons.

### Phase 5: ideal and circuit-accurate birdcage coils

A birdcage needs explicit rungs, end-ring sections, capacitors, nodes, and ports.

First add a fast prescribed-current reference model:

- cylindrical rung/end-ring geometry;
- sinusoidal fundamental current modes;
- cosine/sine degenerate modes and +/-90-degree circular combinations;
- field uniformity and circularity metrics.

Then connect the same topology to the multiport solver:

- low-pass, high-pass, and later hybrid capacitor placement;
- modal frequencies, Q, port drive, matching, and quadrature excitation;
- capacitor tolerance, tuning imbalance, shield/sample loading, and mode
  splitting;
- transmit B1+, reciprocal B1-, and end-to-end array reconstruction.

Validation gates: expected azimuthal current pattern, fundamental-mode
degeneracy, central-field uniformity, correct helicity, analytic ladder
frequencies, energy/Q consistency, and segmentation convergence.

## Implementation priorities

The dependency order is:

1. preserve complex channel sensitivities;
2. carry channels through the forward model;
3. add array combination and SENSE;
4. model multiport coupling and correlated noise;
5. solve full birdcage modes and loaded operation.

This order delivers useful receiver arrays before the electrical network is
complete, while avoiding the central failure mode: several magnitude-only maps
that look like channels but cannot provide physically valid sensitivity
encoding.
