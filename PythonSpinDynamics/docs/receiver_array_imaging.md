# Receiver-array CPMG imaging

Phase 2 adds an uncoupled, channel-preserving receive model to ideal
phase-encoded CPMG imaging. It builds on the reciprocal complex \(B_1^-\) maps
introduced in Phase 1; it does not replace or reinterpret the existing
single-coil \(B_1/I\) calculation.

## What is modeled

For each phase encode, the Bloch/isochromat dynamics are propagated once. The
per-isochromat transverse response is then projected into all receiver channels:

\[
  y_c = \sum_r s_c(r)\,m_\perp(r),
\]

where \(s_c\) is the channel's complex reciprocal \(B_1^-\) sensitivity. This
keeps relative channel phase and avoids repeating spin propagation for every
coil. Result arrays are channel-leading:

```text
channel_kspace : (n_rx, px, pz, n_echo)
channel_image  : (n_rx, px, pz, n_echo)
sensitivity    : (n_rx, px, pz)
```

The geometry solver retains absolute T/A maps in
`ReceiveSensitivityMaps.b1_minus_t_per_a`. The experiment facade uses
`normalized_complex`, which divides each complex map by the same rho-weighted
magnitude reference used by the legacy single-coil normalization. Absolute
scaling and calibration therefore remain explicit.

## Declarative experiment

```python
import numpy as np

from spin_dynamics.experiment import (
    Acquisition, CPMGImaging, Experiment, Hardware, ImagingPlane,
    Phantom, RxArray, RxCoil, Sample, SolenoidCoil,
)

rho = np.ones((8, 8))
receivers = RxArray((
    RxCoil(SolenoidCoil(0.015, 0.03, 10, axis="x")),
    RxCoil(SolenoidCoil(0.015, 0.03, 10, axis="y")),
))
covariance = 4e-4 * np.array([[1.0, 0.3], [0.3, 1.0]])

record = Experiment(
    sequence=CPMGImaging(num_echoes=2, ny=3),
    sample=Sample(phantom=Phantom(rho)),
    hardware=Hardware(
        probe="ideal",
        rx_coil=receivers,
        plane=ImagingPlane(plane="xy", extent_m=(0.025, 0.025)),
    ),
    acquisition=Acquisition(
        receiver_noise_covariance=covariance,
        receiver_noise_seed=7,
    ),
).run()

result = record.result
raw = result.channel_image
rss = result.rss_magnitude
optimal = result.roemer_combined_image
```

Use `Acquisition(receiver_noise_std=sigma)` for independent, equal-variance
channels. Use `receiver_noise_covariance=Psi` for an absolute Hermitian
positive-semidefinite covariance \(\Psi=E[n n^H]\). These options are mutually
exclusive. The older `Acquisition.noise` field remains the scalar/probe-aware
single-channel noise interface.

## Direct workflow and combinations

When sensitivity maps are already available, call
`run_ideal_receiver_array_cpmg_imaging` directly. It accepts arbitrary complex
channel maps, so measured or externally simulated sensitivities can be used as
well as repository coil geometries.

The result includes:

- `channel_kspace`, `channel_image`, and `channel_magnitude`: raw channels;
- `rss_magnitude`: root-sum-of-squares display;
- `sensitivity_combined_image`:
  \(\sum_c s_c^*y_c / \sum_c |s_c|^2\);
- `roemer_combined_image`:
  \(s^H\Psi^{-1}y / (s^H\Psi^{-1}s)\), using a pseudoinverse for a
  positive-semidefinite covariance;
- matching `*_noisy` fields when array noise is enabled.

The existing `result.kspace`, `result.image`, and `result.magnitude` names are
compatibility views of the Roemer combination. With one unit-sensitivity
channel they match the established ideal CPMG result numerically.

The standalone combination helpers `sum_of_squares`,
`sensitivity_weighted_combine`, `roemer_combine`, and
`add_receiver_array_noise` are available from `spin_dynamics.workflows`.

## Runnable example

```bash
python examples/plot_receiver_array_cpmg.py \
  --pixels 9 --ny 1 --noise-std 0.25 --correlation 0.35 \
  --output results/receiver_array_cpmg.png
```

The compact CPMG phase-encode model uses a symmetric gradient grid. The example
therefore requires an odd image matrix and selects the matching FOV so that
k=0 is sampled and the gradient phases lie on the discrete Fourier grid. It
prints a scale-independent clean-reconstruction shape error as a guard against
spatial-encoding regressions.

The figure shows the phantom and both reciprocal sensitivities on the top row.
The bottom row compares clean and noisy Roemer reconstructions on one intensity
scale, then displays `Re(noisy - clean)` on its own symmetric scale so the
image-space noise cannot be mistaken for anatomy. The two orthogonal solenoids
also provide a convention check: their central receive phases differ by
approximately 90 degrees.

The Phase 2 path remains the uncoupled default. An `RxArray.network` now enables
Phase 4 loaded transfer functions, mutual coupling, and circuit-derived
correlated noise; see [Coupled receiver networks](receiver_networks.md).
Receiver arrays still use `probe="ideal"` because the explicit network owns the
receive tuning/loading instead of the older scalar tuned/matched probe kernels.
Cartesian sensitivity encoding and undersampling are implemented in Phase 3;
see [Cartesian SENSE imaging](sense_imaging.md).