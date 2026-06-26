# Internal / Susceptibility Gradients

`spin_dynamics.susceptibility` generates the static internal field set up by
magnetic-susceptibility contrast between a solid matrix and the pore fluid. In
porous media this internal gradient, not the applied gradient, is usually the
dominant field inhomogeneity: it accelerates CPMG decay (diffusion in internal
gradients) and biases diffusion measurements. The module produces the internal
off-resonance field for cylindrical grains that fit the package's
two-dimensional motion field maps, and summarizes the pore-space internal
gradient, so the existing walker pipeline can simulate its effect.

## Geometry and field

Each `CylindricalInclusion` is an infinitely long cylinder perpendicular to the
`(x, z)` map plane, magnetized by a uniform applied field lying in the plane.
Outside the cylinder the parallel field perturbation is the classic 2D dipole

```text
dB_parallel(rho, phi) = B0 * (delta_chi / 2) * (a / rho)**2 * cos(2 phi)
```

with cylinder radius `a`, distance `rho`, and angle `phi` measured from the
in-plane applied-field direction. Contributions from several cylinders superpose
in the dilute (`|delta_chi| << 1`) limit.

```python
import numpy as np
from spin_dynamics.susceptibility import (
    CylindricalInclusion,
    susceptibility_offresonance_map,
    internal_gradient_distribution,
    make_susceptibility_field_maps,
)

axis = np.linspace(-50e-6, 50e-6, 161)
grains = [CylindricalInclusion(cx, cz, 12e-6)
          for cx in (-30e-6, 30e-6) for cz in (-30e-6, 30e-6)]

field = susceptibility_offresonance_map(
    axis, axis, grains,
    b0_tesla=2.0,
    susceptibility_difference=1e-6,   # grain minus fluid SI volume susceptibility
)
```

`field.offresonance_rad` is the angular off-resonance map (rad/s) in the
convention used by `spin_dynamics.motion`, `field.offresonance_hz` the same in
Hz, and `field.inclusion_mask` marks grid points inside solid grains (usually
excluded from the mobile pore fluid).

## Internal-gradient distribution

`internal_gradient_distribution` reports the pore-space distribution of the
internal-gradient magnitude in tesla per metre, dropping grid points adjacent to
a grain boundary where the discrete gradient straddles the susceptibility jump:

```python
dist = internal_gradient_distribution(field, bins=48)
print(dist.rms, dist.mean, dist.maximum)   # tesla / metre
```

`internal_gradient_maps(field)` returns the `(g_x, g_z, g_magnitude)` maps
directly when a full field is needed.

## Driving the walker pipeline

`make_susceptibility_field_maps(field)` wraps the field as `MotionFieldMaps2D`,
so the internal field drops straight into the moving-isochromat sequence
helpers. With no applied gradient, CPMG decay then arises purely from diffusion
through the internal gradient and grows with the echo spacing -- the standard
`g_internal` signature that background-gradient-suppressing sequences are
designed to cancel:

```python
from spin_dynamics.sequences.motion import run_motion_cpmg_sequence

maps = make_susceptibility_field_maps(field)
# build a pore-fluid walker ensemble, then:
result = run_motion_cpmg_sequence(
    ensemble, maps, num_echoes=24, echo_spacing=3e-3,
    excitation_duration=40e-6, refocusing_duration=80e-6,
    gradient=(0.0, 0.0),
)
```

See `examples/plot_internal_gradients.py` for an end-to-end packed-grain demo
that plots the internal field, the internal-gradient distribution, and
echo-spacing-dependent CPMG decay.

## Scope and limits

- Static, linear, dilute susceptibility perturbation: the leading dipole term
  only, no `chi`-squared corrections or self-consistent demagnetization.
- Cylinders perpendicular to the map plane (2D). Spheres and other 3D geometries
  are out of scope for the 2D motion maps.
- Background-gradient-suppressing sequences (bipolar / Cotts 13-interval PGSTE)
  are the natural next addition; the internal field this module produces is the
  input those sequences are designed to cancel.
