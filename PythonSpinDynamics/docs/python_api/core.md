# Core Numerical Functions

Core functions live under `spin_dynamics.core`.

Most application code should call `spin_dynamics.workflows` first. The modules
documented here are lower-level building blocks for validation, debugging, and
continued MATLAB-to-Python conversion work.

## Echo Utilities

```python
from spin_dynamics.core.echo import (
    calc_fid_time_domain,
    calc_time_domain_echo,
    calc_time_domain_echo_arb,
)
```

- `calc_time_domain_echo` mirrors `calc_echo/calc_time_domain_echo.m`.
- `calc_time_domain_echo_arb` mirrors `calc_echo/calc_time_domain_echo_arb.m`.
- `calc_fid_time_domain` mirrors `calc_FID_decay/calc_FID_time_domain.m`.

## Rotations

```python
from spin_dynamics.core.rotations import (
    calc_rotation_matrix,
    calc_rot_axis_arba3,
    calc_rot_axis_arba4,
    calc_v0crit,
    sim_spin_dynamics_asymp_mag3,
    sim_spin_dynamics_exc,
)
```

These functions port the current Version 2 rotation and asymptotic-magnetization
helpers used by the ideal CPMG path. `calc_v0crit` mirrors
`calc_rot/calc_v0crit.m` for ideal refocusing-pulse optimization, and
`sim_spin_dynamics_exc` mirrors the excitation-vector helper used by the
v0crit-plus-excitation optimization scripts.

## Kernels

```python
from spin_dynamics.core.kernels import (
    sim_spin_dynamics_arb7,
    sim_spin_dynamics_arb10,
    sim_spin_dynamics_arb10_chunked,
)
```

- `sim_spin_dynamics_arb10` is the current arbitrary-pulse kernel using
  precomputed RF pulse matrices.
- `sim_spin_dynamics_arb10_chunked` splits large isochromat vectors into
  contiguous chunks and runs the same kernel through a thread pool.
- `sim_spin_dynamics_arb7` is retained for compatibility with the current MATLAB
  ideal FID workflow, including its acquisition-window convolution behavior.

## Isochromat Grid Checks

```python
from spin_dynamics.core.isochromats import (
    analyze_rephasing,
    check_rephasing,
    recommended_numpts_for_rephasing,
)
```

These helpers estimate angular-grid rephasing at approximately
`2*pi / spacing`. Finite CPMG train workflows use them to warn, raise, or refine
the offset grid before long simulations.

## Moving Isochromats

```python
from spin_dynamics.motion import (
    advect_diffuse_positions,
    free_precession_with_motion_step,
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
    make_semipermeable_plane,
    receive_signal,
    transverse_b1_magnitude,
)
from spin_dynamics.sequences.motion import (
    MotionSequenceStep,
    run_motion_cpmg_sequence,
    run_motion_sequence,
)
```

`spin_dynamics.motion` is an opt-in Lagrangian layer for advection and
diffusion physics. It keeps the validated fixed-isochromat kernels unchanged,
but adds reusable building blocks for particle/walker simulations where spins
move through B0, transmit-B1, and receive-B1 maps. Deterministic velocity
fields update positions through advection, diffusion adds seeded Brownian
steps, and field maps are sampled at the particles' current positions before
magnetization updates. Boundary callbacks can impose hard pore walls or
semi-permeable interfaces such as `make_semipermeable_plane` for slow exchange.
Long sequence intervals should be split into smaller steps when fields,
velocities, or membranes require finer spatial resolution.

Scalar B1 maps are treated as already perpendicular to local B0. When field
exports contain vector maps, pass `b0_vector_map` and `b1_*_vector_map` to
`make_motion_field_maps_2d`, or call `transverse_b1_magnitude` directly, to
compute the local transverse B1 sensitivity.

`spin_dynamics.sequences.motion` adds the first sequence-level driver on top of
those primitives. A `MotionSequenceStep` can combine duration, rectangular RF
amplitude/phase, gradient, acquisition sampling, and substep count. The driver
moves particles through B0/B1 maps, samples the local fields, updates
magnetization, and records receive samples at requested acquisition times.
`run_motion_cpmg_sequence` builds and runs a rectangular-pulse CPMG train with
one receive sample per echo.

## Tuned Probe

```python
from spin_dynamics.probes.tuned import (
    calc_masy_tuned_probe_lp_orig,
    calc_rot_axis_tuned_probe_lp_orig2,
    tuned_probe_lp_orig,
    tuned_probe_rx,
)
```

These functions mirror the original/reference tuned-probe CPMG path used by
`CPMG_Asymp_Examples/TunedProbeEffects_CPMG_Asymp.m`.

## Untuned Probe

```python
from spin_dynamics.probes.untuned import (
    calc_masy_untuned_probe_lp,
    calc_rot_axis_untuned_probe_lp,
    untuned_probe_lp,
    untuned_probe_rx,
)
```

These functions mirror the original/reference untuned-probe CPMG path used by
`CPMG_Asymp_Examples/UntunedProbeEffects_CPMG_Asymp.m`.

## Matched Probe

```python
from spin_dynamics.probes.matched import (
    calc_masy_matched_probe_orig,
    calc_rot_axis_matched_probe,
    find_coil_current,
    matched_probe_rx,
    matching_network_design2,
)
```

These functions mirror the original/reference matched-probe CPMG path used by
`CPMG_Asymp_Examples/MatchedProbeEffects_CPMG_Asymp.m`. The Python port uses
a NumPy-only Newton solve and fixed-step RK4 probe response so it does not need
SciPy.

## Pulse Utilities

```python
from spin_dynamics.pulses import (
    adjust_untuned_segment_lengths,
    create_wurst_pulse,
    matched_rectangular_pulse_response,
    matched_wurst_pulse_response,
    quantize_phase,
    tuned_rectangular_pulse_response,
    untuned_rectangular_pulse_response,
)
```

These helpers mirror the non-plotting array outputs from `Pulse Shape/*.m` and
the timing core of `opt_pulse/untuned_pulse_adjust.m`. They are intended as the
validated pulse layer for WURST, OCT, and SPA work. `create_wurst_pulse`
returns the WURST amplitude envelope, angular frequency sweep, and integrated
rotating-frame phase. `matched_wurst_pulse_response` applies the matched-probe
transmit model to that frequency-swept RF block.
