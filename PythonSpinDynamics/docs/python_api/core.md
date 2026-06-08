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
    sim_spin_dynamics_asymp_mag3,
)
```

These functions port the current Version 2 rotation and asymptotic-magnetization
helpers used by the ideal CPMG path.

## Kernels

```python
from spin_dynamics.core.kernels import (
    sim_spin_dynamics_arb7,
    sim_spin_dynamics_arb10,
)
```

- `sim_spin_dynamics_arb10` is the current arbitrary-pulse kernel using
  precomputed RF pulse matrices.
- `sim_spin_dynamics_arb7` is retained for compatibility with the current MATLAB
  ideal FID workflow, including its acquisition-window convolution behavior.

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
