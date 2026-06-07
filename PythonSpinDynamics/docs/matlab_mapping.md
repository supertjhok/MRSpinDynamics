# MATLAB-to-Python Module Mapping

## Active MATLAB Source

The recommended MATLAB source tree is:

```text
../SpinDynamicsUpdated/Version_2/code
```

## Proposed Python Package Map

| MATLAB area | Python module | Notes |
| --- | --- | --- |
| `Params` | `spin_dynamics.parameters` | Convert `sp`, `pp`, and `params` structs to dataclasses. |
| `calc_rot` | `spin_dynamics.core.rotations` | Effective axes, rotation matrices, and matrix-element helpers. |
| `sim_spin_dynamics_arb` | `spin_dynamics.core.kernels` | Start with `sim_spin_dynamics_arb10.m`. |
| `sim_spin_dynamics_asymp` | `spin_dynamics.core.rotations` or `spin_dynamics.workflows.cpmg` | Keep low-level propagation separate from CPMG workflow code. |
| `calc_echo` | `spin_dynamics.core.echo` | Start with `calc_time_domain_echo.m`. |
| `calc_masy` | `spin_dynamics.workflows.cpmg` | High-level CPMG magnetization helpers; may call probe modules. |
| `calc_macq` | `spin_dynamics.sequences` and `spin_dynamics.core.kernels` | Split sequence construction from kernel calls. |
| `calc_macq_diff` | `spin_dynamics.diffusion` later | Defer until ideal/probe kernels are validated. |
| `circuit_simulation/matched_probe` | `spin_dynamics.probes.matched` | Matched transmit/receive and matching network helpers. |
| `circuit_simulation/tuned_probe` | `spin_dynamics.probes.tuned` | Tuned transmit/receive helpers. |
| `circuit_simulation/untuned_probe` | `spin_dynamics.probes.untuned` | Untuned transmit/receive helpers. |
| `CPMG_Asymp_Examples` | `spin_dynamics.workflows.cpmg` | Canonical smoke tests and examples. |
| `FID_Example`, `Sim_FID` | `spin_dynamics.workflows.fid` | Ideal FID should be an early workflow. |
| `Sim_CPMG`, `Imaging_demo` | `spin_dynamics.imaging` later | Defer until CPMG and probe models are stable. |
| `OCT_Pulse_Examples`, `opt_pulse` | `spin_dynamics.optimization` later | Defer until kernels are fast and trusted. |

## First Port Candidates

| Priority | MATLAB reference | Python target |
| --- | --- | --- |
| 1 | `calc_echo/calc_time_domain_echo.m` | `spin_dynamics.core.echo` |
| 2 | Matrix-element helpers inside `sim_spin_dynamics_asymp_mag3.m` | `spin_dynamics.core.rotations` |
| 3 | `Params/set_params_ideal.m` | `spin_dynamics.parameters.constructors` |
| 4 | `calc_masy/calc_masy_ideal.m` | `spin_dynamics.workflows.cpmg` |
| 5 | `sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m` | `spin_dynamics.core.kernels` |

Current status: priorities 1-5 have initial NumPy ports validated against
Octave-generated fixtures from the MATLAB originals. `calc_time_domain_echo_arb`
is also validated and available for direct-sum echoes from arbitrary acquired
magnetization.

## Naming Conventions

- Use `snake_case` for Python functions.
- Preserve MATLAB names in docstrings as references.
- Use explicit units in parameter names where possible, such as `_seconds` or
  `_normalized`.
- Keep complex-valued arrays as NumPy complex arrays.
- Keep MATLAB's `M0`, `M-`, `M+` coherence ordering documented wherever it is
  used internally.
