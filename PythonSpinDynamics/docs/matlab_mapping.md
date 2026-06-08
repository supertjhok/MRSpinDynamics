# MATLAB-to-Python Module Mapping

## Active MATLAB Source

The recommended MATLAB source tree is:

```text
../SpinDynamicsUpdated/Version_2/code
```

## Current Python Package Map

| MATLAB area | Python module | Notes |
| --- | --- | --- |
| `Params` | `spin_dynamics.parameters` | Convert `sp`, `pp`, and `params` structs to dataclasses. |
| `calc_rot` | `spin_dynamics.core.rotations` | Effective axes, rotation matrices, and matrix-element helpers. |
| `sim_spin_dynamics_arb` | `spin_dynamics.core.kernels` | Start with `sim_spin_dynamics_arb10.m`. |
| `sim_spin_dynamics_asymp` | `spin_dynamics.core.rotations` or `spin_dynamics.workflows.cpmg` | Keep low-level propagation separate from CPMG workflow code. |
| `calc_echo` | `spin_dynamics.core.echo` | Start with `calc_time_domain_echo.m`. |
| `calc_masy` | `spin_dynamics.workflows.cpmg` | High-level CPMG magnetization helpers; may call probe modules. |
| CPMG example workflows | `spin_dynamics.workflows` | Public `run_*_cpmg` helpers returning `CPMGResult`. |
| `calc_macq` | `spin_dynamics.sequences` and `spin_dynamics.core.kernels` | Split sequence construction from kernel calls. |
| `calc_macq_diff` | `spin_dynamics.diffusion` later | Defer until ideal/probe kernels are validated. |
| `circuit_simulation/matched_probe` | `spin_dynamics.probes.matched` | Matched transmit/receive and matching network helpers. |
| `circuit_simulation/tuned_probe` | `spin_dynamics.probes.tuned` | Tuned transmit/receive helpers. |
| `circuit_simulation/untuned_probe` | `spin_dynamics.probes.untuned` | Untuned transmit/receive helpers. |
| `CPMG_Asymp_Examples` | `spin_dynamics.workflows.cpmg` | Canonical smoke tests and examples. |
| `CompareQ`, `CompareMistuned` | `spin_dynamics.workflows.sweeps` | Probe Q and tuning/matching frequency sweeps returning array results. |
| `FID_Example`, `Sim_FID` | `spin_dynamics.workflows.fid` | Ideal FID should be an early workflow. |
| `Sim_CPMG`, `Imaging_demo` | `spin_dynamics.imaging` later | Defer until CPMG and probe models are stable. |
| `OCT_Pulse_Examples`, `opt_pulse` | `spin_dynamics.optimization` later | Defer until kernels are fast and trusted. |

## Initial Port Candidates and Status

| Priority | MATLAB reference | Python target |
| --- | --- | --- |
| 1 | `calc_echo/calc_time_domain_echo.m` | `spin_dynamics.core.echo` |
| 2 | Matrix-element helpers inside `sim_spin_dynamics_asymp_mag3.m` | `spin_dynamics.core.rotations` |
| 3 | `Params/set_params_ideal.m` | `spin_dynamics.parameters.constructors` |
| 4 | `calc_masy/calc_masy_ideal.m` | `spin_dynamics.workflows.cpmg` |
| 5 | `sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m` | `spin_dynamics.core.kernels` |

Current status: priorities 1-5 have NumPy ports validated against small fixtures
from the MATLAB originals. `calc_time_domain_echo_arb` is also validated and
available for direct-sum echoes from arbitrary acquired magnetization.

| Area | Python status |
| --- | --- |
| Ideal CPMG | `set_params_ideal`, `calc_masy_ideal`, `calc_time_domain_echo`, and `run_ideal_cpmg` are available. |
| Finite ideal CPMG | `run_ideal_cpmg_train` is available for finite no-probe echo trains with relaxation. |
| Finite tuned CPMG | `run_tuned_cpmg_train` is available for homogeneous finite tuned-probe echo trains with tuned pulse shaping, receiver filtering, and relaxation. |
| Finite untuned CPMG | `run_untuned_cpmg_train` is available for homogeneous finite untuned-probe echo trains with untuned pulse shaping, receiver filtering, and relaxation. |
| Finite matched CPMG | `run_matched_cpmg_train` is available for homogeneous finite matched-probe echo trains with matching-network pulse shaping, receiver filtering, and relaxation. |
| Ideal FID | `set_params_ideal_fid`, `calc_macq_fid`, `sim_spin_dynamics_arb7`, `calc_fid_time_domain`, and `sim_fid_ideal` are available. |
| Ideal finite acquisition | `calc_macq_ideal_probe_relax4` is available for assembled arbitrary sequences with relaxation during free precession. |
| Probe finite-acquisition wrappers | `calc_macq_tuned_probe_relax4`, `calc_macq_untuned_probe_relax4`, and `calc_macq_matched_probe_relax4` are available. Tuned and matched are MATLAB-fixture validated; untuned follows the same receiver-map contract. |
| Arbitrary-pulse kernel | `sim_spin_dynamics_arb10` is available. |
| Tuned-probe CPMG | `set_params_tuned_orig`, `tuned_probe_lp_orig`, `tuned_probe_rx`, `calc_rot_axis_tuned_probe_lp_orig2`, `calc_masy_tuned_probe_lp_orig`, and `run_tuned_cpmg` are available. |
| Untuned-probe CPMG | `set_params_untuned_orig`, `untuned_probe_lp`, `untuned_probe_rx`, `calc_rot_axis_untuned_probe_lp`, `calc_masy_untuned_probe_lp`, and `run_untuned_cpmg` are available. |
| Matched-probe CPMG | `set_params_matched_orig`, `matching_network_design2`, `find_coil_current`, `matched_probe_rx`, `calc_rot_axis_matched_probe`, `calc_masy_matched_probe_orig`, and `run_matched_cpmg` are available. |
| Probe Q/mistuning sweeps | `run_tuned_q_sweep`, `run_matched_q_sweep`, `run_tuned_mistuning_sweep`, and `run_matched_mistuning_sweep` are available. |
| Diffusion, imaging, OCT/SPA | Still MATLAB reference-only. |

## Naming Conventions

- Use `snake_case` for Python functions.
- Preserve MATLAB names in docstrings as references.
- Use explicit units in parameter names where possible, such as `_seconds` or
  `_normalized`.
- Keep complex-valued arrays as NumPy complex arrays.
- Keep MATLAB's `M0`, `M-`, `M+` coherence ordering documented wherever it is
  used internally.
