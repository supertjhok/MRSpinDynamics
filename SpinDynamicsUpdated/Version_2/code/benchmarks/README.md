# Version 2 Benchmark Suite

This folder contains small MATLAB benchmarks for the active Version 2 numerical
code. The suite is intended to capture relative timing changes before and after
kernel edits. It is not a full validation suite.

Run from MATLAB:

```matlab
cd SpinDynamicsUpdated/Version_2/code/benchmarks
results = run_spin_dynamics_benchmarks;
```

Useful options:

```matlab
results = run_spin_dynamics_benchmarks('UseTimeit', true);
results = run_spin_dynamics_benchmarks('UseTimeit', false, 'Repetitions', 5);
results = run_spin_dynamics_benchmarks('SaveResults', true);
```

Benchmarks currently included:

| Benchmark | Purpose |
| --- | --- |
| `arb10_kernel_small` | Current arbitrary-pulse reference kernel. |
| `arb8_legacy_convolution_small` | Older precomputed-rotation kernel that still uses acquisition convolution. |
| `arb_relax_diff_small` | Active diffusion-aware kernel, including the current convolution path. |
| `tiny_imaging_serial_phase_loop` | Small phase-encoding loop that mimics imaging workload structure without launching a long image simulation. |
| `time_domain_echo_small` | Time-domain echo construction from acquired magnetization. |
| `oct_objective_like_small` | Single OCT-like objective evaluation based on rotation-axis calculation and acquisition-window filtering. |

Notes:

- The runner adds `SpinDynamicsUpdated/Version_2/code` and its subfolders to the
  MATLAB path.
- Each benchmark uses persistent setup data, so the timed region focuses on the
  numerical operation.
- Plotting is disabled in all benchmark cases.
- `timeit` gives more stable numbers, while the repetition mode is faster to
  understand and easier to inspect during development.
