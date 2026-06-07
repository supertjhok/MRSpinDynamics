# Speed Audit

This is a static audit of performance-sensitive MATLAB code paths. It is not a
runtime profile yet. The goal is to identify which routines are already
optimized, which older variants are likely inefficient, and which functions are
good candidates for profiling, MEX compilation, or future Python/C translation.

## High-Level Findings

1. The active imaging simulators in `SpinDynamicsUpdated/Version_2/code/Sim_CPMG`
   are already heavily optimized relative to older code. They precompute pulse
   rotation matrices, allocate the image output up front, and use `parfor` over
   image rows.
2. `sim_spin_dynamics_arb10.m` is the current best reference kernel. It is
   vectorized, uses precomputed RF pulse matrices, removes acquisition-window
   convolution, and is marked `%#codegen`.
3. Several older arbitrary-pulse simulators still convolve acquired spectra with
   an acquisition window. The code comments correctly identify this as a major
   bottleneck because direct convolution scales as O(numpts^2).
4. The active diffusion-aware arbitrary-pulse path still resembles an older
   kernel and is likely the highest-value active modernization target if
   diffusion simulations matter.
5. Legacy optimization scripts contain many optimizer calls and older simulator
   calls. They should be treated as expensive workflows rather than as clean
   numerical kernels.
6. The oldest GPU experiment uses historical GPU syntax and nested loops over
   isochromats and time samples. It is useful historically, but should not be
   treated as the current acceleration strategy.

## Already Optimized Code Paths

### `sim_spin_dynamics_arb10.m`

Path:
`SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb10.m`

Strengths:

- marked `%#codegen`;
- stores magnetization as 3 x numpts arrays instead of looping over individual
  isochromats;
- uses precomputed pulse rotation matrices through `params.Rtot`;
- handles gradient offsets through `del_w0 + grad(j)*del_wg`;
- preallocates `macq`;
- stores acquired spectra directly rather than convolving with an acquisition
  window.

This should be the reference implementation for future compiled kernels and for
the first Python port of arbitrary-pulse spin dynamics.

### Version 2 Imaging Simulators

Paths:

- `SpinDynamicsUpdated/Version_2/code/Sim_CPMG/sim_cpmg_ideal_probe_img.m`
- `SpinDynamicsUpdated/Version_2/code/Sim_CPMG/sim_cpmg_matched_probe_img.m`
- `SpinDynamicsUpdated/Version_2/code/Sim_CPMG/sim_cpmg_tuned_probe_img.m`

Strengths:

- use `parfor` over the outer image-row loop;
- precompute RF pulse rotation matrices before the phase-encoding loops;
- preallocate `echo_int_all`;
- keep per-worker copies of `sp` and `pp`;
- move final time-domain echo construction into matrix products such as
  `isoc*mrx'`.

Remaining cost:

- runtime scales approximately as O(N^4) with image dimension because every
  phase-encoding point evaluates a spin-dynamics problem over the sample map;
- each image pixel/encoding point performs multiple phase-cycle simulations;
- progress `disp` calls inside nested loops can add overhead, especially in
  parallel pools;
- plotting at the end should remain optional for benchmark runs.

Recommendation:
Keep these as the current imaging reference. Add small benchmark cases before
changing them, because the current structure already reflects careful
optimization.

## Active Modernization Targets

### Diffusion-Aware Arbitrary-Pulse Kernel

Path:
`SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m`

Likely issue:

- still constructs an acquisition `window = sinc(...)`;
- still executes `conv(mvect(2,:),window,'same')` for every acquisition;
- does not use precomputed `Rtot` pulse matrices in the same way as
  `sim_spin_dynamics_arb10.m`;
- returns through structure-valued matrix-element helpers, which are less
  convenient for code generation than the scalar-array style used in `arb10`.

Recommended next step:
Create an `arb10`-style diffusion kernel:

- preserve the diffusion attenuation in free precession;
- remove acquisition-window convolution when downstream code computes echoes in
  the time domain;
- accept precomputed pulse rotation matrices where possible;
- mark the result `%#codegen`;
- validate against the current diffusion examples for small inputs.

### Diffusion Acquisition Wrapper

Paths:

- `SpinDynamicsUpdated/Version_2/code/calc_macq_diff/calc_macq_matched_probe_relax_diff.m`
- `SpinDynamicsUpdated/Version_2/code/calc_macq_diff/calc_macq_matched_probe_relax_diff_noRx.m`

Likely issue:

- builds `tp_curr`, `phi_curr`, `amp_curr`, and `acq_curr` by repeated vector
  concatenation inside a loop;
- calls the slower diffusion arbitrary-pulse kernel;
- recalculates receiver/time-domain echoes in a loop over acquisitions.

Recommended next step:
Profile this wrapper after the diffusion kernel is modernized. If it remains
hot, replace repeated concatenation with preallocation or a two-pass sequence
builder.

## Legacy or Historical Hot Spots

### Older Arbitrary-Pulse Kernels

Examples:

- `SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb6.m`
- `SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb7.m`
- `SpinDynamicsUpdated/Version_2/code/sim_spin_dynamics_arb/sim_spin_dynamics_arb8.m`
- `SpinDynamics/code/basic/sim_spin_dynamics_arb*.m`
- `SpinDynamics/code/matched_probe/sim_spin_dynamics_arb*.m`

Observed patterns:

- acquisition-window convolution remains in many variants;
- some older versions compute pulse matrix elements for every sequence segment
  instead of reusing precomputed rotation matrices;
- older files may include plotting side effects inside numerical routines;
- historical variants use different parameter conventions.

Recommendation:
Do not optimize these broadly. Document them as legacy/variant implementations
and migrate active workflows to `arb10`-style kernels.

### Older Diffusion Coherence Enumeration

Paths:

- `SpinDynamics/code/diffusion/coherences.m`
- `SpinDynamics/code/diffusion/coherences_new.m`

Observed patterns:

- deeply nested pathway enumeration;
- runtime grows quickly with echo number;
- comments already warn that higher `Nfinal` values become slow.

Recommendation:
Keep these as reference implementations unless diffusion pathway enumeration
becomes an active research path again. If revived, profile separately and
consider algorithmic pruning or memoization before low-level optimization.

### Historical GPU Attempt

Path:
`SpinDynamics/code/basic/sim_spin_dynamics_allpw_gpu.m`

Observed patterns:

- uses historical `gzeros`, `gsingle`, `gfor`, and `gend` syntax;
- loops over parameter chunks, pulses, isochromats, and echo time samples;
- performs matrix-vector products per isochromat inside loops.

Recommendation:
Treat this as historical. Modern acceleration should use `arb10`-style
vectorized kernels plus MATLAB Coder/MEX, Parallel Computing Toolbox, or a
future Python/C/Numba/CUDA implementation.

### Pulse Optimization Scripts

Examples:

- `SpinDynamicsUpdated/Version_2/code/OCT_Pulse_Examples/TunedProbe_OCT.m`
- `SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_tuned.m`
- `SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_untuned.m`
- many older `SpinDynamics/code/basic/opt_*` scripts.

Observed patterns:

- many optimizer calls (`fmin*`, `optim*`, etc.);
- often call spin-dynamics kernels repeatedly inside objective functions;
- some older optimization scripts use legacy kernels with convolution.

Recommendation:
Speed optimization workflows by improving the objective-function kernels first.
Then consider parallel multi-start evaluation, caching of repeated pulse/circuit
calculations, and benchmark-sized regression tests for optimized pulse quality.

## Existing Compiled Artifacts

Compiled artifacts and codegen output exist for the active arbitrary-pulse
kernel, including `sim_spin_dynamics_arb10_mex.mexw64` under
`SpinDynamicsUpdated/Version_2/code/mex`.

Recommendation:
Add a small build note later that records:

- required MATLAB Coder version;
- build command or script;
- expected input structure fields and fixed/dynamic sizes;
- how to verify MEX output against MATLAB output;
- whether generated artifacts should be committed or rebuilt by users.

## Suggested Benchmark Set

A small repeatable benchmark suite now lives in
`SpinDynamicsUpdated/Version_2/code/benchmarks`. Run it from MATLAB with:

```matlab
cd SpinDynamicsUpdated/Version_2/code/benchmarks
results = run_spin_dynamics_benchmarks;
```

It covers the initial benchmark targets:

| Benchmark | Purpose |
| --- | --- |
| `arb10_kernel_small` | Baseline `arb10` correctness and speed. |
| `arb8_legacy_convolution_small` | Measures an older acquisition-convolution path for comparison. |
| `arb_relax_diff_small` | Tracks the likely active bottleneck in `arb_relax_diff`. |
| `tiny_imaging_serial_phase_loop` | Exercises phase encoding and repeated `arb10` calls without a long image run. |
| `time_domain_echo_small` | Measures time-domain echo construction separately. |
| `oct_objective_like_small` | Measures a pulse-design objective-like calculation separately from optimizer behavior. |

Use MATLAB `timeit` for single-call timings and `profile on` / `profile viewer`
for call-tree diagnosis. Keep plotting disabled during benchmarks.

## Priority Matrix

| Priority | Target | Why |
| --- | --- | --- |
| High | Benchmark active workflows | Needed before edits so speed changes do not break physics. |
| High | Modernize `sim_spin_dynamics_arb_relax_diff.m` | Active diffusion path still has known O(numpts^2) convolution. |
| Medium | Add MEX build/verification notes for `arb10` | Supports compiled-routine goal and reproducibility. |
| Medium | Clean diffusion acquisition wrappers after profiling | Repeated concatenation may matter after kernel speedups. |
| Medium | Add serial/parallel imaging switch | Makes imaging easier to run without editing code while preserving `parfor`. |
| Low | Legacy `SpinDynamics/code/basic` optimization | Mostly historical; migrate users to active Version 2 kernels instead. |
| Low | Historical GPU code | Superseded by current vectorized/codegen approach. |

## Next Recommended Work

1. Use the benchmark suite to capture baseline MATLAB timings.
2. Implement an `arb10`-style diffusion kernel and compare output against the
   existing diffusion example.
3. Re-run the benchmark suite and document speed/correctness results.
