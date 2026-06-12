# MATLAB-to-Python Migration Status and Plan

## Reference Policy

- Keep `SpinDynamicsUpdated/Version_2/code` as the active MATLAB reference.
- Keep `SpinDynamicsUpdated/Version_1` and `SpinDynamics` as legacy references.
- Do not move or rewrite MATLAB files as part of the Python port.

## Completed Phase 1: Baseline and Fixtures

- Small reference outputs are stored under `validation/fixtures`.
- Fixture generation is scripted in `validation/octave/generate_basic_fixtures.m`.
- The same script can be run from MATLAB or Octave.
- MATLAB-generated fixtures are used for matched-probe cases that require
  optimization toolbox behavior not available in a stock Octave install.
- The current Python test suite contains 61 checks against fixtures, public
  workflow result shapes, compatibility helpers, and example smoke paths.

## Completed Phase 2: Low-Level Numerical Helpers

- Free-precession matrix elements and RF pulse matrix elements are available in
  `spin_dynamics.core.kernels` and `spin_dynamics.core.rotations`.
- Effective rotation-axis helpers from `calc_rot` are available in
  `spin_dynamics.core.rotations`.
- Echo and FID time-domain conversion helpers are available in
  `spin_dynamics.core.echo`.

These functions remain the best first place to debug numerical drift because
their inputs and outputs are small, array-based, and close to NumPy's strengths.

## Completed Phase 3: Parameter and Sequence API

- MATLAB `sp`, `pp`, and `params` structures are represented by Python
  dataclasses.
- Validated constructors include:
  - `set_params_ideal`
  - `set_params_ideal_fid`
  - `set_params_tuned_orig`
  - `set_params_untuned_orig`
  - `set_params_matched_orig`
- Units remain explicit in the API and docs: ideal "bare" spin-dynamics helpers
  use normalized `w1` time, while probe helpers mirror MATLAB's absolute-time
  circuit conventions where applicable.

## Completed Phase 4: Ideal Workflows

- The ideal CPMG asymptotic path is ported and validated:
  `set_params_ideal` -> `calc_masy_ideal` -> `calc_time_domain_echo`.
- The ideal FID path is ported and validated:
  `set_params_ideal_FID` -> `simFID_ideal`.
- Public examples and workflow documentation are available under
  `examples/` and `docs/python_api/`.

## Completed Phase 5: Core Arbitrary-Pulse Kernel

- `sim_spin_dynamics_arb10.m` has a clear NumPy implementation.
- MATLAB coherence ordering is preserved and documented as `M0`, `M-`, `M+`.
- Precomputed pulse rotation matrix semantics are retained before any optimized
  backend is introduced.
- `sim_spin_dynamics_arb10_chunked` can split large isochromat grids into
  contiguous chunks and evaluate them across a thread pool while preserving the
  serial kernel's numerical result.
- The legacy-compatible `sim_spin_dynamics_arb7` path used by ideal FID is also
  available.

## Completed Phase 6: Original/Reference Probe CPMG Models

- Tuned, untuned, and matched original/reference CPMG paths are ported.
- Public runners are available:
  - `run_tuned_cpmg`
  - `run_untuned_cpmg`
  - `run_matched_cpmg`
- Lower-level probe modules expose transmit response, receive filtering,
  effective-axis helpers, received spectra, asymptotic spectra, and SNR where
  the MATLAB path provides it.
- The matched-probe port uses a NumPy-only Newton solve and fixed-step RK4 probe
  response to avoid adding SciPy as a required dependency. It is validated
  against MATLAB fixtures with practical tolerances appropriate for the
  independent solver.

## Started Phase 7: Relaxation, Acquisition Variants, and Sweeps

- `calc_macq_ideal_probe_relax4` is ported for assembled ideal-probe arbitrary
  sequences with relaxation during free-precession intervals.
- `calc_macq_tuned_probe_relax4` and `calc_macq_matched_probe_relax4` are
  ported and fixture-validated. `calc_macq_untuned_probe_relax4` is available
  as a Python analogue using the same receiver-map contract.
- `run_ideal_cpmg_train` provides a public finite ideal CPMG acquisition
  workflow returning acquired spectra, direct-summed echoes, and echo integrals.
- `run_tuned_cpmg_train`, `run_untuned_cpmg_train`, and
  `run_matched_cpmg_train` provide public finite probe CPMG trains with probe
  pulse shaping, receiver filtering, relaxation, direct-summed echoes, and echo
  integrals.
- `run_matched_cpmg_ir_train` extends the finite matched-probe train into an
  inversion-recovery workflow over `tauvect`, following
  `Sim_CPMG_IR/sim_cpmg_ir_matched_probe_relax4.m`.
- Finite train workflows now estimate isochromat-grid rephasing time, warn or
  raise when the grid is too coarse, optionally refine `numpts` before building
  pulse matrices, and pass long isochromat vectors through the chunked backend
  with `num_workers`.
- `run_tuned_q_sweep`, `run_matched_q_sweep`, `run_tuned_mistuning_sweep`, and
  `run_matched_mistuning_sweep` port the plotting-oriented MATLAB Q and
  mistuning scripts into array-returning workflow APIs.
- `run_tuned_finite_q_sweep`, `run_untuned_finite_q_sweep`,
  `run_matched_finite_q_sweep`, and the corresponding finite mistuning sweeps
  are Python-native wrappers around the relaxation-aware finite train runners.
  They support `auto_refine_grid`, sweep-point parallelism, and chunked
  isochromat propagation.
- `run_matched_z_magnetization_q_sweep` ports the matched-probe z-magnetization
  Q sweep from `z_mag/z_Mag_Q.m`.
- `run_ideal_time_varying_cpmg_final` and
  `run_ideal_time_varying_amplitude_sweep` port the ideal time-varying-field
  final-echo workflow from `time_varying_field/sim_cpmg_ideal_tv_final.m` and
  its comparison scripts into array-returning APIs.
- `examples/probe_parameter_sweeps.py` provides a compact non-plot smoke path
  for the sweep APIs.
- `examples/ideal_time_varying_cpmg.py` provides a compact non-plot smoke path
  for ideal time-varying-field amplitude sweeps.
- `examples/matched_cpmg_ir_train.py` provides a compact non-plot smoke path
  for matched-probe CPMG-IR echo-integral arrays.
- `examples/finite_probe_train_sweeps.py` provides a compact non-plot smoke
  path for finite-train Q and mistuning sweep arrays.
- Keep workflow-level APIs returning small typed result containers, following
  `CPMGResult`.

## Started Phase 8: Diffusion, Imaging, and Optimization

- `sim_spin_dynamics_arb10_diffusion` adds the diffusion free-precession
  attenuation term to the `arb10` kernel shape while preserving precomputed RF
  matrices and avoiding the older acquisition-window convolution.
- `calc_macq_matched_probe_relax_diffusion`,
  `run_matched_diffusion_cpmg`, and `run_matched_diffusion_q_sweep` provide the
  first compact matched-probe diffusion CPMG workflows, following
  `DIffusion_Example/Diff_Echo_Q.m` and
  `Sim_Diffusion/sim_dif_matched_CPMG_noRx.m`.
- Keep broad/high-Q diffusion sweeps behind additional solver validation,
  because the current NumPy matched-probe transient solver can become stiff for
  very high Q values.
- `run_ideal_cpmg_imaging`, `run_tuned_cpmg_imaging`, and
  `run_matched_cpmg_imaging` port the compact CPMG imaging workflows from
  `Imaging_demo/imaging_example_ideal.m` and `Sim_CPMG/*_probe_img.m` into
  array-returning APIs.
- Compact MATLAB-generated k-space fixtures validate the ideal, tuned, and
  matched CPMG imaging workflows end to end.
- `examples/plot_ideal_imaging.py` plots the flower phantom, CPMG k-space, and
  reconstructed image for ideal, tuned, or matched probe models.
- `spin_dynamics.pulses` provides fixture-validated JMR rectangular pulse
  responses for tuned, untuned, and matched probes, phase quantization, and the
  untuned segment-length adjustment used before OCT/SPA pulse work.
- `spin_dynamics.optimization` provides the fixed SPA refocusing pulse catalog
  and MATLAB-style normalized SNR/FOM metric bookkeeping used by the
  `SPA_optimization_*` scripts. It also includes the first non-plotting tuned
  arbitrary-refocusing evaluator mirroring `plot_masy_arbref_tuned.m`.
  Untuned/matched evaluator variants and optimizer loops remain MATLAB
  reference-only.
- Port OCT/SPA optimization last; these workflows depend on fast, trusted
  kernels.

## Later Phase 9: Acceleration

- Start with NumPy/SciPy.
- Add Numba, Cython, compiled C/C++, or GPU backends only behind the same public
  API and only after baseline tests pass.
- Keep benchmarks small, repeatable, and independent of plotting.
