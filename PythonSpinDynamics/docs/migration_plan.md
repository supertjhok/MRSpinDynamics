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
- The current Python test suite contains 161 checks against fixtures, public
  workflow result shapes, compatibility helpers, optional SciPy-backed result
  loaders, imaging modes, and example smoke paths.

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
- `run_ideal_cpmg_ir_train`, `run_tuned_cpmg_ir_train`,
  `run_untuned_cpmg_ir_train`, and `run_matched_cpmg_ir_train` extend finite
  CPMG trains into inversion-recovery workflows over `tauvect`, following the
  assembly pattern in `Sim_CPMG_IR/sim_cpmg_ir_matched_probe_relax4.m`.
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
- `run_tuned_time_varying_cpmg_final`, `run_untuned_time_varying_cpmg_final`,
  and `run_matched_time_varying_cpmg_final` extend the same final-echo
  assembly to probe-shaped refocusing pulses and receiver transfer functions,
  with matching amplitude-sweep wrappers.
- `examples/probe_parameter_sweeps.py` provides a compact non-plot smoke path
  for the sweep APIs.
- `examples/ideal_time_varying_cpmg.py` provides a compact non-plot smoke path
  for ideal time-varying-field amplitude sweeps.
- `examples/matched_cpmg_ir_train.py` provides a compact non-plot smoke path
  for CPMG-IR echo-integral arrays.
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
- Keep broad diffusion sweeps and Q>2000 cases behind additional solver
  validation, because the current NumPy matched-probe transient solver can
  become stiff for very high Q values.
- `run_ideal_phase_encoded_cpmg_imaging`,
  `run_tuned_phase_encoded_cpmg_imaging`, and
  `run_matched_phase_encoded_cpmg_imaging` port the compact phase-encoded CPMG
  imaging workflows from
  `Imaging_demo/imaging_example_ideal.m` and `Sim_CPMG/*_probe_img.m` into
  array-returning APIs.
- Tuned imaging preserves the MATLAB raw-current k-space convention by default
  and also exposes a receive-weighted mode that applies the tuned receiver
  transfer function and `b1_rx_map`.
- Ideal T1-prepared phase-encoded imaging adds an inversion-recovery
  preparation before phase encoding and CPMG, with post-processing helpers for
  selected-echo, echo-summed, fitted-rho, and fitted-T2 image formation.
- Compact MATLAB-generated k-space fixtures validate the ideal, tuned, and
  matched CPMG imaging workflows end to end.
- `examples/plot_ideal_imaging.py` plots the flower phantom, CPMG k-space, and
  reconstructed image for ideal, tuned, or matched probe models.
- `spin_dynamics.pulses` provides fixture-validated JMR rectangular pulse
  responses for tuned, untuned, and matched probes, phase quantization, and the
  untuned segment-length adjustment used before OCT/SPA pulse work.
- `spin_dynamics.optimization` provides the fixed SPA refocusing pulse catalog
  and MATLAB-style normalized SNR/FOM metric bookkeeping used by the
  `SPA_optimization_*` scripts. It also includes non-plotting tuned, untuned,
  and matched arbitrary-refocusing evaluators mirroring the
  `plot_masy_arbref_*` MATLAB helpers, plus array-returning SPA summary
  workflows for tuned, untuned, and matched probes. A lightweight discrete
  phase-program optimizer scaffold is available for small candidate searches.
  The ideal no-probe `v0crit`, excitation-aware `v0crit`, and
  time-varying-field refocusing objectives from
  `opt_ref_pulse_ideal_v0crit*.m` and `opt_ref_pulse_ideal_tv*.m` are available
  as array-returning evaluators, bounded phase optimizers, and multi-start
  drivers. The excitation-aware path includes the default ideal excitation
  vector preparation from `opt_ref_pulse_ideal_v0crit_exc_repeat.m`.
  Bounded refocusing phase optimizer wrappers are available around the existing
  tuned, untuned, and matched SNR evaluators; full MATLAB-equivalent OCT/SPA
  optimizer loops remain reference-only. Tuned excitation-pulse evaluation and
  bounded phase optimization are available for supplied refocusing axes.
  Array-returning multi-start driver scaffolds provide seeded repeated starts
  and best-result selection for the available refocusing, tuned excitation, and
  tuned inverse-excitation optimizers. The inverse driver starts from the
  phase-flipped target, then follows the MATLAB repeat workflow by perturbing
  the current best inverse pulse. The phase optimizers support an optional SciPy
  continuous bounded backend through the `opt` package extra, with a NumPy
  pattern-search fallback for minimal installations. Default phase bounds and
  random starts follow the MATLAB `0` to `2*pi` convention. Tuned
  inverse-excitation evaluation and bounded phase optimization are available
  for target received spectra, with compact MATLAB optimizer-result fixtures
  covering objective improvement. Strong inverse-cancellation workflows remain
  a validation target. A plotting-free pipeline helper now connects selected
  refocusing results to tuned excitation and inverse-excitation multi-start
  searches.
- MATLAB-style result-cell conversion, `.npz` archive load/export,
  script-aware `plot_opt_*_results.m` layout analysis, selected
  score/program/metadata inspection, tuned original/inverse result-pair
  comparison, and optional SciPy-backed `.mat` import/export are available for
  multi-start outputs. Keep broader real MATLAB optimizer-result fixture
  coverage, exact historical `params`/`sp`/`pp` file parity, and broad MATLAB
  `fmincon` result parity as later validation steps.

## Started Phase 9: Receiver Noise Modeling

- Keep all deterministic workflows backwards-compatible by making noise opt-in.
  Existing result fields such as `mrx`, `echo`, `echo_integrals`, `kspace`, and
  `echo_stack` should remain noiseless unless a noise configuration is passed.
- `spin_dynamics.noise.NoiseSpec` carries the model name, absolute amplitude or
  target SNR, complex-vs-real convention, scale factor, and a NumPy `Generator`
  or seed. Seeded runs are reproducible without changing global random state.
- Deterministic signal arrays are preserved, with noisy companions and metadata
  added only when requested. CPMG results expose `mrx_noisy`, `echo_noisy`, and
  optional noisy echo integrals; imaging results expose `kspace_noisy`,
  `image_noisy`, and `magnitude_noisy`.
- Ideal workflows support additive complex white Gaussian noise on the received
  spectrum or k-space, with absolute standard deviation and target-SNR modes.
- Tuned, untuned, and matched CPMG workflows can reuse the existing receiver
  noise-power calculations already used by the matched-filter SNR routines.
  These `pnoise` arrays are treated as output-referred spectral noise density
  after resistor, amplifier, and probe transfer-function filtering. The
  generated complex Gaussian spectral noise is then passed through the same
  echo path as the noiseless signal.
- Keep a later extension point for explicit source-level circuit noise. That
  mode can generate independent Johnson or amplifier noise sources before the
  receiver network and propagate them through each probe transfer function, but
  the first implementation should use output-referred `pnoise` to stay aligned
  with the current MATLAB SNR formulas.
- For imaging, inject noise in k-space or per-echo k-space before
  reconstruction so image noise, echo summation, and voxel-wise rho/T2 fitting
  inherit the correct correlations. The current implementation supports white
  k-space noise and probe-colored tuned/matched imaging noise generated from
  receiver-output spectral density before echo integration. Tuned probe-colored
  imaging noise requires `receive_mode="weighted"` because the default `raw`
  mode is a MATLAB display convention rather than a physical receiver-output
  signal.
- Validation currently covers default zero-noise parity through the existing
  fixtures, seeded reproducibility, measured white-noise variance, probe
  spectral noise from receiver density, k-space imaging noise, probe-colored
  tuned imaging noise, matched-filter SNR consistency against analytical
  `pnoise`, direct time-domain echo noise, noisy image-formation helpers,
  repeated-trial imaging statistics, and the user-facing received-noise
  example. Follow-on validation should add broader repeated-trial checks over
  fitted rho/T2 variance.

## Started Phase 10: Analysis and Inverse Laplace Processing

- `spin_dynamics.analysis` adds post-processing helpers rather than changing
  simulator parity paths. The first analysis surface covers 1D T1/T2 and
  separable 2D T1-T2/D-T2 inverse Laplace transforms.
- Kernels are explicit and reusable: T2 decay, T1 saturation recovery, T1
  inversion recovery, and diffusion attenuation. Generic 1D and 2D solvers can
  also accept precomputed kernel matrices.
- Tikhonov regularization is adjustable by strength and penalty order, with
  per-axis strengths for 2D distributions. SNR-informed selector helpers scan
  candidate strengths and apply a discrepancy-principle rule against the
  expected noise residual. Non-negative solves use SciPy's `nnls`;
  unconstrained least-squares remains NumPy-only for minimal installations.
- `examples/plot_inverse_laplace.py` generates synthetic T1, T2, T1-T2, and
  D-T2 data, adds Gaussian noise at configurable SNR levels, and plots the
  recovered distributions with either manual or automatically selected
  regularization.
- Current validation covers kernel values, unconstrained NumPy solves, compact
  SciPy-backed non-negative 1D peak recovery, compact 2D T1-T2 and D-T2 peak
  recovery, SNR-based regularization selection, and CLI exposure for the
  plotting example. Follow-on work should add L-curve/GCV regularization
  selection and real workflow handoffs from CPMG-IR, imaging echo stacks, and
  diffusion acquisitions.

## Later Phase 11: Acceleration

- Start with NumPy/SciPy.
- Add Numba, Cython, compiled C/C++, or GPU backends only behind the same public
  API and only after baseline tests pass.
- Keep benchmarks small, repeatable, and independent of plotting.
