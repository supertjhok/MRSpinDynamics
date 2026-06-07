# MATLAB-to-Python Migration Plan

## Reference Policy

- Keep `SpinDynamicsUpdated/Version_2/code` as the active MATLAB reference.
- Keep `SpinDynamicsUpdated/Version_1` and `SpinDynamics` as legacy references.
- Do not move or rewrite MATLAB files as part of the Python port.

## Phase 1: Baseline and Fixtures

- Run the MATLAB quick-start ideal CPMG example.
- Save small reference outputs for `sp`, `pp`, `masy`, `echo_asy`, and `tvect`.
- Run or inspect the MATLAB benchmark suite in `Version_2/code/benchmarks`.
- Define numerical tolerances for complex spectra and echoes.

## Phase 2: Low-Level Numerical Helpers

- Port free-precession matrix elements.
- Port RF pulse matrix elements.
- Port effective rotation-axis helpers from `calc_rot`.
- Port `calc_time_domain_echo`.

These functions are good early tests because their inputs and outputs are small,
array-based, and close to NumPy's strengths.

## Phase 3: Parameter and Sequence API

- Convert MATLAB `sp`, `pp`, and `params` structures to Python dataclasses.
- Start with `set_params_ideal` and `set_params_ideal_FID`.
- Add probe-specific constructors after the ideal CPMG and FID paths pass.
- Keep units explicit: distinguish absolute seconds from normalized `w1` time.

## Phase 4: Ideal Workflows

- Port the ideal CPMG asymptotic path:
  `set_params_ideal` -> `calc_masy_ideal` -> `calc_time_domain_echo`.
- Port the ideal FID path:
  `set_params_ideal_FID` -> `simFID_ideal`.
- Add tests that compare Python arrays with MATLAB fixtures.

## Phase 5: Core Arbitrary-Pulse Kernel

- Port `sim_spin_dynamics_arb10.m` to a clear NumPy implementation.
- Preserve the MATLAB coherence ordering: `M0`, `M-`, `M+`.
- Preserve precomputed pulse rotation matrix semantics through a Python data
  structure before optimizing.
- Validate against MATLAB benchmark-sized cases.

## Phase 6: Probe Models

- Port untuned, tuned, and matched probe circuit models in that order.
- Recreate the CPMG probe-effect examples as Python workflow tests.
- Keep receiver filtering and SNR calculations separately testable.

## Phase 7: Diffusion, Imaging, and Optimization

- Port diffusion only after the non-diffusion kernel is stable.
- Revisit the diffusion kernel design using the `arb10` structure, since the
  MATLAB speed audit identifies the active diffusion path as a modernization
  target.
- Port imaging workflows after probe and CPMG paths are validated.
- Port OCT/SPA optimization last; these workflows depend on fast, trusted
  kernels.

## Phase 8: Acceleration

- Start with NumPy/SciPy.
- Add Numba, Cython, compiled C/C++, or GPU backends only behind the same public
  API and only after baseline tests pass.
- Keep benchmarks small, repeatable, and independent of plotting.
