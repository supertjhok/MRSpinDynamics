# Performance Worklist

This document captures the 2026-06 source-tree performance audit and tracks
implementation work. The package currently favors MATLAB parity and readable
NumPy implementations, so each item should be validated against the existing
physics and fixture tests before being marked complete.

## Work Items

1. Tuned phase-encoded imaging does redundant branch work.
   - Finding: `_probe_imaging` computes four acquisitions per k-space point for
     tuned imaging, but the tuned signal path only uses branches 1 and 3.
   - Files: `src/spin_dynamics/workflows/imaging.py`.
   - Plan: skip branches 2 and 4 for tuned raw/weighted receive modes while
     preserving matched-probe four-branch phase cycling.
   - Status: implemented and focused tests passing.

2. NQR selective-pulse sweeps rebuild the same tiny propagators many times.
   - Finding: `simulate_slse` calls `apply_selective_pulse` once per echo, which
     rebuilds the selective Hamiltonian and eigensolves the propagator for every
     echo. Offset sweeps also repeatedly rebuild spin matrices.
   - Files: `src/spin_dynamics/nqr/simulation.py`,
     `src/spin_dynamics/nqr/operators.py`.
   - Plan: cache spin matrices and reuse one selective-pulse unitary per
     orientation/sequence when relaxation is not active.
   - Status: implemented and focused tests passing.

3. Coupled isochromat simulation repeats per-step validation and static
   Hamiltonian construction inside the isochromat loop.
   - Finding: `_step_isochromat_values` is called for every isochromat and every
     step, even though each result is an isochromat-sized array. Scalar-coupling
     Hamiltonians are also independent of local B0 offsets.
   - Files: `src/spin_dynamics/coupling/isochromats.py`.
   - Plan: precompute per-step B0/B1 arrays once and reuse a static J
     Hamiltonian.
   - Status: implemented and focused tests passing.

4. Core `arb10` kernels allocate and recompute heavily in repeated free
   precession segments.
   - Finding: free-precession updates build full `MatrixElements`, zero arrays,
     and exponentials per segment. Repeated CPMG intervals could reuse factors
     or use a diagonal free-step update.
   - Files: `src/spin_dynamics/core/kernels.py`,
     `src/spin_dynamics/workflows/acquisition.py`.
   - Plan: add a specialized free-precession path and scratch-buffer reuse, then
     benchmark against finite CPMG and diffusion workflows.
   - Status: partially implemented and focused tests passing. Serial free
     precession now avoids full matrix-element construction; broader
     scratch-buffer reuse remains backlog.

5. Echo reconstruction can dominate compact CPMG runs.
   - Finding: `_echo_train_from_spectra` materializes an `nacq x numpts`
     isochromat phase matrix and repeats it for clean/noisy spectra.
   - Files: `src/spin_dynamics/workflows/cpmg.py`.
   - Plan: reuse/carry the phase matrix when reconstructing clean and noisy
     echoes; consider chunked or FFT-based reconstruction for large uniform
     grids.
   - Status: partially implemented and focused tests passing. Clean/noisy finite
     CPMG train reconstruction now reuses the phase matrix; chunked/FFT
     reconstruction remains backlog.

6. 2D inverse Laplace transforms form dense Kronecker systems.
   - Finding: `invert_laplace_2d` builds `np.kron(k2, k1)` and regularization
     selectors repeat full solves for each candidate strength.
   - Files: `src/spin_dynamics/analysis/ilt.py`,
     `src/spin_dynamics/analysis/regularization.py`.
   - Plan: evaluate a `LinearOperator`/separable solver path and cache kernels
     and penalty structures across candidate strengths.
   - Status: partially implemented and focused tests passing. Regularization
     selectors now validate axes and build kernels/design matrices once per
     sweep; a `LinearOperator`/separable solver remains backlog for large 2D
     inversions.

7. Threading thresholds are too eager for compact cases.
   - Finding: existing and quick audit benchmarks show chunked worker overhead
     can dominate below large isochromat counts.
   - Files: `src/spin_dynamics/core/kernels.py`, workflow docs and benchmarks.
   - Plan: choose workers from estimated work (`numpts * sequence_steps`) and
     document avoiding nested sweep/phase and isochromat workers for small runs.
   - Status: partially implemented. Chunked kernels now require larger default
     chunks before using multiple workers; richer sequence-step-aware selection
     remains backlog.

## Validation Notes

For code changes, run focused tests first, then the PythonSpinDynamics smoke
workflow before pushing:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '/mnt/c/Users/super/OneDrive - Brookhaven National Laboratory/Codex/NMR/PythonSpinDynamics' && python3 -m pytest tests/test_nqr.py tests/test_coupling.py tests/test_imaging_frequency.py"
```

Before pushing, reproduce the GitHub smoke job:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '/mnt/c/Users/super/OneDrive - Brookhaven National Laboratory/Codex/NMR/PythonSpinDynamics' && bash scripts/setup_dev_env_wsl.sh && . .venv-wsl/bin/activate && python -m unittest tests.smoke_tests && python -m ruff check src tests examples && python docs/generate_api_reference.py && git diff --exit-code docs/python_api/api_reference.md"
```
