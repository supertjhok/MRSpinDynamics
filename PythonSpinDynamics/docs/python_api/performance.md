# Performance

The Python port prioritizes MATLAB parity and readable NumPy implementations,
then adds acceleration where benchmarks show useful wins. Performance-sensitive
workflows use vectorized NumPy kernels, explicit worker chunking over isochromat
grids, and optional Numba/JAX backends for selected large numerical kernels.

## Benchmark Commands

Create or update the persistent development environment before benchmarking:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev_env.ps1
& ".\.venv-win\Scripts\Activate.ps1"
python scripts\verify_dev_env.py --strict
```

For CUDA-enabled JAX benchmarks, use WSL:

```bash
JAX_CUDA=13 bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict --require-jax-gpu
```

For small GPU smoke runs, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid
JAX reserving most of the device memory before the benchmark starts.

Run compact benchmark checks from `PythonSpinDynamics`:

```powershell
python -B benchmarks\long_cpmg_workers.py --sizes 1001,4001 --workers 1,2 --num-echoes 64 --repeats 2
python -B benchmarks\diffusion_high_q_validation.py --q-values 100,1000,2000,2500 --numpts 17 --num-echoes 2
python examples\porous_rock_cpmg_walkers.py --grid 24 --z-cells 32 --pores 90 --walkers-per-voxel 2 --num-echoes 6 --substeps 2 --benchmark-backends --plot-output .tmp\porous_dt2.png
```

Longer benchmark sweeps and historical results are documented in
`benchmarks/README.md`.

The full porous-rock walker challenge should be run with the JAX backend rather
than `--benchmark-backends`, because benchmarking all backends would run the
NumPy path on millions of walkers:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/porous_rock_cpmg_walkers.py --backend jax --plot-output .tmp/porous_rock_challenge.png --output .tmp/porous_rock_challenge.npz
```

## Interpreting Results

Timing results depend on CPU, operating system, Python version, NumPy/SciPy
builds, BLAS threading, and worker count. Compare before/after runs on the same
machine with the same command line. Treat committed timing CSV files as
historical guidance rather than portable pass/fail thresholds.

The chunked core kernels default to larger per-worker chunks so compact
isochromat grids stay serial unless there is enough work to offset process
startup and data-transfer overhead. For small sweeps, prefer outer-level
parallelism over simultaneously enabling sweep workers and isochromat workers.

SNR-based inverse-Laplace regularization selectors reuse their validated kernel
matrices across candidate strengths. Large 2D inversions still build the dense
Kronecker design matrix for the solve, so very large maps remain better treated
as benchmark targets than interactive workloads.

The matched-diffusion high-Q benchmark is a solver-validation boundary, not a
physical limit. The current pure-Python matched transient calculation is
validated for compact cases through Q=2000 and warns above that boundary.

## Acceleration Roadmap

Numba/JAX backends are opt-in and must stay tied to NumPy-reference parity
tests and timing baselines. Current and near-term acceleration targets include:

- large finite CPMG echo trains;
- matched-probe transient response calculations;
- broad diffusion and imaging sweeps;
- moving-isochromat and voxel-walker sequence simulations.
