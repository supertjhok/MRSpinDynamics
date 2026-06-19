# Performance

The Python port prioritizes MATLAB parity and readable NumPy implementations
before aggressive acceleration. Performance-sensitive workflows currently use
vectorized NumPy kernels and, where useful, explicit worker chunking over
isochromat grids.

## Benchmark Commands

Install benchmark dependencies with:

```powershell
python -m pip install -e ".[bench,opt,plot]"
```

Run compact benchmark checks from `PythonSpinDynamics`:

```powershell
python -B benchmarks\long_cpmg_workers.py --sizes 1001,4001 --workers 1,2 --num-echoes 64 --repeats 2
python -B benchmarks\diffusion_high_q_validation.py --q-values 100,1000,2000,2500 --numpts 17 --num-echoes 2
```

Longer benchmark sweeps and historical results are documented in
`benchmarks/README.md`.

## Interpreting Results

Timing results depend on CPU, operating system, Python version, NumPy/SciPy
builds, BLAS threading, and worker count. Compare before/after runs on the same
machine with the same command line. Treat committed timing CSV files as
historical guidance rather than portable pass/fail thresholds.

The matched-diffusion high-Q benchmark is a solver-validation boundary, not a
physical limit. The current pure-Python matched transient calculation is
validated for compact cases through Q=2000 and warns above that boundary.

## Acceleration Roadmap

Compiled, JIT, or GPU backends are intentionally deferred until the reference
coverage is stronger. Good candidates for future acceleration include:

- large finite CPMG echo trains;
- matched-probe transient response calculations;
- broad diffusion and imaging sweeps;
- moving-isochromat sequence simulations.
