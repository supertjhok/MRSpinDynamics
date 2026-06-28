# Benchmarks

Benchmark Python kernels against the active MATLAB benchmark suite:

```text
../MATLABSpinDynamics/SpinDynamicsUpdated/Version_2/code/benchmarks
```

Start with correctness-oriented tiny cases. Add timing comparisons only after
the NumPy implementation reproduces MATLAB output within agreed tolerances.

## Environment

Run benchmarks from the persistent PythonSpinDynamics development environment,
not from whichever `python` happens to be first on `PATH`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev_env.ps1
& ".\.venv-win\Scripts\Activate.ps1"
python scripts\verify_dev_env.py --strict
```

For WSL benchmarking, use `Ubuntu-24.04` and the WSL setup script:

```bash
bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict
```

For CUDA-enabled JAX benchmarking in WSL:

```bash
JAX_CUDA=13 bash scripts/setup_dev_env_wsl.sh
source .venv-wsl/bin/activate
python scripts/verify_dev_env.py --strict --require-jax-gpu
```

For small GPU smoke runs on memory-limited cards, set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX does not reserve most device
memory up front.

Save the verifier output with curated timing results so NumPy/SciPy, Numba,
JAX, `jaxlib`, and device differences are visible.

## Benchmark Policy

Benchmarks serve two different purposes in this repository:

- **Validation benchmarks** record solver boundaries or numerical stability
  behavior. Commit compact CSV outputs when they document a meaningful
  validation boundary, such as the matched-diffusion high-Q limit.
- **Performance benchmarks** compare runtime across implementation choices.
  Treat these as host-specific measurements. Commit only summary CSVs that
  support a documented recommendation, and include the date, Python runtime,
  NumPy/SciPy versions, operating system, and CPU class in the surrounding
  documentation.

Use `benchmarks/results/` for curated results only. Scratch timing sweeps,
profile dumps, and exploratory runs should stay outside the checkout or under
`.tmp/`.

Before changing performance-sensitive code, run a small baseline and save the
command line. After the change, rerun the same command on the same machine.
Prefer medians over single timings, and avoid comparing runs across different
BLAS threading settings.

Recommended quick checks:

```powershell
python -B benchmarks\long_cpmg_workers.py --sizes 1001,4001 --workers 1,2 --num-echoes 64 --repeats 2
python -B benchmarks\diffusion_high_q_validation.py --q-values 100,1000,2000,2500 --numpts 17 --num-echoes 2
python examples\porous_rock_cpmg_walkers.py --grid 24 --z-cells 32 --pores 90 --walkers-per-voxel 2 --num-echoes 6 --substeps 2 --benchmark-backends --plot-output .tmp\porous_dt2.png
```

## Long CPMG Isochromat Worker Sweep

`long_cpmg_workers.py` benchmarks the public finite ideal CPMG train workflow
while varying the isochromat vector size and the number of chunked `arb10`
workers:

```powershell
python -B benchmarks\long_cpmg_workers.py `
  --sizes 8001,16001,32001 `
  --workers 1,2,4,8 `
  --num-echoes 256 `
  --repeats 2 `
  --warmups 1 `
  --output benchmarks\results\long_cpmg_workers_heavy_2026-06-08.csv
```

The benchmark pins common BLAS thread-count environment variables to `1` before
importing NumPy, so the sweep measures the explicit isochromat chunking rather
than nested BLAS threading. Timings below were measured on 2026-06-08 with the
bundled Codex Python runtime, NumPy 2.3.5, and a 24-logical-CPU Windows host.

### 64 Echo Smoke Sweep

Small grids complete so quickly that chunking overhead dominates:

| Isochromats | 1 worker | 2 workers | 4 workers | 8 workers | Best speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 501 | 0.026 s | 0.041 s | 0.042 s | 0.047 s | 1.00x |
| 1,001 | 0.043 s | 0.059 s | 0.085 s | 0.088 s | 1.00x |
| 2,001 | 0.075 s | 0.140 s | 0.117 s | 0.162 s | 1.00x |
| 4,001 | 0.191 s | 0.229 s | 0.435 s | 0.242 s | 1.00x |

### 256 Echo Long-Train Sweep

For longer trains and larger grids, chunking begins to help. The crossover on
this host is around 32k isochromats:

| Isochromats | 1 worker | 2 workers | 4 workers | 8 workers | Best speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,001 | 1.181 s | 1.244 s | 2.143 s | 3.347 s | 1.00x |
| 16,001 | 2.046 s | 2.044 s | 3.136 s | 4.140 s | 1.00x |
| 32,001 | 4.852 s | 3.032 s | 3.540 s | 5.624 s | 1.60x |
| 64,001 | 9.718 s | 5.966 s | 5.167 s | 6.644 s | 1.88x |

The scaling is real but limited. Two workers were best at 32,001 isochromats;
four workers were best at 64,001. Eight workers slowed down relative to the
best case, likely from memory bandwidth pressure and thread scheduling overhead.
For production long-train runs, start with `num_workers=2` or `4` rather than
blindly using all cores, then benchmark the specific echo count and grid size.

## Forward Kernel & Optimizer Baseline (acceleration Phase 0)

`forward_kernel.py` is the pre-acceleration baseline for the JAX/Numba work
described in `../../docs/performance.md`. It has two timing groups matching the
two halves of that plan:

```powershell
python -B benchmarks\forward_kernel.py --group kernel --sizes 1001,4001 --num-echoes 64 --repeats 3
python -B benchmarks\forward_kernel.py --group optimizer --segments 8,16,32 --optimizer pattern
```

- **`kernel`** times the core segment-loop propagator end to end through the
  finite ideal CPMG train, across isochromat-grid sizes. Phase 1 (Numba) and
  Phase 2 (JAX) target this.
- **`optimizer`** times a bounded refocusing phase optimization across the
  number of phase segments and records the forward-objective **eval count** per
  run. The eval count growing with `num_segments` is exactly the
  finite-difference-gradient cost that Phase 3 (autodiff) removes. The default
  `pattern` backend needs no optional dependency; pass `--optimizer scipy`
  (requires the `opt` extra) to time the `L-BFGS-B` finite-difference path, or
  `--optimizer jax` (requires the `jax` extra) for the autodiff path.

  Autodiff results (ideal v0crit objective, numpts=801, compile cached,
  `results/jax_optimizer_2026-06-28.csv`): at 16 segments finite-difference
  L-BFGS takes 0.74 s / 749 evals vs autodiff 0.059 s / 45 evals (12.6×); at 32
  segments 2.00 s / 1189 evals vs 0.060 s / 37 evals (33×). FD costs ~N extra
  forward evals per gradient; autodiff is one reverse pass regardless of N. Note
  the objective is multimodal and FD gradients are inaccurate on its stiff
  `1/v0crit` term, so FD and autodiff runs from one start may reach different
  optima — global quality comes from multistart, which autodiff makes cheap.

As with the other performance benchmarks, treat the numbers as host-specific:
run a baseline and save the command before an acceleration change, then rerun
the same command on the same host afterward. Each later backend is additionally
held to bit-for-bit (within tolerance) parity with the NumPy reference via the
`tests/test_perf_golden.py` golden fixture.

A third group, `rawkernel`, times `sim_spin_dynamics_arb10` directly on a
prebuilt CPMG parameter set, isolating the segment-loop kernel from workflow
assembly (rotation-matrix build, PAP phase cycling, echo construction). Use it
to measure backend (`--backend numpy|numba`) changes:

```bash
python -B benchmarks/forward_kernel.py --group rawkernel --backend numba --sizes 4001,16001,64001 --num-echoes 256
```

### Numba Backend Results (Phase 1)

Measured 2026-06-28 in WSL2 Ubuntu 24.04 (the Windows dev env lacks Numba),
Python 3.12.3, NumPy 2.4.6, Numba 0.65.1, 20-logical-CPU host, BLAS pinned to 1
thread. Curated medians in `results/numba_rawkernel_2026-06-28.csv`.

Raw kernel, single thread (`num_echoes=256`):

| Isochromats | numpy | numba | Speedup |
| ---: | ---: | ---: | ---: |
| 1,001 | 0.0178 s | 0.0107 s | 1.66× |
| 4,001 | 0.0598 s | 0.0378 s | 1.58× |
| 16,001 | 0.2675 s | 0.1662 s | 1.61× |
| 64,001 | 1.3846 s | 0.7300 s | 1.90× |

Chunked (`nogil`) threading at 64,001 isochromats:

| Path | 1 worker | 4 workers | 8 workers |
| --- | ---: | ---: | ---: |
| numpy | 1.391 s | 0.656 s (2.1×) | — |
| numba | 0.695 s | 0.215 s | 0.145 s (**9.6× vs numpy 1w**) |

The single-thread Numba win is moderate (the NumPy kernel is already vectorized;
Numba removes per-segment temporaries). The larger gain is that `nogil=True`
lets the isochromat-chunking pool scale where NumPy's partial GIL release stalls
near 2×. Recommendation: prefer Numba for large grids / long trains, and combine
it with `num_workers=4`–`8` via the chunked acquisition path.

### JAX Backend Results (Phase 2)

Same host, jax 0.10.2 (CPU jaxlib), x64 enabled, 2026-06-28. Curated medians in
`results/jax_rawkernel_2026-06-28.csv`. Raw kernel, `num_echoes=256`:

| Isochromats | numpy | numba | jax | jax vs numpy |
| ---: | ---: | ---: | ---: | ---: |
| 4,001 | 0.0737 s | 0.0426 s | 0.0546 s | 1.35× |
| 16,001 | 0.2417 s | 0.1546 s | 0.1805 s | 1.34× |
| 64,001 | 1.4149 s | 0.7609 s | 0.4863 s | 2.91× |

JAX uses `lax.scan` (one compilation regardless of train length) and XLA's own
intra-op threading, so it scales best on large grids — overtaking single-thread
Numba at 64k — while dispatch overhead makes it slowest on small grids. Warmups
absorb the one-time JIT compilation; report post-warmup medians only. The CPU
figures understate JAX: its real wins are GPU, `vmap` over sweeps, and autodiff
(`jax.grad`) for the optimizer (Phase 3). Recommendation: Numba for CPU forward
runs today; JAX where you need GPU, batched sweeps, or gradients.

#### GPU note (RTX 4060 Ti, jax[cuda12] 0.10.2)

A single forward run is **slower on this GPU than on CPU** (3.36 s vs 0.58 s at
64k isochromats) — a lone simulation is a sequential `lax.scan` of small steps
that starves the device. The cause is *not* FP64 (only a 1.6× complex128-vs-64
gap on a fused micro-benchmark); it is the thin, sequential, control-flow-heavy
per-step work. Batching many simulations with `vmap` flips this: GPU wins **6–7×**
even at complex128 (batch=256: 0.49 s GPU vs 3.40 s CPU). So reserve the GPU for
`vmap`-batched sweeps / optimizer multistarts, not single runs. Full numbers in
`results/jax_gpu_2026-06-28.csv`; discussion in `../../docs/performance.md`.

#### Batched kernel (`--group batch`, Phase 2b)

`sim_spin_dynamics_arb10_batched` runs many same-structure simulations in one
`vmap`. `--group batch --batch N` times it; the device is set with
`JAX_PLATFORMS=cpu` / default (gpu). Per-case time, CPMG 4001×64 echoes, batch
128 (`results/jax_batched_2026-06-28.csv`):

| path | per-case |
| --- | ---: |
| numpy single (loop) | 15.0 ms |
| numba single (loop) | 10.0 ms |
| JAX CPU batched | 7.5 ms |
| JAX GPU batched | 3.07 ms |

The batched runner dispatches by device: a branchless, scatter-free kernel on
GPU (avoids the `cond`/`dynamic_update_slice` that crush GPU throughput) and the
memory-light `cond` kernel on CPU. Batching helps CPU ~2× and GPU ~4.9× vs the
NumPy single-run baseline.

## Matched Diffusion High-Q Validation

## Matched Diffusion High-Q Validation

`diffusion_high_q_validation.py` runs the compact matched diffusion CPMG
workflow across coil Q values and records whether the echo arrays remain finite:

```powershell
python -B benchmarks\diffusion_high_q_validation.py `
  --q-values 20,50,80,100,200,500,1000,2000,2500,5000 `
  --numpts 17 `
  --num-echoes 2 `
  --output benchmarks\results\diffusion_high_q_validation_2026-06-16.csv
```

On 2026-06-16, after the analytic positive-capacitance match solution and
matched transient substepping updates, the small validation case remained
finite through Q=2000 and became non-finite at Q=2500 and above:

| Q | Finite | Runtime | Warnings | Peak \|integral\| |
| ---: | :---: | ---: | ---: | ---: |
| 20 | yes | 11.863 s | 0 | 17.397 |
| 50 | yes | 12.793 s | 0 | 17.257 |
| 80 | yes | 12.607 s | 0 | 17.097 |
| 100 | yes | 8.265 s | 0 | 16.980 |
| 200 | yes | 9.519 s | 0 | 16.304 |
| 500 | yes | 13.837 s | 0 | 15.133 |
| 1,000 | yes | 11.235 s | 0 | 14.339 |
| 2,000 | yes | 10.560 s | 0 | 11.550 |
| 2,500 | no | 10.468 s | 1177 | NaN |
| 3,000 | no | 10.120 s | 1039 | NaN |
| 5,000 | no | 9.992 s | 1024 | NaN |

A slightly larger smoke case with 33 offsets and 3 echoes also remained finite
through Q=2000. The older 2026-06-09 result file remains in
`benchmarks/results` as a historical record of the previous Q=100 boundary.

The public workflow exposes this as a solver-validation boundary through
`check_matched_diffusion_q_stability`. Calls above Q=2000 warn by default, can
raise with `q_stability_action="raise"`, or can bypass the early warning with
`q_stability_action="ignore"` for exploratory benchmark sweeps.

These results are a solver-validation boundary, not a physics conclusion. The
current pure-Python matched transient calculation uses fixed-step RK4, while
the MATLAB reference uses adaptive ODE tooling. Broad or very high-Q diffusion
studies should use a hardened transient solver before treating the Python
output as production data.
