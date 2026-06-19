# Benchmarks

Benchmark Python kernels against the active MATLAB benchmark suite:

```text
../MATLABSpinDynamics/SpinDynamicsUpdated/Version_2/code/benchmarks
```

Start with correctness-oriented tiny cases. Add timing comparisons only after
the NumPy implementation reproduces MATLAB output within agreed tolerances.

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
