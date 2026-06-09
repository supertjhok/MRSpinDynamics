# Benchmarks

Benchmark Python kernels against the active MATLAB benchmark suite:

```text
../SpinDynamicsUpdated/Version_2/code/benchmarks
```

Start with correctness-oriented tiny cases. Add timing comparisons only after
the NumPy implementation reproduces MATLAB output within agreed tolerances.

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
  --q-values 20,50,80,100,200,500,1000,5000 `
  --numpts 17 `
  --num-echoes 2 `
  --output benchmarks\results\diffusion_high_q_validation_2026-06-09.csv
```

On 2026-06-09, the small validation case remained finite through Q=100 and
became non-finite at Q=200 and above:

| Q | Finite | Runtime | Warnings | Peak \|integral\| |
| ---: | :---: | ---: | ---: | ---: |
| 20 | yes | 1.497 s | 0 | 8.034 |
| 50 | yes | 1.726 s | 0 | 17.253 |
| 80 | yes | 2.382 s | 0 | 17.097 |
| 100 | yes | 2.486 s | 0 | 16.979 |
| 200 | no | 2.537 s | 1054 | NaN |
| 500 | no | 2.721 s | 1026 | NaN |
| 1,000 | no | 2.695 s | 1026 | NaN |
| 5,000 | no | 2.671 s | 1026 | NaN |

A MATLAB-like tiny sweep over Q = 50, 500, 5000, 50000, and 500000 confirmed
the same pattern: Q=50 remained finite, while larger Q values produced
non-finite transient outputs. The matching-network Newton solve now falls back
to a least-squares step if the finite-difference Jacobian is singular, so the
validation script records high-Q transient instability instead of aborting at
network design.

These results are a solver-validation boundary, not a physics conclusion. The
current pure-Python matched transient calculation uses fixed-step RK4, while
the MATLAB reference uses adaptive ODE tooling. Broad or very high-Q diffusion
studies should use a hardened transient solver before treating the Python
output as production data.
