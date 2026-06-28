# MRSpinDynamics — Performance Acceleration Plan (JAX / Numba)

_Last updated: 2026-06-28_

This document maps the current computational situation in the workspace and lays
out a phased plan for a compiled / autodiff acceleration backend. It expands on
opportunity #3 of the [repository roadmap](roadmap.md) ("JAX/Numba isochromat
backend") and the "compiled or GPU acceleration backends" item in
`PythonSpinDynamics/docs/python_api/known_gaps.md`.

The engineering payoff is twofold and the two halves should not be conflated:

1. **Forward-model speed** — making each isochromat propagation faster.
2. **Optimization speed via autodiff** — making each *gradient* cheaper, which is
   a much larger win for the pulse-optimization workflows than raw kernel speed
   alone.

## 1. Where the time actually goes

Almost the entire simulation engine collapses to **one kernel shape**: a Python
`for` loop over pulse-sequence segments, each applying a 3×3 coherence-basis
rotation to a magnetization state that is **already vectorized over
isochromats**.

| Hot path | Location | Shape | Notes |
|---|---|---|---|
| Core propagator | `core/kernels.py` → `sim_spin_dynamics_arb10` | Python loop over `~3·num_echoes` segments × 9 complex elementwise mults over `numpts` (≤ ~64k) | The inner loop behind every CPMG / FID / finite-acquisition workflow. |
| Pulse-matrix build | `core/rotations.py` → `rf_matrix_elements`, `calc_rotation_matrix` | elementwise `sqrt/sin/cos/exp` over the offset grid + a Python composition loop | Rebuilt per pulse for the `arb7`/sweep paths. |
| Per-isochromat eig | `core/kernels.py` → `_matrix_elements_power` | `for idx in range(numpts)`: a 3×3 `np.linalg.eig` + `inv` **per isochromat** | Pathological scaling; radiation-damping path only, but the worst Python-level offender. |
| Dense diagonalization | `nqr/hamiltonians.py` → `diagonalize_site`; ESR Zeeman | small dense `eigh` (dim 3–4) | Cheap per call, but multiplied across powder-orientation and site scans. |

### Current acceleration (NumPy + threads only)

The package requires only `numpy>=1.24`; `scipy` is an optional extra. Two
threading layers exist today:

- **Isochromat chunking** — `sim_spin_dynamics_arb10_chunked`
  (`core/kernels.py`) splits the offset grid across a `ThreadPoolExecutor`.
- **Sweep-case threading** — `_run_sweep_cases` (`workflows/sweeps.py`).

Both rely on NumPy releasing the GIL during large array ops. Measured scaling is
real but limited and only helps for large grids: the committed benchmark
(`benchmarks/README.md`, 2026-06-08, 24-CPU Windows host) shows **no benefit
below ~32k isochromats**, ~1.6× at 32k and ~1.9× at 64k, with >8 workers
regressing on memory bandwidth.

### The optimization bottleneck

`optimization/_bounded.py` → `scipy_maximize` calls SciPy `L-BFGS-B`
**without an analytic Jacobian**, so each gradient step costs `N+1` full forward
simulations (finite differences over `N` phase segments). That is then
multiplied by multi-start counts (`num_starts=24` in `optimization/drivers.py`)
and passes. For a 30–55-segment SPA pulse this finite-difference overhead — not
the forward model — is the dominant cost in the repository, and autodiff
removes it outright.

## 2. Why the codebase is already well-suited for this

The hard part is mostly done. The kernels are:

- **pure functions** over plain arrays (no hidden global state);
- **vectorized over isochromats** (the natural batch/`vmap` axis);
- backed by a **NumPy reference plus an extensive fixture/parity test suite**, so
  any new backend can be validated bit-for-bit (within tolerance) against the
  established output.

The single structural obstacle is the **data layout**: `MatrixElements`
(`core/rotations.py`) is a frozen dataclass of nine named complex arrays. A
compiled/JIT backend wants a stacked `(numpts, 3, 3)` (or `(3, 3, numpts)`)
array and a `scan`/`njit`-friendly state tuple. The plan therefore introduces a
stacked representation at the kernel boundary while keeping the dataclass as the
public/reference type.

## 3. Two wins, two tools

| Win | Best tool | Touches |
|---|---|---|
| Raw kernel speed (forward sims, sweeps) | **Numba** (CPU, native Windows) *or* **JAX** (CPU/GPU) | segment loop + matrix-element build |
| Faster optimization via autodiff | **JAX only** | replaces finite-difference `L-BFGS-B` gradients |

Numba is the cheapest, lowest-risk speedup: no functional rewrite, native
Windows support, immediate constant-factor gains on the per-segment loop. JAX is
the strategic target because it is the *only* path that also delivers autodiff
(the actual lever on optimization cost), GPU execution, and `vmap`-over-sweeps.

### Platform caveats (the host is Windows 11)

- **JAX GPU on Windows** effectively requires WSL2; CPU wheels run natively on
  Windows. Numba runs natively on Windows with no caveat.
- **JAX defaults to float32.** Complex128 parity with the NumPy reference
  requires `jax.config.update("jax_enable_x64", True)` — mandatory for passing
  the parity harness.
- New backends must stay **optional extras** so the minimal-install philosophy
  and the 3-Python × 2-OS CI matrix are preserved.

## 4. Phased implementation plan

**Phase 0 — Baseline & regression gate (prerequisite).** _In progress._
Add `numba`/`jax` as optional extras; build a deterministic benchmark for the
forward kernel and an optimizer run; and lock golden numerical outputs from the
current NumPy path so every later phase validates against them. Deliverables:
`benchmarks/forward_kernel.py`, `tests/_perf_scenarios.py`,
`tests/test_perf_golden.py`, and committed golden fixtures.

**Phase 1 — Numba quick win (low risk).** _Landed (compiled path pending
numba-host validation)._ Implemented:

- batched `_matrix_elements_power` (`core/kernels.py`) — the per-isochromat
  Python `eig` loop is gone, replaced by NumPy's broadcasting `eig`/`inv`
  (dependency-free; validated against a per-isochromat reference and the
  radiation-damping suite);
- stacked `(num_pulses,3,3,numpts)` representation (`_stack_matrix_elements`)
  and a `@njit(nogil=True)` segment-loop core (`core/_numba_kernels.py`) wrapped
  as an opt-in backend, selected by `set_arb10_backend("numba")` / the
  `backend=` kwarg and honored by both the serial and chunked acquisition paths;
  the public API is unchanged and `"numpy"` stays the default.

The compiled core's *algorithm* is validated on every host by running it in
pure-Python (numba-absent) mode against the NumPy kernel
(`tests/test_numba_backend.py`, matches to ~1e-17); the JIT path is gated behind
`@skipUnless(NUMBA_AVAILABLE)`.

#### Measured results

Validated in WSL2 Ubuntu 24.04, Python 3.12.3, NumPy 2.4.6, **Numba 0.65.1**
(installs cleanly against NumPy 2.4.6 — the Windows dev env simply lacks it),
20-logical-CPU host, BLAS pinned to 1 thread, 2026-06-28. The skipped JIT parity
tests all pass under this venv.

Raw kernel, single thread (`forward_kernel.py --group rawkernel`,
`num_echoes=256`, median seconds):

| isochromats | numpy | numba | speedup |
| ---: | ---: | ---: | ---: |
| 1,001 | 0.0178 | 0.0107 | 1.66× |
| 4,001 | 0.0598 | 0.0378 | 1.58× |
| 16,001 | 0.2675 | 0.1662 | 1.61× |
| 64,001 | 1.3846 | 0.7300 | 1.90× |

The single-thread win is moderate because the NumPy kernel is already vectorized;
Numba mainly removes the per-segment temporaries. The larger win is **threading**:
`nogil=True` lets the existing isochromat-chunking pool scale where NumPy's
partial GIL release cannot (64,001 isochromats):

| path | 1 worker | 4 workers | 8 workers |
| --- | ---: | ---: | ---: |
| numpy | 1.391 s | 0.656 s (2.1×) | — |
| numba | 0.695 s | 0.215 s | 0.145 s |

Best Numba (8 workers, 0.145 s) vs the original NumPy single-thread baseline
(1.391 s) is **9.6×**. End to end through `run_ideal_cpmg_train` (which also does
PAP phase cycling and echo assembly outside the kernel) the single-thread gain is
a more modest ~1.3×, so the kernel swap helps most for large grids and long
trains driven through the chunked path.

**Phase 2 — JAX backend behind a protocol.** _Landed._ The
`backend="numpy"|"numba"|"jax"` selector (`set_arb10_backend`) now also routes to
a JAX kernel (`core/_jax_kernels.py`): a `jax.lax.scan` over segments,
`jit`-compiled with x64 enabled. The sequence *structure* (pulse/free mask,
pulse index, acquisition mask, per-segment time/gradient) is passed as the
scan's per-step inputs, so the loop body is traced once and **one compilation
serves every sequence and train length** — and the kernel is `vmap`-able over
sweeps and differentiable for Phase 3.

> Design note: the first attempt unrolled the Python segment loop at trace time.
> That compiled the entire train into one XLA graph, so a 256-echo run took
> >14 min and ~7 GB to compile. Switching to `lax.scan` made compilation O(1) in
> train length (the parity suite went from hanging to 0.34 s).

#### Measured results

Same WSL2 host as Phase 1, **jax 0.10.2 (CPU jaxlib)**, x64 on, 2026-06-28. All
JAX parity tests pass (raw kernel and the full CPMG workflow match NumPy).
Raw kernel, `num_echoes=256`, median seconds (`results/jax_rawkernel_2026-06-28.csv`):

| isochromats | numpy | numba | jax | jax vs numpy |
| ---: | ---: | ---: | ---: | ---: |
| 4,001 | 0.0737 | 0.0426 | 0.0546 | 1.35× |
| 16,001 | 0.2417 | 0.1546 | 0.1805 | 1.34× |
| 64,001 | 1.4149 | 0.7609 | 0.4863 | 2.91× |

On CPU, JAX wins at large grids (XLA fusion + its own intra-op threading, which
the BLAS pin does not constrain), overtaking even single-thread Numba at 64k; at
small grids dispatch overhead makes it the slowest. The CPU numbers understate
its value — GPU execution, `vmap` over sweeps, and `jax.grad` (Phase 3) are the
real reasons to carry this backend.

#### GPU (NVIDIA RTX 4060 Ti, WSL2) — batch, don't single-shot

Tested with `jax[cuda12]` 0.10.2 on the RTX 4060 Ti (8 GB), x64 on, 2026-06-28.
GPU parity matches NumPy. The headline finding: **a single forward run is
*slower* on this GPU than on CPU, and the GPU only wins when work is batched.**
(`results/jax_gpu_2026-06-28.csv`.)

| workload | GPU | CPU | verdict |
| --- | ---: | ---: | --- |
| single forward run (64k isochromats, 256 echoes) | 3.36 s | 0.58 s | GPU 5.8× **slower** |
| single forward run (128k) | 6.67 s | 0.86 s | GPU 7.8× **slower** |
| fused matmul-only micro (64k, no cond/acq) | 0.032 s | 0.131 s | GPU 4× faster |
| `vmap` batch=64 parallel sims (2k×769) | 0.055 s | 0.353 s | GPU 6.4× faster |
| `vmap` batch=256 parallel sims | 0.489 s | 3.40 s | GPU 7.0× faster |

Diagnosis: the slowdown is **not** float64. A consumer GeForce throttles FP64,
but the fused-op micro shows only a 1.6× complex128-vs-complex64 gap, so the
kernel is memory/latency-bound, not FP64-throughput-bound. The real cause is
structural: one simulation is a **sequential `lax.scan` of ~769 small steps**
(each only a few ops over the isochromat vector) with a per-step `cond` and
`dynamic_update_slice`. That launches a long chain of tiny GPU kernels and
starves the device — hence the 100× gap between the single-run kernel (3.36 s)
and the fused-op micro (0.032 s). CPU executes the same short sequential chain
with far less per-step overhead.

The fix is not precision but **parallel width**: `vmap` many independent
simulations into one wide program and the GPU wins 6–7× even at complex128. That
is exactly the shape of optimizer multistarts (Phase 3) and parameter sweeps
(Phase 2b). **Recommendation:** keep CPU (Numba, or JAX-CPU) as the default for
single forward runs; reserve the GPU for `vmap`-batched workloads. Routing a lone
`sim_spin_dynamics_arb10` call to the GPU is a pessimization and should not be a
default.

**Phase 2b — `vmap` the batched workloads (GPU enablement).** _Landed._ Added
`sim_spin_dynamics_arb10_batched` (`core/kernels.py`) over `run_arb10_batched`
(`core/_jax_kernels.py`): many same-structured simulations (shared pulse program;
per-case `del_w`, `T1n/T2n`, `m0/mth`, `Rtot`) run as one `jax.vmap` program.
Validated against looped single NumPy runs on both CPU and GPU. **Wired into a
concrete workflow:** `run_ideal_cpmg_relaxation_sweep`
(`workflows/batched_sweeps.py`) evaluates the ideal finite CPMG echo train over a
(T1, T2) grid in one batched call (per-branch batch of the PAP phase cycle, then
combine), reusing the validated cpmg.py construction helpers and matching looped
`run_ideal_cpmg_train` to <1e-8.

> Design note: getting the GPU to actually win took two kernel changes that a
> single-run kernel doesn't need. The per-step `lax.cond` and especially the
> `dynamic_update_slice` scatter into the carried acquisition buffer crippled the
> GPU (the full kernel batched at 17 ms/case on GPU). A **branchless** variant —
> `jnp.where` instead of `cond`, and emit every step's transverse component then
> gather the acquired rows instead of scattering — dropped that to ~3 ms/case
> (the scatter removal alone was ~34×). The branchless kernel computes both
> branches and materializes all `nseg` steps, which is *slower* on CPU, so
> `run_arb10_batched` **dispatches by device**: branchless on GPU, the
> memory-light `cond` kernel on CPU.

#### Measured results

Per-case time, CPMG `numpts=4001`, `num_echoes=64`, `batch=128`
(`results/jax_batched_2026-06-28.csv`):

| path | per-case | vs numpy single |
| --- | ---: | ---: |
| numpy single (loop) | 15.0 ms | 1.0× |
| numba single (loop) | 10.0 ms | 1.5× |
| JAX CPU batched (`vmap`, cond) | 7.5 ms | 2.0× |
| JAX GPU batched (`vmap`, branchless) | 3.07 ms | 4.9× |

So batching helps CPU ~2× (XLA fusion) and GPU ~4.9× vs the NumPy baseline
(2.4× over the best CPU).

End-to-end through `run_ideal_cpmg_relaxation_sweep`
(`results/jax_sweep_2026-06-28.csv`), the win tracks how much of the workflow is
the kernel:

| sweep | loop (numpy) | batched | device | speedup |
| --- | ---: | ---: | --- | ---: |
| N=64, numpts=8001, 64 echoes | 9.03 s | 2.53 s | GPU | 3.57× |
| N=64, numpts=8001, 64 echoes | 9.46 s | 4.20 s | CPU | 2.25× |
| N=128, numpts=201, 32 echoes | 2.63 s | 2.55 s | GPU | 1.03× |

The honest caveat: at **small grids** (numpts ≈ 100–200, common for many CPMG
sweeps) the kernel is trivial and the workflow's per-case Python overhead
(param assembly, echo construction) dominates, so batching the kernel alone is a
wash. The batched-sweep win shows up at kernel-dominated sizes (large `numpts` /
long trains). Pushing the small-grid case further would mean vectorizing the
surrounding workflow (echo construction, param assembly) too — a future
refinement. The probe *asymptotic* CPMG paths are not arb10-based and stay on
CPU; the optimizer multistart drivers are the other natural adopter (Phase 3).

**Phase 3 — Autodiff optimization (the headline).** _Landed for the ideal
v0crit refocusing objective._ Ported `evaluate_ideal_v0crit_refocusing_pulse`'s
math (the `calc_rot_axis_arba4` rotation-axis composition, `calc_v0crit`, and the
windowed score) to JAX in `optimization/_jax_objectives.py`, exposing a
`jax.value_and_grad` factory. `optimize_ideal_v0crit_refocusing_phases(...,
optimizer="jax")` feeds that analytic gradient to `L-BFGS-B` via the new
`scipy_maximize_with_grad` backend (`optimization/_bounded.py`). The pulse
structure is host-baked; only the segment phases are traced.

Validation: the JAX score matches NumPy to 1e-6; the JAX gradient matches a
central finite difference of the NumPy objective to <5e-3 relative
(`tests/test_jax_optimizer.py`).

#### Measured results

CPU, ideal v0crit objective, `numpts=801`, compile cached across runs
(`results/jax_optimizer_2026-06-28.csv`):

| segments | finite-diff L-BFGS | autodiff L-BFGS | speedup |
| ---: | ---: | ---: | ---: |
| 16 | 0.736 s (749 evals) | 0.059 s (45 evals) | 12.6× |
| 32 | 2.005 s (1189 evals) | 0.060 s (37 evals) | 33× |

Finite differencing costs ~`N` extra forward evals per gradient, so its eval
count (and time) grows with the number of phase segments; autodiff's gradient is
one reverse pass regardless of `N`, so its eval count stays ~the iteration count.
The walltime advantage therefore widens with `N`.

Two implementation lessons:

> **Cache the compiled objective.** The factory is `lru_cache`d by configuration.
> Without that, every `optimize()` call built a fresh `jax.jit` closure and paid
> a full ~1–2 s XLA compile — which made autodiff *slower* than finite
> differencing despite ~25× fewer evals. With caching, the compile is amortized
> over a multistart (or repeated runs) and the table above holds. At a tiny
> single problem (`numpts=101`, 8 segments) the unamortized compile still
> dominates, so autodiff only pays off once the compile is amortized or the
> forward eval is non-trivial.

> **Autodiff changes gradient *cost* and *quality*, not which optimum is found.**
> This objective is highly multimodal, and finite-difference gradients are
> additionally *inaccurate* on its stiff `1/v0crit` term — so an FD run and an
> autodiff run from the same start legitimately land in different optima. Global
> quality still comes from multistart; autodiff makes each start far cheaper and
> its gradients exact. The batched primitive (Phase 2b) is the natural way to run
> those multistarts in parallel.

**Scope:** the tuned/untuned/matched refocusing objectives go through
probe-specific NumPy machinery and are not yet JAX-ported; the ideal v0crit path
(used by the multistart drivers and the optimizer benchmark) is the
implemented one.

**Phase 4 — Batched dense solvers.** _Landed for NQR powder scans._ Refactored
`nqr/hamiltonians.py`: the transition-extraction logic is now a shared helper, and
`diagonalize_sites_over_b0(site, b0_vectors, backend=...)` builds every
Hamiltonian at once (`batched_nqr_hamiltonians`, one contraction for the Zeeman
term) and runs a single batched Hermitian eigensolve — NumPy's `eigh` broadcasts
over the leading axis; the `"jax"` backend (`nqr/_jax_eigh.py`) adds GPU. The
per-orientation `diagonalize_site` loop in `simulate_weak_b0_spectrum` now goes
through it (`backend=` param, results identical). Validated against the
per-orientation loop for spin-1 and spin-3/2, and jax-vs-numpy
(`tests/test_batched_nqr.py`).

#### Measured results

Powder grid of N=4000 orientations, spin-1 (3×3 Hamiltonians)
(`results/batched_eigh_2026-06-28.csv`):

| stage | loop | batched NumPy | batched JAX |
| --- | ---: | ---: | ---: |
| `eigh` only | 16.2 ms | 2.8 ms (5.8×) | 6.0 ms CPU / 10.1 ms GPU |
| full powder diagonalize | 402 ms | 255 ms (1.6×) | — |

Two honest takeaways: (1) **batched NumPy `eigh` is the right tool** here — 5.8×
on the eigensolve via LAPACK over the stack, no extra dependency. (2) **JAX/GPU
does not help for 3×3 matrices**: dispatch overhead exceeds the work, exactly the
"thin work starves the GPU" pattern from Phase 2b. The `"jax"` backend is wired
and validated, but only expected to pay off for *larger* Hilbert spaces (spin ≥
5/2, multi-spin, or ESR hyperfine systems) — that is the case worth revisiting
when those land. The full powder scan is 1.6× because, once `eigh` is batched,
the per-orientation **transition extraction** (the `dim²` dipole/label work plus
object construction) dominates; vectorizing that — or working directly on the
batched eigenarrays instead of per-orientation `NQRTransition` objects — is the
next refinement.

## 5. Sequencing recommendation

Do **Phase 0 → 1 (Numba)** first for an immediate, safe CPU win, then
**Phase 2 → 3 (JAX)** for the autodiff optimizer payoff — the larger strategic
prize and the direct answer to "make optimizations much faster." Phase 4 follows
opportunistically alongside the higher-spin NQR work.
