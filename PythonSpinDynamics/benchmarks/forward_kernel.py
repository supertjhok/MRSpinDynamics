"""Pre-acceleration baseline for the JAX/Numba performance work (Phase 0).

Two timing groups, the two halves of the acceleration plan
(see ``docs/performance.md``):

* ``kernel`` — the core segment-loop propagator, timed end to end through the
  finite ideal CPMG train across isochromat-grid sizes and echo counts. This is
  what Phase 1 (Numba) and Phase 2 (JAX) speed up.
* ``optimizer`` — a bounded refocusing phase optimization, timed across the
  number of phase segments, recording how many forward objective evaluations
  each run costs. This is what Phase 3 (autodiff) collapses: today every
  gradient step costs O(num_segments) forward sims via finite differences.

Run before any acceleration change and save the command; rerun the same command
on the same host afterward. Prefer medians; do not compare across BLAS thread
settings.

Examples::

    python -B benchmarks/forward_kernel.py --group kernel --sizes 1001,4001 --num-echoes 64
    python -B benchmarks/forward_kernel.py --group optimizer --segments 8,16,32 --optimizer scipy
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path
import statistics
import sys
import time

# Pin BLAS so the measurement reflects the Python kernels, not nested threading.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from spin_dynamics.core.kernels import (  # noqa: E402
    set_arb10_backend,
    sim_spin_dynamics_arb10,
    sim_spin_dynamics_arb10_batched,
)
from spin_dynamics.core.rotations import rf_matrix_elements  # noqa: E402
from spin_dynamics.optimization.refocusing import (  # noqa: E402
    optimize_ideal_v0crit_refocusing_phases,
)
from spin_dynamics.workflows import run_ideal_cpmg_train  # noqa: E402


def _positive_int_list(value: str) -> list[int]:
    vals = [int(part) for part in value.split(",") if part.strip()]
    if not vals or any(item <= 0 for item in vals):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return vals


def _median_time(fn, repeats: int, warmups: int) -> tuple[float, float, float]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples), max(samples)


def _build_cpmg_arb10_params(numpts: int, num_echoes: int, maxoffs: float = 10.0) -> dict:
    """A prebuilt CPMG-style arb10 parameter set (no workflow assembly).

    One excitation pulse followed by ``num_echoes`` blocks of
    (free, refocus, free) with a single acquisition per echo. Times the kernel
    itself, isolating the Phase-1 segment-loop change from workflow overhead.
    """

    del_w = np.linspace(-maxoffs, maxoffs, numpts)
    exc = rf_matrix_elements(del_w, w1=1.0, tp=np.pi / 2, phi=0.0)
    ref = rf_matrix_elements(del_w, w1=1.0, tp=np.pi, phi=np.pi / 2)
    tp = [np.pi / 2]
    pul = [1]
    amp = [1.0]
    acq = [False]
    grad = [0.0]
    for _ in range(num_echoes):
        tp += [2.0, np.pi, 2.0]
        pul += [0, 2, 0]
        amp += [0.0, 1.0, 0.0]
        acq += [False, False, True]
        grad += [0.0, 0.0, 0.0]
    return {
        "tp": np.asarray(tp, dtype=np.float64),
        "pul": np.asarray(pul, dtype=np.int64),
        "amp": np.asarray(amp, dtype=np.float64),
        "acq": np.asarray(acq, dtype=bool),
        "grad": np.asarray(grad, dtype=np.float64),
        "Rtot": [exc, ref],
        "del_w": del_w,
        "del_wg": np.ones(numpts),
        "w_1": np.ones(numpts),
        "T1n": np.full(numpts, 1e6),
        "T2n": np.full(numpts, 1e6),
        "m0": np.ones(numpts, dtype=np.complex128),
        "mth": np.zeros(numpts, dtype=np.complex128),
    }


def _bench_rawkernel(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    print(f"[rawkernel] sizes={args.sizes} num_echoes={args.num_echoes} repeats={args.repeats}")
    for numpts in args.sizes:
        params = _build_cpmg_arb10_params(numpts, args.num_echoes)

        def case(params: dict = params) -> None:
            sim_spin_dynamics_arb10(params)

        median, lo, hi = _median_time(case, args.repeats, args.warmups)
        rows.append(
            {
                "group": "rawkernel",
                "numpts": str(numpts),
                "num_echoes": str(args.num_echoes),
                "segments": str(params["tp"].size),
                "optimizer": "",
                "evals": "",
                "median_seconds": f"{median:.6f}",
                "min_seconds": f"{lo:.6f}",
                "max_seconds": f"{hi:.6f}",
            }
        )
        print(f"  numpts={numpts:6d}  median={median:8.4f}s  ({lo:.4f}-{hi:.4f})")
    return rows


def _bench_batch(args: argparse.Namespace) -> list[dict[str, str]]:
    """Batched (vmap) JAX kernel across a batch of same-structure cases.

    This is the GPU-enabling path (Phase 2b): one wide program over `batch`
    independent simulations instead of a Python/thread loop. Always uses JAX;
    set the device with JAX_PLATFORMS=cpu / default (gpu).
    """

    rows: list[dict[str, str]] = []
    print(
        f"[batch] sizes={args.sizes} batch={args.batch} num_echoes={args.num_echoes} "
        f"repeats={args.repeats}"
    )
    for numpts in args.sizes:
        base = _build_cpmg_arb10_params(numpts, args.num_echoes)
        cases = [
            {**base, "T2n": base["T2n"] * (1.0 + 0.01 * i)} for i in range(args.batch)
        ]

        def case(cases: list = cases) -> None:
            sim_spin_dynamics_arb10_batched(cases)

        median, lo, hi = _median_time(case, args.repeats, args.warmups)
        rows.append(
            {
                "group": "batch",
                "numpts": str(numpts),
                "num_echoes": str(args.num_echoes),
                "segments": str(base["tp"].size),
                "optimizer": f"batch={args.batch}",
                "evals": "",
                "median_seconds": f"{median:.6f}",
                "min_seconds": f"{lo:.6f}",
                "max_seconds": f"{hi:.6f}",
            }
        )
        print(
            f"  numpts={numpts:6d} batch={args.batch:4d}  median={median:8.4f}s "
            f"({lo:.4f}-{hi:.4f})  per-case={median / args.batch * 1e3:7.2f} ms"
        )
    return rows


def _bench_kernel(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    print(f"[kernel] sizes={args.sizes} num_echoes={args.num_echoes} repeats={args.repeats}")
    for numpts in args.sizes:
        def case(numpts: int = numpts) -> None:
            run_ideal_cpmg_train(
                numpts=numpts,
                maxoffs=10.0,
                num_echoes=args.num_echoes,
                t1_seconds=1.7,
                t2_seconds=1.1,
                num_workers=1,
                auto_refine_grid=False,
                rephase_action="ignore",
            )

        median, lo, hi = _median_time(case, args.repeats, args.warmups)
        rows.append(
            {
                "group": "kernel",
                "numpts": str(numpts),
                "num_echoes": str(args.num_echoes),
                "segments": "",
                "optimizer": "",
                "evals": "",
                "median_seconds": f"{median:.6f}",
                "min_seconds": f"{lo:.6f}",
                "max_seconds": f"{hi:.6f}",
            }
        )
        print(f"  numpts={numpts:6d}  median={median:8.4f}s  ({lo:.4f}-{hi:.4f})")
    return rows


def _bench_optimizer(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    print(
        f"[optimizer] segments={args.segments} optimizer={args.optimizer} "
        f"numpts={args.opt_numpts} repeats={args.repeats}"
    )
    for num_segments in args.segments:
        rng = np.random.default_rng(0)
        initial = rng.uniform(0.0, 2 * np.pi, size=num_segments)
        last_evals = {"n": 0, "method": ""}

        def case(initial: np.ndarray = initial) -> None:
            result = optimize_ideal_v0crit_refocusing_phases(
                initial,
                numpts=args.opt_numpts,
                optimizer=args.optimizer,
            )
            last_evals["n"] = int(result.history_scores.size)
            last_evals["method"] = result.optimizer_method

        median, lo, hi = _median_time(case, args.repeats, args.warmups)
        rows.append(
            {
                "group": "optimizer",
                "numpts": str(args.opt_numpts),
                "num_echoes": "",
                "segments": str(num_segments),
                "optimizer": last_evals["method"],
                "evals": str(last_evals["n"]),
                "median_seconds": f"{median:.6f}",
                "min_seconds": f"{lo:.6f}",
                "max_seconds": f"{hi:.6f}",
            }
        )
        print(
            f"  segments={num_segments:3d}  median={median:8.4f}s  "
            f"evals={last_evals['n']:5d}  method={last_evals['method']}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=["rawkernel", "kernel", "batch", "optimizer", "all"],
        default="all",
    )
    parser.add_argument("--batch", type=int, default=64, help="batch size for --group batch")
    parser.add_argument(
        "--backend",
        choices=["numpy", "numba", "jax"],
        default="numpy",
        help="arb10 kernel backend; 'numba'/'jax' require their optional extras",
    )
    parser.add_argument("--sizes", type=_positive_int_list, default=[501, 1001, 2001, 4001])
    parser.add_argument("--num-echoes", type=int, default=64)
    parser.add_argument("--segments", type=_positive_int_list, default=[8, 16, 32])
    parser.add_argument(
        "--optimizer", choices=["auto", "pattern", "scipy", "jax"], default="pattern"
    )
    parser.add_argument("--opt-numpts", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.num_echoes <= 0 or args.opt_numpts <= 1:
        raise SystemExit("num-echoes must be positive and opt-numpts must exceed 1")
    if args.repeats <= 0 or args.warmups < 0:
        raise SystemExit("repeats must be positive and warmups non-negative")

    set_arb10_backend(args.backend)
    print(
        f"NumPy {np.__version__}; cpu_count={os.cpu_count()}; "
        f"BLAS threads pinned to {os.environ.get('OMP_NUM_THREADS')}; "
        f"backend={args.backend}"
    )

    if args.batch <= 0:
        raise SystemExit("batch must be positive")

    rows: list[dict[str, str]] = []
    if args.group in ("rawkernel", "all"):
        rows.extend(_bench_rawkernel(args))
    if args.group in ("kernel", "all"):
        rows.extend(_bench_kernel(args))
    if args.group == "batch":
        rows.extend(_bench_batch(args))
    if args.group in ("optimizer", "all"):
        rows.extend(_bench_optimizer(args))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
