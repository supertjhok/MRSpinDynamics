"""Benchmark long finite CPMG trains with chunked isochromat propagation."""

from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path
import statistics
import sys
import time

# Keep BLAS from competing with the explicit worker sweep.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows import run_ideal_cpmg_train  # noqa: E402


def _positive_int_list(value: str) -> list[int]:
    vals = [int(part) for part in value.split(",") if part.strip()]
    if not vals or any(item <= 0 for item in vals):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return vals


def _run_case(numpts: int, num_echoes: int, workers: int) -> float:
    start = time.perf_counter()
    run_ideal_cpmg_train(
        numpts=numpts,
        maxoffs=10.0,
        num_echoes=num_echoes,
        t1_seconds=1.7,
        t2_seconds=1.1,
        num_workers=workers,
        auto_refine_grid=False,
        rephase_action="ignore",
    )
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_positive_int_list, default=[501, 1001, 2001, 4001])
    parser.add_argument("--workers", type=_positive_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--num-echoes", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.num_echoes <= 0 or args.repeats <= 0 or args.warmups < 0:
        raise SystemExit("num-echoes and repeats must be positive; warmups must be non-negative")

    cpu_count = os.cpu_count() or 1
    workers = [worker for worker in args.workers if worker <= cpu_count]
    if 1 not in workers:
        workers.insert(0, 1)

    rows: list[dict[str, str]] = []
    print(
        f"CPU count: {cpu_count}; sizes={args.sizes}; workers={workers}; "
        f"num_echoes={args.num_echoes}; repeats={args.repeats}"
    )
    for numpts in args.sizes:
        baseline = None
        for worker in workers:
            for _ in range(args.warmups):
                _run_case(numpts, args.num_echoes, worker)
            samples = []
            for _ in range(args.repeats):
                gc.collect()
                samples.append(_run_case(numpts, args.num_echoes, worker))
            median = statistics.median(samples)
            if worker == 1:
                baseline = median
            speedup = (baseline / median) if baseline else 1.0
            row = {
                "numpts": str(numpts),
                "num_echoes": str(args.num_echoes),
                "workers": str(worker),
                "median_seconds": f"{median:.6f}",
                "min_seconds": f"{min(samples):.6f}",
                "max_seconds": f"{max(samples):.6f}",
                "speedup_vs_1_worker": f"{speedup:.3f}",
            }
            rows.append(row)
            print(
                f"numpts={numpts:5d} workers={worker:2d} "
                f"median={median:8.3f}s speedup={speedup:5.2f}x"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
