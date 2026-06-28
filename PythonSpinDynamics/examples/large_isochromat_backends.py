"""Compare the arb10 backends (NumPy / Numba / JAX) on a large isochromat grid.

This demonstrates the acceleration backends added in the performance workstream
(see ``docs/performance.md``): the *same* finite ideal CPMG echo train is run
through each available backend on a large offset grid, with timing and a
parity check of the echo integrals against the NumPy reference.

Numba and JAX are optional extras — install with ``pip install -e .[perf]``.
Missing backends are skipped with a note, so the example always runs on a plain
NumPy install. The backend is selected through the public
``spin_dynamics.core.kernels.set_arb10_backend`` switch, which the finite-train
workflow honors.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.core._jax_kernels import JAX_AVAILABLE
from spin_dynamics.core._numba_kernels import NUMBA_AVAILABLE
from spin_dynamics.core.kernels import set_arb10_backend
from spin_dynamics.workflows import run_ideal_cpmg_train


def _backend_available(name: str) -> bool:
    if name == "numpy":
        return True
    if name == "numba":
        return NUMBA_AVAILABLE
    if name == "jax":
        return JAX_AVAILABLE
    raise ValueError(f"unknown backend: {name}")


def _time_run(run, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        run()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=50001, help="Number of offset isochromats.")
    parser.add_argument("--num-echoes", type=int, default=64, help="Number of echoes.")
    parser.add_argument("--maxoffs", type=float, default=10.0, help="Maximum normalized offset.")
    parser.add_argument("--t1", type=float, default=1.7, help="T1 in seconds.")
    parser.add_argument("--t2", type=float, default=1.1, help="T2 in seconds.")
    parser.add_argument(
        "--backends",
        type=str,
        default="numpy,numba,jax",
        help="Comma-separated backends to compare (numpy is always the reference).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats (median reported).")
    args = parser.parse_args()

    requested = [b.strip() for b in args.backends.split(",") if b.strip()]
    if "numpy" not in requested:
        requested.insert(0, "numpy")

    def run_case() -> object:
        return run_ideal_cpmg_train(
            numpts=args.numpts,
            maxoffs=args.maxoffs,
            num_echoes=args.num_echoes,
            t1_seconds=args.t1,
            t2_seconds=args.t2,
            num_workers=1,
            rephase_action="ignore",
        )

    print("arb10 backend comparison on a large isochromat grid")
    print(f"isochromats: {args.numpts}   echoes: {args.num_echoes}   maxoffs: {args.maxoffs}")
    print(f"{'backend':>8} {'available':>10} {'time(s)':>10} {'speedup':>9} {'max|d| vs numpy':>16}")

    reference = None
    numpy_time = None
    for backend in requested:
        if not _backend_available(backend):
            print(f"{backend:>8} {'no':>10} {'-':>10} {'-':>9} {'(extra not installed)':>16}")
            continue

        set_arb10_backend(backend)
        # Numba and JAX pay a one-time compile on the first call; absorb it.
        warmups = 1 if backend in ("numba", "jax") else 0
        elapsed = _time_run(run_case, args.repeats, warmups)
        result = run_case()
        integrals = np.asarray(result.echo_integrals)

        if backend == "numpy":
            reference = integrals
            numpy_time = elapsed
            diff = 0.0
        else:
            diff = float(np.max(np.abs(integrals - reference)))
        speedup = (numpy_time / elapsed) if numpy_time else 1.0
        print(f"{backend:>8} {'yes':>10} {elapsed:>10.4f} {speedup:>8.2f}x {diff:>16.3e}")

    set_arb10_backend("numpy")
    print(
        "\nParity is reported as the maximum absolute echo-integral difference "
        "from the NumPy reference; all backends should agree to ~1e-9."
    )


if __name__ == "__main__":
    main()
