"""Run a compact matched-probe diffusion CPMG Q sweep."""

from __future__ import annotations

import argparse

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_matched_diffusion_q_sweep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=21)
    parser.add_argument("--num-echoes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sweep-workers", type=int, default=1)
    args = parser.parse_args()

    result = run_matched_diffusion_q_sweep(
        q_values=[20, 50],
        num_echoes=args.num_echoes,
        numpts=args.numpts,
        num_workers=args.workers,
        sweep_workers=args.sweep_workers,
    )
    print("Matched diffusion CPMG Q sweep")
    print(f"q values: {result.values.size}")
    print(f"num offsets: {result.del_w.size}")
    print(f"num echoes: {result.sequence_time.size}")
    print(f"echo shape: {result.echo.shape}")
    print(f"echo integral shape: {result.echo_integrals.shape}")
    print(f"max |integral|: {abs(result.echo_integrals).max():.6g}")


if __name__ == "__main__":
    main()
