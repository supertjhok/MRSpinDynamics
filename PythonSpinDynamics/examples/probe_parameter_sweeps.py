"""Run compact tuned and matched CPMG probe-parameter sweeps."""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import (  # noqa: E402
    run_matched_mistuning_sweep,
    run_matched_q_sweep,
    run_matched_z_magnetization_q_sweep,
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
)


def _summary(name: str, result) -> None:
    # Parameter sweep results stack one row per sweep value.
    best = int(result.snr.argmax())
    print(f"{name}:")
    print(f"  values: {result.values.size}")
    print(f"  mrx shape: {result.mrx.shape}")
    print(f"  echo shape: {result.echo.shape}")
    print(f"  best {result.value_label}: {result.values[best]:.6g}")
    print(f"  best SNR: {result.snr[best]:.6g}")


def _z_summary(name: str, result) -> None:
    # Z-magnetization sweeps track nutation/inversion depth instead of SNR.
    depth = 1 - np.abs(result.mz)
    best = np.unravel_index(int(np.argmax(depth)), depth.shape)
    print(f"{name}:")
    print(f"  values: {result.values.size}")
    print(f"  mz shape: {result.mz.shape}")
    print(f"  pulse samples: {result.tvect.size}")
    print(f"  max inversion Q: {result.values[best[0]]:.6g}")
    print(f"  max inversion depth: {depth[best]:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Keep default sweeps deliberately small so this file stays a quick smoke
    # test as well as a readable starting point.
    q_values = [20, 50, 80]
    offsets = [-2, 0, 2]
    _summary(
        "Tuned Q sweep",
        run_tuned_q_sweep(q_values=q_values, numpts=args.numpts, num_workers=args.workers),
    )
    _summary(
        "Tuned mistuning sweep",
        run_tuned_mistuning_sweep(
            offsets=offsets,
            numpts=args.numpts,
            num_workers=args.workers,
        ),
    )
    _summary(
        "Matched Q sweep",
        run_matched_q_sweep(q_values=q_values, numpts=args.numpts, num_workers=args.workers),
    )
    _summary(
        "Matched mistuning sweep",
        run_matched_mistuning_sweep(
            offsets=offsets,
            numpts=args.numpts,
            num_workers=args.workers,
        ),
    )
    _z_summary(
        "Matched Z magnetization Q sweep",
        run_matched_z_magnetization_q_sweep(
            q_values=q_values,
            numpts=args.numpts,
            num_workers=args.workers,
        ),
    )


if __name__ == "__main__":
    main()
