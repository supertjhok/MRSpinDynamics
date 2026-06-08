"""Run compact tuned and matched CPMG probe-parameter sweeps."""

from __future__ import annotations

import argparse

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import (  # noqa: E402
    run_matched_mistuning_sweep,
    run_matched_q_sweep,
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
)


def _summary(name: str, result) -> None:
    best = int(result.snr.argmax())
    print(f"{name}:")
    print(f"  values: {result.values.size}")
    print(f"  mrx shape: {result.mrx.shape}")
    print(f"  echo shape: {result.echo.shape}")
    print(f"  best {result.value_label}: {result.values[best]:.6g}")
    print(f"  best SNR: {result.snr[best]:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
