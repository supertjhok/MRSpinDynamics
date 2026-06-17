"""Run compact finite CPMG train probe-parameter sweeps."""

from __future__ import annotations

import argparse

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import (  # noqa: E402
    run_matched_finite_q_sweep,
    run_tuned_finite_mistuning_sweep,
    run_untuned_finite_q_sweep,
)


def _summary(label: str, result) -> None:
    # Finite sweep arrays use (sweep value, echo, ...) leading dimensions.
    print(label)
    print(f"  probe: {result.probe}")
    print(f"  sweep: {result.sweep}")
    print(f"  values: {result.values.size}")
    print(f"  offsets: {result.del_w.size}")
    print(f"  echoes: {result.sequence_time.size}")
    print(f"  mrx shape: {result.mrx.shape}")
    print(f"  echo integral shape: {result.echo_integrals.shape}")
    print(f"  max |integral|: {abs(result.echo_integrals).max():.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=21)
    parser.add_argument("--num-echoes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sweep-workers", type=int, default=1)
    args = parser.parse_args()

    # Common settings keep the three example sweeps comparable. Rephasing
    # warnings are ignored here to keep the compact smoke case quiet.
    common = {
        "numpts": args.numpts,
        "num_echoes": args.num_echoes,
        "num_workers": args.workers,
        "sweep_workers": args.sweep_workers,
        "rephase_action": "ignore",
    }
    # These are intentionally tiny sweeps; expand the value lists for studies.
    _summary(
        "Matched finite Q sweep",
        run_matched_finite_q_sweep([20, 50], **common),
    )
    _summary(
        "Untuned finite Q sweep",
        run_untuned_finite_q_sweep([20, 50], **common),
    )
    _summary(
        "Tuned finite mistuning sweep",
        run_tuned_finite_mistuning_sweep([-1, 1], **common),
    )


if __name__ == "__main__":
    main()
