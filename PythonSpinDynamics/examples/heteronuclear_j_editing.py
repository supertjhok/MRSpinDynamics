"""Demonstrate analytic heteronuclear J-editing models."""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.coupling import (  # noqa: E402
    fit_known_j_spectrum,
    j_modulation_curve,
    tango_b_filter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=33, help="Number of tau samples.")
    parser.add_argument("--max-time-ms", type=float, default=12.8, help="Maximum encoding time.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of J-encoding cycles.")
    args = parser.parse_args()

    times = np.linspace(0.0, args.max_time_ms * 1e-3, args.points)
    couplings = np.array([125.0, 160.0])
    amplitudes = np.array([0.85, 0.15])
    signal = j_modulation_curve(
        times,
        couplings,
        amplitudes,
        cycles=args.cycles,
    )
    fit = fit_known_j_spectrum(
        times,
        signal,
        couplings,
        cycles=args.cycles,
        include_background=False,
    )
    filter_response = tango_b_filter(couplings, target_coupling_hz=160.0, order=3)

    print("Heteronuclear J-editing example")
    print(f"encoding points: {times.size}")
    print(f"encoding max: {times[-1] * 1e3:.6g} ms")
    print(f"couplings: {', '.join(f'{value:.6g} Hz' for value in couplings)}")
    print(f"input amplitudes: {', '.join(f'{value:.6g}' for value in amplitudes)}")
    print(f"fit amplitudes: {', '.join(f'{value:.6g}' for value in fit.amplitudes)}")
    print(f"residual norm: {fit.residual_norm:.6g}")
    print(
        "TANGO-B order-3 filter at 160 Hz: "
        + ", ".join(f"{value:.6g}" for value in filter_response)
    )


if __name__ == "__main__":
    main()
