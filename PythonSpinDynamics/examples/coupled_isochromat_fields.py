"""Run a scalar-coupled sequence over inhomogeneous B0/B1 isochromats."""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.coupling import (  # noqa: E402
    coupled_isochromat_ensemble,
    coupled_spin_system,
    free_precession_step,
    rf_step,
    simulate_coupled_isochromat_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=21, help="Isochromat samples.")
    parser.add_argument("--max-b0-hz", type=float, default=3.0, help="B0 half-width in Hz.")
    parser.add_argument("--j-hz", type=float, default=7.0, help="Two-spin J coupling in Hz.")
    parser.add_argument("--free-time-ms", type=float, default=20.0, help="Free-precession time.")
    args = parser.parse_args()

    system = coupled_spin_system(
        offsets_hz=[-0.35, 0.35],
        couplings_hz=[[0.0, args.j_hz], [args.j_hz, 0.0]],
        labels=["A", "B"],
    )
    b0_offsets = np.linspace(-args.max_b0_hz, args.max_b0_hz, args.points)
    weights = np.exp(-0.5 * (b0_offsets / (0.45 * args.max_b0_hz)) ** 2)
    weights = weights / weights.sum()
    b1_tx = 1.0 + 0.15 * b0_offsets / max(args.max_b0_hz, np.finfo(float).eps)

    ensemble = coupled_isochromat_ensemble(
        system,
        b0_offsets,
        weights=weights,
        b1_tx_scale=b1_tx,
        b1_rx_scale=1.0,
    )
    free_time = args.free_time_ms * 1e-3
    static = simulate_coupled_isochromat_sequence(
        ensemble,
        [
            rf_step(duration=0.25, nutation_hz=1.0, phase=np.pi / 2.0),
            free_precession_step(free_time),
        ],
        initial_axis="x",
        detect_axis="x",
    )
    refocused = simulate_coupled_isochromat_sequence(
        ensemble,
        [
            free_precession_step(0.5 * free_time, b0_offsets_hz=b0_offsets),
            free_precession_step(0.5 * free_time, b0_offsets_hz=-b0_offsets),
        ],
        initial_axis="x",
        detect_axis="x",
    )

    print(f"isochromats: {ensemble.nisochromats}")
    print(f"B0 range: {b0_offsets[0]:.3g} to {b0_offsets[-1]:.3g} Hz")
    print(f"B1 tx range: {b1_tx.min():.3g} to {b1_tx.max():.3g}")
    print(f"static-field signal: {static.signal.real:.6g} + {static.signal.imag:.6g}j")
    print(
        "time-varying refocused signal: "
        f"{refocused.signal.real:.6g} + {refocused.signal.imag:.6g}j"
    )


if __name__ == "__main__":
    main()
