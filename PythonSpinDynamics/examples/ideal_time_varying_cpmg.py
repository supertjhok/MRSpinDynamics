"""Run a compact ideal CPMG sweep with time-varying B0 offsets."""

from __future__ import annotations

import argparse

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import (  # noqa: E402
    run_ideal_time_varying_amplitude_sweep,
    sinusoidal_field_waveform,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--num-echoes", type=int, default=16, help="Number of echoes.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel amplitude workers.")
    args = parser.parse_args()

    # Build a normalized B0 fluctuation waveform and test several amplitudes.
    # The runner reports the final echo and a matched-signal scalar per amplitude.
    waveform = sinusoidal_field_waveform(args.num_echoes)
    result = run_ideal_time_varying_amplitude_sweep(
        amplitudes=[0.0, 0.5, 1.0, 2.0],
        waveform=waveform,
        numpts=args.numpts,
        num_workers=args.workers,
    )
    # Pick the amplitude with the largest absolute matched-filter signal.
    best = int(abs(result.matched_signal).argmax())
    print("Ideal time-varying CPMG sweep")
    print(f"num offsets: {result.del_w.size}")
    print(f"num echoes: {result.waveform.size}")
    print(f"amplitudes: {result.amplitudes.size}")
    print(f"echo shape: {result.echo.shape}")
    print(f"best amplitude: {result.amplitudes[best]:.6g}")
    print(f"best matched signal: {result.matched_signal[best]:.6g}")


if __name__ == "__main__":
    main()
