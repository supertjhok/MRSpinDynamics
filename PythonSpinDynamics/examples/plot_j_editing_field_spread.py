"""Plot J-editing echo amplitudes under B0/B1 field spread."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling import (  # noqa: E402
    coupled_isochromat_ensemble,
    coupled_spin_system,
    free_precession_step,
    rf_step,
    simulate_coupled_isochromat_sequence,
)


def _csv_floats(text: str, name: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(f"{name} must contain finite values")
    return values


def _b0_spreads(text: str) -> list[float]:
    values = _csv_floats(text, "b0 spreads")
    if any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("b0 spreads must be non-negative")
    return values


def _b1_spreads(text: str) -> list[float]:
    values = _csv_floats(text, "b1 spreads")
    if any(value < 0.0 or value >= 1.0 for value in values):
        raise argparse.ArgumentTypeError("b1 spreads must satisfy 0 <= value < 1")
    return values


def _ensemble(system, *, b0_half_width_hz: float, b1_half_spread: float, points: int):
    axis = np.linspace(-1.0, 1.0, points)
    weights = np.ones(points, dtype=np.float64) / points
    return coupled_isochromat_ensemble(
        system,
        b0_offsets_hz=b0_half_width_hz * axis,
        weights=weights,
        b1_tx_scale=1.0 + b1_half_spread * axis,
        b1_rx_scale=1.0,
    )


def _echo_curve(
    system,
    encoding_times: np.ndarray,
    *,
    b0_half_width_hz: float,
    b1_half_spread: float,
    isochromats: int,
    nutation_hz: float,
    baseline: float,
) -> np.ndarray:
    ensemble = _ensemble(
        system,
        b0_half_width_hz=b0_half_width_hz,
        b1_half_spread=b1_half_spread,
        points=isochromats,
    )
    pi_duration = 1.0 / (2.0 * nutation_hz)
    values = []
    for encoding_time in encoding_times:
        result = simulate_coupled_isochromat_sequence(
            ensemble,
            [
                free_precession_step(0.5 * encoding_time),
                rf_step(pi_duration, nutation_hz, phase=0.0),
                free_precession_step(0.5 * encoding_time),
            ],
            initial_axis="x",
            detect_axis="x",
            j_mode="secular",
        )
        values.append(result.signal.real / baseline)
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j-hz", type=float, default=7.0, help="Two-spin J coupling in Hz.")
    parser.add_argument("--points", type=int, default=81, help="Encoding-time samples.")
    parser.add_argument("--map-points", type=int, default=41, help="Samples used for the heatmap metric.")
    parser.add_argument("--isochromats", type=int, default=21, help="Isochromats per field-spread case.")
    parser.add_argument("--max-time-ms", type=float, default=160.0, help="Maximum J-encoding time.")
    parser.add_argument(
        "--b0-spreads",
        type=_b0_spreads,
        default=_b0_spreads("0,10,30,60"),
        help="Comma-separated B0 half-widths in Hz.",
    )
    parser.add_argument(
        "--b1-spreads",
        type=_b1_spreads,
        default=_b1_spreads("0,0.05,0.15,0.30"),
        help="Comma-separated relative B1 half-spreads, e.g. 0.15 for +/-15%%.",
    )
    parser.add_argument(
        "--nutation-hz",
        type=float,
        default=200.0,
        help="Nominal 180-degree pulse nutation frequency.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    if args.points < 2:
        raise SystemExit("--points must be at least 2")
    if args.map_points < 2:
        raise SystemExit("--map-points must be at least 2")
    if args.isochromats < 1:
        raise SystemExit("--isochromats must be positive")
    if args.j_hz <= 0.0 or args.nutation_hz <= 0.0 or args.max_time_ms <= 0.0:
        raise SystemExit("--j-hz, --nutation-hz, and --max-time-ms must be positive")

    plt = load_matplotlib(headless=args.output is not None)

    system = coupled_spin_system(
        offsets_hz=[0.0, 0.0],
        couplings_hz=[[0.0, args.j_hz], [args.j_hz, 0.0]],
        labels=["A", "B"],
    )
    encoding_times = np.linspace(0.0, args.max_time_ms * 1e-3, args.points)
    map_times = np.linspace(0.0, args.max_time_ms * 1e-3, args.map_points)

    ideal = _echo_curve(
        system,
        encoding_times,
        b0_half_width_hz=0.0,
        b1_half_spread=0.0,
        isochromats=1,
        nutation_hz=args.nutation_hz,
        baseline=1.0,
    )
    baseline = float(ideal[0])
    ideal = ideal / baseline

    b0_curves = [
        _echo_curve(
            system,
            encoding_times,
            b0_half_width_hz=spread,
            b1_half_spread=0.0,
            isochromats=args.isochromats,
            nutation_hz=args.nutation_hz,
            baseline=baseline,
        )
        for spread in args.b0_spreads
    ]
    b1_curves = [
        _echo_curve(
            system,
            encoding_times,
            b0_half_width_hz=0.0,
            b1_half_spread=spread,
            isochromats=args.isochromats,
            nutation_hz=args.nutation_hz,
            baseline=baseline,
        )
        for spread in args.b1_spreads
    ]
    ideal_map = _echo_curve(
        system,
        map_times,
        b0_half_width_hz=0.0,
        b1_half_spread=0.0,
        isochromats=1,
        nutation_hz=args.nutation_hz,
        baseline=baseline,
    )
    distortion = np.empty((len(args.b1_spreads), len(args.b0_spreads)), dtype=np.float64)
    for row, b1_spread in enumerate(args.b1_spreads):
        for col, b0_spread in enumerate(args.b0_spreads):
            curve = _echo_curve(
                system,
                map_times,
                b0_half_width_hz=b0_spread,
                b1_half_spread=b1_spread,
                isochromats=args.isochromats,
                nutation_hz=args.nutation_hz,
                baseline=baseline,
            )
            distortion[row, col] = float(np.sqrt(np.mean((curve - ideal_map) ** 2)))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    time_ms = 1e3 * encoding_times

    axes[0, 0].plot(time_ms, ideal, color="0.25", linestyle="--", label="ideal")
    for spread, curve in zip(args.b0_spreads, b0_curves):
        axes[0, 0].plot(time_ms, curve, label=f"+/-{spread:g} Hz")
    axes[0, 0].set_title("B0 Spread, Ideal B1")
    axes[0, 0].set_xlabel("J-encoding time (ms)")
    axes[0, 0].set_ylabel("Normalized echo amplitude")
    axes[0, 0].legend(title="B0 half-width")

    axes[0, 1].plot(time_ms, ideal, color="0.25", linestyle="--", label="ideal")
    for spread, curve in zip(args.b1_spreads, b1_curves):
        axes[0, 1].plot(time_ms, curve, label=f"+/-{100.0 * spread:g}%")
    axes[0, 1].set_title("B1 Spread, Uniform B0")
    axes[0, 1].set_xlabel("J-encoding time (ms)")
    axes[0, 1].set_ylabel("Normalized echo amplitude")
    axes[0, 1].legend(title="B1 half-spread")

    image = axes[1, 0].imshow(
        distortion,
        origin="lower",
        aspect="auto",
    )
    fig.colorbar(image, ax=axes[1, 0], label="RMS change vs ideal")
    axes[1, 0].set_title("Combined Field-Spread Distortion")
    axes[1, 0].set_xticks(np.arange(len(args.b0_spreads)))
    axes[1, 0].set_xticklabels([f"{value:g}" for value in args.b0_spreads])
    axes[1, 0].set_yticks(np.arange(len(args.b1_spreads)))
    axes[1, 0].set_yticklabels([f"{100.0 * value:g}" for value in args.b1_spreads])
    axes[1, 0].set_xlabel("B0 half-width (Hz)")
    axes[1, 0].set_ylabel("B1 half-spread (%)")

    axes[1, 1].axis("off")
    summary = (
        "Sequence per encoding time:\n"
        "free(t/2) - 180_x - free(t/2)\n\n"
        f"J = {args.j_hz:g} Hz\n"
        f"180 nutation = {args.nutation_hz:g} Hz\n"
        f"180 duration = {1e3 / (2.0 * args.nutation_hz):.4g} ms\n"
        f"isochromats/case = {args.isochromats}\n"
        "B0 changes local offsets during\n"
        "free evolution and pulses.\n"
        "B1 scales the finite 180-degree pulse."
    )
    axes[1, 1].text(0.0, 0.98, summary, va="top", family="monospace", fontsize=11)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
