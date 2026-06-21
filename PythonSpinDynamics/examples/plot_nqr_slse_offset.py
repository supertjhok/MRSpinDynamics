"""Plot spin-1 NQR SLSE response versus RF irradiation offset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    NQRRelaxationModel,
    QuadrupolarSite,
    powder_average_grid,
    simulate_slse_offset_sweep,
    single_crystal_orientation,
)


def _pulse_duration(angle_degrees: float, nutation_hz: float) -> float:
    return np.deg2rad(angle_degrees) / (2.0 * np.pi * nutation_hz)


def _orientations(args):
    if args.orientation == "single":
        return single_crystal_orientation(alpha=args.alpha, beta=args.beta)
    return powder_average_grid(args.n_theta, args.n_phi)


def _draw_period_markers(
    axis,
    offset_khz: np.ndarray,
    echo_spacing_seconds: float,
) -> None:
    period_khz = 1.0 / echo_spacing_seconds / 1e3
    if period_khz <= 0:
        return
    limit = float(np.max(np.abs(offset_khz)))
    for marker in np.arange(-limit, limit + period_khz, period_khz):
        axis.axvline(marker, color="0.75", linewidth=0.8, zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Spin support: this pulsed example currently supports spin=1 only.",
    )
    parser.add_argument("--transition", choices=["x", "y", "z"], default="x")
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--quadrupole-khz", type=float, default=900.0)
    parser.add_argument("--nutation-khz", type=float, default=10.0)
    parser.add_argument("--pulse-angle", type=float, default=90.0)
    parser.add_argument("--echo-spacing-us", type=float, default=500.0)
    parser.add_argument("--num-echoes", type=int, default=16)
    parser.add_argument("--echo-index", type=int, default=-1)
    parser.add_argument("--max-offset-khz", type=float, default=5.0)
    parser.add_argument("--points", type=int, default=81)
    parser.add_argument("--t2-ms", type=float, default=20.0)
    parser.add_argument("--t1-ms", type=float, default=np.inf)
    parser.add_argument(
        "--orientation",
        choices=["powder", "single"],
        default="powder",
        help="Use a powder average or one fixed EFG orientation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Single-crystal azimuthal orientation angle in radians.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=np.pi / 2.0,
        help="Single-crystal polar orientation angle in radians.",
    )
    parser.add_argument("--n-theta", type=int, default=6)
    parser.add_argument("--n-phi", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    site = QuadrupolarSite(
        spin=1,
        isotope="14N",
        quadrupole_frequency_hz=args.quadrupole_khz * 1e3,
        eta=args.eta,
    )
    orientations = _orientations(args)
    nutation_hz = args.nutation_khz * 1e3
    echo_spacing_seconds = args.echo_spacing_us * 1e-6

    # The offset grid is centered on the selected zero-field NQR transition.
    # In the book's SLSE discussion, this is the axis where MSLSE develops its
    # characteristic 1 / tc modulation.
    offsets_hz = np.linspace(
        -args.max_offset_khz * 1e3,
        args.max_offset_khz * 1e3,
        args.points,
    )
    result = simulate_slse_offset_sweep(
        site,
        args.transition,
        offsets_hz,
        pulse_duration_seconds=_pulse_duration(args.pulse_angle, nutation_hz),
        nutation_hz=nutation_hz,
        echo_spacing_seconds=echo_spacing_seconds,
        num_echoes=args.num_echoes,
        orientations=orientations,
        relaxation=NQRRelaxationModel(
            t1_seconds=args.t1_ms * 1e-3,
            t2_seconds=args.t2_ms * 1e-3,
        ),
        echo_index=args.echo_index,
    )

    offsets_khz = result.sweep_values / 1e3
    amplitude = result.selected_echo_amplitudes
    magnitude = np.abs(amplitude)
    magnitude_scale = float(np.max(magnitude)) if magnitude.size else 0.0
    normalized_magnitude = (
        magnitude / magnitude_scale if magnitude_scale > 0 else magnitude
    )
    phase = (
        np.angle(amplitude[int(np.argmax(magnitude))])
        if magnitude_scale > 0
        else 0.0
    )
    aligned_real = np.real(amplitude * np.exp(-1j * phase))
    if magnitude_scale > 0:
        aligned_real = aligned_real / magnitude_scale

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    _draw_period_markers(axes[0], offsets_khz, echo_spacing_seconds)
    axes[0].plot(offsets_khz, normalized_magnitude, label="magnitude")
    axes[0].plot(offsets_khz, aligned_real, label="phase-aligned real")
    axes[0].set_xlabel("RF offset from selected line (kHz)")
    axes[0].set_ylabel("Normalized selected-echo amplitude")
    axes[0].set_title("SLSE Offset Response")
    axes[0].legend()

    # The current relaxation model is phenomenological. This panel is mainly a
    # diagnostic for the Liouville cycle eigenmodes that will later receive the
    # book-inspired dipolar/spin-lattice rates.
    axes[1].plot(offsets_khz, result.effective_t2eff_seconds * 1e3)
    axes[1].set_xlabel("RF offset from selected line (kHz)")
    axes[1].set_ylabel("Effective cycle decay time (ms)")
    axes[1].set_title("Cycle-Derived T2eff")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
