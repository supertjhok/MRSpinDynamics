"""Replicate key Konnai 2008 14N SORC trends with theory overlays.

The example follows the dimethylnitramine SORC measurements from Konnai,
Odano, and Asaji (2008): offset modulation at fixed pulse repetition time,
pulse-spacing modulation at fixed offset, and pulse-width dependence compared
with the spin-1 powder FID response.

Run with ``--output nqr_sorc_konnai2008.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    diagonalize_site,
    fid_powder_theory_signal,
    powder_average_grid,
    simulate_sorc,
    sorc_powder_theory_signal,
    sorc_sequence,
)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return values / scale if scale > 0 else values


def _paper_nutation_hz(powder_90_us: float) -> float:
    # Konnai et al. call the powder "90 deg" pulse phi = 0.66*pi.
    return 0.33 / (float(powder_90_us) * 1e-6)


def _simulate_sweep(
    *,
    site: QuadrupolarSite,
    transition: str,
    line_frequency_hz: float,
    offsets_hz: np.ndarray,
    two_tau_seconds: np.ndarray,
    pulse_width_seconds: np.ndarray,
    nutation_hz: float,
    num_pulses: int,
    orientations,
) -> np.ndarray:
    amplitudes = np.empty(offsets_hz.size, dtype=np.float64)
    for idx, (offset, two_tau, pulse_width) in enumerate(
        zip(offsets_hz, two_tau_seconds, pulse_width_seconds)
    ):
        sequence = sorc_sequence(
            transition,
            pulse_duration_seconds=float(pulse_width),
            nutation_hz=nutation_hz,
            half_spacing_seconds=0.5 * float(two_tau),
            num_pulses=num_pulses,
            rf_frequency_hz=line_frequency_hz + float(offset),
        )
        result = simulate_sorc(site, sequence, orientations=orientations)
        amplitudes[idx] = abs(result.signal_amplitudes[-1])
    return amplitudes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Spin support: this reduced SORC example currently supports spin=1 only.",
    )
    parser.add_argument("--line-khz", type=float, default=4604.25)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--transition", choices=["x", "y", "z"], default="x")
    parser.add_argument(
        "--powder-90-us",
        type=float,
        default=20.0,
        help="Pulse width corresponding to the paper's powder 90 deg pulse.",
    )
    parser.add_argument("--num-pulses", type=int, default=96)
    parser.add_argument("--n-theta", type=int, default=6)
    parser.add_argument("--n-phi", type=int, default=12)
    parser.add_argument(
        "--simulation-points",
        type=int,
        default=31,
        help="Sparse numerical SORC markers per sweep.",
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="Only plot closed-form theory curves.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_pulses <= 0:
        raise SystemExit("--num-pulses must be positive")
    if args.simulation_points <= 1:
        raise SystemExit("--simulation-points must be greater than one")

    plt = load_matplotlib(headless=args.output is not None)

    target_line_hz = args.line_khz * 1e3
    if args.transition == "x":
        quadrupole_hz = target_line_hz / (1.0 + args.eta / 3.0)
    elif args.transition == "y":
        quadrupole_hz = target_line_hz / (1.0 - args.eta / 3.0)
    else:
        quadrupole_hz = target_line_hz / ((2.0 / 3.0) * args.eta)
    site = QuadrupolarSite(
        spin=1,
        isotope="14N",
        quadrupole_frequency_hz=quadrupole_hz,
        eta=args.eta,
    )
    line_frequency_hz = diagonalize_site(site).transition(args.transition).frequency_hz
    nutation_hz = _paper_nutation_hz(args.powder_90_us)
    orientations = powder_average_grid(args.n_theta, args.n_phi)

    offset_dense = np.linspace(-3.0e3, 3.0e3, 601)
    offset_sparse = np.linspace(-3.0e3, 3.0e3, args.simulation_points)
    two_tau_offset = 1.6e-3
    width_offset = 12.0e-6
    phi_offset = 2.0 * np.pi * nutation_hz * width_offset
    offset_theory = sorc_powder_theory_signal(
        offset_dense,
        0.5 * two_tau_offset,
        phi_offset,
        normalize=True,
    )

    two_tau_dense = np.linspace(0.5e-3, 2.8e-3, 501)
    two_tau_sparse = np.linspace(0.5e-3, 2.8e-3, args.simulation_points)
    spacing_offset_hz = -2.05e3
    width_spacing = 20.0e-6
    phi_spacing = 2.0 * np.pi * nutation_hz * width_spacing
    spacing_theory = np.array(
        [
            sorc_powder_theory_signal(
                [spacing_offset_hz],
                0.5 * two_tau,
                phi_spacing,
            )[0]
            for two_tau in two_tau_dense
        ],
        dtype=np.float64,
    )
    spacing_theory = _normalize(spacing_theory)

    widths_dense = np.linspace(0.0, 120.0e-6, 501)
    widths_sparse = np.linspace(0.0, 120.0e-6, args.simulation_points)
    two_tau_width = 1.71e-3
    width_offset_hz = -2.05e3
    width_theory = np.array(
        [
            sorc_powder_theory_signal(
                [width_offset_hz],
                0.5 * two_tau_width,
                2.0 * np.pi * nutation_hz * width,
            )[0]
            for width in widths_dense
        ],
        dtype=np.float64,
    )
    width_theory = _normalize(width_theory)
    fid_theory = fid_powder_theory_signal(
        2.0 * np.pi * nutation_hz * widths_dense,
        normalize=True,
    )
    fid_abs_theory = fid_powder_theory_signal(
        2.0 * np.pi * nutation_hz * widths_dense,
        normalize=True,
        absolute=True,
    )

    offset_sim = spacing_sim = width_sim = None
    if not args.no_simulation:
        offset_sim = _simulate_sweep(
            site=site,
            transition=args.transition,
            line_frequency_hz=line_frequency_hz,
            offsets_hz=offset_sparse,
            two_tau_seconds=np.full_like(offset_sparse, two_tau_offset),
            pulse_width_seconds=np.full_like(offset_sparse, width_offset),
            nutation_hz=nutation_hz,
            num_pulses=args.num_pulses,
            orientations=orientations,
        )
        spacing_sim = _simulate_sweep(
            site=site,
            transition=args.transition,
            line_frequency_hz=line_frequency_hz,
            offsets_hz=np.full_like(two_tau_sparse, spacing_offset_hz),
            two_tau_seconds=two_tau_sparse,
            pulse_width_seconds=np.full_like(two_tau_sparse, width_spacing),
            nutation_hz=nutation_hz,
            num_pulses=args.num_pulses,
            orientations=orientations,
        )
        width_sim = _simulate_sweep(
            site=site,
            transition=args.transition,
            line_frequency_hz=line_frequency_hz,
            offsets_hz=np.full_like(widths_sparse, width_offset_hz),
            two_tau_seconds=np.full_like(widths_sparse, two_tau_width),
            pulse_width_seconds=widths_sparse,
            nutation_hz=nutation_hz,
            num_pulses=args.num_pulses,
            orientations=orientations,
        )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)

    axes[0].plot(offset_dense / 1e3, offset_theory, color="C0", label="SORC theory")
    if offset_sim is not None:
        axes[0].plot(
            offset_sparse / 1e3,
            _normalize(offset_sim),
            "o",
            ms=3,
            color="C1",
            label="density matrix",
        )
    axes[0].set_xlabel("frequency offset delta f (kHz)")
    axes[0].set_ylabel("normalized signal magnitude")
    axes[0].set_title("Offset Sweep, 2 tau = 1.6 ms")
    axes[0].legend()

    axes[1].plot(two_tau_dense * 1e3, spacing_theory, color="C0")
    if spacing_sim is not None:
        axes[1].plot(two_tau_sparse * 1e3, _normalize(spacing_sim), "o", ms=3, color="C1")
    axes[1].set_xlabel("pulse separation 2 tau (ms)")
    axes[1].set_ylabel("normalized signal magnitude")
    axes[1].set_title("Spacing Sweep, delta f = -2.05 kHz")

    axes[2].plot(widths_dense * 1e6, width_theory, color="C0", label="SORC theory")
    axes[2].plot(widths_dense * 1e6, fid_theory, "--", color="C2", label="FID theory")
    axes[2].plot(
        widths_dense * 1e6,
        fid_abs_theory,
        ":",
        color="C3",
        label="abs(FID theory)",
    )
    if width_sim is not None:
        axes[2].plot(widths_sparse * 1e6, _normalize(width_sim), "o", ms=3, color="C1")
    axes[2].set_xlabel("pulse width tw (us)")
    axes[2].set_ylabel("normalized signal")
    axes[2].set_title("Pulse-Width Sweep")
    axes[2].legend()

    fig.suptitle(
        "14N SORC response after Konnai et al. 2008 "
        f"({line_frequency_hz / 1e6:.4f} MHz line)"
    )

    print(f"14N line: {line_frequency_hz / 1e6:.5f} MHz")
    print(f"effective nutation: {nutation_hz / 1e3:.2f} kHz")
    print(f"SORC offset minima period: {1.0 / two_tau_offset / 1e3:.3f} kHz")
    print(f"SORC spacing minima period: {1.0 / abs(spacing_offset_hz) * 1e3:.3f} ms")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
