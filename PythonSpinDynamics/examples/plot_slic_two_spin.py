"""Plot a two-spin SLIC spectrum around the J-coupling resonance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling import (  # noqa: E402
    coupled_spin_system,
    simulate_slic_spectrum,
    two_spin_slic_transfer_time,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j-hz", type=float, default=7.0, help="Two-spin J coupling in Hz.")
    parser.add_argument(
        "--delta-hz",
        type=float,
        default=0.7,
        help="Resonance-frequency difference between the two spins in Hz.",
    )
    parser.add_argument("--points", type=int, default=241, help="Nutation-frequency samples.")
    parser.add_argument("--min-nutation", type=float, default=1.0, help="Minimum nutation frequency.")
    parser.add_argument("--max-nutation", type=float, default=14.0, help="Maximum nutation frequency.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    system = coupled_spin_system(
        offsets_hz=[-0.5 * args.delta_hz, 0.5 * args.delta_hz],
        couplings_hz=[[0.0, args.j_hz], [args.j_hz, 0.0]],
        labels=["A", "B"],
    )
    frequencies = np.linspace(args.min_nutation, args.max_nutation, args.points)
    optimal_time = two_spin_slic_transfer_time(args.delta_hz)
    spectrum = simulate_slic_spectrum(
        system,
        frequencies,
        spin_lock_time=optimal_time,
    )

    time_factors = np.linspace(0.25, 1.75, 41)
    dip_map = np.vstack(
        [
            simulate_slic_spectrum(
                system,
                frequencies,
                spin_lock_time=factor * optimal_time,
            ).dip
            for factor in time_factors
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    axes[0, 0].plot(frequencies, spectrum.normalized_mx, label="remaining Mx")
    axes[0, 0].axvline(args.j_hz, color="tab:red", linestyle="--", label="J")
    axes[0, 0].set_title("Two-Spin SLIC Response")
    axes[0, 0].set_xlabel("Spin-lock nutation frequency (Hz)")
    axes[0, 0].set_ylabel("Normalized Mx")
    axes[0, 0].legend()

    axes[0, 1].plot(frequencies, spectrum.dip, color="tab:orange")
    axes[0, 1].axvline(args.j_hz, color="tab:red", linestyle="--")
    axes[0, 1].set_title(
        f"Deepest Dip at {spectrum.strongest_dip_frequency_hz:.3g} Hz"
    )
    axes[0, 1].set_xlabel("Spin-lock nutation frequency (Hz)")
    axes[0, 1].set_ylabel("Signal loss")

    image = axes[1, 0].imshow(
        dip_map,
        aspect="auto",
        origin="lower",
        extent=[frequencies[0], frequencies[-1], time_factors[0], time_factors[-1]],
    )
    fig.colorbar(image, ax=axes[1, 0], label="Signal loss")
    axes[1, 0].axvline(args.j_hz, color="white", linestyle="--", alpha=0.85)
    axes[1, 0].axhline(1.0, color="white", linestyle=":", alpha=0.85)
    axes[1, 0].set_title("Dip vs Spin-Lock Duration")
    axes[1, 0].set_xlabel("Spin-lock nutation frequency (Hz)")
    axes[1, 0].set_ylabel("Duration / ideal transfer time")

    axes[1, 1].axis("off")
    summary = (
        f"J = {args.j_hz:g} Hz\n"
        f"delta nu = {args.delta_hz:g} Hz\n"
        f"ideal transfer time = {optimal_time:.4g} s\n"
        f"strongest dip = {spectrum.dip.max():.4g}\n"
        f"dip frequency = {spectrum.strongest_dip_frequency_hz:.4g} Hz"
    )
    axes[1, 1].text(0.0, 0.95, summary, va="top", family="monospace", fontsize=12)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
