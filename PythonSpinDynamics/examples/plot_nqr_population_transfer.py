"""Plot a diagnostic two-frequency NQR population-transfer map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    SelectivePulse,
    diagonalize_site,
    powder_average_grid,
    simulate_population_transfer,
    slse_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eta",
        type=float,
        default=0.3,
        help="EFG asymmetry parameter for the spin-1 site.",
    )
    parser.add_argument(
        "--quadrupole-khz",
        type=float,
        default=900.0,
        help="Quadrupole frequency parameter in kHz.",
    )
    parser.add_argument(
        "--nutation-khz",
        type=float,
        default=10.0,
        help="Nominal RF nutation frequency in kHz.",
    )
    parser.add_argument(
        "--perturb-angle",
        type=float,
        default=300.0,
        help="Nominal perturbation pulse angle in degrees.",
    )
    parser.add_argument(
        "--detect-angle",
        type=float,
        default=120.0,
        help="Nominal SLSE detection pulse angle in degrees.",
    )
    parser.add_argument("--n-theta", type=int, default=10, help="Powder polar samples.")
    parser.add_argument("--n-phi", type=int, default=20, help="Powder azimuthal samples.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    site = QuadrupolarSite(
        spin=1,
        isotope="14N",
        quadrupole_frequency_hz=args.quadrupole_khz * 1e3,
        eta=args.eta,
    )
    eigensystem = diagonalize_site(site)
    transitions = tuple(sorted(eigensystem.transitions, key=lambda item: item.label))
    labels = [transition.label for transition in transitions]
    frequencies_khz = np.array([transition.frequency_hz / 1e3 for transition in transitions])
    orientations = powder_average_grid(args.n_theta, args.n_phi)
    nutation_hz = args.nutation_khz * 1e3
    perturb_duration = np.deg2rad(args.perturb_angle) / (2.0 * np.pi * nutation_hz)
    detect_duration = np.deg2rad(args.detect_angle) / (2.0 * np.pi * nutation_hz)

    matrix = np.empty((len(labels), len(labels)), dtype=np.float64)
    for row, perturb_label in enumerate(labels):
        perturbation = SelectivePulse(
            perturb_label,
            duration_seconds=perturb_duration,
            nutation_hz=nutation_hz,
        )
        for col, detect_label in enumerate(labels):
            detection = slse_sequence(
                detect_label,
                pulse_duration_seconds=detect_duration,
                nutation_hz=nutation_hz,
                echo_spacing_seconds=1e-3,
                num_echoes=1,
            )
            result = simulate_population_transfer(
                site,
                perturbation,
                detection,
                orientations=orientations,
            )
            matrix[row, col] = np.real(result.normalized_difference[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    vmax = max(0.1, float(np.max(np.abs(matrix))))
    image = axes[0].imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_xticks(np.arange(len(labels)), labels)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].set_xlabel("Detection transition")
    axes[0].set_ylabel("Perturbation transition")
    axes[0].set_title("Normalized Difference: S / S0 - 1")
    for row in range(len(labels)):
        for col in range(len(labels)):
            axes[0].text(
                col,
                row,
                f"{matrix[row, col]:+.2f}",
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(image, ax=axes[0], shrink=0.9)

    axes[1].bar(labels, frequencies_khz, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1].set_xlabel("Transition")
    axes[1].set_ylabel("Frequency (kHz)")
    axes[1].set_title("Spin-1 NQR Lines")
    for idx, value in enumerate(frequencies_khz):
        axes[1].text(idx, value, f"{value:.1f}", ha="center", va="bottom")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
