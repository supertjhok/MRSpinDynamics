"""Plot a simple ESR hyperfine doublet from one coupled spin-1/2 nucleus."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    BOHR_MAGNETON_HZ_PER_T,
    ESRSpinSystem,
    NuclearSite,
    diagonalize_hyperfine_system,
    electron_nuclear_system,
    resonance_field_tesla,
    simulate_hyperfine_field_sweep,
)


def _normalized(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return values / scale if scale > 0 else values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g", type=float, default=2.00231930436256)
    parser.add_argument(
        "--hyperfine-mhz",
        type=float,
        default=20.0,
        help="Isotropic hyperfine coupling A in MHz.",
    )
    parser.add_argument("--microwave-ghz", type=float, default=9.5)
    parser.add_argument(
        "--nuclear-gamma-mhz-per-t",
        type=float,
        default=42.57747892,
        help="Nuclear gamma / 2pi in MHz/T; use 0 for the pure high-field doublet.",
    )
    parser.add_argument("--broadening-mhz", type=float, default=0.5)
    parser.add_argument("--points", type=int, default=801)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    system = electron_nuclear_system(
        [args.hyperfine_mhz * 1e6],
        nuclei=[
            NuclearSite(
                "H1",
                isotope="1H",
                gamma_hz_per_t=args.nuclear_gamma_mhz_per_t * 1e6,
            )
        ],
        g_tensor=args.g,
    )
    microwave_hz = args.microwave_ghz * 1e9

    # In the high-field/secular picture, one spin-1/2 nucleus splits the ESR
    # line into two components separated by A in frequency, or approximately
    # A / ((mu_B/h) g) in field. The dense Hamiltonian below also includes the
    # small nonsecular SxIx + SyIy terms, so it remains valid as B0 is lowered.
    result = simulate_hyperfine_field_sweep(
        system,
        microwave_hz,
        broadening_hz=args.broadening_mhz * 1e6,
        points=args.points,
    )

    # For the branch plot, diagonalize a smaller set of field values and keep
    # transitions with appreciable x-polarized microwave intensity.
    fields = np.linspace(result.fields_tesla[0], result.fields_tesla[-1], 101)
    branch_fields: list[float] = []
    branch_frequencies: list[float] = []
    branch_intensities: list[float] = []
    for field in fields:
        eigensystem = diagonalize_hyperfine_system(system, [0.0, 0.0, field])
        for transition in eigensystem.transitions:
            intensity = abs(transition.dipole_vector[0]) ** 2
            if intensity < 1e-5:
                continue
            branch_fields.append(field)
            branch_frequencies.append(transition.frequency_hz)
            branch_intensities.append(float(intensity))

    center_field = resonance_field_tesla(ESRSpinSystem(g_tensor=args.g), microwave_hz)
    high_field_split_mt = (
        args.hyperfine_mhz * 1e6 / (BOHR_MAGNETON_HZ_PER_T * args.g) * 1e3
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    axes[0].plot(
        1e3 * result.fields_tesla,
        _normalized(result.spectrum),
        color="tab:blue",
    )
    axes[0].axvline(1e3 * center_field, color="0.4", linestyle="--", label="no A")
    axes[0].set_xlabel("Static field B0 (mT)")
    axes[0].set_ylabel("Normalized absorption")
    axes[0].set_title("Field-Swept Hyperfine Doublet")
    axes[0].legend()

    scatter = axes[1].scatter(
        1e3 * np.asarray(branch_fields),
        np.asarray(branch_frequencies) / 1e9,
        c=np.asarray(branch_intensities),
        s=18,
        cmap="viridis",
    )
    axes[1].axhline(args.microwave_ghz, color="tab:red", linestyle="--")
    axes[1].set_xlabel("Static field B0 (mT)")
    axes[1].set_ylabel("Transition frequency (GHz)")
    axes[1].set_title("ESR-Active Transition Branches")
    fig.colorbar(scatter, ax=axes[1], label="x-polarized intensity")

    axes[0].text(
        0.02,
        0.95,
        f"high-field split ~ {high_field_split_mt:.3f} mT",
        transform=axes[0].transAxes,
        va="top",
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
