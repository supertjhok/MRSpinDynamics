"""Plot history-dependent NaNO2 dynamics during an up-and-down B0 sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    FieldDependentRelaxationModel,
    QuadrupolarSite,
    nqr_hamiltonian,
    simulate_field_sweep_history,
)


ROOT = Path(__file__).resolve().parents[2]
NANO2_SUMMARY = ROOT / "QuadrupolarDFT" / "results" / "nano2_efg_summary.csv"


def _nano2_site() -> tuple[QuadrupolarSite, str]:
    with NANO2_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == "nano2_icsd82857_efg" and row["isotope"] == "14N":
                cq_hz = float(row["mean_abs_cq_mhz"]) * 1.0e6
                eta = float(row["mean_eta"])
                return (
                    QuadrupolarSite(
                        spin=1.0,
                        isotope="14N",
                        label="NaNO2 N",
                        quadrupole_frequency_hz=0.75 * cq_hz,
                        eta=eta,
                        gamma_hz_per_t=3.0766e6,
                    ),
                    f"ABINIT ICSD 82857: C_Q={cq_hz / 1e6:.4f} MHz, eta={eta:.4f}",
                )
    raise RuntimeError(f"could not find NaNO2 parameters in {NANO2_SUMMARY}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-per-leg", type=int, default=101)
    parser.add_argument("--minimum-ratio", type=float, default=0.02)
    parser.add_argument("--maximum-ratio", type=float, default=3.0)
    parser.add_argument("--fast-duration-ms", type=float, default=0.2)
    parser.add_argument("--slow-duration-ms", type=float, default=100.0)
    parser.add_argument("--thermalization-ms", type=float, default=20.0)
    parser.add_argument("--dephasing-ms", type=float, default=2.0)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _prepared_ground_state(site: QuadrupolarSite, field: np.ndarray) -> np.ndarray:
    _, vectors = np.linalg.eigh(nqr_hamiltonian(site, field))
    ground = vectors[:, 0]
    return np.outer(ground, ground.conj())


def main() -> None:
    args = _parse_args()
    if args.points_per_leg < 3 or args.substeps <= 0:
        raise ValueError("points-per-leg must be at least three and substeps positive")
    if not 0.0 < args.minimum_ratio < args.maximum_ratio:
        raise ValueError("ratios must satisfy 0 < minimum < maximum")
    if min(
        args.fast_duration_ms,
        args.slow_duration_ms,
        args.thermalization_ms,
        args.dephasing_ms,
    ) <= 0.0:
        raise ValueError("durations and relaxation times must be positive")

    site, provenance = _nano2_site()
    direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    up = np.linspace(args.minimum_ratio, args.maximum_ratio, args.points_per_leg)
    ratios = np.concatenate([up, up[-2::-1]])
    fields_tesla = ratios * site.quadrupole_frequency_hz / abs(site.gamma_hz_per_t)
    fields = fields_tesla[:, np.newaxis] * direction[np.newaxis, :]
    initial = _prepared_ground_state(site, fields[0])
    relaxation = FieldDependentRelaxationModel(
        temperature_kelvin=300.0,
        thermalization_time_seconds=args.thermalization_ms * 1.0e-3,
        dephasing_time_seconds=args.dephasing_ms * 1.0e-3,
    )

    def run(duration_ms: float):
        return simulate_field_sweep_history(
            site,
            np.linspace(0.0, duration_ms * 1.0e-3, ratios.size),
            fields,
            relaxation=relaxation,
            initial_density=initial,
            substeps_per_interval=args.substeps,
        )

    fast = run(args.fast_duration_ms)
    slow = run(args.slow_duration_ms)
    turn = args.points_per_leg - 1

    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), constrained_layout=True)
    normalized_time = np.linspace(0.0, 1.0, ratios.size)
    axes[0, 0].plot(normalized_time, ratios, color="black")
    axes[0, 0].axvline(normalized_time[turn], color="0.5", linestyle="--")
    axes[0, 0].set_xlabel("Normalized sweep time")
    axes[0, 0].set_ylabel(r"$\nu_L/\nu_Q$")
    axes[0, 0].set_title("Applied triangular field history")

    for result, label, color in (
        (fast, f"fast ({args.fast_duration_ms:g} ms)", "tab:blue"),
        (slow, f"slow ({args.slow_duration_ms:g} ms)", "tab:orange"),
    ):
        axes[0, 1].plot(
            ratios[: turn + 1],
            result.equilibrium_deviation_norm[: turn + 1],
            color=color,
            label=f"{label}, up",
        )
        axes[0, 1].plot(
            ratios[turn:],
            result.equilibrium_deviation_norm[turn:],
            color=color,
            linestyle="--",
            label=f"{label}, down",
        )
        axes[1, 0].plot(
            ratios,
            result.instantaneous_populations[:, 0],
            color=color,
            label=label,
        )
        spin_along_field = result.spin_expectation_pas @ direction
        axes[1, 1].plot(normalized_time, spin_along_field, color=color, label=label)

    axes[0, 1].set_xlabel(r"$\nu_L/\nu_Q$")
    axes[0, 1].set_ylabel(r"$\|\rho-\rho_{eq}\|_F$")
    axes[0, 1].set_title("History-dependent lag and hysteresis")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_xlabel(r"$\nu_L/\nu_Q$")
    axes[1, 0].set_ylabel("Instantaneous ground-state population")
    axes[1, 0].set_title("Instantaneous eigenstate population retains history")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_xlabel("Normalized sweep time")
    axes[1, 1].set_ylabel(r"$\langle I_{B_0}\rangle$")
    axes[1, 1].set_title("Spin polarization retains sweep history")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        r"NaNO$_2$ $^{14}$N prepared-state field history: coherent passage plus relaxation"
        + "\n"
        + provenance
        + f"; T={relaxation.temperature_kelvin:g} K, "
        + f"T1={args.thermalization_ms:g} ms, Tphi={args.dephasing_ms:g} ms"
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved {args.output}")
        print(
            "final equilibrium deviation: "
            f"fast={fast.equilibrium_deviation_norm[-1]:.6g}, "
            f"slow={slow.equilibrium_deviation_norm[-1]:.6g}"
        )


if __name__ == "__main__":
    main()
