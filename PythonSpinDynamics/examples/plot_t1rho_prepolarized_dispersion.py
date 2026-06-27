"""Plot prepolarized T1rho relaxation dispersion from a BPP spectral density."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.prepolarization import prepolarized_state  # noqa: E402
from spin_dynamics.relaxation import (  # noqa: E402
    BPPRelaxationModel,
    spectral_density_lorentzian,
)


def t1rho_rate(
    spin_lock_angular_rad_per_s: np.ndarray,
    larmor_angular_rad_per_s: float,
    correlation_time_seconds: np.ndarray,
    *,
    coupling_scale_per_second2: float,
    lock_coefficient: float = 1.0,
    omega0_coefficient: float = 2.5,
    two_omega0_coefficient: float = 1.0,
    baseline_rate_per_second: float = 0.0,
) -> np.ndarray:
    """Return a simple on-resonance spin-lock relaxation-dispersion model."""

    omega1 = np.asarray(spin_lock_angular_rad_per_s, dtype=np.float64)
    tau = np.asarray(correlation_time_seconds, dtype=np.float64)
    j_lock = spectral_density_lorentzian(omega1[np.newaxis, :], tau[:, np.newaxis])
    jw = spectral_density_lorentzian(larmor_angular_rad_per_s, tau)[:, np.newaxis]
    j2w = spectral_density_lorentzian(2.0 * larmor_angular_rad_per_s, tau)[
        :, np.newaxis
    ]
    return (
        float(coupling_scale_per_second2)
        * (
            float(lock_coefficient) * j_lock
            + float(omega0_coefficient) * jw
            + float(two_omega0_coefficient) * j2w
        )
        + float(baseline_rate_per_second)
    )


def build_t1rho_dispersion(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Build the prepolarized T1rho dispersion arrays for plotting."""

    temperatures = np.linspace(args.temp_min_k, args.temp_max_k, int(args.temperatures))
    spin_lock_khz = np.geomspace(
        args.spin_lock_min_khz,
        args.spin_lock_max_khz,
        int(args.spin_locks),
    )
    omega0 = 2.0 * np.pi * args.larmor_mhz * 1e6
    omega1 = 2.0 * np.pi * spin_lock_khz * 1e3
    model = BPPRelaxationModel(
        angular_frequency_rad_per_s=omega0,
        tau_ref_seconds=args.tau_ref_ns * 1e-9,
        reference_temperature_kelvin=args.reference_temp_k,
        activation_energy_j_per_mol=args.activation_energy_kj_mol * 1e3,
        coupling_scale_per_second2=args.coupling_scale,
        baseline_r1_per_second=args.baseline_r1,
        baseline_r2_per_second=args.baseline_r2,
    )
    rates = model.rates(temperatures)
    prepared = prepolarized_state(
        polarizing_field_tesla=args.prepolarizing_field_t,
        detection_field_tesla=args.detection_field_t,
        prepolarization_time_seconds=args.prepolarization_time_s,
        t1_seconds=rates.t1_seconds,
    )
    r1rho = t1rho_rate(
        omega1,
        omega0,
        rates.correlation_time_seconds,
        coupling_scale_per_second2=args.coupling_scale,
        lock_coefficient=args.lock_coefficient,
        omega0_coefficient=args.omega0_coefficient,
        two_omega0_coefficient=args.two_omega0_coefficient,
        baseline_rate_per_second=args.baseline_r1rho,
    )
    t1rho = np.divide(
        1.0,
        r1rho,
        out=np.full_like(r1rho, np.inf, dtype=np.float64),
        where=r1rho > 0.0,
    )
    signal = prepared.m0[:, np.newaxis] * np.exp(-args.spin_lock_time_s * r1rho)
    return {
        "temperatures": temperatures,
        "spin_lock_khz": spin_lock_khz,
        "rates_t1": rates.t1_seconds,
        "rates_t2": rates.t2_seconds,
        "tau_c": rates.correlation_time_seconds,
        "prepared_m0": prepared.m0,
        "r1rho": r1rho,
        "t1rho": t1rho,
        "signal": signal,
    }


def _nearest_indices(axis: np.ndarray, values: list[float]) -> list[int]:
    return [int(np.argmin(np.abs(axis - value))) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The T1rho formula is a compact on-resonance dispersion model with "
            "configurable coefficients multiplying J(w1), J(w0), and J(2w0)."
        ),
    )
    parser.add_argument("--temp-min-k", type=float, default=250.0)
    parser.add_argument("--temp-max-k", type=float, default=360.0)
    parser.add_argument("--reference-temp-k", type=float, default=300.0)
    parser.add_argument("--temperatures", type=int, default=121)
    parser.add_argument("--spin-lock-min-khz", type=float, default=0.2)
    parser.add_argument("--spin-lock-max-khz", type=float, default=200.0)
    parser.add_argument("--spin-locks", type=int, default=140)
    parser.add_argument("--spin-lock-time-s", type=float, default=0.02)
    parser.add_argument("--larmor-mhz", type=float, default=20.0)
    parser.add_argument("--tau-ref-ns", type=float, default=800.0)
    parser.add_argument("--activation-energy-kj-mol", type=float, default=16.0)
    parser.add_argument("--coupling-scale", type=float, default=5.0e7)
    parser.add_argument("--baseline-r1", type=float, default=0.0)
    parser.add_argument("--baseline-r2", type=float, default=0.0)
    parser.add_argument("--baseline-r1rho", type=float, default=0.0)
    parser.add_argument("--lock-coefficient", type=float, default=1.0)
    parser.add_argument("--omega0-coefficient", type=float, default=2.5)
    parser.add_argument("--two-omega0-coefficient", type=float, default=1.0)
    parser.add_argument("--prepolarizing-field-t", type=float, default=0.1)
    parser.add_argument("--detection-field-t", type=float, default=0.02)
    parser.add_argument("--prepolarization-time-s", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.spin_lock_min_khz <= 0.0 or args.spin_lock_max_khz <= args.spin_lock_min_khz:
        raise ValueError("spin-lock range must be positive and increasing")

    plt = load_matplotlib(headless=args.output is not None)
    data = build_t1rho_dispersion(args)
    temperatures = data["temperatures"]
    spin_lock_khz = data["spin_lock_khz"]
    selected = _nearest_indices(
        temperatures,
        [
            args.temp_min_k,
            args.reference_temp_k,
            args.temp_max_k,
        ],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    for idx in selected:
        axes[0, 0].semilogx(
            spin_lock_khz,
            data["t1rho"][idx],
            label=f"{temperatures[idx]:.0f} K",
        )
    axes[0, 0].set_xlabel("Spin-lock nutation (kHz)")
    axes[0, 0].set_ylabel("T1rho (s)")
    axes[0, 0].set_title("T1rho Relaxation Dispersion")
    axes[0, 0].legend()

    for idx in selected:
        axes[0, 1].semilogx(
            spin_lock_khz,
            data["signal"][idx],
            label=f"{temperatures[idx]:.0f} K",
        )
    axes[0, 1].set_xlabel("Spin-lock nutation (kHz)")
    axes[0, 1].set_ylabel("Locked signal after spin-lock")
    axes[0, 1].set_title("Prepolarized T1rho Readout")
    axes[0, 1].legend()

    prep_ax = axes[1, 0]
    relax_ax = prep_ax.twinx()
    prep_line = prep_ax.plot(
        temperatures,
        data["prepared_m0"],
        color="tab:green",
        label="prepared M0",
    )
    relax_lines = relax_ax.semilogy(
        temperatures,
        data["rates_t1"],
        color="tab:blue",
        label="T1",
    )
    relax_lines += relax_ax.semilogy(
        temperatures,
        data["rates_t2"],
        color="tab:orange",
        label="T2",
    )
    prep_ax.set_xlabel("Temperature (K)")
    prep_ax.set_ylabel("Prepared M0")
    relax_ax.set_ylabel("Lab-frame relaxation time (s)")
    axes[1, 0].set_title("Prepolarization and Lab-Frame Relaxation")
    all_lines = prep_line + relax_lines
    prep_ax.legend(all_lines, [line.get_label() for line in all_lines])

    image = axes[1, 1].imshow(
        data["signal"],
        origin="lower",
        aspect="auto",
        extent=(
            np.log10(spin_lock_khz[0]),
            np.log10(spin_lock_khz[-1]),
            temperatures[0],
            temperatures[-1],
        ),
    )
    tick_values = np.array([0.2, 1.0, 5.0, 20.0, 100.0, 200.0])
    tick_values = tick_values[
        (tick_values >= spin_lock_khz[0]) & (tick_values <= spin_lock_khz[-1])
    ]
    axes[1, 1].set_xticks(np.log10(tick_values))
    axes[1, 1].set_xticklabels([f"{value:g}" for value in tick_values])
    axes[1, 1].set_xlabel("Spin-lock nutation (kHz)")
    axes[1, 1].set_ylabel("Temperature (K)")
    axes[1, 1].set_title("Signal Map")
    fig.colorbar(image, ax=axes[1, 1], label="locked signal")

    temp_idx = int(np.argmin(np.abs(temperatures - args.reference_temp_k)))
    low = data["t1rho"][temp_idx, 0]
    high = data["t1rho"][temp_idx, -1]
    print("Prepolarized T1rho relaxation dispersion")
    print(f"reference temperature K: {temperatures[temp_idx]:.6g}")
    print(f"prepared M0 at reference: {data['prepared_m0'][temp_idx]:.6g}")
    print(f"T1rho at min spin-lock: {low:.6g} s")
    print(f"T1rho at max spin-lock: {high:.6g} s")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
