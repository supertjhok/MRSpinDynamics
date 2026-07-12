"""Plot receiver-estimated NaNO2 echoes in four NQR--NMR field regimes.

Zero field uses exact finite-sideband Floquet pulses. Nonzero-field powder
runs use the phase-consistent RWA backend after its orientation-level check
against Floquet. Crystallite waveforms are coherently averaged before finite-
bandwidth receiver filtering and matched estimation. A panel is withheld when
the nested orientation-prefix error exceeds five percent.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    FieldDependentDaviesRelaxationModel,
    FieldDependentNonsecularRelaxationModel,
    QuadrupolarSite,
    b0_b1_powder_average_halton,
    diagonalize_site,
    nqr_hamiltonian,
    powder_carrier_frequency_hz,
    select_powder_frequency_slice,
    simulate_crossover_slse,
    simulate_crossover_slse_powder,
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
    parser.add_argument("--echoes", type=int, default=30)
    parser.add_argument("--echo-spacing-us", type=float, default=2000.0)
    parser.add_argument("--nutation-khz", type=float, default=3.3056)
    parser.add_argument("--pulse-us", type=float, default=50.0)
    parser.add_argument("--nmr-nutation-khz", type=float, default=100.0)
    parser.add_argument("--acquisition-window-us", type=float, default=100.0)
    parser.add_argument("--acquisition-points", type=int, default=129)
    parser.add_argument("--receiver-bandwidth-khz", type=float, default=200.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--cluster-width-khz", type=float, default=20.0)
    parser.add_argument("--powder", action="store_true")
    parser.add_argument("--powder-n-theta", type=int, default=4)
    parser.add_argument("--powder-n-phi", type=int, default=8)
    parser.add_argument("--powder-n-chi", type=int, default=4)
    parser.add_argument("--powder-samples", type=int, default=8192)
    parser.add_argument("--powder-active-samples", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--parallel-backend", choices=("process", "thread"), default="process"
    )
    parser.add_argument("--zero-field-n-theta", type=int, default=8)
    parser.add_argument("--zero-field-n-phi", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.echoes <= 0
        or args.echo_spacing_us <= 0.0
        or args.nutation_khz <= 0.0
        or args.pulse_us <= 0.0
        or args.nmr_nutation_khz <= 0.0
        or args.acquisition_window_us <= 0.0
        or args.receiver_bandwidth_khz <= 0.0
    ):
        raise ValueError("echo count, timing, and nutation must be positive")
    if min(
        args.powder_n_theta,
        args.powder_n_phi,
        args.powder_n_chi,
        args.powder_samples,
        args.powder_active_samples,
        args.workers,
        args.zero_field_n_theta,
        args.zero_field_n_phi,
    ) <= 0:
        raise ValueError("powder grid dimensions must be positive")
    site, provenance = _nano2_site()
    direction_b0 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    direction_b1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    regimes = (
        (0.0, "zero-field NQR"),
        (0.1, "Zeeman-perturbed NQR"),
        (1.0, "intermediate crossover"),
        (10.0, "quadrupolar NMR"),
    )
    common_relaxation = dict(
        spin=site.spin,
        temperature_kelvin=args.temperature_k,
        # Common scale anchored to the 332 ms room-temperature NaNO2 SLSE
        # lifetime; the 4:1 channel ratio and tau_c remain model assumptions.
        magnetic_rate_per_second=3.48,
        efg_rate_per_second=0.87,
        correlation_time_seconds=0.2e-6,
        secular_tolerance_hz=1.0e-3,
    )
    unified = FieldDependentNonsecularRelaxationModel(
        **common_relaxation,
        frequency_cluster_width_hz=args.cluster_width_khz * 1.0e3,
    )
    secular = FieldDependentDaviesRelaxationModel(**common_relaxation)
    nutation = args.nutation_khz * 1.0e3
    zero_field_sequence = dict(
        nutation_hz=nutation,
        excitation_duration_seconds=args.pulse_us * 1.0e-6,
        refocus_duration_seconds=args.pulse_us * 1.0e-6,
        echo_spacing_seconds=args.echo_spacing_us * 1.0e-6,
        num_echoes=args.echoes,
        floquet_sidebands=5,
    )

    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), constrained_layout=True)
    for axis, (ratio, regime) in zip(axes.flat, regimes, strict=True):
        field = ratio * site.quadrupole_frequency_hz / abs(site.gamma_hz_per_t)
        b0 = field * direction_b0
        sequence = dict(zero_field_sequence)
        if field > 0.0:
            nmr_nutation = args.nmr_nutation_khz * 1.0e3
            nominal_90 = 1.0 / (4.0 * np.sqrt(2.0) * nmr_nutation)
            sequence.update(
                nutation_hz=nmr_nutation,
                excitation_duration_seconds=nominal_90,
                refocus_duration_seconds=2.0 * nominal_90,
            )
        if args.powder:
            retained_powder_weight = 1.0
            selected_orientations = None
            selected_carrier = diagonalize_site(site).transition("y").frequency_hz
            if field > 0.0:
                candidate_orientations = b0_b1_powder_average_halton(
                    args.powder_samples
                )
                selected_carrier = powder_carrier_frequency_hz(
                    site,
                    field,
                    candidate_orientations,
                    nutation_hz=sequence["nutation_hz"],
                )
                selection_half_width = (
                    0.5 * args.receiver_bandwidth_khz * 1.0e3
                    + 2.0 * sequence["nutation_hz"]
                )
                selected_orientations, retained_powder_weight = (
                    select_powder_frequency_slice(
                        site,
                        field,
                        candidate_orientations,
                        carrier_frequency_hz=selected_carrier,
                        half_width_hz=selection_half_width,
                    )
                )
                active_target = args.powder_active_samples * (
                    4 if ratio <= 1.0 else 8
                )
                selected_orientations = selected_orientations[
                    :active_target
                ]
                print(
                    f"{regime}: carrier={selected_carrier / 1e6:.6g} MHz, "
                    f"slice weight={retained_powder_weight:.4g}, "
                    f"propagating {len(selected_orientations)} orientations"
                )
            powder = dict(
                n_theta=(
                    args.zero_field_n_theta if field == 0.0 else args.powder_n_theta
                ),
                n_phi=(
                    args.zero_field_n_phi if field == 0.0 else args.powder_n_phi
                ),
                n_chi=args.powder_n_chi,
                rf_frequency_hz=selected_carrier,
                acquisition_duration_seconds=args.acquisition_window_us * 1.0e-6,
                acquisition_points=args.acquisition_points,
                receiver_bandwidth_hz=args.receiver_bandwidth_khz * 1.0e3,
                orientations=selected_orientations,
                pulse_model="rwa" if field > 0.0 else "floquet",
                num_workers=args.workers,
                parallel_backend=args.parallel_backend,
                retain_local_results=False,
            )
            nonsecular_result = simulate_crossover_slse_powder(
                site,
                field,
                relaxation=unified,
                **powder,
                **sequence,
            )
            secular_result = None
            convergence_error = 0.0
            if field > 0.0:
                fine_shape = np.abs(nonsecular_result.matched_echo_amplitudes)
                coarse_shape = np.abs(
                    nonsecular_result.prefix_matched_echo_amplitudes
                )
                convergence_error = float(
                    np.linalg.norm(fine_shape - coarse_shape)
                    / max(np.linalg.norm(fine_shape), 1.0e-30)
                )
        else:
            nonsecular_result = simulate_crossover_slse(
                site,
                b0,
                relaxation=unified,
                b1_direction_pas=direction_b1,
                **sequence,
            )
            secular_result = simulate_crossover_slse(
                site,
                b0,
                relaxation=secular,
                b1_direction_pas=direction_b1,
                **sequence,
            )
        plot_times = nonsecular_result.echo_times_seconds
        if args.powder:
            nonsecular_observable = np.abs(nonsecular_result.matched_echo_amplitudes)
            secular_observable = nonsecular_observable
        else:
            nonsecular_observable = np.abs(nonsecular_result.echo_amplitudes)
            secular_observable = np.abs(secular_result.echo_amplitudes)
        signal_scale = max(float(nonsecular_observable[0]), 1.0e-30)
        nonsecular_signal = nonsecular_observable / signal_scale
        secular_signal = secular_observable / signal_scale
        time_ms = plot_times * 1.0e3
        difference = np.linalg.norm(
            nonsecular_observable - secular_observable
        ) / max(np.linalg.norm(secular_observable), 1.0e-30)
        hamiltonian = nqr_hamiltonian(site, b0)
        gibbs_residual = unified.gibbs_stationarity_error(hamiltonian)
        zero_field_fit = ""
        if field == 0.0 and np.all(nonsecular_observable[1:] > 0.0):
            fit_times = plot_times[1:]
            fit_values = np.log(nonsecular_observable[1:])
            slope, intercept = np.polyfit(fit_times, fit_values, 1)
            fit_residual = float(
                np.sqrt(
                    np.mean((fit_values - (slope * fit_times + intercept)) ** 2)
                )
            )
            if slope < 0.0:
                zero_field_fit = (
                    f"\nmatched exponential fit {(-1.0 / slope) * 1.0e3:.0f} ms"
                    f"; log-RMS {fit_residual:.3f}"
                )
        if args.powder and field > 0.0 and convergence_error > 0.05:
            axis.set_title(
                f"{regime}: $\\nu_L/\\nu_Q={ratio:g}$, $B_0={field:.3g}$ T"
            )
            axis.text(
                0.5,
                0.5,
                "Waveform result withheld\n"
                f"orientation-grid error = {100 * convergence_error:.1f}% (>5%)",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            continue
        axis.plot(
            time_ms,
            nonsecular_signal,
            "o-",
            linewidth=1.6,
            markersize=4.0,
            label="unified GKLS",
        )
        if secular_result is not None:
            axis.plot(
                time_ms,
                secular_signal,
                "--",
                linewidth=1.2,
                label="secular Davies",
            )
        axis.set_title(
            f"{regime}: $\\nu_L/\\nu_Q={ratio:g}$, $B_0={field:.3g}$ T"
        )
        axis.text(
            0.97,
            0.95,
            f"carrier {nonsecular_result.rf_frequency_hz / 1e6:.3f} MHz\n"
            + (
                ""
                if args.powder
                else f"relative model difference {100 * difference:.3g}%\n"
            )
            + f"Gibbs residual {gibbs_residual:.2g} s$^{{-1}}$"
            + (
                f"\nwaveform grid error {100 * convergence_error:.2g}%"
                f"; slice weight {retained_powder_weight:.3g}"
                if args.powder and field > 0.0
                else ""
            )
            + zero_field_fit,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        axis.set_xlabel("Echo time (ms)")
        axis.set_ylabel(
            "Matched echo amplitude / first echo"
            if args.powder
            else "Echo-center magnitude / first echo"
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        r"NaNO$_2$ $^{14}$N SLSE across the NQR--NMR crossover" + "\n"
        + provenance
        + f"; cluster width={args.cluster_width_khz:g} kHz; "
        + (
            "coherent powder waveforms with matched receiver"
            if args.powder
            else "tilted single crystal"
        )
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
