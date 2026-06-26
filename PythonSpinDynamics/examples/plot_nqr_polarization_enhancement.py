"""Adiabatic polarization-enhanced NQR in a Halbach fringe field.

This example follows the instrument-level picture of Glickstein and Mandal
(Rev. Sci. Instrum. 89, 093106, 2018): a sample is pre-polarized in a finite
Halbach magnet, then translated through the falling fringe field. Whenever the
proton Larmor frequency ``gamma_H * B0`` crosses a 14N NQR transition, proton
polarization can transfer to the quadrupolar reservoir if the crossing is slow
enough to be adiabatic.

The defaults use melamine-like 14N lines and a four-square-rod Halbach magnet.
Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()

DEFAULT_MELAMINE_CIF = (
    Path(__file__).resolve().parents[2]
    / "QuadrupolarDFT"
    / "structures"
    / "Melamine"
    / "237082.cif"
)


@dataclass(frozen=True)
class EnhancementSweep:
    velocities: np.ndarray
    reference_velocity: float
    enhancements: np.ndarray
    prepolarization_times: np.ndarray
    prepolarization_enhancements: np.ndarray
    sample_lengths: np.ndarray
    length_enhancements: np.ndarray
    reference: object
    coupling_estimate: object | None
    coupling_hz: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate adiabatic polarization-enhanced NQR as a sample moves "
            "through a finite Halbach magnet fringe field."
        )
    )
    parser.add_argument("--velocity", type=float, default=16.67,
                        help="Reference sample speed through crossings (cm/s).")
    parser.add_argument("--speed-min", type=float, default=2.0,
                        help="Minimum speed for sweep (cm/s).")
    parser.add_argument("--speed-max", type=float, default=50.0,
                        help="Maximum speed for sweep (cm/s).")
    parser.add_argument("--speed-points", type=int, default=24,
                        help="Number of velocity sweep points.")
    parser.add_argument("--prepolarization", type=float, default=100.0,
                        help="Reference proton pre-polarization time (s).")
    parser.add_argument("--t1h", type=float, default=48.6,
                        help="Proton T1 in the pre-polarizing magnet (s).")
    parser.add_argument("--t1n", type=float, default=5.0,
                        help="14N longitudinal retention time after transfer (s).")
    parser.add_argument("--proton-linewidth-khz", type=float, default=80.0,
                        help="Effective 1H linewidth in the crossing region (kHz).")
    parser.add_argument("--nh-coupling-hz", type=float,
                        help="Manual effective 1H-14N coupling rate (Hz).")
    parser.add_argument("--cif", type=Path, default=DEFAULT_MELAMINE_CIF,
                        help="CIF file used to estimate 1H-14N coupling.")
    parser.add_argument("--coupling-target", default="N101",
                        help="Quadrupolar atom label in the CIF.")
    parser.add_argument("--coupling-radius", type=float, default=3.0,
                        help="Nearby-proton search radius in Angstrom.")
    parser.add_argument("--sample-length", type=float, default=20.0,
                        help="Sample length along the motion axis (mm).")
    parser.add_argument("--sample-diameter", type=float, default=8.0,
                        help="Sample diameter (mm).")
    parser.add_argument("--axial-points", type=int, default=7,
                        help="Axial quadrature points for sample averaging.")
    parser.add_argument("--center-radius", type=float, default=25.4,
                        help="Halbach rod-center radius (mm).")
    parser.add_argument("--rod-width", type=float, default=25.4,
                        help="Square rod width (mm).")
    parser.add_argument("--magnet-length", type=float, default=101.6,
                        help="Magnet length along the transport axis (mm).")
    parser.add_argument("--remanence", type=float, default=1.15,
                        help="Rod remanence Br (T).")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Transport start coordinate z (mm).")
    parser.add_argument("--stop", type=float, default=100.0,
                        help="Transport stop coordinate z (mm).")
    parser.add_argument("--path-points", type=int, default=301,
                        help="Path samples for crossing detection.")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show the plot.")
    return parser.parse_args()


def _build_objects(args: argparse.Namespace, *, velocity_m_s: float,
                   coupling_hz: float,
                   sample_length_m: float | None = None):
    from spin_dynamics.nqr import (
        CylindricalSampleGeometry,
        HalbachPrepolarizationMagnet,
        LinearTransportMotion,
        PolarizationEnhancedNQRSample,
    )

    sample = PolarizationEnhancedNQRSample(
        name="melamine-like",
        line_labels=("nu+", "nu-", "nu0"),
        line_frequencies_hz=(2.766e6, 2.034e6, 0.732e6),
        protons_per_molecule=6.0,
        nitrogens_per_molecule=6.0,
        proton_t1_seconds=args.t1h,
        nitrogen_t1_seconds=args.t1n,
        proton_linewidth_hz=args.proton_linewidth_khz * 1e3,
        proton_nitrogen_coupling_hz=coupling_hz,
    )
    geometry = CylindricalSampleGeometry(
        length=(
            args.sample_length * 1e-3
            if sample_length_m is None
            else sample_length_m
        ),
        diameter=args.sample_diameter * 1e-3,
        axial_points=args.axial_points,
        radial_rings=0,
    )
    magnet = HalbachPrepolarizationMagnet(
        center_radius=args.center_radius * 1e-3,
        length=args.magnet_length * 1e-3,
        remanence=args.remanence,
        rod_shape="square",
        rod_width=args.rod_width * 1e-3,
        rod_radius=0.5 * args.rod_width * 1e-3,
        n_cross=5,
        n_length=21,
    )
    motion = LinearTransportMotion(
        args.start * 1e-3,
        args.stop * 1e-3,
        velocity=velocity_m_s,
        axis="z",
    )
    return magnet, sample, geometry, motion


def _run_simulation(args: argparse.Namespace) -> EnhancementSweep:
    from spin_dynamics.nqr import simulate_adiabatic_polarization_transfer

    coupling_hz, coupling_estimate = _resolve_coupling(args)
    reference_velocity = args.velocity * 1e-2
    magnet, sample, geometry, motion = _build_objects(
        args,
        velocity_m_s=reference_velocity,
        coupling_hz=coupling_hz,
    )
    reference = simulate_adiabatic_polarization_transfer(
        magnet,
        sample,
        geometry,
        motion,
        prepolarization_time_seconds=args.prepolarization,
        path_points=args.path_points,
    )

    speeds = np.linspace(args.speed_min, args.speed_max, args.speed_points) * 1e-2
    speed_results = []
    for speed in speeds:
        magnet, sample, geometry, motion = _build_objects(
            args,
            velocity_m_s=speed,
            coupling_hz=coupling_hz,
        )
        speed_results.append(
            simulate_adiabatic_polarization_transfer(
                magnet,
                sample,
                geometry,
                motion,
                prepolarization_time_seconds=args.prepolarization,
                path_points=args.path_points,
            ).practical_enhancement
        )

    magnet, sample, geometry, motion = _build_objects(
        args,
        velocity_m_s=reference_velocity,
        coupling_hz=coupling_hz,
    )
    tpol = np.linspace(0.0, max(3.0 * args.t1h, args.prepolarization), 80)
    prepol_results = []
    for duration in tpol:
        prepol_results.append(
            simulate_adiabatic_polarization_transfer(
                magnet,
                sample,
                geometry,
                motion,
                prepolarization_time_seconds=float(duration),
                path_points=args.path_points,
            ).practical_enhancement
        )

    lengths = np.linspace(2.0e-3, max(2.0e-3, 4.0 * args.sample_length * 1e-3), 16)
    length_results = []
    for length in lengths:
        magnet, sample, geometry, motion = _build_objects(
            args,
            velocity_m_s=reference_velocity,
            coupling_hz=coupling_hz,
            sample_length_m=float(length),
        )
        length_results.append(
            simulate_adiabatic_polarization_transfer(
                magnet,
                sample,
                geometry,
                motion,
                prepolarization_time_seconds=args.prepolarization,
                path_points=args.path_points,
            ).practical_enhancement
        )

    return EnhancementSweep(
        velocities=speeds,
        reference_velocity=reference_velocity,
        enhancements=np.asarray(speed_results),
        prepolarization_times=tpol,
        prepolarization_enhancements=np.asarray(prepol_results),
        sample_lengths=lengths,
        length_enhancements=np.asarray(length_results),
        reference=reference,
        coupling_estimate=coupling_estimate,
        coupling_hz=coupling_hz,
    )


def _resolve_coupling(args: argparse.Namespace):
    if args.nh_coupling_hz is not None:
        return float(args.nh_coupling_hz), None
    if args.cif and args.cif.exists():
        from spin_dynamics.nqr import estimate_proton_dipolar_couplings_from_cif

        estimate = estimate_proton_dipolar_couplings_from_cif(
            args.cif,
            args.coupling_target,
            proton_radius_angstrom=args.coupling_radius,
        )
        if estimate.effective_rms_hz > 0.0:
            return estimate.effective_rms_hz, estimate
    return 1000.0, None


def _plot(plt, sweep: EnhancementSweep, args: argparse.Namespace):
    ref = sweep.reference
    labels = ref.line_labels
    colors = ("tab:blue", "tab:orange", "tab:green")

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), constrained_layout=True)

    ax = axes[0, 0]
    z_mm = ref.b0_profile_positions * 1e3
    ax.plot(z_mm, ref.b0_profile_tesla, color="k", label="|B0|")
    for label, crossing, field, color in zip(
        labels, ref.crossing_positions, ref.crossing_fields_tesla, colors
    ):
        ax.axhline(field, color=color, linestyle=":", linewidth=1.0)
        ax.axvline(crossing * 1e3, color=color, linestyle="--", linewidth=1.0)
        ax.text(crossing * 1e3, field, f" {label}", color=color, va="bottom")
    ax.set_xlabel("transport coordinate z (mm)")
    ax.set_ylabel("|B0| (T)")
    ax.set_title("Halbach fringe field and level crossings")

    ax = axes[0, 1]
    gradient = np.abs(ref.b0_profile_gradient_t_per_m)
    ax.plot(z_mm, gradient, color="0.2", label="|dB0/dz|")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("gradient (T/m)")
    ax.set_title("crossing gradients set adiabatic speed")
    ax2 = ax.twinx()
    vmax = (
            args.proton_linewidth_khz
            * 1e3
            * sweep.coupling_hz
        / (42.57747892e6 * np.maximum(gradient, 1.0e-12))
    )
    ax2.plot(z_mm, vmax * 1e2, color="tab:green", alpha=0.6)
    ax2.set_ylabel("illustrative vmax (cm/s)")

    ax = axes[0, 2]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, ref.ideal_enhancement, width, color="0.75", label="ideal")
    ax.bar(
        x + width / 2,
        ref.practical_enhancement,
        width,
        color=colors[: len(labels)],
        label="practical",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("signal enhancement factor")
    ax.set_title("ideal vs transport-limited enhancement")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(
            sweep.velocities * 1e2,
            sweep.enhancements[:, idx],
            color=color,
            label=label,
        )
    ax.axvline(sweep.reference_velocity * 1e2, color="0.3", linestyle="--",
               linewidth=1.0)
    ax.set_xlabel("sample speed (cm/s)")
    ax.set_ylabel("enhancement factor")
    ax.set_title("speed tunes adiabatic transfer")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(
            sweep.prepolarization_times,
            sweep.prepolarization_enhancements[:, idx],
            color=color,
            label=label,
        )
    ax.set_xlabel("pre-polarization time (s)")
    ax.set_ylabel("enhancement factor")
    ax.set_title("proton T1 build-up before transport")

    ax = axes[1, 2]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(
            sweep.sample_lengths * 1e3,
            sweep.length_enhancements[:, idx],
            color=color,
            label=label,
        )
    ax.set_xlabel("sample length (mm)")
    ax.set_ylabel("enhancement factor")
    ax.set_title("finite sample averages the fringe field")

    fig.suptitle(
        "Adiabatic polarization-enhanced NQR transport simulation",
        fontsize=13,
    )
    return fig


def main() -> None:
    args = _parse_args()
    if args.speed_points < 2:
        raise SystemExit("--speed-points must be at least 2")
    if args.path_points < 3:
        raise SystemExit("--path-points must be at least 3")

    plt = load_matplotlib(headless=bool(args.output))
    sweep = _run_simulation(args)
    ref = sweep.reference

    print("polarization-enhanced NQR transport simulation")
    print(f"  effective 1H-14N coupling: {sweep.coupling_hz:.1f} Hz")
    if sweep.coupling_estimate is not None:
        estimate = sweep.coupling_estimate
        print(
            f"  CIF coupling target {estimate.target_label}: "
            f"{len(estimate.proton_couplings)} protons within "
            f"{args.coupling_radius:.1f} A"
        )
        for item in estimate.proton_couplings[:4]:
            print(
                f"    {item.proton_label:>4s} "
                f"r={item.distance_angstrom:.3f} A, "
                f"d={item.coupling_hz:.1f} Hz"
            )
    print(f"  center-field maximum on path: {np.max(ref.b0_profile_tesla):.3f} T")
    print(f"  travel time: {ref.travel_time_seconds:.3f} s")
    for label, freq, crossing, ratio, eff, practical in zip(
        ref.line_labels,
        ref.line_frequencies_hz,
        ref.crossing_positions,
        ref.adiabatic_ratios,
        ref.transfer_efficiency,
        ref.practical_enhancement,
    ):
        print(
            f"  {label:>3s} {freq / 1e6:.3f} MHz: "
            f"z={crossing * 1e3:5.1f} mm, "
            f"adiabatic ratio={ratio:.2f}, "
            f"transfer={eff:.2f}, EF={practical:.2f}"
        )

    fig = _plot(plt, sweep, args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
