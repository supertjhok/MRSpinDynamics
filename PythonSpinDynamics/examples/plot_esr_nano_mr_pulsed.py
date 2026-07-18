"""Compare pure pulsed ESR with exact defect-spin correlation spectroscopy.

The pure-ESR panel shows inhomogeneous Hahn refocusing.  The nano-MR panels
show a population-detected two-block Hahn surface and its 2-D spectrum for one
resolved proton.  Run ``python examples/plot_esr_nano_mr_pulsed.py --help``.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    ESRSpinSystem,
    flip_angle_duration,
    resonance_field_tesla,
    simulate_hahn_echo,
)
from spin_dynamics.nano_mr import (  # noqa: E402
    ResolvedNucleus,
    ResolvedSpinCluster,
    correlation_spectrum_2d,
    diamond_nv_minus,
    simulate_two_block_correlation,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-mt", type=float, default=20.0)
    parser.add_argument("--nutation-mhz", type=float, default=5.0)
    parser.add_argument("--tau-us", type=float, default=2.0)
    parser.add_argument("--mixing-us", type=float, default=1.0)
    parser.add_argument("--rf-flip-deg", type=float, default=180.0)
    parser.add_argument("--proton-distance-nm", type=float, default=2.0)
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.points < 8:
        parser.error("--points must be at least 8")
    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)

    pure_system = ESRSpinSystem(g_tensor=2.0028)
    carrier_hz = 9.5e9
    nutation_hz = args.nutation_mhz * 1.0e6
    t90 = flip_angle_duration(np.pi / 2.0, nutation_hz)
    t180 = flip_angle_duration(np.pi, nutation_hz)
    echo_times = np.linspace(0.0, 2.0 * args.tau_us * 1.0e-6, 301)
    echo = np.zeros(echo_times.size, dtype=np.complex128)
    for offset_hz in np.linspace(-1.0e6, 1.0e6, 31):
        field = resonance_field_tesla(pure_system, carrier_hz + offset_hz)
        trace = simulate_hahn_echo(
            pure_system,
            [0.0, 0.0, field],
            nutation_hz=nutation_hz,
            excitation_duration_seconds=t90,
            refocus_duration_seconds=t180,
            tau_seconds=args.tau_us * 1.0e-6,
            times_seconds=echo_times,
            rf_frequency_hz=carrier_hz,
        )
        echo += trace.signal
    echo /= 31.0

    sensor = diamond_nv_minus(depth_nm=3.0)
    transverse = args.proton_distance_nm / np.sqrt(2.0)
    cluster = ResolvedSpinCluster(
        sensor,
        (
            ResolvedNucleus.from_isotope(
                "1H",
                [transverse, 0.0, transverse],
            ),
        ),
        sensor_position_lab_nm=[0.0, 0.0, 0.0],
    )
    block_times = np.linspace(0.2e-6, 8.0e-6, args.points)
    correlation = simulate_two_block_correlation(
        cluster,
        [0.0, 0.0, args.field_mt * 1.0e-3],
        block_times,
        block_times,
        mixing_seconds=args.mixing_us * 1.0e-6,
        nuclear_rf_flip_angle_rad=np.deg2rad(args.rf_flip_deg),
    )
    spectrum = correlation_spectrum_2d(correlation)

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    axes[0].plot(1.0e6 * echo_times, np.abs(echo), color="tab:blue")
    axes[0].axvline(args.tau_us, color="tab:orange", linestyle="--", label="echo")
    axes[0].set_title("Pure ESR Hahn Echo")
    axes[0].set_xlabel("Time after pi pulse (us)")
    axes[0].set_ylabel("Ensemble signal magnitude")
    axes[0].legend()

    extent_time = [
        1.0e6 * block_times[0],
        1.0e6 * block_times[-1],
        1.0e6 * block_times[0],
        1.0e6 * block_times[-1],
    ]
    image = axes[1].imshow(
        1.0 - correlation.bright_probability,
        origin="lower",
        aspect="auto",
        extent=extent_time,
        cmap="viridis",
    )
    axes[1].set_title("NV-Proton Two-Block Contrast")
    axes[1].set_xlabel("Second Hahn block (us)")
    axes[1].set_ylabel("First Hahn block (us)")
    fig.colorbar(image, ax=axes[1], label="Dark-state population, 1 - Pbright")

    extent_frequency = [
        1.0e-3 * spectrum.frequencies2_hz[0],
        1.0e-3 * spectrum.frequencies2_hz[-1],
        1.0e-3 * spectrum.frequencies1_hz[0],
        1.0e-3 * spectrum.frequencies1_hz[-1],
    ]
    image = axes[2].imshow(
        spectrum.amplitude,
        origin="lower",
        aspect="auto",
        extent=extent_frequency,
        cmap="magma",
    )
    axes[2].set_title("2-D Correlation Spectrum")
    axes[2].set_xlabel("Second-block frequency (kHz)")
    axes[2].set_ylabel("First-block frequency (kHz)")
    fig.colorbar(image, ax=axes[2], label="Fourier magnitude")

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
