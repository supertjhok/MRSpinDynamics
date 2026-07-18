"""Plot a seeded diffusing-liquid dipolar field record and spectrum.

The example places protons in a reflecting liquid nanovolume above a shallow
NV center, propagates Brownian motion plus optional flow, and analyzes the
sensor-axis field. Run
``python examples/plot_nano_mr_diffusing_liquid.py --help``.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nano_mr import (  # noqa: E402
    NuclearBathSpecies,
    field_autocorrelation,
    field_power_spectral_density,
    simulate_diffusing_dipolar_field,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-mt", type=float, default=20.0)
    parser.add_argument("--depth-nm", type=float, default=8.0)
    parser.add_argument("--diffusion-m2-s", type=float, default=2.0e-10)
    parser.add_argument("--drift-mm-s", type=float, default=0.0)
    parser.add_argument("--spins", type=int, default=800)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--dt-ns", type=float, default=20.0)
    parser.add_argument("--motion-substeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.field_mt < 0.0:
        parser.error("--field-mt must be non-negative")
    if args.depth_nm <= 0.0:
        parser.error("--depth-nm must be positive")
    if args.diffusion_m2_s < 0.0:
        parser.error("--diffusion-m2-s must be non-negative")
    if args.spins < 10:
        parser.error("--spins must be at least 10")
    if args.samples < 64:
        parser.error("--samples must be at least 64")
    if args.dt_ns <= 0.0:
        parser.error("--dt-ns must be positive")
    if args.motion_substeps <= 0:
        parser.error("--motion-substeps must be positive")

    bounds_nm = np.array([[-20.0, 20.0], [-20.0, 20.0], [0.0, 40.0]])
    generator = np.random.default_rng(args.seed)
    initial_positions_nm = generator.uniform(
        bounds_nm[:, 0],
        bounds_nm[:, 1],
        size=(args.spins, 3),
    )
    proton = NuclearBathSpecies.from_isotope("1H")
    trajectory = simulate_diffusing_dipolar_field(
        initial_positions_nm,
        proton,
        sensor_position_lab_nm=[0.0, 0.0, -args.depth_nm],
        sensor_axis_lab=[0.0, 0.0, 1.0],
        static_field_lab_tesla=[0.0, 0.0, args.field_mt * 1.0e-3],
        sample_interval_seconds=args.dt_ns * 1.0e-9,
        sample_count=args.samples,
        motion_substeps_per_sample=args.motion_substeps,
        diffusion_coefficient_m2_s=args.diffusion_m2_s,
        drift_velocity_m_s=[args.drift_mm_s * 1.0e-3, 0.0, 0.0],
        bounds_lab_nm=tuple(map(tuple, bounds_nm)),
        boundary="reflect",
        seed=args.seed + 1,
        minimum_distance_nm=args.depth_nm,
    )
    maximum_lag = min(args.samples // 4, 900)
    correlation = field_autocorrelation(
        trajectory,
        max_lag=maximum_lag,
        normalize=True,
    )
    spectrum = field_power_spectral_density(
        trajectory,
        segment_length=min(1024, args.samples),
    )

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), constrained_layout=True)
    initial = trajectory.positions_lab_nm[0]
    final = trajectory.positions_lab_nm[-1]
    stride = max(1, args.spins // 400)
    axes[0, 0].scatter(
        initial[::stride, 0],
        initial[::stride, 2],
        s=7,
        alpha=0.35,
        label="initial",
    )
    axes[0, 0].scatter(
        final[::stride, 0],
        final[::stride, 2],
        s=7,
        alpha=0.35,
        label="final",
    )
    axes[0, 0].scatter(
        [0.0],
        [-args.depth_nm],
        marker="*",
        s=130,
        color="black",
        label="NV",
    )
    axes[0, 0].axhline(0.0, color="0.3", lw=1.0)
    axes[0, 0].set(
        xlabel="x (nm)",
        ylabel="z (nm)",
        title="Reflecting liquid above the sensor",
        xlim=bounds_nm[0],
        ylim=(-args.depth_nm - 3.0, bounds_nm[2, 1]),
    )
    axes[0, 0].legend(loc="upper right", markerscale=1.4)

    time_us = trajectory.times_seconds * 1.0e6
    axes[0, 1].plot(
        time_us,
        trajectory.sensor_axis_field_tesla * 1.0e9,
        lw=0.8,
    )
    axes[0, 1].set(
        xlabel=r"Time ($\mu$s)",
        ylabel=r"$B_\mathrm{NV}$ (nT)",
        title=f"Seeded dipolar field record (seed {args.seed + 1})",
    )

    axes[1, 0].plot(
        correlation.lag_seconds * 1.0e6,
        correlation.autocorrelation_tesla2,
    )
    axes[1, 0].axhline(0.0, color="0.65", lw=0.8)
    axes[1, 0].set(
        xlabel=r"Lag ($\mu$s)",
        ylabel=r"$C_B(\tau)/C_B(0)$",
        title="Diffusion-broadened field correlation",
    )

    positive = spectrum.frequency_hz > 0.0
    axes[1, 1].semilogy(
        spectrum.frequency_hz[positive] * 1.0e-3,
        spectrum.power_spectral_density_tesla2_per_hz[positive] * 1.0e18,
    )
    axes[1, 1].axvline(
        abs(trajectory.larmor_frequency_hz) * 1.0e-3,
        color="tab:orange",
        ls="--",
        label=rf"$^1$H Larmor ({abs(trajectory.larmor_frequency_hz) * 1e-3:.1f} kHz)",
    )
    axes[1, 1].set(
        xlabel="Frequency (kHz)",
        ylabel=r"One-sided PSD (nT$^2$/Hz)",
        title="Trajectory-derived spectrum",
        xlim=(0.0, min(2500.0, 0.5e-3 / trajectory.sample_interval_seconds)),
    )
    axes[1, 1].legend()

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
