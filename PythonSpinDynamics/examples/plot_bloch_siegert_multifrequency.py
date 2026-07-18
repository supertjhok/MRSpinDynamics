"""Mandal-2014 multi-frequency Bloch-Siegert validation and low-B0 mapping.

The first row reproduces the paper's paired-offset phase slope and interleaved
multi-slice correction. The second row deliberately leaves the rotating-wave
approximation: the paired-pulse common phase isolates the counter-rotating term
and is inverted into a noisy low-frequency B0 map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows.bloch_siegert import (
    counter_rotating_common_phase,
    estimate_larmor_hz_from_counter_rotating_phase,
    estimate_nutation_hz,
    mandal_multislice_correction,
    simulate_bloch_siegert_phase_sweep,
)


def _build_b0_map(points: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, points)
    xx, zz = np.meshgrid(axis, axis, indexing="xy")
    frequency = 110.0e3 + 28.0e3 * xx - 18.0e3 * zz
    frequency += 38.0e3 * np.exp(-((xx + 0.30) ** 2 + (zz - 0.20) ** 2) / 0.10)
    frequency -= 24.0e3 * np.exp(-((xx - 0.35) ** 2 + (zz + 0.35) ** 2) / 0.08)
    return np.maximum(frequency, 45.0e3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--data-output", type=Path)
    parser.add_argument("--map-points", type=int, default=64)
    parser.add_argument("--phase-noise-deg", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    plt = load_matplotlib(headless=args.output is not None)

    # Mandal et al. Fig. 15: f0=1.48 MHz, paired offsets +/-25 kHz,
    # and a fitted 90-degree pulse length of 102 us.
    larmor_hz = 1.48e6
    offset_hz = 25.0e3
    t90_s = 102.0e-6
    nutation_hz = 1.0 / (4.0 * t90_s)
    durations_s = np.arange(40.0e-6, 400.0e-6 + 1e-12, 40.0e-6)
    paper = simulate_bloch_siegert_phase_sweep(
        durations_s,
        larmor_hz=larmor_hz,
        offset_hz=offset_hz,
        nutation_hz=nutation_hz,
    )
    inferred_nutation_hz = float(
        estimate_nutation_hz(
            paper.differential_phase_rad[-1],
            durations_s[-1],
            offset_hz=offset_hz,
        )
    )
    inferred_t90_s = 1.0 / (4.0 * inferred_nutation_hz)

    # Eq. 15 phase corrections for the paper's Fig. 14 parameters.
    multislice_t90_s = 140.0e-6
    multislice = mandal_multislice_correction(
        8,
        nutation_hz=1.0 / (4.0 * multislice_t90_s),
        slice_spacing_hz=20.0e3,
        t90_s=multislice_t90_s,
    )

    # Counter-rotating common phase versus field. Exact lab-frame checkpoints
    # show where the perturbative formula is trustworthy and where the RWA's
    # zero-common-phase prediction fails.
    carrier_axis_hz = np.geomspace(45.0e3, 2.0e6, 240)
    common_curve = counter_rotating_common_phase(
        carrier_axis_hz,
        400.0e-6,
        offset_hz=offset_hz,
        nutation_hz=nutation_hz,
    )
    checkpoint_hz = np.array([50, 75, 100, 150, 250, 500, 1000, 1480]) * 1e3
    checkpoint_common = np.array(
        [
            simulate_bloch_siegert_phase_sweep(
                np.array([400.0e-6]),
                larmor_hz=value,
                offset_hz=offset_hz,
                nutation_hz=nutation_hz,
                steps_per_cycle=64,
            ).common_mode_phase_rad[0]
            for value in checkpoint_hz
        ]
    )

    # A synthetic low-field map uses the common phase, with realistic phase
    # noise added before inversion. This is a B0 observable that does not exist
    # in the RWA because its paired-offset common phase is identically zero.
    true_frequency_hz = _build_b0_map(args.map_points)
    clean_common = counter_rotating_common_phase(
        true_frequency_hz,
        400.0e-6,
        offset_hz=offset_hz,
        nutation_hz=nutation_hz,
    )
    rng = np.random.default_rng(args.seed)
    noisy_common = clean_common + np.deg2rad(args.phase_noise_deg) * rng.standard_normal(
        true_frequency_hz.shape
    )
    recovered_frequency_hz = estimate_larmor_hz_from_counter_rotating_phase(
        noisy_common,
        400.0e-6,
        offset_hz=offset_hz,
        nutation_hz=nutation_hz,
    )
    relative_rmse = float(
        np.sqrt(np.mean((recovered_frequency_hz - true_frequency_hz) ** 2))
        / np.mean(true_frequency_hz)
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), constrained_layout=True)
    duration_us = durations_s * 1e6
    axes[0, 0].plot(
        duration_us,
        np.degrees(paper.differential_phase_rad),
        "o",
        label="exact lab frame",
    )
    axes[0, 0].plot(
        duration_us,
        np.degrees(paper.rwa_differential_phase_rad),
        "--",
        label="Mandal Eq. 17 / RWA",
    )
    axes[0, 0].set(
        title="Mandal 2014 Fig. 15",
        xlabel=r"$T_{BS}$ (µs)",
        ylabel=r"paired phase $\Delta\phi_{BS}$ (deg)",
    )
    axes[0, 0].text(
        0.04,
        0.94,
        f"fit: T90 = {inferred_t90_s * 1e6:.1f} µs\nreported: 102 µs",
        transform=axes[0, 0].transAxes,
        va="top",
    )
    axes[0, 0].legend(frameon=False, loc="lower right")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        duration_us,
        np.degrees(paper.lower_offset_phase_rad),
        "o-",
        label=r"carrier $f_0-\Delta f$",
    )
    axes[0, 1].plot(
        duration_us,
        np.degrees(paper.upper_offset_phase_rad),
        "s-",
        label=r"carrier $f_0+\Delta f$",
    )
    axes[0, 1].plot(
        duration_us,
        np.degrees(paper.common_mode_phase_rad),
        "k--",
        label="common (counter-rotating)",
    )
    axes[0, 1].set(
        title="Odd and even paired phases",
        xlabel=r"$T_{BS}$ (µs)",
        ylabel="phase (deg)",
    )
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25)

    slices = multislice.slice_number
    axes[0, 2].plot(
        slices,
        np.degrees(multislice.excitation_phase_error_rad),
        "o-",
        label="uncorrected",
    )
    axes[0, 2].plot(
        slices,
        np.degrees(
            multislice.excitation_phase_error_rad
            + multislice.excitation_phase_correction_rad
        ),
        "s--",
        label="after Eq. 15 correction",
    )
    timing_axis = axes[0, 2].twinx()
    timing_axis.plot(
        slices,
        multislice.excitation_timing_shift_s * 1e6,
        "^:",
        color="tab:green",
        label="Eq. 16 timing",
    )
    axes[0, 2].set(
        title="Interleaved multi-slice CPMG",
        xlabel="slice number",
        ylabel="excitation phase error (deg)",
    )
    timing_axis.set_ylabel("timing shift (µs)")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    handles2, labels2 = timing_axis.get_legend_handles_labels()
    axes[0, 2].legend(handles + handles2, labels + labels2, frameon=False)
    axes[0, 2].grid(alpha=0.25)

    axes[1, 0].semilogx(
        carrier_axis_hz / 1e3,
        np.degrees(common_curve),
        label="second-order counter-rotating",
    )
    axes[1, 0].semilogx(
        checkpoint_hz / 1e3,
        np.degrees(checkpoint_common),
        "o",
        label="exact lab frame",
    )
    axes[1, 0].axhline(0.0, color="black", linestyle="--", label="RWA")
    axes[1, 0].set(
        title="RWA limit at low Larmor frequency",
        xlabel=r"$f_0$ (kHz)",
        ylabel="paired common phase (deg)",
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.25, which="both")

    extent = (-1.0, 1.0, -1.0, 1.0)
    image = axes[1, 1].imshow(
        true_frequency_hz / 1e3,
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes[1, 1].set(
        title="True low-frequency B0 map",
        xlabel="x (normalized)",
        ylabel="z (normalized)",
    )
    fig.colorbar(image, ax=axes[1, 1], label=r"$f_0$ (kHz)", shrink=0.85)

    recovered = axes[1, 2].imshow(
        recovered_frequency_hz / 1e3,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=float(np.min(true_frequency_hz / 1e3)),
        vmax=float(np.max(true_frequency_hz / 1e3)),
    )
    axes[1, 2].set(
        title=(
            f"Counter-rotating B0 recovery\n"
            f"phase noise {args.phase_noise_deg:.2f}°, RMSE {100 * relative_rmse:.2f}%"
        ),
        xlabel="x (normalized)",
        ylabel="z (normalized)",
    )
    fig.colorbar(recovered, ax=axes[1, 2], label=r"estimated $f_0$ (kHz)", shrink=0.85)

    fig.suptitle(
        "Multi-frequency Bloch-Siegert phase: published RWA result and "
        "counter-rotating extension"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved {args.output}")
    else:
        plt.show()
    print(
        f"Mandal Fig. 15 endpoint: {np.degrees(paper.differential_phase_rad[-1]):.2f} deg; "
        f"T90 fit {inferred_t90_s * 1e6:.2f} us"
    )
    print(
        f"Noisy low-frequency B0 recovery: {100 * relative_rmse:.3f}% RMSE "
        f"at {args.phase_noise_deg:.3f} deg phase noise"
    )

    if args.data_output is not None:
        args.data_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.data_output,
            durations_s=durations_s,
            exact_differential_phase_rad=paper.differential_phase_rad,
            rwa_differential_phase_rad=paper.rwa_differential_phase_rad,
            exact_common_phase_rad=paper.common_mode_phase_rad,
            carrier_axis_hz=carrier_axis_hz,
            counter_rotating_common_phase_rad=common_curve,
            true_larmor_hz=true_frequency_hz,
            noisy_common_phase_rad=noisy_common,
            recovered_larmor_hz=recovered_frequency_hz,
        )
        print(f"saved {args.data_output}")


if __name__ == "__main__":
    main()
