"""Loaded/matched birdcage ports, reciprocal B1-, noise, and reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import (
    BirdcageBranchLoading,
    BirdcageGeometry,
    BirdcageMultiport,
    birdcage_branch_mutual_inductance_matrix,
    birdcage_conductive_loading_resistance,
    calibrate_birdcage_conductor_quality_factor,
    design_independent_l_match,
    retune_loaded_birdcage,
    solve_birdcage_receive_sensitivities,
    tuned_low_pass_birdcage,
)
from spin_dynamics.workflows import (
    add_receiver_array_noise,
    centered_kspace_from_images,
    reconstruct_receiver_images,
    roemer_combine,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a mutually coupled, sample-loaded, matched two-port "
            "birdcage and its reciprocal receive reconstruction."
        )
    )
    parser.add_argument("--frequency-mhz", type=float, default=63.87)
    parser.add_argument("--pixels", type=int, default=33)
    parser.add_argument("--loading-grid", type=int, default=7)
    parser.add_argument("--conductivity", type=float, default=0.5)
    parser.add_argument("--unloaded-q", type=float, default=180.0)
    parser.add_argument("--mutual-scale", type=float, default=0.10)
    parser.add_argument("--noise-std", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _phantom(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    rho = (((xx / 0.060) ** 2 + (yy / 0.050) ** 2) <= 1.0).astype(np.float64)
    rho += 0.65 * (((xx + 0.020) / 0.012) ** 2 + ((yy - 0.012) / 0.010) ** 2 <= 1.0)
    rho *= 1.0 - 0.55 * (
        ((xx - 0.022) / 0.010) ** 2 + ((yy + 0.018) / 0.013) ** 2 <= 1.0
    )
    return rho


def main() -> None:
    args = _parse_args()
    if args.frequency_mhz <= 0.0:
        raise ValueError("--frequency-mhz must be positive")
    if args.pixels < 17:
        raise ValueError("--pixels must be at least 17")
    if args.loading_grid < 3:
        raise ValueError("--loading-grid must be at least 3")
    if args.conductivity < 0.0:
        raise ValueError("--conductivity must be non-negative")
    if args.unloaded_q <= 0.0:
        raise ValueError("--unloaded-q must be positive")
    if args.mutual_scale < 0.0:
        raise ValueError("--mutual-scale must be non-negative")
    if args.noise_std < 0.0:
        raise ValueError("--noise-std must be non-negative")

    plt = load_matplotlib(headless=args.output is not None)
    assert plt is not None

    frequency_hz = args.frequency_mhz * 1.0e6
    geometry = BirdcageGeometry(
        radius=0.15,
        length=0.30,
        n_rungs=16,
        ring_segments_per_section=4,
    )
    reference = tuned_low_pass_birdcage(
        geometry,
        frequency_hz,
        rung_inductance_h=180.0e-9,
        end_ring_inductance_h=35.0e-9,
        rung_resistance_ohm=0.04,
        end_ring_resistance_ohm=0.008,
    )
    mutual = args.mutual_scale * birdcage_branch_mutual_inductance_matrix(geometry)
    mutual_only = BirdcageBranchLoading(
        3 * geometry.n_rungs,
        inductance_coupling_h=mutual,
    )
    unloaded = calibrate_birdcage_conductor_quality_factor(
        retune_loaded_birdcage(reference, mutual_only, frequency_hz),
        args.unloaded_q,
    )

    loading_axis = np.linspace(-0.06, 0.06, args.loading_grid)
    lx, ly, lz = np.meshgrid(
        loading_axis,
        loading_axis,
        loading_axis,
        indexing="ij",
    )
    sample_mask = (lx**2 + ly**2 <= 0.06**2) & (np.abs(lz) <= 0.06)
    loading_points = np.stack((lx, ly, lz), axis=-1)
    loading_spacing = float(loading_axis[1] - loading_axis[0])
    sample_resistance = birdcage_conductive_loading_resistance(
        geometry,
        frequency_hz,
        loading_points,
        conductivity_s_per_m=args.conductivity * sample_mask,
        cell_volume_m3=loading_spacing**3,
    )
    loading = BirdcageBranchLoading(
        3 * geometry.n_rungs,
        inductance_coupling_h=mutual,
        resistance_ohm=sample_resistance,
    )
    loaded = retune_loaded_birdcage(
        unloaded.circuit,
        loading,
        frequency_hz,
    )
    ports = BirdcageMultiport(
        loaded,
        (0, geometry.n_rungs // 4),
        labels=("I port", "Q port"),
    )
    coil_impedance = ports.impedance_matrix_ohm(frequency_hz)
    matching = design_independent_l_match(
        coil_impedance,
        frequency_hz,
    )
    matched_impedance = matching.input_impedance_ohm(coil_impedance)
    scattering = matching.scattering_matrix(coil_impedance)

    image_axis = np.linspace(-0.07, 0.07, args.pixels)
    xx, yy = np.meshgrid(image_axis, image_axis, indexing="ij")
    points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    rho = _phantom(xx, yy)
    receive = solve_birdcage_receive_sensitivities(
        ports,
        frequency_hz,
        points,
        matching=matching,
        normalization_weights=rho,
    )
    sensitivities = receive.normalized_complex
    channel_images = sensitivities[..., np.newaxis] * rho[np.newaxis, ..., np.newaxis]
    channel_kspace = centered_kspace_from_images(channel_images)
    relative_covariance = (args.noise_std**2) * receive.noise_correlation
    if args.noise_std > 0.0:
        noisy_kspace = add_receiver_array_noise(
            channel_kspace,
            relative_covariance,
            seed=args.seed,
        )
    else:
        noisy_kspace = channel_kspace
    noisy_channels = reconstruct_receiver_images(noisy_kspace)
    reconstruction = roemer_combine(
        noisy_channels,
        sensitivities,
        (receive.noise_correlation if args.noise_std > 0.0 else np.eye(ports.n_ports)),
    )[..., 0]

    unloaded_mode = unloaded.modal_analysis().azimuthal_modes(1)[0]
    loaded_modes = loaded.modal_analysis().azimuthal_modes(1)
    loaded_mode = loaded_modes[0]
    phase_difference = np.rad2deg(np.angle(sensitivities[1] * sensitivities[0].conj()))
    support = rho > 0.0
    reconstruction_scale = float(
        np.sum(rho * np.abs(reconstruction))
        / max(np.sum(np.abs(reconstruction) ** 2), np.finfo(float).tiny)
    )
    reconstructed_magnitude = reconstruction_scale * np.abs(reconstruction)
    relative_error = float(
        np.linalg.norm(reconstructed_magnitude - rho) / np.linalg.norm(rho)
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    mutual_artist = axes[0, 0].imshow(
        mutual * 1.0e9,
        origin="lower",
        cmap="coolwarm",
    )
    axes[0, 0].set(
        xlabel="Branch index",
        ylabel="Branch index",
        title="Geometry-derived mutual inductance (nH)",
    )
    figure.colorbar(mutual_artist, ax=axes[0, 0])

    axes[0, 1].bar(
        ("unloaded", "sample loaded"),
        (unloaded_mode.quality_factor, loaded_mode.quality_factor),
        color=("tab:blue", "tab:orange"),
    )
    axes[0, 1].set(
        ylabel="Fundamental-mode Q",
        title=(
            f"Calibrated unloaded Q={args.unloaded_q:g}\n"
            f"loaded f={loaded_mode.frequency_hz * 1e-6:.3f} MHz"
        ),
    )
    axes[0, 1].grid(axis="y", alpha=0.25)

    scattering_db = 20.0 * np.log10(np.maximum(np.abs(scattering), 1.0e-8))
    scattering_artist = axes[0, 2].imshow(
        scattering_db,
        origin="lower",
        cmap="viridis",
        vmin=-80.0,
        vmax=0.0,
    )
    for row in range(ports.n_ports):
        for column in range(ports.n_ports):
            axes[0, 2].text(
                column,
                row,
                f"{scattering_db[row, column]:.1f}",
                ha="center",
                va="center",
                color="white",
            )
    axes[0, 2].set(
        xticks=range(ports.n_ports),
        yticks=range(ports.n_ports),
        xticklabels=ports.labels,
        yticklabels=ports.labels,
        title="Matched port S parameters (dB)",
    )
    figure.colorbar(scattering_artist, ax=axes[0, 2])

    phase_artist = axes[1, 0].imshow(
        np.ma.masked_where(~support, phase_difference).T,
        origin="lower",
        extent=(
            image_axis[0] * 100.0,
            image_axis[-1] * 100.0,
            image_axis[0] * 100.0,
            image_axis[-1] * 100.0,
        ),
        cmap="twilight",
        vmin=-180.0,
        vmax=180.0,
    )
    axes[1, 0].set(
        xlabel="x (cm)",
        ylabel="y (cm)",
        title=(
            "Reciprocal port phase: Q relative to I\n"
            f"noise corr={abs(receive.noise_correlation[0, 1]):.4f}"
        ),
    )
    figure.colorbar(phase_artist, ax=axes[1, 0], label="degree")

    image_limit = max(float(np.max(rho)), float(np.max(reconstructed_magnitude)))
    phantom_artist = axes[1, 1].imshow(
        rho.T,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=image_limit,
    )
    axes[1, 1].set(
        xticks=(),
        yticks=(),
        title="Spin-density reference",
    )
    figure.colorbar(phantom_artist, ax=axes[1, 1])

    reconstruction_artist = axes[1, 2].imshow(
        reconstructed_magnitude.T,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=image_limit,
    )
    axes[1, 2].set(
        xticks=(),
        yticks=(),
        title=f"Noise-aware Roemer reconstruction\nNRMSE={relative_error:.3f}",
    )
    figure.colorbar(reconstruction_artist, ax=axes[1, 2])
    figure.suptitle(
        "Loaded, matched birdcage: circuit → reciprocal ports → reconstruction",
        fontsize=14,
    )

    print(
        f"Calibrated unloaded/sample-loaded Q: "
        f"{unloaded_mode.quality_factor:.2f} / "
        f"{loaded_mode.quality_factor:.2f}"
    )
    print(
        "Loaded fundamental frequencies: "
        + ", ".join(f"{mode.frequency_hz * 1e-6:.6f} MHz" for mode in loaded_modes)
    )
    for label, summary in zip(ports.labels, matching.component_summary()):
        print(f"{label}: {summary}")
    print("Matched input impedance (ohm):")
    print(matched_impedance)
    print("Matched |S| (dB):")
    print(scattering_db)
    print("Noise correlation:")
    print(receive.noise_correlation)
    print(f"Reconstruction NRMSE: {relative_error:.6f}")

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
