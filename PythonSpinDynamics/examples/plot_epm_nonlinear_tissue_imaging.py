"""Nonlinear EPM imaging and reconstruction of a simple tissue phantom."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import illustrative_hybrid_epm_array  # noqa: E402
from spin_dynamics.workflows import (  # noqa: E402
    build_epm_nonlinear_encoding,
    random_epm_encoding_states,
    run_epm_nonlinear_imaging,
    simple_tissue_phantom,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a two-tissue phantom with the nonlinear field profiles of "
            "the illustrative 72-control electropermanent array."
        )
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=16,
        help="Square phantom matrix size (default: 16).",
    )
    parser.add_argument(
        "--encodings",
        type=int,
        default=384,
        help="Number of retained-state acquisitions (default: 384).",
    )
    parser.add_argument(
        "--phase-encoding-us",
        type=float,
        default=300.0,
        help="Phase accumulation per retained state (default: 300 us).",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=35.0,
        help="Complex acquisition RMS SNR (default: 35 dB).",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-4,
        help="Dimensionless Tikhonov fraction (default: 1e-4).",
    )
    parser.add_argument(
        "--panel-gap-mm",
        type=float,
        default=150.0,
        help="Hybrid-array panel gap (default: 150 mm).",
    )
    parser.add_argument("--seed", type=int, default=4, help="Encoding/noise seed.")
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.matrix_size < 10:
        raise ValueError("--matrix-size must be at least 10")
    if args.encodings < args.matrix_size**2:
        raise ValueError("--encodings must be at least the voxel count")
    if not np.isfinite(args.phase_encoding_us) or args.phase_encoding_us <= 0.0:
        raise ValueError("--phase-encoding-us must be positive")
    if not np.isfinite(args.snr_db) or args.snr_db <= 0.0:
        raise ValueError("--snr-db must be positive")
    if not np.isfinite(args.regularization) or args.regularization < 0.0:
        raise ValueError("--regularization must be non-negative")
    if not np.isfinite(args.panel_gap_mm) or args.panel_gap_mm <= 80.0:
        raise ValueError("--panel-gap-mm must exceed 80")


def _centroid_mm(
    image: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(x_m, y_m, indexing="xy")
    weights = np.where(mask, np.maximum(image, 0.0), 0.0)
    total = float(np.sum(weights))
    if total == 0.0:
        return np.asarray((np.nan, np.nan))
    return 1e3 * np.asarray(
        (
            np.sum(weights * x_grid) / total,
            np.sum(weights * y_grid) / total,
        )
    )


def _extent_mm(axis: np.ndarray) -> tuple[float, float, float, float]:
    return (1e3 * axis[0], 1e3 * axis[-1], 1e3 * axis[0], 1e3 * axis[-1])


def main() -> None:
    args = _parser().parse_args()
    _validate(args)

    phantom = simple_tissue_phantom(args.matrix_size, field_of_view_m=0.040)
    array = illustrative_hybrid_epm_array(panel_gap_m=args.panel_gap_mm * 1e-3)
    basis = array.build_field_basis(
        phantom.points_m,
        n_cross=1,
        n_length=7,
    )
    states = random_epm_encoding_states(
        basis,
        args.encodings,
        amplitude_fraction=0.60,
        seed=args.seed,
    )
    encoding = build_epm_nonlinear_encoding(
        basis,
        states,
        image_shape=phantom.shape,
        phase_encoding_s=args.phase_encoding_us * 1e-6,
    )
    expected = phantom.spin_echo_image(
        repetition_time_s=1.2,
        echo_time_s=40e-3,
    )
    result = run_epm_nonlinear_imaging(
        encoding,
        expected,
        regularization=args.regularization,
        snr_db=args.snr_db,
        seed=args.seed + 1,
    )

    tissue_mean = float(np.mean(expected[phantom.tissue_labels == 1]))
    target_mean = float(np.mean(expected[phantom.target_mask]))
    target_threshold = 0.5 * (tissue_mean + target_mean)
    reconstructed_target = result.reconstructed_image > target_threshold
    true_center = _centroid_mm(
        expected,
        phantom.x_m,
        phantom.y_m,
        phantom.target_mask,
    )
    reconstructed_center = _centroid_mm(
        result.reconstructed_image,
        phantom.x_m,
        phantom.y_m,
        reconstructed_target,
    )
    localization_error_mm = float(np.linalg.norm(reconstructed_center - true_center))

    extent = _extent_mm(phantom.x_m)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.5), constrained_layout=True)
    fig.suptitle(
        "Nonlinear electropermanent-array imaging of a simple tissue phantom",
        fontsize=15,
    )

    ax = axes[0, 0]
    image = ax.imshow(expected, origin="lower", extent=extent, cmap="magma")
    ax.contour(
        phantom.x_m * 1e3,
        phantom.y_m * 1e3,
        phantom.target_mask,
        levels=(0.5,),
        colors="cyan",
        linewidths=1.7,
    )
    ax.plot(true_center[0], true_center[1], "c+", ms=11, mew=2)
    ax.set_title("spin-echo tissue contrast\ncyan: image-guided target")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(image, ax=ax, label="relative signal")

    ax = axes[0, 1]
    phase_indices = np.linspace(1, args.encodings - 1, 4, dtype=int)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for index, color in zip(phase_indices, colors, strict=True):
        phase_map = encoding.phase_rad[index].reshape(phantom.shape)
        levels = np.linspace(np.percentile(phase_map, 15), np.percentile(phase_map, 85), 6)
        ax.contour(
            phantom.x_m * 1e3,
            phantom.y_m * 1e3,
            phase_map,
            levels=np.unique(levels),
            colors=color,
            linewidths=0.8,
            alpha=0.85,
        )
    ax.set_title("four measured nonlinear phase profiles")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    ax = axes[0, 2]
    real_system = np.vstack((encoding.matrix.real, encoding.matrix.imag))
    singular_values = np.linalg.svd(real_system, compute_uv=False)
    ax.semilogy(singular_values / singular_values[0], color="tab:purple", lw=1.8)
    ax.axhline(
        singular_values[-1] / singular_values[0],
        color="0.35",
        ls="--",
        lw=1.0,
    )
    ax.set_title(f"encoding spectrum; condition = {encoding.condition_number:.2f}")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("normalized singular value")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    sample_count = min(120, args.encodings)
    sample_index = np.arange(sample_count)
    ax.plot(sample_index, result.signal.real[:sample_count], label="real", lw=1.2)
    ax.plot(sample_index, result.signal.imag[:sample_count], label="imag", lw=1.2)
    ax.plot(
        sample_index,
        np.abs(result.signal_clean[:sample_count]),
        color="0.2",
        alpha=0.55,
        label="clean magnitude",
        lw=1.0,
    )
    ax.set_title(f"encoded signal with {args.snr_db:.0f} dB RMS SNR")
    ax.set_xlabel("retained-state acquisition")
    ax.set_ylabel("encoded signal")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    image = ax.imshow(
        result.reconstructed_image,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=float(np.max(expected)),
    )
    ax.contour(
        phantom.x_m * 1e3,
        phantom.y_m * 1e3,
        reconstructed_target,
        levels=(0.5,),
        colors="white",
        linewidths=1.5,
    )
    ax.plot(reconstructed_center[0], reconstructed_center[1], "w+", ms=11, mew=2)
    ax.set_title(f"nonnegative reconstruction\nNRMSE = {100 * result.nrmse:.2f}%")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(image, ax=ax, label="relative signal")

    ax = axes[1, 2]
    error = result.reconstructed_image - expected
    limit = max(float(np.max(np.abs(error))), 1e-12)
    image = ax.imshow(
        error,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.plot(true_center[0], true_center[1], "c+", ms=11, mew=2, label="true")
    ax.plot(
        reconstructed_center[0],
        reconstructed_center[1],
        "kx",
        ms=8,
        mew=1.8,
        label="localized",
    )
    ax.set_title(f"reconstruction error\ntarget error = {localization_error_mm:.2f} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(fontsize=8)
    fig.colorbar(image, ax=ax, label="reconstructed - expected")

    print("Nonlinear EPM tissue imaging")
    print(f"  phantom matrix: {args.matrix_size} x {args.matrix_size}")
    print(f"  retained-state acquisitions: {encoding.encoding_count}")
    print(f"  encoding time: {1e6 * encoding.phase_encoding_s:.1f} us")
    print(
        "  field span across ROI: "
        f"{1e6 * np.min(encoding.field_span_t):.1f} to "
        f"{1e6 * np.max(encoding.field_span_t):.1f} uT"
    )
    print(f"  real-system condition number: {encoding.condition_number:.3f}")
    print(f"  reconstruction NRMSE: {100 * result.nrmse:.3f}%")
    print(f"  target localization error: {localization_error_mm:.3f} mm")
    print(f"  projected iterations/converged: {result.iterations}/{result.converged}")
    print("  next: use the localized target in magnetic nanoparticle transport")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
    else:
        plt.show()


if __name__ == "__main__":
    main()
