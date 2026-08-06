"""Plot R=2 Cartesian SENSE with two reciprocal receive-coil maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot two-channel Cartesian SENSE reconstruction."
    )
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.20,
        help="Per-channel complex sampled-k-space noise standard deviation.",
    )
    parser.add_argument("--correlation", type=float, default=0.25)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the figure instead of opening an interactive window.",
    )
    return parser.parse_args()


def _phantom(pixels: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, pixels)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    rho = (x**2 + z**2 <= 0.82**2).astype(np.float64)
    rho += 0.7 * (((x + 0.3) / 0.2) ** 2 + ((z - 0.2) / 0.16) ** 2 <= 1.0)
    rho *= 1.0 - 0.5 * (((x - 0.25) / 0.17) ** 2 + ((z + 0.25) / 0.2) ** 2 <= 1.0)
    return rho


def main() -> None:
    args = _parse_args()
    if args.pixels < 4 or args.pixels % 2:
        raise ValueError("pixels must be an even integer of at least four")
    if args.noise_std < 0.0:
        raise ValueError("noise-std must be non-negative")
    if not -1.0 < args.correlation < 1.0:
        raise ValueError("correlation must lie strictly between -1 and 1")
    if args.regularization < 0.0:
        raise ValueError("regularization must be non-negative")

    plt = load_matplotlib(headless=args.output is not None)
    assert plt is not None

    from spin_dynamics.experiment import (
        Hardware,
        ImagingPlane,
        Phantom,
        RxArray,
        RxCoil,
        SolenoidCoil,
        solve_receive_sensitivities,
    )
    from spin_dynamics.workflows import (
        CartesianSENSEEncoding,
        add_receiver_array_noise,
        reconstruct_cartesian_sense,
        uniform_cartesian_mask,
    )

    rho = _phantom(args.pixels)
    receivers = RxArray(
        (
            RxCoil(
                SolenoidCoil(
                    0.013,
                    0.028,
                    10,
                    center_m=(0.0, -0.006, 0.0),
                    axis="x",
                )
            ),
            RxCoil(
                SolenoidCoil(
                    0.013,
                    0.028,
                    10,
                    center_m=(0.0, 0.006, 0.0),
                    axis="x",
                )
            ),
        )
    )
    hardware = Hardware(
        rx_coil=receivers,
        plane=ImagingPlane(plane="xy", extent_m=(0.025, 0.025)),
    )
    sensitivities = solve_receive_sensitivities(
        Phantom(rho),
        hardware,
    ).normalized_complex
    covariance = (args.noise_std**2) * np.array(
        [[1.0, args.correlation], [args.correlation, 1.0]],
        dtype=np.complex128,
    )
    combination_covariance = (
        covariance
        if args.noise_std > 0.0
        else np.eye(sensitivities.shape[0], dtype=np.complex128)
    )
    sampling_mask = uniform_cartesian_mask(
        rho.shape,
        2,
        axis="z",
    )
    encoding = CartesianSENSEEncoding(
        sensitivities,
        sampling_mask,
        axis="z",
        noise_covariance=combination_covariance,
    )
    sampled_kspace = encoding.forward(rho)
    clean = reconstruct_cartesian_sense(
        sampled_kspace,
        sensitivities,
        sampling_mask,
        axis="z",
        noise_covariance=combination_covariance,
        regularization=args.regularization,
    )
    if args.noise_std > 0.0:
        noisy_kspace = add_receiver_array_noise(
            sampled_kspace,
            covariance,
            seed=args.seed,
        )
        noisy_kspace *= sampling_mask[np.newaxis, ..., np.newaxis]
        noisy = reconstruct_cartesian_sense(
            noisy_kspace,
            sensitivities,
            sampling_mask,
            axis="z",
            noise_covariance=combination_covariance,
            regularization=args.regularization,
        )
    else:
        noisy = clean

    echo = 0
    clean_image = clean.image[:, :, echo]
    noisy_image = noisy.image[:, :, echo]
    noise_residual = noisy_image - clean_image
    zero_filled = np.sqrt(np.sum(np.abs(clean.zero_filled_channel_image) ** 2, axis=0))[
        :, :, echo
    ]
    reconstruction_limit = max(
        float(np.max(rho)),
        float(np.max(np.abs(clean_image))),
        float(np.max(np.abs(noisy_image))),
    )
    residual_component = np.real(noise_residual)
    residual_limit = max(
        float(np.max(np.abs(residual_component))),
        np.finfo(np.float64).eps,
    )
    g_display = np.where(np.isfinite(clean.g_factor), clean.g_factor, np.nan)
    panels = (
        (rho, "Spin density reference", "gray", 0.0, reconstruction_limit),
        (
            np.abs(clean_image),
            "R=2 SENSE reconstruction (clean)",
            "magma",
            0.0,
            reconstruction_limit,
        ),
        (
            np.abs(noisy_image),
            "R=2 SENSE reconstruction (noisy)",
            "magma",
            0.0,
            reconstruction_limit,
        ),
        (zero_filled, "R=2 zero-filled RSS", "magma", None, None),
        (
            residual_component,
            "Image-space noise: Re(noisy - clean)",
            "coolwarm",
            -residual_limit,
            residual_limit,
        ),
        (g_display, "Predicted g-factor", "viridis", None, None),
    )

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), constrained_layout=True)
    for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        handle = axis.imshow(
            image,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Two-coil Cartesian SENSE: explicit P F S encoding, R=2")
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        plt.close(fig)

    clean_magnitude = np.abs(clean_image)
    shape_scale = float(np.vdot(rho, clean_magnitude).real / np.vdot(rho, rho).real)
    shape_error = float(
        np.linalg.norm(clean_magnitude - shape_scale * rho)
        / np.linalg.norm(clean_magnitude)
    )
    image_noise_rms = float(np.sqrt(np.mean(np.abs(noise_residual) ** 2)))
    finite_g = clean.g_factor[np.isfinite(clean.g_factor)]
    rank_deficient = int(np.count_nonzero(clean.rank < clean.acceleration))
    print(f"clean reconstruction shape error: {shape_error:.4g}")
    print(f"image-space complex noise RMS: {image_noise_rms:.6g}")
    print(f"sampling fraction: {np.mean(clean.sampling_mask):.3f}")
    print(
        f"maximum finite g-factor: {np.max(finite_g) if finite_g.size else np.inf:.3f}"
    )
    print(f"rank-deficient voxels: {rank_deficient}")
    if args.output is not None:
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
