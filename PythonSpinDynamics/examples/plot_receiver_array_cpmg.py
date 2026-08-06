"""Plot channel-resolved ideal CPMG imaging for a two-coil Rx array.

The spin dynamics are propagated once per phase encode. Complex reciprocal
B1- maps are then applied to both receive channels, followed by correlated
k-space noise and RSS, sensitivity-weighted, and Roemer combinations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a two-channel reciprocal Rx-array CPMG simulation."
    )
    parser.add_argument("--pixels", type=int, default=9)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.25,
        help="Per-channel complex k-space noise standard deviation.",
    )
    parser.add_argument("--correlation", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the figure instead of opening an interactive window.",
    )
    return parser.parse_args()


def _phantom(pixels: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, pixels)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    rho = (x**2 + y**2 <= 0.82**2).astype(np.float64)
    rho += 0.65 * (((x + 0.28) / 0.2) ** 2 + ((y - 0.22) / 0.16) ** 2 <= 1.0)
    rho *= 1.0 - 0.55 * (((x - 0.28) / 0.16) ** 2 + ((y + 0.22) / 0.2) ** 2 <= 1.0)
    return rho


def main() -> None:
    args = _parse_args()
    if args.pixels < 3 or args.pixels % 2 == 0:
        raise ValueError("pixels must be an odd integer of at least three")
    if args.ny < 1:
        raise ValueError("ny must be positive")
    if args.noise_std < 0.0:
        raise ValueError("noise-std must be non-negative")
    if not -1.0 < args.correlation < 1.0:
        raise ValueError("correlation must lie strictly between -1 and 1")
    plt = load_matplotlib(headless=args.output is not None)
    assert plt is not None

    from spin_dynamics.experiment import (
        Acquisition,
        CPMGImaging,
        Experiment,
        Hardware,
        ImagingPlane,
        Phantom,
        RxArray,
        RxCoil,
        Sample,
        SolenoidCoil,
    )

    rho = _phantom(args.pixels)
    receivers = RxArray(
        (
            RxCoil(SolenoidCoil(0.015, 0.03, 10, axis="x")),
            RxCoil(SolenoidCoil(0.015, 0.03, 10, axis="y")),
        )
    )
    covariance = (args.noise_std**2) * np.array(
        [[1.0, args.correlation], [args.correlation, 1.0]],
        dtype=np.complex128,
    )
    # The legacy phase-encode gradients are symmetric about k=0. For an odd
    # matrix, this FOV makes their phase increments exactly match the DFT grid.
    dft_matched_fov = args.pixels**3 / (args.pixels - 1) ** 2
    experiment = Experiment(
        sequence=CPMGImaging(
            num_echoes=1,
            ny=args.ny,
            maxoffs=0.5,
            fov=(dft_matched_fov, dft_matched_fov),
        ),
        sample=Sample(phantom=Phantom(rho)),
        hardware=Hardware(
            rx_coil=receivers,
            plane=ImagingPlane(plane="xy", extent_m=(0.025, 0.025)),
        ),
        acquisition=Acquisition(
            receiver_noise_covariance=(covariance if args.noise_std > 0.0 else None),
            receiver_noise_seed=args.seed,
        ),
    )
    result = experiment.run().result

    echo = 0
    clean_reconstruction = result.roemer_combined_image[:, :, echo]
    noisy_reconstruction = (
        result.roemer_combined_image_noisy[:, :, echo]
        if result.roemer_combined_image_noisy is not None
        else clean_reconstruction
    )
    noise_residual = noisy_reconstruction - clean_reconstruction
    reconstruction_limit = max(
        float(np.max(np.abs(clean_reconstruction))),
        float(np.max(np.abs(noisy_reconstruction))),
    )
    residual_component = np.real(noise_residual)
    residual_limit = max(
        float(np.max(np.abs(residual_component))),
        np.finfo(np.float64).eps,
    )
    panels = (
        (rho, "Spin density", "gray", None, None),
        (
            np.abs(result.receiver_sensitivities[0]),
            "Receive sensitivity |B1-|: rx0",
            "viridis",
            None,
            None,
        ),
        (
            np.abs(result.receiver_sensitivities[1]),
            "Receive sensitivity |B1-|: rx1",
            "viridis",
            None,
            None,
        ),
        (
            np.abs(clean_reconstruction),
            "Roemer reconstruction (clean)",
            "magma",
            0.0,
            reconstruction_limit,
        ),
        (
            np.abs(noisy_reconstruction),
            "Roemer reconstruction (noisy)",
            "magma",
            0.0,
            reconstruction_limit,
        ),
        (
            residual_component,
            "Image-space noise: Re(noisy - clean)",
            "coolwarm",
            -residual_limit,
            residual_limit,
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.5), constrained_layout=True)
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
    fig.suptitle(
        "Two-channel reciprocal Rx array: one spin solve, complex channel projection"
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        plt.close(fig)

    center = (args.pixels // 2, args.pixels // 2)
    phase = np.angle(
        result.receiver_sensitivities[(1, *center)]
        / result.receiver_sensitivities[(0, *center)]
    )
    clean_magnitude = np.abs(clean_reconstruction)
    shape_scale = float(np.vdot(rho, clean_magnitude).real / np.vdot(rho, rho).real)
    shape_error = float(
        np.linalg.norm(clean_magnitude - shape_scale * rho)
        / np.linalg.norm(clean_magnitude)
    )
    image_noise_rms = float(np.sqrt(np.mean(np.abs(noise_residual) ** 2)))
    print(f"clean reconstruction shape error: {shape_error:.4f}")
    print(f"image-space complex noise RMS: {image_noise_rms:.6g}")
    print(f"channel k-space shape: {result.channel_kspace.shape}")
    print(f"central rx1/rx0 phase: {np.degrees(phase):.1f} deg")
    print(f"noise covariance:\n{result.noise_covariance}")
    if args.output is not None:
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
