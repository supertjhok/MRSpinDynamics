"""Plot channel-resolved ideal CPMG imaging for a two-coil Rx array.

The spin dynamics are propagated once per phase encode. Complex reciprocal
B1- maps are then applied to both receive channels, followed by correlated
k-space noise and RSS, sensitivity-weighted, and Roemer combinations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a two-channel reciprocal Rx-array CPMG simulation."
    )
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--noise-std", type=float, default=0.02)
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
    if args.pixels < 2 or args.ny < 1:
        raise ValueError("pixels must be at least two and ny must be positive")
    if args.noise_std < 0.0:
        raise ValueError("noise-std must be non-negative")
    if not -1.0 < args.correlation < 1.0:
        raise ValueError("correlation must lie strictly between -1 and 1")

    import matplotlib.pyplot as plt

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
    experiment = Experiment(
        sequence=CPMGImaging(num_echoes=1, ny=args.ny, maxoffs=0.5),
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
    noisy_channels = (
        result.channel_image_noisy
        if result.channel_image_noisy is not None
        else result.channel_image
    )
    noisy_roemer = (
        result.roemer_combined_image_noisy
        if result.roemer_combined_image_noisy is not None
        else result.roemer_combined_image
    )
    panels = (
        (rho, "Spin density", "gray"),
        (np.abs(result.receiver_sensitivities[0]), "|B1-| rx0", "viridis"),
        (np.abs(result.receiver_sensitivities[1]), "|B1-| rx1", "viridis"),
        (np.abs(noisy_channels[0, :, :, echo]), "Raw rx0", "magma"),
        (result.rss_magnitude[:, :, echo], "RSS (clean)", "magma"),
        (np.abs(noisy_roemer[:, :, echo]), "Roemer (with noise)", "magma"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.5), constrained_layout=True)
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        handle = axis.imshow(image, origin="lower", cmap=cmap)
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
    print(f"channel k-space shape: {result.channel_kspace.shape}")
    print(f"central rx1/rx0 phase: {np.degrees(phase):.1f} deg")
    print(f"noise covariance:\n{result.noise_covariance}")
    if args.output is not None:
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
