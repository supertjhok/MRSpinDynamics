"""Plot R=2 Cartesian SENSE with two reciprocal receive-coil maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot two-channel Cartesian SENSE reconstruction."
    )
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--noise-std", type=float, default=0.015)
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
    if args.ny < 1:
        raise ValueError("ny must be positive")
    if args.noise_std < 0.0:
        raise ValueError("noise-std must be non-negative")
    if not -1.0 < args.correlation < 1.0:
        raise ValueError("correlation must lie strictly between -1 and 1")
    if args.regularization < 0.0:
        raise ValueError("regularization must be non-negative")

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
    covariance = (args.noise_std**2) * np.array(
        [[1.0, args.correlation], [args.correlation, 1.0]],
        dtype=np.complex128,
    )
    result = Experiment(
        sequence=CPMGImaging(num_echoes=1, ny=args.ny, maxoffs=0.5),
        sample=Sample(phantom=Phantom(rho)),
        hardware=Hardware(
            rx_coil=receivers,
            plane=ImagingPlane(plane="xy", extent_m=(0.025, 0.025)),
        ),
        acquisition=Acquisition(
            receiver_noise_covariance=(covariance if args.noise_std > 0.0 else None),
            receiver_noise_seed=args.seed,
            sense_acceleration=2,
            sense_axis="z",
            sense_regularization=args.regularization,
        ),
    ).run().result

    echo = 0
    zero_filled = np.sqrt(
        np.sum(np.abs(result.sense_zero_filled_channel_image) ** 2, axis=0)
    )
    noisy = (
        result.sense_image_noisy
        if result.sense_image_noisy is not None
        else result.sense_image
    )
    g_display = np.where(np.isfinite(result.sense_g_factor), result.sense_g_factor, np.nan)
    condition_display = np.where(
        np.isfinite(result.sense_condition_number),
        np.log10(np.maximum(result.sense_condition_number, 1.0)),
        np.nan,
    )
    panels = (
        (np.abs(result.sense_reference_image[:, :, echo]), "Reference", "gray"),
        (zero_filled[:, :, echo], "R=2 zero-filled RSS", "magma"),
        (np.abs(result.sense_image[:, :, echo]), "SENSE (clean)", "magma"),
        (np.abs(noisy[:, :, echo]), "SENSE (with noise)", "magma"),
        (g_display, "Predicted g-factor", "viridis"),
        (condition_display, "log10 condition", "viridis"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), constrained_layout=True)
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        handle = axis.imshow(image, origin="lower", cmap=cmap)
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

    finite_g = result.sense_g_factor[np.isfinite(result.sense_g_factor)]
    rank_deficient = int(np.count_nonzero(result.sense_rank < result.sense_acceleration))
    print(f"sampling fraction: {np.mean(result.sampling_mask):.3f}")
    print(f"maximum finite g-factor: {np.max(finite_g) if finite_g.size else np.inf:.3f}")
    print(f"rank-deficient voxels: {rank_deficient}")
    if args.output is not None:
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
