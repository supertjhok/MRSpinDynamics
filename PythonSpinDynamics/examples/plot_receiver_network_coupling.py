"""Plot Phase 4 multiport coupling, loaded sensitivity, and correlated noise."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spin_dynamics.experiment import (
    Acquisition,
    CPMGImaging,
    Experiment,
    Hardware,
    ImagingPlane,
    Phantom,
    ReceiverNetwork,
    RxArray,
    RxCoil,
    Sample,
    SolenoidCoil,
)
from spin_dynamics.fields import (
    extract_multiport_impedance,
    helical_solenoid,
)


def _phantom(pixels: int) -> Phantom:
    axis = np.linspace(-1.0, 1.0, pixels)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    rho = (
        1.0 * ((xx + 0.35) ** 2 + (yy + 0.15) ** 2 < 0.28**2)
        + 0.7 * ((xx - 0.25) ** 2 + (yy - 0.25) ** 2 < 0.4**2)
    )
    if not np.any(rho):
        rho[pixels // 2, pixels // 2] = 1.0
    return Phantom(rho)


def _receiver_array(
    frequency_hz: float,
    noise_bandwidth_hz: float,
) -> tuple[RxArray, np.ndarray, np.ndarray]:
    geometries = (
        SolenoidCoil(
            radius_m=0.012,
            length_m=0.025,
            turns=4,
            center_m=(0.0, -0.007, 0.0),
            axis="x",
            n_segments=12,
        ),
        SolenoidCoil(
            radius_m=0.012,
            length_m=0.025,
            turns=4,
            center_m=(0.0, 0.007, 0.0),
            axis="x",
            n_segments=12,
        ),
    )
    channels = tuple(RxCoil(geometry) for geometry in geometries)

    def peec_conductor(geometry: SolenoidCoil):
        conductor = helical_solenoid(
            diameter=2.0 * geometry.radius_m,
            length=geometry.length_m,
            turns=geometry.turns,
            wire_radius=0.4e-3,
            n_per_turn=geometry.n_segments,
            n_radial=1,
            n_angular=4,
            axis=geometry.axis,
        )
        return dataclasses.replace(
            conductor,
            path_points=conductor.path_points + np.asarray(geometry.center_m),
        )

    conductors = tuple(peec_conductor(geometry) for geometry in geometries)
    peec = extract_multiport_impedance(conductors, [frequency_hz])
    # PEEC currently supplies conductor loss. Add an illustrative reciprocal
    # sample-loss term so the example makes shared thermal noise visible.
    sample_loss_ohm = np.array([[6.0, 3.0], [3.0, 6.0]])
    z_coil = peec.impedance[0] + sample_loss_ohm

    # Tune each isolated port by cancelling its diagonal inductive reactance.
    # The off-diagonal mutual reactance remains and mixes the loaded channels.
    z_series = np.diag(-1j * np.imag(np.diag(z_coil)))
    network = ReceiverNetwork(
        frequency_hz=frequency_hz,
        coil_impedance_ohm=z_coil,
        series_impedance_ohm=z_series,
        load_impedance_ohm=50.0,
        temperature_k=293.15,
        noise_bandwidth_hz=noise_bandwidth_hz,
        preamp_voltage_noise_covariance_v2_per_hz=np.diag(
            [0.7e-9**2, 0.9e-9**2]
        ),
    )
    return RxArray(channels=channels, network=network), z_coil, peec.inductance[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixels", type=int, default=6)
    parser.add_argument("--frequency-mhz", type=float, default=2.0)
    parser.add_argument("--noise-bandwidth-khz", type=float, default=10.0)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pixels < 2:
        raise ValueError("--pixels must be at least 2")
    if args.frequency_mhz <= 0.0:
        raise ValueError("--frequency-mhz must be positive")
    if args.noise_bandwidth_khz <= 0.0:
        raise ValueError("--noise-bandwidth-khz must be positive")
    if args.noise_std < 0.0:
        raise ValueError("--noise-std must be non-negative")

    receivers, z_coil, inductance = _receiver_array(
        args.frequency_mhz * 1.0e6,
        args.noise_bandwidth_khz * 1.0e3,
    )
    experiment = Experiment(
        sequence=CPMGImaging(num_echoes=1, ny=1, maxoffs=0.1),
        sample=Sample(phantom=_phantom(args.pixels)),
        hardware=Hardware(
            rx_coil=receivers,
            plane=ImagingPlane(
                extent_m=(0.035, 0.035),
                plane="xy",
            ),
        ),
        acquisition=Acquisition(
            receiver_noise_std=args.noise_std,
            receiver_noise_seed=args.seed,
        ),
    )
    result = experiment.run().result

    geometric_rss = np.sqrt(
        np.sum(np.abs(result.geometric_receiver_sensitivities) ** 2, axis=0)
    )
    loaded_rss = np.sqrt(
        np.sum(np.abs(result.receiver_sensitivities) ** 2, axis=0)
    )
    image = (
        result.magnitude_noisy[..., 0]
        if result.magnitude_noisy is not None
        else result.magnitude[..., 0]
    )

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = (
        (np.abs(z_coil), r"$|Z_\mathrm{coil}|$ (ohm)", "magma"),
        (np.abs(result.receiver_transfer_matrix), r"$|H|$", "viridis"),
        (geometric_rss, "Geometric sensitivity RSS", "viridis"),
        (loaded_rss, "Loaded sensitivity RSS", "viridis"),
        (
            np.abs(result.receiver_network_noise_correlation),
            "Noise correlation magnitude",
            "magma",
        ),
        (image, "Noise-aware Roemer image", "gray"),
    )
    for axis, (values, title, cmap) in zip(axes.flat, panels):
        artist = axis.imshow(values, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("channel / x")
        axis.set_ylabel("channel / y")
        figure.colorbar(artist, ax=axis, shrink=0.8)

    coupling = inductance[0, 1] / np.sqrt(inductance[0, 0] * inductance[1, 1])
    figure.suptitle(
        f"Phase 4 coupled receiver network: "
        f"f={args.frequency_mhz:g} MHz, k={coupling:.3f}"
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)

    print(f"mutual coupling coefficient: {coupling:.6f}")
    print(
        "maximum off-diagonal loaded transfer: "
        f"{abs(result.receiver_transfer_matrix[0, 1]):.6f}"
    )
    print(
        "noise correlation magnitude: "
        f"{abs(result.receiver_network_noise_correlation[0, 1]):.6f}"
    )


if __name__ == "__main__":
    main()
