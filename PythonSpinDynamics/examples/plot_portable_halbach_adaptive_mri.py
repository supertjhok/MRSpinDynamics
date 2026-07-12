"""Book-calibrated Halbach MRI, thermal drift, noisy CS, and auto-stop demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spin_dynamics.analysis.compressed_sensing import centered_ifft2
from spin_dynamics.workflows.portable_halbach import (
    PortableHalbachMRIConfig,
    simulate_portable_halbach_mri,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/portable_halbach_adaptive_mri.png"))
    parser.add_argument("--data-output", type=Path)
    args = parser.parse_args()

    result = simulate_portable_halbach_mri(
        PortableHalbachMRIConfig(matrix_size=args.matrix_size)
    )
    adaptive = result.adaptive
    zero_fill = centered_ifft2(
        np.where(adaptive.acquired_mask, result.measured_kspace, 0.0)
    )
    extent = np.array([-1, 1, -1, 1]) * result.config.field_of_view_m * 500.0

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    im = axes[0, 0].imshow(result.b1_map, origin="lower", extent=extent)
    spacing = result.config.field_of_view_m / (args.matrix_size - 1)
    gx_gradient = np.gradient(result.gx_field_map_t_per_a, spacing, axis=0)
    center = args.matrix_size // 2
    gradient_coordinates = (
        result.gx_field_map_t_per_a / gx_gradient[center, center]
    )
    axes[0, 0].contour(
        gradient_coordinates * 1e3,
        levels=np.linspace(-4, 4, 9),
        colors="white",
        linewidths=0.55,
        extent=extent,
    )
    axes[0, 0].set(title="Tx×Rx sensitivity + Gx contours", xlabel="z (mm)", ylabel="x (mm)")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.82)

    minutes = result.acquisition_times_s / 60.0
    axes[0, 1].plot(minutes, result.coil_temperature_k - 273.15, label="RF coil")
    axes[0, 1].plot(minutes, result.magnet_temperature_k - 273.15, label="ferrite")
    stop_minutes = result.stopped_time_s / 60.0
    axes[0, 1].axvline(stop_minutes, color="black", linestyle="--", label="auto-stop")
    axes[0, 1].set(title="Self-heating", xlabel="elapsed time (min)", ylabel="temperature (°C)")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(minutes, result.larmor_drift_hz / 1e3, color="tab:red", label="Larmor drift")
    gain_axis = axes[0, 2].twinx()
    gain_axis.plot(minutes, result.receiver_gain, color="tab:blue", label="receiver gain")
    axes[0, 2].axvline(stop_minutes, color="black", linestyle="--")
    axes[0, 2].set(title="Thermal detuning", xlabel="elapsed time (min)", ylabel="frequency shift (kHz)")
    gain_axis.set_ylabel("normalized gain")

    axes[1, 0].imshow(np.abs(zero_fill), cmap="gray", origin="lower", extent=extent)
    axes[1, 0].set(title=f"Noisy zero-fill ({100 * adaptive.sampling_fractions[-1]:.0f}% acquired)", xlabel="z (mm)", ylabel="x (mm)")
    axes[1, 1].imshow(np.abs(adaptive.image), cmap="gray", origin="lower", extent=extent)
    axes[1, 1].set(title=f"L1-Haar CS (NRMSE {result.reference_nrmse:.3f})", xlabel="z (mm)", ylabel="x (mm)")

    fraction = 100.0 * adaptive.sampling_fractions
    axes[1, 2].plot(fraction, adaptive.quality, "o-", label="held-out quality")
    axes[1, 2].axvline(fraction[-1], color="black", linestyle="--", label="auto-stop")
    axes[1, 2].set(title="Real-time stopping metric", xlabel="k-space acquired (%)", ylabel="1 / (1 + validation NRMSE)")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend(frameon=False)

    fig.suptitle(
        f"Low-cost Halbach MRI: loaded SNR={result.predicted_single_scan_snr:.0f} "
        f"(measured {result.measured_reference_snr:g}), "
        f"Q={result.receive_coil_q_factor:.0f}, "
        f"stopped after {stop_minutes:.1f} min ({result.adaptive.stop_reason})"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")
    if args.data_output is not None:
        args.data_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.data_output,
            image=adaptive.image,
            phantom=result.phantom,
            acquired_mask=adaptive.acquired_mask,
            sampling_fraction=adaptive.sampling_fractions,
            quality=adaptive.quality,
            validation_nrmse=adaptive.validation_nrmse,
            larmor_drift_hz=result.larmor_drift_hz,
            coil_temperature_k=result.coil_temperature_k,
            magnet_temperature_k=result.magnet_temperature_k,
        )
        print(f"saved {args.data_output}")


if __name__ == "__main__":
    main()
