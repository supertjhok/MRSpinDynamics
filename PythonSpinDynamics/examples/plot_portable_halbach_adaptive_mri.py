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
    summarize_portable_halbach_design,
)


def _print_design_tables(design) -> None:
    rf = design.rf_coils
    grad = design.gradients
    rx = design.receiver
    weight = design.weight
    print("\nRF coils")
    print("coil  L (uH)  R (ohm)  Q       B1 center (mT/A)")
    print(
        f"Tx    {rf.transmit_inductance_h * 1e6:6.2f}  "
        f"{rf.transmit_ac_resistance_ohm:7.3f}  "
        f"{rf.transmit_q_factor:6.1f}  {rf.transmit_b1_center_t_per_a * 1e3:9.3f}"
    )
    print(
        f"Rx    {rf.receive_inductance_h * 1e6:6.2f}  "
        f"{rf.receive_loaded_resistance_ohm:7.3f}  "
        f"{rf.receive_loaded_q_factor:6.1f}  {rf.receive_b1_center_t_per_a * 1e3:9.3f}"
    )
    print("\nGradient and receiver requirements")
    print(f"Gx/Gz efficiency: {grad.gx_efficiency_t_per_m_per_a * 1e3:.2f} / "
          f"{grad.gz_efficiency_t_per_m_per_a * 1e3:.2f} mT/m/A")
    print(f"Peak gradient current / voltage: {grad.peak_current_a:.2f} A / "
          f"{grad.peak_voltage_v:.2f} V")
    print(f"Required receiver gain: {rx.required_gain_db:.1f} dB "
          f"for {rx.target_adc_peak_v:.2f} V ADC peak")
    print(f"System mass: {weight.total_kg:.1f} kg "
          f"({weight.portable_without_baseplate_kg:.1f} kg without baseplate)")


def _plot_design_dashboard(design, output: Path) -> None:
    pulse_us = design.pulse_sweep.pulse_lengths_s * 1e6
    sweep = design.pulse_sweep
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.8), constrained_layout=True)

    axes[0, 0].plot(pulse_us, sweep.peak_forward_power_w, "o-")
    axes[0, 0].axhline(200.0, color="tab:red", linestyle="--", label="200 W hardware")
    axes[0, 0].set(title="90° pulse power", xlabel="90° pulse length (µs)", ylabel="peak forward power (W)")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(pulse_us, sweep.predicted_snr, "o-", label="predicted")
    axes[0, 1].axhline(84.0, color="tab:red", linestyle="--", label="measured reference")
    axes[0, 1].set(title="Single-echo SNR", xlabel="90° pulse length (µs)", ylabel="SNR per scan")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(pulse_us, sweep.active_sample_volume_m3 * 1e6, "o-", color="tab:green")
    slice_axis = axes[0, 2].twinx()
    slice_axis.plot(pulse_us, sweep.effective_slice_thickness_m * 1e3, "s--", color="tab:purple")
    axes[0, 2].set(title="Excited sample", xlabel="90° pulse length (µs)", ylabel="active volume (mL)")
    slice_axis.set_ylabel("effective slice (mm)")
    axes[0, 2].grid(alpha=0.25)

    rf = design.rf_coils
    axes[1, 0].bar(
        ["Tx Q", "Rx copper Q", "Rx loaded Q"],
        [rf.transmit_q_factor, rf.receive_copper_q_factor, rf.receive_loaded_q_factor],
        color=["tab:orange", "tab:blue", "tab:cyan"],
    )
    axes[1, 0].set(title="RF loss budget", ylabel="quality factor")
    axes[1, 0].tick_params(axis="x", rotation=15)

    grad = design.gradients
    axes[1, 1].bar(
        ["current (A)", "voltage (V)", "peak I²R (W)"],
        [grad.peak_current_a, grad.peak_voltage_v, grad.peak_resistive_power_w],
        color=["tab:blue", "tab:orange", "tab:red"],
    )
    axes[1, 1].set(title="Gradient driver peak requirements")
    axes[1, 1].text(
        0.03,
        0.95,
        f"G/I = {grad.gx_efficiency_t_per_m_per_a * 1e3:.2f} mT/m/A\n"
        f"L = {grad.inductance_h * 1e6:.2f} µH, R = {grad.resistance_ohm:.2f} Ω",
        transform=axes[1, 1].transAxes,
        va="top",
    )

    weight = design.weight
    labels = ["magnet", "transmitter", "other elec.", "batteries", "baseplate"]
    masses = [
        weight.magnet_kg,
        weight.transmitter_kg,
        weight.other_electronics_kg,
        weight.batteries_kg,
        weight.baseplate_kg,
    ]
    axes[1, 2].bar(labels, masses, color="tab:gray")
    axes[1, 2].set(title=f"Mass budget: {weight.total_kg:.1f} kg total", ylabel="mass (kg)")
    axes[1, 2].tick_params(axis="x", rotation=25)
    axes[1, 2].text(
        0.03,
        0.95,
        f"Receiver gain: {design.receiver.required_gain_db:.1f} dB\n"
        f"ADC target: {design.receiver.target_adc_peak_v:.2f} V peak",
        transform=axes[1, 2].transAxes,
        va="top",
    )

    fig.suptitle("Portable Halbach MRI capstone — designer dashboard")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/portable_halbach_adaptive_mri.png"))
    parser.add_argument(
        "--design-output",
        type=Path,
        default=Path("results/portable_halbach_design_dashboard.png"),
    )
    parser.add_argument("--data-output", type=Path)
    args = parser.parse_args()

    result = simulate_portable_halbach_mri(
        PortableHalbachMRIConfig(matrix_size=args.matrix_size)
    )
    design = summarize_portable_halbach_design(result)
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
    _plot_design_dashboard(design, args.design_output)
    _print_design_tables(design)
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
            pulse_lengths_s=design.pulse_sweep.pulse_lengths_s,
            peak_rf_power_w=design.pulse_sweep.peak_forward_power_w,
            pulse_snr=design.pulse_sweep.predicted_snr,
            active_sample_volume_m3=design.pulse_sweep.active_sample_volume_m3,
            effective_slice_thickness_m=(
                design.pulse_sweep.effective_slice_thickness_m
            ),
        )
        print(f"saved {args.data_output}")


if __name__ == "__main__":
    main()
