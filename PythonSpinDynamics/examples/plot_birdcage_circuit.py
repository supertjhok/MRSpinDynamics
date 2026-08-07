"""Compare birdcage circuit modes, tolerance splitting, and quadrature drive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import (
    BirdcageCircuit,
    BirdcageGeometry,
    birdcage_field_metrics,
    birdcage_quadrature_port_voltages,
    solve_birdcage_field,
    tuned_high_pass_birdcage,
    tuned_low_pass_birdcage,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot low/high-pass birdcage resonances, capacitor-tolerance "
            "mode splitting, and the resulting dual-port quadrature field."
        )
    )
    parser.add_argument("--frequency-mhz", type=float, default=63.87)
    parser.add_argument("--rungs", type=int, default=16)
    parser.add_argument("--cap-tolerance-percent", type=float, default=3.0)
    parser.add_argument("--grid", type=int, default=51)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _fundamental_frequency(circuit: BirdcageCircuit) -> float:
    family = circuit.modal_analysis().azimuthal_modes(1)
    return float(np.mean([mode.frequency_hz for mode in family]))


def _response(
    circuit: BirdcageCircuit,
    frequencies_hz: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.norm(
                circuit.solve_drive(frequency, source).currents.rung_currents_a
            )
            for frequency in frequencies_hz
        ]
    )


def main() -> None:
    args = _parse_args()
    if args.frequency_mhz <= 0.0:
        raise ValueError("--frequency-mhz must be positive")
    if args.rungs < 8 or args.rungs % 4:
        raise ValueError("--rungs must be at least 8 and divisible by four")
    if args.cap_tolerance_percent < 0.0:
        raise ValueError("--cap-tolerance-percent must be non-negative")
    if args.grid < 21:
        raise ValueError("--grid must be at least 21")

    plt = load_matplotlib(headless=args.output is not None)
    assert plt is not None

    target_hz = args.frequency_mhz * 1.0e6
    geometry = BirdcageGeometry(
        radius=0.15,
        length=0.30,
        n_rungs=args.rungs,
        ring_segments_per_section=8,
    )
    circuit_arguments = {
        "geometry": geometry,
        "resonance_frequency_hz": target_hz,
        "rung_inductance_h": 180.0e-9,
        "end_ring_inductance_h": 35.0e-9,
        "rung_resistance_ohm": 0.08,
        "end_ring_resistance_ohm": 0.015,
    }
    low_pass = tuned_low_pass_birdcage(**circuit_arguments)
    high_pass = tuned_high_pass_birdcage(**circuit_arguments)

    perturbed_capacitance = np.array(low_pass.rung_capacitance_f, copy=True)
    perturbed_capacitance[0] *= 1.0 + args.cap_tolerance_percent / 100.0
    perturbed = BirdcageCircuit(
        geometry=geometry,
        rung_inductance_h=low_pass.rung_inductance_h,
        end_ring_inductance_h=low_pass.end_ring_inductance_h,
        rung_capacitance_f=perturbed_capacitance,
        rung_resistance_ohm=low_pass.rung_resistance_ohm,
        end_ring_resistance_ohm=low_pass.end_ring_resistance_ohm,
    )

    source = birdcage_quadrature_port_voltages(
        geometry,
        voltage_v=1.0,
        handedness=1,
    )
    frequency_axis = np.linspace(0.985, 1.015, 501) * target_hz
    ideal_response = _response(low_pass, frequency_axis, source)
    perturbed_response = _response(perturbed, frequency_axis, source)
    ideal_drive = low_pass.solve_drive(target_hz, source, label="balanced dual-port")
    perturbed_drive = perturbed.solve_drive(
        _fundamental_frequency(perturbed),
        source,
        label="one capacitor high",
    )

    coordinate = np.linspace(-0.70, 0.70, args.grid) * geometry.radius
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    roi = xx**2 + yy**2 <= (0.4 * geometry.radius) ** 2
    outside = xx**2 + yy**2 >= (0.98 * geometry.radius) ** 2
    ideal_field = solve_birdcage_field(geometry, ideal_drive.currents, points)
    perturbed_field = solve_birdcage_field(
        geometry,
        perturbed_drive.currents,
        points,
    )
    ideal_metrics = birdcage_field_metrics(
        ideal_field,
        roi_mask=roi,
        target_handedness=1,
    )
    perturbed_metrics = birdcage_field_metrics(
        perturbed_field,
        roi_mask=roi,
        target_handedness=1,
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    low_analysis = low_pass.modal_analysis()
    high_analysis = high_pass.modal_analysis()
    for analysis, label, marker in (
        (low_analysis, "low-pass", "o"),
        (high_analysis, "high-pass", "s"),
    ):
        indices = [mode.dominant_azimuthal_index for mode in analysis.modes]
        axes[0, 0].scatter(
            indices,
            analysis.frequencies_hz * 1.0e-6,
            label=label,
            marker=marker,
            alpha=0.8,
        )
    axes[0, 0].set(
        xlabel="Dominant azimuthal mode m",
        ylabel="Resonance (MHz)",
        title="Uniform ladder eigenmodes",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        frequency_axis * 1.0e-6,
        ideal_response / np.max(ideal_response),
        label="uniform capacitors",
    )
    axes[0, 1].plot(
        frequency_axis * 1.0e-6,
        perturbed_response / np.max(perturbed_response),
        label=f"C[0] +{args.cap_tolerance_percent:g}%",
    )
    split_khz = perturbed.modal_analysis().splitting_hz(1) * 1.0e-3
    axes[0, 1].set(
        xlabel="Frequency (MHz)",
        ylabel="Normalized rung-current norm",
        title=f"Driven response; m=1 split = {split_khz:.1f} kHz",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    azimuth_deg = np.rad2deg(geometry.azimuth_rad)
    axes[0, 2].plot(
        azimuth_deg,
        np.abs(ideal_drive.currents.rung_currents_a),
        "o-",
        label="uniform",
    )
    axes[0, 2].plot(
        azimuth_deg,
        np.abs(perturbed_drive.currents.rung_currents_a),
        "s-",
        label="perturbed",
    )
    axes[0, 2].set(
        xlabel="Rung azimuth (degree)",
        ylabel="Current magnitude (A)",
        title="Dual-port current balance",
    )
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)

    extent_cm = tuple(
        value * 100.0
        for value in (
            coordinate[0],
            coordinate[-1],
            coordinate[0],
            coordinate[-1],
        )
    )
    field_scale = np.mean(np.abs(ideal_field.b1_plus_t)[roi])
    for axis, field, metrics, title in (
        (axes[1, 0], ideal_field, ideal_metrics, "Balanced circuit B1+"),
        (
            axes[1, 1],
            perturbed_field,
            perturbed_metrics,
            "Tolerance-perturbed B1+",
        ),
    ):
        normalized = np.ma.masked_where(
            outside,
            np.abs(field.b1_plus_t) / field_scale,
        )
        artist = axis.imshow(
            normalized.T,
            origin="lower",
            extent=extent_cm,
            cmap="viridis",
            vmin=0.8,
            vmax=1.2,
        )
        axis.contour(
            xx.T * 100.0,
            yy.T * 100.0,
            roi.T.astype(float),
            levels=(0.5,),
            colors="white",
            linewidths=0.8,
        )
        axis.set(
            xlabel="x (cm)",
            ylabel="y (cm)",
            title=(
                f"{title}\nCV={100 * metrics.coefficient_of_variation:.2f}%, "
                f"isolation={metrics.circularity_db:.1f} dB"
            ),
        )
        figure.colorbar(artist, ax=axis, label="B1+ / balanced ROI mean")

    ratio = np.abs(perturbed_field.b1_minus_t) / np.maximum(
        np.abs(perturbed_field.b1_plus_t),
        np.finfo(np.float64).tiny,
    )
    isolation = np.ma.masked_where(
        outside,
        -20.0 * np.log10(np.maximum(ratio, 1.0e-6)),
    )
    artist = axes[1, 2].imshow(
        isolation.T,
        origin="lower",
        extent=extent_cm,
        cmap="magma",
        vmin=20.0,
        vmax=60.0,
    )
    axes[1, 2].set(
        xlabel="x (cm)",
        ylabel="y (cm)",
        title="Perturbed circular-component isolation",
    )
    figure.colorbar(artist, ax=axes[1, 2], label="B1+ / B1- (dB)")
    figure.suptitle(
        "Birdcage circuit layer: modes → tolerance splitting → driven B1",
        fontsize=14,
    )

    print(
        "Low-pass rung capacitance: "
        f"{low_pass.rung_capacitance_f[0] * 1e12:.3f} pF"
    )
    print(
        "High-pass end-ring capacitance: "
        f"{high_pass.end_ring_capacitance_f[0] * 1e12:.3f} pF"
    )
    print(f"Perturbed m=1 splitting: {split_khz:.3f} kHz")
    print(
        "Balanced circuit Q: "
        f"{low_analysis.azimuthal_modes(1)[0].quality_factor:.1f}"
    )
    print(f"Balanced circular isolation: {ideal_metrics.circularity_db:.2f} dB")
    print(f"Perturbed circular isolation: {perturbed_metrics.circularity_db:.2f} dB")

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
