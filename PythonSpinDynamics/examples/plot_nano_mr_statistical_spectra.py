"""Plot multi-isotope statistical nano-NMR and sensor-depth scaling.

Run ``python examples/plot_nano_mr_statistical_spectra.py --help`` for field,
layer, depth, and output options.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nano_mr import (  # noqa: E402
    NuclearBathSpecies,
    SurfaceGeometry,
    UniformBathComponent,
    UniformNuclearLayer,
    diamond_nv_minus,
    simulate_statistical_spectrum,
)


def sample_layer(thickness_nm: float | None) -> UniformNuclearLayer:
    """Return an illustrative mixed proton/fluorine surface layer."""

    proton = NuclearBathSpecies.from_isotope(
        "1H",
        correlation_time_seconds=150.0e-6,
    )
    fluorine = NuclearBathSpecies.from_isotope(
        "19F",
        polarization_mode="thermal",
        temperature_kelvin=300.0,
        correlation_time_seconds=150.0e-6,
    )
    return UniformNuclearLayer(
        SurfaceGeometry([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        (
            UniformBathComponent(proton, 5.0e28),
            UniformBathComponent(fluorine, 2.5e28),
        ),
        thickness_nm=thickness_nm,
        label="mixed 1H/19F layer",
    )


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-mt", type=float, default=20.0)
    parser.add_argument("--depth-nm", type=float, default=5.0)
    parser.add_argument("--layer-thickness-nm", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=2401)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.field_mt <= 0.0:
        parser.error("--field-mt must be positive")
    if args.depth_nm <= 0.0:
        parser.error("--depth-nm must be positive")
    if args.layer_thickness_nm <= 0.0:
        parser.error("--layer-thickness-nm must be positive")
    if args.points < 101:
        parser.error("--points must be at least 101")

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    field_tesla = args.field_mt * 1.0e-3
    layer = sample_layer(args.layer_thickness_nm)
    larmor_hz = np.array(
        [
            component.species.gamma_hz_per_t * field_tesla
            for component in layer.components
        ]
    )
    span_hz = max(15.0e3, 0.2 * np.ptp(larmor_hz))
    frequencies_hz = np.linspace(
        np.min(larmor_hz) - span_hz,
        np.max(larmor_hz) + span_hz,
        args.points,
    )
    spectrum = simulate_statistical_spectrum(
        diamond_nv_minus(depth_nm=args.depth_nm),
        layer,
        [0.0, 0.0, field_tesla],
        2.0 * np.pi * frequencies_hz,
    )

    depths_nm = np.geomspace(2.0, 20.0, 80)
    variances_t2 = []
    for depth_nm in depths_nm:
        result = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=depth_nm),
            layer,
            [0.0, 0.0, field_tesla],
            np.array([-1.0, 1.0]),
        )
        variances_t2.append(result.total_field_variance_t2)
    variances_t2 = np.asarray(variances_t2)
    half_space = sample_layer(None)
    half_space_variances_t2 = []
    for depth_nm in depths_nm:
        result = simulate_statistical_spectrum(
            diamond_nv_minus(depth_nm=depth_nm),
            half_space,
            [0.0, 0.0, field_tesla],
            np.array([-1.0, 1.0]),
        )
        half_space_variances_t2.append(result.total_field_variance_t2)
    half_space_variances_t2 = np.asarray(half_space_variances_t2)

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    for index, isotope in enumerate(spectrum.isotopes):
        axes[0].plot(
            frequencies_hz * 1.0e-3,
            spectrum.component_psd_t2_s[index] * 1.0e18,
            label=isotope,
        )
        axes[0].axvline(larmor_hz[index] * 1.0e-3, color="0.6", lw=0.8, ls=":")
    axes[0].set(
        xlabel="Frequency (kHz)",
        ylabel=r"Two-sided field PSD (nT$^2$/Hz)",
        title=f"Mixed-isotope spectrum at {args.field_mt:g} mT",
    )
    axes[0].legend()

    axes[1].loglog(
        depths_nm,
        variances_t2 * 1.0e18,
        label=f"{args.layer_thickness_nm:g} nm layer",
    )
    axes[1].loglog(
        depths_nm,
        half_space_variances_t2 * 1.0e18,
        label="half-space",
    )
    reference = half_space_variances_t2[0] * (depths_nm[0] / depths_nm) ** 3
    axes[1].loglog(
        depths_nm,
        reference * 1.0e18,
        ":",
        color="0.35",
        label=r"$d^{-3}$ reference",
    )
    axes[1].set(
        xlabel="Sensor depth (nm)",
        ylabel=r"AC field variance (nT$^2$)",
        title="Planar-layer depth response",
    )
    axes[1].legend()

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
