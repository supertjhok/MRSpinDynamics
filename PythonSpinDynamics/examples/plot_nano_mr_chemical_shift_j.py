"""Plot coherent thermal chemical-shift/J spectra and optional DNP.

Run ``python examples/plot_nano_mr_chemical_shift_j.py --help`` for magnetic
field, acquisition, diffusion, clock, DNP, and output options.
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
    ClockModel,
    CoherentNMRSite,
    DNPModel,
    NuclearBathSpecies,
    dnp_polarization,
    simulate_coherent_nmr_spectrum,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-t", type=float, default=3.0)
    parser.add_argument("--duration-s", type=float, default=1.5)
    parser.add_argument("--sample-rate-hz", type=float, default=2400.0)
    parser.add_argument("--sample-t2-s", type=float, default=0.8)
    parser.add_argument("--diffusion-time-s", type=float, default=1.2)
    parser.add_argument("--clock-instability-ppb", type=float, default=0.0)
    parser.add_argument("--j-hz", type=float, default=7.0)
    parser.add_argument("--dnp-enhancement", type=float, default=100.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=2043)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    for name in (
        "field_t",
        "duration_s",
        "sample_rate_hz",
        "sample_t2_s",
        "diffusion_time_s",
        "temperature_k",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.clock_instability_ppb < 0.0:
        parser.error("--clock-instability-ppb must be non-negative")
    if args.j_hz == 0.0:
        parser.error("--j-hz must be non-zero")
    if args.dnp_enhancement == 0.0:
        parser.error("--dnp-enhancement must be non-zero")

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    count = max(256, int(round(args.duration_s * args.sample_rate_hz)))
    times = np.arange(count, dtype=np.float64) / args.sample_rate_hz
    proton = NuclearBathSpecies.from_isotope(
        "1H",
        polarization_mode="thermal",
        temperature_kelvin=args.temperature_k,
    )
    methyl_scale = proton.polarization_scaling(args.field_t, 3.0)
    methylene_scale = proton.polarization_scaling(args.field_t, 2.0)
    scale = max(
        abs(methyl_scale.coherent_mean_projection),
        np.finfo(float).tiny,
    )
    sites = (
        CoherentNMRSite(
            "1H",
            1.20,
            amplitude=methyl_scale.coherent_mean_projection / scale,
            transverse_relaxation_seconds=args.sample_t2_s,
            scalar_couplings_hz=(args.j_hz, args.j_hz),
            label="CH3 triplet",
        ),
        CoherentNMRSite(
            "1H",
            3.70,
            amplitude=methylene_scale.coherent_mean_projection / scale,
            transverse_relaxation_seconds=0.8 * args.sample_t2_s,
            scalar_couplings_hz=(args.j_hz, args.j_hz, args.j_hz),
            label="CH2 quartet",
        ),
    )
    reference_hz = (
        proton.gamma_hz_per_t
        * args.field_t
        * (1.0 + 2.45e-6)
    )
    ideal = simulate_coherent_nmr_spectrum(
        sites,
        args.field_t,
        times,
        reference_frequency_hz=reference_hz,
        window=True,
    )
    broadened = simulate_coherent_nmr_spectrum(
        sites,
        args.field_t,
        times,
        reference_frequency_hz=reference_hz,
        diffusion_correlation_seconds=args.diffusion_time_s,
        clock=ClockModel(
            interval_fractional_frequency_instability=(
                args.clock_instability_ppb * 1.0e-9
            ),
        ),
        seed=args.seed,
        window=True,
    )

    dnp_model = DNPModel(
        enhancement_factor=args.dnp_enhancement,
        buildup_time_seconds=0.8,
        nuclear_t1_seconds=3.0,
    )
    thermal_polarization = (
        proton.mean_spin_projection(args.field_t) / proton.spin
    )
    dnp_times = np.linspace(0.0, 4.0, 300)
    buildup = dnp_polarization(
        dnp_model,
        dnp_times,
        thermal_polarization,
    )
    decay = dnp_polarization(
        dnp_model,
        dnp_times,
        thermal_polarization,
        pumping=False,
        initial_polarization=buildup.steady_state_polarization,
    )

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    display = ideal.times_seconds <= min(0.15, ideal.times_seconds[-1])
    axes[0, 0].plot(
        ideal.times_seconds[display],
        ideal.fid.real[display],
        label="coherent thermal FID",
    )
    axes[0, 0].plot(
        broadened.times_seconds[display],
        broadened.fid.real[display],
        "--",
        label="with diffusion/clock",
    )
    axes[0, 0].set(
        xlabel="Acquisition time (s)",
        ylabel="In-phase signal (normalized)",
        title="First-order J modulation",
    )
    axes[0, 0].legend()

    offsets_ideal = ideal.frequencies_hz - ideal.reference_frequency_hz
    offsets_broadened = broadened.frequencies_hz - broadened.reference_frequency_hz
    mask_ideal = np.abs(offsets_ideal) <= 250.0
    mask_broad = np.abs(offsets_broadened) <= 250.0
    axes[0, 1].plot(
        offsets_ideal[mask_ideal],
        ideal.spectrum[mask_ideal] / np.max(ideal.spectrum[mask_ideal]),
        label="sample coherence only",
    )
    axes[0, 1].plot(
        offsets_broadened[mask_broad],
        broadened.spectrum[mask_broad] / np.max(broadened.spectrum[mask_broad]),
        "--",
        label="plus diffusion/clock",
    )
    axes[0, 1].set(
        xlabel="Frequency offset from 2.45 ppm (Hz)",
        ylabel="Normalized amplitude",
        title=f"Chemical shift and {args.j_hz:g} Hz scalar coupling",
    )
    axes[0, 1].legend()

    component_start = 0
    for site_index, site in enumerate(sites):
        center = (
            proton.gamma_hz_per_t
            * args.field_t
            * (1.0 + site.chemical_shift_ppm * 1.0e-6)
        )
        raw_offsets = np.array([0.0])
        for coupling in site.scalar_couplings_hz:
            raw_offsets = np.concatenate(
                (
                    raw_offsets - 0.5 * coupling,
                    raw_offsets + 0.5 * coupling,
                )
            )
        component_count = np.unique(np.round(raw_offsets, decimals=12)).size
        component_stop = component_start + component_count
        offsets = (
            ideal.component_frequencies_hz[component_start:component_stop]
            - center
        )
        weights = np.abs(
            ideal.component_amplitudes[component_start:component_stop]
        )
        axes[1, 0].stem(
            offsets,
            weights,
            linefmt=f"C{site_index}-",
            markerfmt=f"C{site_index}o",
            basefmt=" ",
            label=site.label,
        )
        component_start = component_stop
    axes[1, 0].set(
        xlabel="Offset from each chemical-shift center (Hz)",
        ylabel="First-order line weight",
        title="Triplet and quartet component inventory",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        dnp_times,
        buildup.polarization,
        label="microwave pumping on",
    )
    axes[1, 1].plot(
        dnp_times,
        decay.polarization,
        "--",
        label="pumping off: nuclear T1",
    )
    axes[1, 1].axhline(
        thermal_polarization,
        color="0.45",
        ls=":",
        label="thermal polarization",
    )
    axes[1, 1].set(
        xlabel="Preparation / relaxation time (s)",
        ylabel="Nuclear polarization",
        title=(
            "Optional DNP is explicit preparation\n"
            f"steady state {buildup.steady_state_polarization:.3g}"
        ),
    )
    axes[1, 1].legend()

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
