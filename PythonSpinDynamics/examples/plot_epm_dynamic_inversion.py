"""Dynamic-inversion trap stability and actuator-architecture consequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows import (  # noqa: E402
    DynamicInversionHardwareConfig,
    FerromagneticParticle,
    assess_dynamic_inversion_hardware,
    nacev_2015_sequence,
    simulate_dynamic_inversion,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ferromagnetic-particle shape and size under the Nacev "
            "polarize-delay-antialigned-gradient sequence and compare fast "
            "coils, EPM switching, and a hybrid actuator."
        )
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Simulated trap duration per particle case (default: 60 s).",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=48,
        help="Particles in each seeded stability run (default: 48).",
    )
    parser.add_argument(
        "--polarizing-mt",
        type=float,
        default=500.0,
        help="Illustrative uniform polarizing field (default: 500 mT).",
    )
    parser.add_argument(
        "--gradient-field-mt",
        type=float,
        default=200.0,
        help="Illustrative gradient-source field at trap center (default: 200 mT).",
    )
    parser.add_argument("--seed", type=int, default=21, help="Brownian seed.")
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if not np.isfinite(args.duration_s) or args.duration_s < 1.0:
        raise ValueError("--duration-s must be at least 1")
    if args.particles < 8:
        raise ValueError("--particles must be at least 8")
    for name in ("polarizing_mt", "gradient_field_mt"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _particle(
    shape: str,
    length_m: float,
    diameter_m: float,
    label: str,
    *,
    internal_relaxation_time_s: float | None = None,
) -> FerromagneticParticle:
    return FerromagneticParticle(
        shape=shape,
        length_m=length_m,
        diameter_m=diameter_m,
        volume_susceptibility=0.65,
        saturation_magnetization_a_m=1.4e6,
        remanent_magnetization_a_m=1.0e6,
        fluid_viscosity_pa_s=0.7e-3,
        temperature_k=298.0,
        internal_relaxation_time_s=internal_relaxation_time_s,
        label=label,
    )


def main() -> None:
    args = _parser().parse_args()
    _validate(args)
    sequence = nacev_2015_sequence(
        polarizing_field_t=1e-3 * args.polarizing_mt,
        gradient_field_at_center_t=1e-3 * args.gradient_field_mt,
        actuator_radius_m=8e-3,
    )
    particles = (
        _particle("sphere", 2e-6, 2e-6, "2 um sphere"),
        _particle("rod", 2e-6, 0.25e-6, "2 x 0.25 um rod"),
        _particle("rod", 8e-6, 0.25e-6, "8 x 0.25 um rod"),
        _particle("rod", 20e-6, 1e-6, "20 x 1 um rod"),
        _particle("rod", 200e-6, 0.2e-6, "200 x 0.2 um rod"),
        _particle(
            "rod",
            8e-6,
            0.25e-6,
            "8 um rod, 100 ns internal",
            internal_relaxation_time_s=100e-9,
        ),
    )
    rng = np.random.default_rng(args.seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, args.particles)
    radii = np.sqrt(rng.uniform((1e-3) ** 2, (4e-3) ** 2, args.particles))
    initial = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    target_radius = 2e-3
    results = tuple(
        simulate_dynamic_inversion(
            sequence,
            particle,
            initial,
            duration_s=args.duration_s,
            target_radius_m=target_radius,
            bounds_m=((-5e-3, 5e-3), (-5e-3, 5e-3)),
            seed=args.seed + index,
            save_every_full_cycles=10,
        )
        for index, particle in enumerate(particles)
    )
    paper_particle = particles[4]
    hardware = tuple(
        assess_dynamic_inversion_hardware(
            sequence,
            paper_particle,
            duration_s=9.1 * 60.0,
            config=DynamicInversionHardwareConfig(
                architecture=architecture,
                epm_parallel_channels=18,
            ),
        )
        for architecture in ("coils", "epm", "hybrid")
    )

    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(results)))
    labels = [particle.label for particle in particles]
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.0), constrained_layout=True)
    fig.suptitle(
        "Dynamic-inversion trap: particle stability and actuator consequences",
        fontsize=15,
    )

    ax = axes[0, 0]
    scale = 1e6
    start = 0.0
    segments = (
        (sequence.polarizing_duration_s, 1.0, "polarize", "tab:purple"),
        (sequence.delay_s, 0.0, "delay", "0.6"),
        (sequence.gradient_duration_s, -0.75, "antialigned gradient", "tab:green"),
    )
    for width, height, label, color in segments:
        ax.fill_between(
            scale * np.asarray((start, start + width)),
            0.0,
            height,
            color=color,
            alpha=0.75,
            label=label,
        )
        start += width
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(
        xlabel="time within active sequence element (us)",
        ylabel="normalized field",
        title=(
            "reported timing; %.1f ms element period"
            % (1e3 * sequence.element_period_s)
        ),
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    lengths = np.geomspace(0.5e-6, 300e-6, 100)
    geometry_curves = (
        ("sphere", lambda length: length, "sphere"),
        ("rod", lambda length: length / 8.0, "rod, aspect 8"),
        ("rod", lambda _length: 0.25e-6, "rod, diameter 0.25 um"),
        ("rod", lambda _length: 1.0e-6, "rod, diameter 1 um"),
    )
    for (shape, diameter_for_length, label), color in zip(
        geometry_curves,
        plt.cm.viridis(np.linspace(0.1, 0.9, len(geometry_curves))),
        strict=True,
    ):
        memory = []
        for length in lengths:
            diameter = diameter_for_length(length)
            if shape == "rod" and length / diameter < 2.0:
                memory.append(np.nan)
                continue
            particle = _particle(shape, length, diameter, "sweep")
            memory.append(
                particle.orientation_memory_time_s(
                    sequence.gradient_field_at_center_t,
                    polarizing_field_t=sequence.polarizing_field_t,
                )
            )
        ax.loglog(
            1e6 * lengths,
            1e6 * np.asarray(memory),
            color=color,
            label=label,
        )
    ax.axhline(1e6 * sequence.gradient_duration_s, color="tab:red", ls="--", label="50 us gradient")
    ax.set(
        xlabel="particle length (um)",
        ylabel="field-driven orientation-memory time (us)",
        title="shape and size set the usable repulsion window",
    )
    ax.legend(fontsize=8)
    ax.grid(which="both", alpha=0.2)

    ax = axes[0, 2]
    gains = np.asarray([result.stability.concentration_gain for result in results])
    repulsive = np.asarray(
        [result.stability.repulsive_gradient_fraction for result in results]
    )
    x = np.arange(len(results))
    ax.bar(x, gains, color=colors, alpha=0.8)
    ax.axhline(1.0, color="black", lw=1)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="initial RMS radius / final RMS radius",
        title=f"{args.duration_s:g} s concentration gain (>1 contracts)",
    )
    ax.tick_params(axis="x", rotation=38, labelsize=8)
    repulsion_axis = ax.twinx()
    repulsion_axis.plot(x, 100 * repulsive, "ko--", ms=4, label="repulsive samples")
    repulsion_axis.set_ylabel("gradient samples still repulsive (%)")
    repulsion_axis.set_ylim(0.0, 105.0)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    for result, color in zip((results[4], results[5]), ("tab:blue", "tab:red"), strict=True):
        selected = np.arange(0, args.particles, max(1, args.particles // 16))
        for index in selected:
            path = 1e3 * result.positions_m[:, index]
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.38, lw=0.8)
        ax.scatter(
            1e3 * result.final_positions_m[:, 0],
            1e3 * result.final_positions_m[:, 1],
            s=10,
            color=color,
            alpha=0.6,
            label=result.particle.label,
        )
    ax.add_patch(Circle((0.0, 0.0), 1e3 * target_radius, fill=False, color="black", lw=1.5))
    for direction in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        ax.add_patch(
            FancyArrowPatch(
                (5.3 * direction[0], 5.3 * direction[1]),
                (4.2 * direction[0], 4.2 * direction[1]),
                arrowstyle="-|>",
                mutation_scale=10,
                color="0.35",
            )
        )
    ax.set(
        xlim=(-5.5, 5.5),
        ylim=(-5.5, 5.5),
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="rigid long rod contracts; rapid internal relaxation escapes",
    )
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    architectures = [item.config.architecture for item in hardware]
    pulse_counts = np.asarray(
        [
            item.coil_pulse_count + item.epm_channel_pulse_count
            for item in hardware
        ],
        dtype=float,
    )
    bars = ax.bar(architectures, pulse_counts, color=("tab:blue", "tab:orange", "tab:green"))
    ax.set_yscale("log")
    ax.set(
        ylabel="driver or channel pulses in 9.1 min",
        title="EPM-only multiplies retained-state programming",
    )
    for bar, item in zip(bars, hardware, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.12,
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(5.0, bar.get_height() / 25.0),
            "viable" if item.viable else "limited",
            ha="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    ax.grid(axis="y", which="both", alpha=0.2)

    ax = axes[1, 2]
    ax.axis("off")
    rows = []
    for item in hardware:
        rows.append(
            [
                item.config.architecture,
                f"{1e6 * item.field_transition_s:.0f}",
                f"{1e6 * item.effective_antialigned_delay_s:.0f}",
                f"{item.epm_retained_state_changes:,}",
                "yes" if item.waveform_fidelity_feasible else "no",
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=(
            "source",
            "transition\n(us)",
            "effective delay\n(us)",
            "EPM states",
            "pulse fidelity",
        ),
        cellLoc="center",
        loc="upper center",
        bbox=(0.0, 0.50, 1.0, 0.43),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    ax.text(
        0.0,
        0.42,
        "Hardware interpretation",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.35,
        "Coils: faithful fast waveform; repeated copper/driver loss.",
        fontsize=9,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.27,
        "EPM only: polarize, gradient, and off state every 60.6 ms;",
        fontsize=9,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.21,
        "programming transients are uncontrolled particle fields.",
        fontsize=9,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.13,
        "Hybrid: program EPM bias/shaping slowly; use coils for inversion.",
        fontsize=9,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.03,
        "Energy is intentionally unreported until coil R/L and EPM pulse energy are calibrated.",
        fontsize=8,
        color="0.3",
        transform=ax.transAxes,
    )

    print("Dynamic-inversion particle and hardware study")
    print(f"  sequence={sequence.provenance}")
    print(
        "  fields are illustrative, not reconstructed from the paper: "
        f"Bp={1e3 * sequence.polarizing_field_t:.1f} mT, "
        f"Bg(center)={1e3 * sequence.gradient_field_at_center_t:.1f} mT, "
        f"|grad Bg|(center)={sequence.center_gradient_t_m:.1f} T/m"
    )
    for result in results:
        stability = result.stability
        print(
            f"  {result.particle.label}: memory="
            f"{1e6 * result.particle.orientation_memory_time_s(sequence.gradient_field_at_center_t, polarizing_field_t=sequence.polarizing_field_t):.2f} us, "
            f"repulsive={100 * stability.repulsive_gradient_fraction:.1f}%, "
            f"gain={stability.concentration_gain:.4f}, "
            f"target={100 * stability.final_target_fraction:.1f}%"
        )
    for item in hardware:
        print(
            f"  {item.config.architecture}: viable={item.viable}, "
            f"coil pulses={item.coil_pulse_count:,}, "
            f"EPM states={item.epm_retained_state_changes:,}, "
            f"EPM channel pulses={item.epm_channel_pulse_count:,}; "
            f"{item.consequence}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
