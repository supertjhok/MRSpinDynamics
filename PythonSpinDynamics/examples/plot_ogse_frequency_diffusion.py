"""Oscillating-gradient spin echo (OGSE): the frequency-resolved diffusion spectrum.

PGSE varies the diffusion *time*; OGSE instead varies the oscillation
*frequency* of a cosine-modulated gradient, reaching much shorter effective
diffusion times. The encoding power sits at ``omega = 2*pi*f``, so sweeping the
frequency maps the diffusion spectrum ``D(omega)``. In a restricted pore the
apparent diffusion coefficient rises from the low-frequency (long-time,
tortuosity-limited) value toward the bulk value as the frequency increases and
the spins no longer have time to feel the walls; free diffusion stays flat at
the bulk value.

This example uses the random-walker OGSE backend (``run_ogse_walkers``):

1. Waveform -- the cosine gradient lobes around the refocusing pulse.
2. D(omega) -- the apparent diffusion coefficient versus frequency for free
   diffusion and for reflecting slab pores of two widths; the smaller pore's
   restriction persists to higher frequency.

Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


GAMMA = 2.675e8  # rad/(s*T), proton gyromagnetic ratio
NUM_PERIODS = 2
SAMPLES_PER_PERIOD = 12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-walker OGSE: map the diffusion spectrum D(omega) by sweeping "
            "the oscillating-gradient frequency in restricted and free media."
        )
    )
    parser.add_argument("--slab-widths", type=str, default="4e-6,8e-6",
                        help="Comma-separated reflecting slab widths L (m).")
    parser.add_argument("--diffusion-coefficient", type=float, default=2.0e-9,
                        help="Bulk diffusion coefficient (m^2/s).")
    parser.add_argument("--freq-min", type=float, default=20.0,
                        help="Lowest oscillation frequency (Hz).")
    parser.add_argument("--freq-max", type=float, default=600.0,
                        help="Highest oscillation frequency (Hz).")
    parser.add_argument("--num-freqs", type=int, default=9,
                        help="Number of frequencies (log-spaced).")
    parser.add_argument("--b-target", type=float, default=3.0e8,
                        help="Target b-value (s/m^2) held fixed across frequency.")
    parser.add_argument("--num-cells", type=int, default=15,
                        help="Spatial cells across the slab seeding walkers.")
    parser.add_argument("--walkers-per-cell", type=int, default=160,
                        help="Random walkers per spatial cell.")
    parser.add_argument("--substeps", type=int, default=6,
                        help="Diffusion substeps per waveform sample.")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed, reused across frequencies.")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show the plot.")
    return parser.parse_args()


def _gradient_for_b(b_target: float, frequency: float) -> float:
    """Cosine-OGSE amplitude giving the target b: b = (gamma G / omega)^2 N / f."""

    omega = 2.0 * np.pi * frequency
    return float(omega / GAMMA * np.sqrt(b_target * frequency / NUM_PERIODS))


def _slab(width: float, num_cells: int):
    x = np.linspace(-0.5 * width, 0.5 * width, int(num_cells))
    z = np.array([-0.5e-6, 0.5e-6])
    rho = np.ones((x.size, z.size), dtype=np.float64)
    return rho, x, z


def _d_app_vs_frequency(args, frequencies, width):
    """Apparent diffusion coefficient vs frequency for a slab (or free if inf)."""

    from spin_dynamics.motion import make_motion_field_maps_2d
    from spin_dynamics.workflows import run_ogse_walkers

    restricted = np.isfinite(width)
    rho, x, z = _slab(width if restricted else 1.0e-3, args.num_cells)
    fields = make_motion_field_maps_2d(x, z) if restricted else None
    norm = float(rho.sum())

    d_app = np.zeros_like(frequencies)
    for index, frequency in enumerate(frequencies):
        result = run_ogse_walkers(
            rho=rho, x_axis=x, z_axis=z, fields=fields,
            gradient_amplitude=_gradient_for_b(args.b_target, float(frequency)),
            oscillation_frequency=float(frequency), num_periods=NUM_PERIODS,
            samples_per_period=SAMPLES_PER_PERIOD,
            diffusion_coefficient=args.diffusion_coefficient,
            walkers_per_cell=args.walkers_per_cell, seed=args.seed, jitter=True,
            excitation_duration=40.0e-6, refocusing_duration=80.0e-6,
            boundary="reflect", substeps_per_interval=args.substeps,
        )
        echo = abs(result.signal[0]) / norm
        d_app[index] = -np.log(max(echo, np.finfo(float).eps)) / result.b_value
    return d_app


def _waveform(frequency: float, refocusing_duration: float = 80.0e-6):
    """Reconstruct the OGSE gradient waveform g(t)/G for plotting."""

    period = 1.0 / frequency
    lobe_t = np.linspace(0.0, NUM_PERIODS * period, 400)
    lobe_g = np.cos(2.0 * np.pi * frequency * lobe_t)
    gap = refocusing_duration
    t = np.concatenate([
        lobe_t,
        lobe_t[-1] + np.array([0.0, gap]),
        lobe_t[-1] + gap + lobe_t,
    ])
    g = np.concatenate([lobe_g, np.array([0.0, 0.0]), lobe_g])
    return t, g, lobe_t[-1], gap


def _plot(plt, args, *, frequencies, widths, d_app_free, d_app_slabs):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # Panel 1: the OGSE waveform (two cosine lobes around the refocusing pulse).
    f_demo = float(np.sqrt(args.freq_min * args.freq_max))
    t, g, lobe_end, gap = _waveform(f_demo)
    axes[0].plot(t * 1e3, g, color="#1f77b4", linewidth=1.3)
    axes[0].axvspan(lobe_end * 1e3, (lobe_end + gap) * 1e3, color="0.85",
                    label="180 pulse")
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("g(t) / G")
    axes[0].set_title(f"OGSE waveform, f = {f_demo:.0f} Hz, {NUM_PERIODS} periods")
    axes[0].legend(fontsize="small", loc="upper right")
    axes[0].grid(True, alpha=0.25)

    # Panel 2: diffusion spectrum D(omega) -- restricted rises toward bulk.
    colors = ["#d62728", "#2ca02c", "#9467bd"]
    axes[1].semilogx(frequencies, d_app_free, "ks-", markersize=4, label="free")
    for (width, d_app), color in zip(zip(widths, d_app_slabs), colors):
        axes[1].semilogx(frequencies, d_app / args.diffusion_coefficient, "o-",
                         color=color, markersize=4, label=f"slab {width*1e6:.0f} um")
    axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_xlabel("oscillation frequency f (Hz)")
    axes[1].set_ylabel("D_app(f) / D_bulk")
    axes[1].set_title("Diffusion spectrum from OGSE")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize="small", loc="lower right")

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    widths = [float(w) for w in args.slab_widths.split(",") if w.strip()]
    frequencies = np.logspace(np.log10(args.freq_min), np.log10(args.freq_max),
                              int(args.num_freqs))

    d_app_free = _d_app_vs_frequency(args, frequencies, np.inf) / args.diffusion_coefficient
    d_app_slabs = [_d_app_vs_frequency(args, frequencies, w) for w in widths]

    print(f"frequency range: {frequencies[0]:.0f}-{frequencies[-1]:.0f} Hz")
    print(f"free D_app/D_bulk: {d_app_free.min():.2f}-{d_app_free.max():.2f} (flat ~1)")
    for width, d_app in zip(widths, d_app_slabs):
        ratio = d_app / args.diffusion_coefficient
        print(f"slab {width*1e6:.0f} um D_app/D_bulk: {ratio[0]:.2f} (low f) -> "
              f"{ratio[-1]:.2f} (high f)")

    fig = _plot(plt, args, frequencies=frequencies, widths=widths,
                d_app_free=d_app_free, d_app_slabs=d_app_slabs)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
