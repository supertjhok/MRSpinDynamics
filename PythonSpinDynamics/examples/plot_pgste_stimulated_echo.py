"""Pulsed-gradient stimulated echo (PGSTE): T1-limited long-time diffusion.

A stimulated echo splits the diffusion encoding across three 90-degree pulses:
the first gradient lobe encodes position, the second 90-degree pulse stores one
quadrature *along the longitudinal axis*, and after a long storage interval a
third 90-degree pulse reads it back for the second lobe. Because the encoded
magnetization is longitudinal during storage, the reachable diffusion time is
limited by ``T1`` instead of ``T2`` -- the whole reason PGSTE is used for slow
diffusion, large pores, and low-field / internal-gradient samples where ``T2``
is short.

The random-walker backend models the phase-cycled stimulated-echo pathway
(``run_pgste_walkers``), so two textbook features appear directly:

1. The diffusion attenuation still follows Stejskal-Tanner, ``E(b) ~ exp(-b D)``,
   but at **half** the spin-echo amplitude (the other coherence pathway is
   spoiled away). Panel 1 shows the PGSTE points on the ``0.5 exp(-b D)`` curve
   and the ordinary PGSE spin echo on ``exp(-b D)``.
2. At fixed ``b`` but increasing diffusion time, the PGSE spin echo decays with
   ``T2`` (transverse the whole time) while PGSTE decays only with ``T1``.
   Panel 2 sweeps the diffusion time with a short ``T2`` and long ``T1``: the
   spin echo collapses while the stimulated echo survives.

Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


GAMMA = 2.675e8  # rad/(s*T), proton gyromagnetic ratio
D_FREE = 2.3e-9  # m^2/s, bulk water at room temperature
GRADIENT_DURATION = 1.0e-3  # delta (s)
# A wide slab so the unwanted anti-echo pathway (which encodes ~2*q*x) dephases
# across the sample; free diffusion is translation-invariant so this is unbiased.
SLAB_WIDTH = 2.0e-3
EXCITATION_DURATION = 50.0e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-walker pulsed-gradient stimulated echo (PGSTE): the "
            "stimulated echo tracks exp(-b D) at half amplitude and reaches "
            "long diffusion times limited by T1 rather than T2."
        )
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=64,
        help="Spatial cells across the slab used to seed walkers along x.",
    )
    parser.add_argument(
        "--walkers-per-cell",
        type=int,
        default=32,
        help="Random walkers per spatial cell. Higher means smoother curves.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=8,
        help="Diffusion substeps per sequence interval.",
    )
    parser.add_argument(
        "--attenuation-diffusion-time",
        type=float,
        default=30.0e-3,
        help="Diffusion time Delta (s) for the E(b) attenuation panel.",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=400.0e-3,
        help="Longitudinal relaxation time T1 (s) for the diffusion-time panel.",
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=15.0e-3,
        help="Transverse relaxation time T2 (s) for the diffusion-time panel.",
    )
    parser.add_argument(
        "--fixed-b",
        type=float,
        default=4.0e7,
        help="b-value (s/m^2) held fixed while sweeping the diffusion time.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed. Reused across the sweeps for smooth curves.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _slab(num_cells: int):
    x_axis = np.linspace(-0.5 * SLAB_WIDTH, 0.5 * SLAB_WIDTH, int(num_cells))
    z_axis = np.array([-1.0e-6, 1.0e-6])
    rho = np.ones((x_axis.size, z_axis.size), dtype=np.float64)
    return rho, x_axis, z_axis


def _gradient_for_b(b_value: float, diffusion_time: float) -> float:
    """Gradient amplitude giving the requested rectangular-PGSE b-value."""

    moment = b_value / (diffusion_time - GRADIENT_DURATION / 3.0)
    return float(np.sqrt(max(moment, 0.0)) / (GAMMA * GRADIENT_DURATION))


def _attenuation_sweep(args: argparse.Namespace):
    """E(b) for the stimulated echo and the ordinary spin echo."""

    from spin_dynamics.workflows import run_pgse_walkers, run_pgste_walkers

    rho, x_axis, z_axis = _slab(args.num_cells)
    norm = float(rho.sum())
    delta_big = args.attenuation_diffusion_time
    gradients = np.linspace(0.05, 0.60, 10)

    b_values = np.zeros_like(gradients)
    e_ste = np.zeros_like(gradients)
    e_pgse = np.zeros_like(gradients)
    common = dict(
        rho=rho, x_axis=x_axis, z_axis=z_axis, gradient_duration=GRADIENT_DURATION,
        diffusion_time=delta_big, diffusion_coefficient=D_FREE,
        walkers_per_cell=args.walkers_per_cell, seed=args.seed, jitter=True,
        excitation_duration=EXCITATION_DURATION, substeps_per_interval=args.substeps,
    )
    for index, gradient in enumerate(gradients):
        ste = run_pgste_walkers(gradient_amplitude=float(gradient), **common)
        pgse = run_pgse_walkers(
            gradient_amplitude=float(gradient), refocusing_duration=100.0e-6, **common
        )
        b_values[index] = ste.b_value
        e_ste[index] = abs(ste.signal[0]) / norm
        e_pgse[index] = abs(pgse.signal[0]) / norm
    return b_values, e_ste, e_pgse


def _diffusion_time_sweep(args: argparse.Namespace):
    """Signal vs diffusion time at fixed b: T1-limited STE vs T2-limited PGSE."""

    from spin_dynamics.workflows import run_pgse_walkers, run_pgste_walkers

    rho, x_axis, z_axis = _slab(args.num_cells)
    norm = float(rho.sum())
    diffusion_times = np.linspace(6.0e-3, 130.0e-3, 10)

    s_ste = np.zeros_like(diffusion_times)
    s_pgse = np.zeros_like(diffusion_times)
    te_pgse = np.zeros_like(diffusion_times)
    ts_ste = np.zeros_like(diffusion_times)
    common = dict(
        rho=rho, x_axis=x_axis, z_axis=z_axis, gradient_duration=GRADIENT_DURATION,
        diffusion_coefficient=D_FREE, walkers_per_cell=args.walkers_per_cell,
        seed=args.seed, jitter=True, excitation_duration=EXCITATION_DURATION,
        t1_seconds=args.t1, t2_seconds=args.t2, substeps_per_interval=args.substeps,
    )
    for index, delta_big in enumerate(diffusion_times):
        gradient = _gradient_for_b(args.fixed_b, float(delta_big))
        ste = run_pgste_walkers(
            gradient_amplitude=gradient, diffusion_time=float(delta_big), **common
        )
        pgse = run_pgse_walkers(
            gradient_amplitude=gradient, diffusion_time=float(delta_big),
            refocusing_duration=100.0e-6, **common
        )
        s_ste[index] = abs(ste.signal[0]) / norm
        s_pgse[index] = abs(pgse.signal[0]) / norm
        te_pgse[index] = float(pgse.echo_times[0])
        ts_ste[index] = float(ste.storage_time)
    return diffusion_times, s_ste, s_pgse, te_pgse, ts_ste


def _plot_results(
    plt,
    *,
    args: argparse.Namespace,
    b_values: np.ndarray,
    e_ste: np.ndarray,
    e_pgse: np.ndarray,
    diffusion_times: np.ndarray,
    s_ste: np.ndarray,
    s_pgse: np.ndarray,
    te_pgse: np.ndarray,
    ts_ste: np.ndarray,
):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel 1: diffusion attenuation, spin echo vs stimulated echo.
    b_axis = b_values * 1e-9
    axes[0].semilogy(b_axis, e_pgse, "s", color="#1f77b4", markersize=5, label="PGSE spin echo")
    axes[0].semilogy(b_axis, e_ste, "o", color="#d62728", markersize=5, label="PGSTE stim. echo")
    fine = np.linspace(b_values[0], b_values[-1], 200)
    axes[0].semilogy(fine * 1e-9, np.exp(-fine * D_FREE), "-", color="#1f77b4",
                     linewidth=1.0, alpha=0.6, label="exp(-b D)")
    axes[0].semilogy(fine * 1e-9, 0.5 * np.exp(-fine * D_FREE), "--", color="#d62728",
                     linewidth=1.0, alpha=0.7, label="0.5 exp(-b D)")
    axes[0].set_xlabel("b (10^9 s/m^2)")
    axes[0].set_ylabel("E = |S| / M0")
    axes[0].set_title(f"Diffusion attenuation, Delta = {args.attenuation_diffusion_time*1e3:.0f} ms")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize="small")

    # Panel 2: signal vs diffusion time at fixed b -- T1 (STE) beats T2 (PGSE).
    delta_axis = diffusion_times * 1e3
    diff_atten = np.exp(-args.fixed_b * D_FREE)
    axes[1].semilogy(delta_axis, s_pgse, "s", color="#1f77b4", markersize=5, label="PGSE spin echo")
    axes[1].semilogy(delta_axis, s_ste, "o", color="#d62728", markersize=5, label="PGSTE stim. echo")
    axes[1].semilogy(delta_axis, diff_atten * np.exp(-te_pgse / args.t2), "-",
                     color="#1f77b4", linewidth=1.0, alpha=0.6,
                     label=f"exp(-TE/T2), T2={args.t2*1e3:.0f} ms")
    axes[1].semilogy(delta_axis, 0.5 * diff_atten * np.exp(-ts_ste / args.t1), "--",
                     color="#d62728", linewidth=1.0, alpha=0.7,
                     label=f"0.5 exp(-Ts/T1), T1={args.t1*1e3:.0f} ms")
    axes[1].set_ylim(1e-3, 1.2)
    axes[1].set_xlabel("diffusion time Delta (ms)")
    axes[1].set_ylabel("|S| / M0  (fixed b)")
    axes[1].set_title(f"Long-time reach at b = {args.fixed_b*1e-9:.3f} x10^9 s/m^2")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    b_values, e_ste, e_pgse = _attenuation_sweep(args)
    diffusion_times, s_ste, s_pgse, te_pgse, ts_ste = _diffusion_time_sweep(args)

    print(f"b range: {b_values[0]:.3e} to {b_values[-1]:.3e} s/m^2")
    print(f"spin-echo / stim-echo amplitude ratio (low b): "
          f"{e_pgse[0] / max(e_ste[0], 1e-9):.2f} (expected ~2.0)")
    print(f"at Delta = {diffusion_times[-1]*1e3:.0f} ms (T2={args.t2*1e3:.0f} ms, "
          f"T1={args.t1*1e3:.0f} ms): "
          f"PGSE |S|/M0 = {s_pgse[-1]:.2e}, PGSTE |S|/M0 = {s_ste[-1]:.2e}, "
          f"gain x{s_ste[-1] / max(s_pgse[-1], 1e-12):.0f}")

    fig = _plot_results(
        plt,
        args=args,
        b_values=b_values,
        e_ste=e_ste,
        e_pgse=e_pgse,
        diffusion_times=diffusion_times,
        s_ste=s_ste,
        s_pgse=s_pgse,
        te_pgse=te_pgse,
        ts_ste=ts_ste,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
