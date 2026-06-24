"""Diffusive diffraction from a 2D circular pore with random-walker PGSE.

This extends the slab-pore restricted-diffusion example to a genuinely
two-dimensional geometry: diffusion confined to a disc with a reflecting wall.
Confinement is supplied by ``make_circular_reflector``, a curved-boundary
callback now accepted by the motion engine, so the walkers bounce off the
circular pore wall instead of a rectangular box.

In the narrow-pulse, long-mixing limit (the "q-space" regime of Callaghan), the
normalized echo no longer decays monotonically. Once spins have fully sampled
the pore the echo approaches the squared magnitude of the pore form factor, so
``E(q)`` develops *diffusive diffraction* minima at the zeros of the disc
structure factor. For a uniform disc of radius ``a`` the form factor is the
2D Fourier transform of a top-hat,

    E(q) -> |2 * J1(q_ang * a) / (q_ang * a)|^2 ,

with minima at the Bessel-function zeros ``q_ang * a = 3.83, 7.02, 10.17, ...``.

Definition of q (note the factor-of-2pi ambiguity in the literature):

* Angular wavenumber  ``q_ang = gamma * G * delta``      [rad/m].
  The diffusion phase a spin accrues across one gradient lobe is
  ``phi = q_ang * x``, and the b-value is ``b = q_ang^2 * (Delta - delta/3)``.
* Reciprocal-space wavenumber (Callaghan)  ``q = q_ang / (2*pi)``   [1/m],
  defined so the encoding kernel is ``exp(i 2*pi*q*x)`` and a pore of size ``a``
  produces its first diffraction feature near ``q * a ~ 1``.

This script plots the x-axis in the Callaghan ``q = gamma*G*delta/(2*pi)``
convention and marks the disc-structure-factor minima.

Two methods are available via ``--method``:

* ``walker-sweep`` (default): re-run the full PGSE walker simulation at every q.
  Slow, but it includes finite-gradient-pulse blurring of the fringes.
* ``sgp-propagator``: run the confined diffusion once and obtain ``E(q)`` for all
  q analytically from each walker's net displacement. Tens of times faster; it
  is the ideal narrow-pulse (``delta -> 0``) limit and agrees with the sweep when
  ``delta << a^2 / D``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


GAMMA = 2.675e8  # rad/(s*T), proton gyromagnetic ratio
D_FREE = 2.3e-9  # m^2/s, bulk water at room temperature

# Zeros of the first-order Bessel function J1 -> disc diffraction minima in
# the angular product q_ang * a.
J1_ZEROS = np.array([3.8317, 7.0156, 10.1735])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-walker PGSE inside a 2D circular pore, showing diffusive "
            "diffraction minima in the echo attenuation E(q)."
        )
    )
    parser.add_argument(
        "--pore-radius",
        type=float,
        default=5.0e-6,
        help="Circular pore radius a (m).",
    )
    parser.add_argument(
        "--diffusion-time",
        type=float,
        default=80.0e-3,
        help="Diffusion time Delta (s). Must be long vs a^2/D for sharp fringes.",
    )
    parser.add_argument(
        "--gradient-duration",
        type=float,
        default=0.4e-3,
        help="Gradient-pulse duration delta (s). Keep short vs a^2/D (narrow pulse).",
    )
    parser.add_argument(
        "--num-q",
        type=int,
        default=28,
        help="Number of q samples in the diffraction sweep.",
    )
    parser.add_argument(
        "--max-qa",
        type=float,
        default=11.0,
        help="Largest dimensionless q_ang*a probed (11 reaches the first 2-3 minima).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=21,
        help="Spatial cells per axis used to seed walkers across the disc.",
    )
    parser.add_argument(
        "--walkers-per-cell",
        type=int,
        default=28,
        help="Random walkers per spatial cell. Higher deepens the fringe minima.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=80,
        help="Diffusion substeps per interval (refine so the per-step hop << a).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed. Reused across q so the diffraction curve is smooth.",
    )
    parser.add_argument(
        "--method",
        choices=["walker-sweep", "sgp-propagator"],
        default="walker-sweep",
        help=(
            "walker-sweep re-runs the full PGSE walker simulation per gradient "
            "(slow, includes finite-pulse effects). sgp-propagator runs the "
            "confined diffusion once and computes E(q) for all q analytically "
            "from the net displacement (fast, ideal narrow-pulse limit)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _bessel_j1():
    """Return scipy's J1 if available, else None (theory overlay is skipped)."""

    try:
        from scipy.special import j1
    except ImportError:
        return None
    return j1


def _disc_density(radius: float, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a uniform-disc spin density on a square grid spanning [-a, a]."""

    axis = np.linspace(-radius, radius, int(grid))
    xx, zz = np.meshgrid(axis, axis, indexing="ij")
    rho = (xx**2 + zz**2 <= radius**2).astype(np.float64)
    return rho, axis, axis


def _run_disc_pgse(
    *,
    gradient_amplitude: float,
    args: argparse.Namespace,
    rho: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
):
    """One random-walker PGSE experiment inside the reflecting circular pore."""

    from spin_dynamics.motion import make_circular_reflector, make_motion_field_maps_2d
    from spin_dynamics.workflows import run_pgse_walkers

    reflector = make_circular_reflector((0.0, 0.0), args.pore_radius)
    fields = make_motion_field_maps_2d(x_axis, z_axis)

    return run_pgse_walkers(
        rho=rho,
        x_axis=x_axis,
        z_axis=z_axis,
        fields=fields,
        gradient_amplitude=float(gradient_amplitude),
        gradient_duration=args.gradient_duration,
        diffusion_time=args.diffusion_time,
        diffusion_coefficient=D_FREE,
        gamma=GAMMA,
        gradient_axis="x",
        walkers_per_cell=int(args.walkers_per_cell),
        seed=int(args.seed),
        jitter=True,
        boundary=reflector,
        substeps_per_interval=int(args.substeps),
    )


def _walker_sweep(args: argparse.Namespace):
    """Echo magnitude versus q from a full PGSE walker run per gradient.

    Faithful but slow: it re-simulates the confined ensemble (RF pulses, finite
    gradient lobes, reflection) for every q, so it captures finite-pulse blurring
    of the fringes. Use ``--method sgp-propagator`` for the fast alternative.
    """

    rho, x_axis, z_axis = _disc_density(args.pore_radius, args.grid)

    q_ang = np.linspace(0.0, args.max_qa / args.pore_radius, int(args.num_q))
    gradients = q_ang / (GAMMA * args.gradient_duration)

    echo = np.zeros_like(gradients)
    baseline_positions = None
    baseline_weights = None
    for index, gradient in enumerate(gradients):
        result = _run_disc_pgse(
            gradient_amplitude=float(gradient),
            args=args,
            rho=rho,
            x_axis=x_axis,
            z_axis=z_axis,
        )
        echo[index] = float(np.abs(result.signal[0]))
        if index == 0:  # q = 0 baseline; reuse its walker cloud for the pore map.
            final = result.sequence.final_ensemble
            baseline_positions = final.positions
            baseline_weights = final.weights

    attenuation = echo / max(echo[0], np.finfo(float).eps)
    return q_ang, attenuation, baseline_positions, baseline_weights


def _propagator_sweep(args: argparse.Namespace):
    """Fast q-space method: one confined-diffusion run, all q computed analytically.

    In the short-gradient-pulse (narrow-pulse) limit the PGSE echo is the Fourier
    transform of the average propagator,
    ``E(q) = |<exp(i q_ang * (x_final - x_initial))>|``. Because the walker
    trajectories do not depend on q, a single confined-diffusion run over the
    diffusion time yields every walker's net displacement, and ``E(q)`` for all q
    follows from one vectorized sum -- tens of times faster than re-running the
    ensemble per gradient. This models the ideal ``delta -> 0`` limit, so unlike
    the walker sweep it omits finite-pulse blurring; the two agree when
    ``delta << a^2 / D``.
    """

    from spin_dynamics.motion import (
        advect_diffuse_positions,
        initialize_ensemble_from_density,
        make_circular_reflector,
    )

    rho, x_axis, z_axis = _disc_density(args.pore_radius, args.grid)
    ensemble = initialize_ensemble_from_density(
        rho,
        x_axis,
        z_axis,
        walkers_per_cell=int(args.walkers_per_cell),
        diffusion_coefficient=D_FREE,
        seed=int(args.seed),
        jitter=True,
    )
    reflector = make_circular_reflector((0.0, 0.0), args.pore_radius)
    bounds = (
        (-args.pore_radius, args.pore_radius),
        (-args.pore_radius, args.pore_radius),
    )

    # Pick enough substeps that the per-step hop stays well below the pore.
    target_hop = args.pore_radius / 6.0
    n_steps = max(1, int(np.ceil(2.0 * D_FREE * args.diffusion_time / target_hop**2)))
    dt = args.diffusion_time / n_steps
    print(f"propagator method: {n_steps} diffusion substeps "
          f"(per-step hop {np.sqrt(2.0 * D_FREE * dt) * 1e6:.2f} um)")

    rng = np.random.default_rng(int(args.seed))
    start_x = ensemble.positions[:, 0].copy()
    positions = ensemble.positions
    for _ in range(n_steps):
        positions = advect_diffuse_positions(
            positions,
            dt,
            diffusion_coefficient=D_FREE,
            rng=rng,
            bounds=bounds,
            boundary=reflector,
        )
    displacement = positions[:, 0] - start_x
    weights = ensemble.weights

    q_ang = np.linspace(0.0, args.max_qa / args.pore_radius, int(args.num_q))
    signal = np.exp(1j * q_ang[:, None] * displacement[None, :]) @ weights
    attenuation = np.abs(signal) / max(float(weights.sum()), np.finfo(float).eps)
    return q_ang, attenuation, positions, weights


def _plot_results(
    plt,
    *,
    args: argparse.Namespace,
    q_ang: np.ndarray,
    attenuation: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
):
    j1 = _bessel_j1()
    radius_um = args.pore_radius * 1e6
    # Callaghan reciprocal-space q = gamma*G*delta/(2*pi), shown in 1/um.
    q_callaghan_per_um = q_ang / (2.0 * np.pi) * 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A: walkers confined to the disc (form factor of this is the fringes).
    edges = np.linspace(-radius_um * 1.05, radius_um * 1.05, 70)
    hist = axes[0].hist2d(
        positions[:, 0] * 1e6,
        positions[:, 1] * 1e6,
        bins=edges,
        weights=weights,
        cmap="magma",
    )
    circle = plt.Circle(
        (0.0, 0.0), radius_um, fill=False, color="white", linewidth=1.4, linestyle="--"
    )
    axes[0].add_patch(circle)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("z (um)")
    axes[0].set_title(f"Walkers in disc, a = {radius_um:.1f} um")
    fig.colorbar(hist[3], ax=axes[0], fraction=0.046, pad=0.04, label="spin density")

    # Panel B: diffusive diffraction E(q) with the disc form-factor theory.
    method_label = (
        "SGP propagator" if args.method == "sgp-propagator" else "walker sweep"
    )
    axes[1].semilogy(
        q_callaghan_per_um, attenuation, "o-", color="#1f77b4", markersize=4,
        label=method_label,
    )
    if j1 is not None:
        x = np.maximum(q_ang * args.pore_radius, 1e-9)
        theory = (2.0 * j1(x) / x) ** 2
        axes[1].semilogy(
            q_callaghan_per_um, theory, "k--", linewidth=1.2,
            label="|2 J1(qa)/(qa)|^2",
        )
    for zero in J1_ZEROS:
        if zero <= args.max_qa:
            q_min = zero / (2.0 * np.pi * args.pore_radius) * 1e-6
            axes[1].axvline(q_min, color="0.6", linestyle=":", linewidth=1.0)
            axes[1].text(
                q_min, 1.4, f"qa={zero:.1f}", rotation=90,
                va="bottom", ha="center", fontsize="x-small", color="0.4",
            )
    axes[1].set_ylim(1e-3, 2.0)
    axes[1].set_xlabel("q = gamma G delta / 2pi  (1/um)")
    axes[1].set_ylabel("E(q) = |S(q)| / |S(0)|")
    axes[1].set_title(f"Diffusive diffraction, Delta = {args.diffusion_time * 1e3:.0f} ms")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize="small", loc="upper right")

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    if args.method == "sgp-propagator":
        q_ang, attenuation, positions, weights = _propagator_sweep(args)
    else:
        q_ang, attenuation, positions, weights = _walker_sweep(args)

    averaging_ratio = args.diffusion_time / (args.pore_radius**2 / D_FREE)
    print(f"method: {args.method}")
    print(f"pore radius a: {args.pore_radius * 1e6:.1f} um")
    print(f"motional-averaging ratio Delta / (a^2/D): {averaging_ratio:.1f} (want >> 1)")
    print(f"narrow-pulse ratio delta / (a^2/D): "
          f"{args.gradient_duration / (args.pore_radius**2 / D_FREE):.3f} (want << 1)")
    first_min = J1_ZEROS[0] / (2.0 * np.pi * args.pore_radius) * 1e-6
    print(f"predicted first diffraction minimum: q = {first_min:.3f} 1/um "
          f"(q_ang*a = {J1_ZEROS[0]:.2f})")
    # Locate the first fringe: the deepest sample within a window around the
    # first Bessel zero, rather than the global argmin at the highest order.
    qa = q_ang * args.pore_radius
    window = (qa > 0.5 * J1_ZEROS[0]) & (qa < 0.5 * (J1_ZEROS[0] + J1_ZEROS[1]))
    if np.any(window):
        candidates = np.where(window)[0]
        dip = candidates[int(np.argmin(attenuation[candidates]))]
        print(f"observed first minimum: q = {q_ang[dip] / (2.0 * np.pi) * 1e-6:.3f} 1/um "
              f"(q_ang*a = {qa[dip]:.2f}), E = {attenuation[dip]:.3e}")
        print("free diffusion would instead give E ~ exp(-b D) -> 0 here; the "
              "non-monotonic fringe is the signature of restriction.")

    fig = _plot_results(
        plt,
        args=args,
        q_ang=q_ang,
        attenuation=attenuation,
        positions=positions,
        weights=weights,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
