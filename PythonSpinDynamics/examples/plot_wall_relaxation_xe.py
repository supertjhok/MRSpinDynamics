"""Plot microscopic wall-collision relaxation for 129Xe gas.

The model treats wall relaxation as a stochastic sequence of gas-wall
encounters. Kinetic theory sets the encounter rate,

    k_wall = p_accommodation * vbar * (S/V) / 4,

and each encounter applies a small spin depolarization channel. The resulting
Liouville generator is a Poisson jump process, not an assigned ``T1``. Run with
``--output wall_relaxation_xe.png`` to save, or omit it to show the figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.relaxation import (  # noqa: E402
    WallCollisionRelaxationModel,
    cube_surface_to_volume_per_m,
    cylinder_surface_to_volume_per_m,
    gas_mean_speed_m_per_s,
    liouville_superoperator,
    matrix_exponential,
    single_spin_matrices,
    sphere_surface_to_volume_per_m,
    wall_collision_rate_per_second,
)


GAMMA_129XE_HZ_PER_T = -11.777e6
MASS_129XE_AMU = 128.9047808611


@dataclass(frozen=True)
class WallRelaxationResult:
    """Wall-induced 129Xe collision rates and Liouville decays."""

    sizes_m: np.ndarray
    selected_diameters_m: np.ndarray
    times_seconds: np.ndarray
    temperature_kelvin: float
    mass_amu: float
    accommodation_probability: float
    depolarization_probability: float
    mean_speed_m_per_s: float
    cylinder_aspect: float
    offset_hz: float
    surface_to_volume: dict[str, np.ndarray]
    collision_rates_per_second: dict[str, np.ndarray]
    t1_seconds: dict[str, np.ndarray]
    transverse_decays: dict[float, np.ndarray]
    longitudinal_decays: dict[float, np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-k", type=float, default=295.0)
    parser.add_argument("--mass-amu", type=float, default=MASS_129XE_AMU)
    parser.add_argument(
        "--accommodation-probability",
        type=float,
        default=1.0,
        help="Fraction of kinetic wall encounters that sample the relaxing surface.",
    )
    parser.add_argument(
        "--depolarization-probability",
        type=float,
        default=1.0e-8,
        help="Spin depolarization probability for one relaxing wall encounter.",
    )
    parser.add_argument(
        "--sizes-mm",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0],
        help="Container diameter/edge sweep in mm.",
    )
    parser.add_argument(
        "--selected-diameters-mm",
        type=float,
        nargs="+",
        default=[2.0, 10.0, 50.0],
        help="Spherical-cell diameters shown in the Liouville decay panel.",
    )
    parser.add_argument(
        "--cylinder-aspect",
        type=float,
        default=2.0,
        help="Cylinder length divided by diameter.",
    )
    parser.add_argument(
        "--max-time-hours",
        type=float,
        default=6.0,
        help="Maximum time shown in the Liouville decay panel.",
    )
    parser.add_argument("--points", type=int, default=320)
    parser.add_argument(
        "--offset-hz",
        type=float,
        default=0.0,
        help="Rotating-frame offset for the transverse Liouville decay.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _model_for_sphere(
    args: argparse.Namespace,
    diameter_m: float,
) -> WallCollisionRelaxationModel:
    surface_to_volume = float(sphere_surface_to_volume_per_m(diameter_m))
    return WallCollisionRelaxationModel.from_geometry(
        0.5,
        surface_to_volume_per_m=surface_to_volume,
        temperature_kelvin=float(args.temperature_k),
        mass_amu=float(args.mass_amu),
        accommodation_probability=float(args.accommodation_probability),
        depolarization_probability=float(args.depolarization_probability),
    )


def _normalized_observable_decay(
    *,
    model: WallCollisionRelaxationModel,
    times_seconds: np.ndarray,
    offset_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    ops = single_spin_matrices(0.5)
    hamiltonian = 2.0 * np.pi * offset_hz * ops.iz
    generator = liouville_superoperator(hamiltonian, model)

    transverse0 = ops.ix.reshape(-1, order="F")
    longitudinal0 = ops.iz.reshape(-1, order="F")
    transverse_observable = ops.ix + 1j * ops.iy
    longitudinal_observable = ops.iz

    transverse = np.empty_like(times_seconds, dtype=np.float64)
    longitudinal = np.empty_like(times_seconds, dtype=np.float64)
    for idx, time_seconds in enumerate(times_seconds):
        propagator = matrix_exponential(generator, float(time_seconds))
        rho_t = (propagator @ transverse0).reshape(ops.ix.shape, order="F")
        rho_z = (propagator @ longitudinal0).reshape(ops.iz.shape, order="F")
        transverse[idx] = abs(np.trace(rho_t @ transverse_observable))
        longitudinal[idx] = float(np.real(np.trace(rho_z @ longitudinal_observable)))

    transverse /= transverse[0]
    longitudinal /= longitudinal[0]
    return transverse, longitudinal


def _simulate(args: argparse.Namespace) -> WallRelaxationResult:
    sizes_m = np.asarray(args.sizes_mm, dtype=np.float64) * 1.0e-3
    selected_m = np.asarray(args.selected_diameters_mm, dtype=np.float64) * 1.0e-3
    times_seconds = np.linspace(
        0.0,
        float(args.max_time_hours) * 3600.0,
        int(args.points),
    )
    mean_speed = float(gas_mean_speed_m_per_s(args.temperature_k, args.mass_amu))

    surface_to_volume = {
        "sphere": sphere_surface_to_volume_per_m(sizes_m),
        (
            f"cylinder L/D={float(args.cylinder_aspect):g}"
        ): cylinder_surface_to_volume_per_m(
            sizes_m,
            aspect=float(args.cylinder_aspect),
        ),
        "cube": cube_surface_to_volume_per_m(sizes_m),
    }
    collision_rates = {
        name: wall_collision_rate_per_second(
            sv,
            temperature_kelvin=float(args.temperature_k),
            mass_amu=float(args.mass_amu),
            accommodation_probability=float(args.accommodation_probability),
        )
        for name, sv in surface_to_volume.items()
    }
    t1_seconds = {
        name: np.divide(
            1.0,
            rates * float(args.depolarization_probability),
            out=np.full_like(rates, np.inf, dtype=np.float64),
            where=rates > 0.0,
        )
        for name, rates in collision_rates.items()
    }

    transverse_decays: dict[float, np.ndarray] = {}
    longitudinal_decays: dict[float, np.ndarray] = {}
    for diameter_m in selected_m:
        model = _model_for_sphere(args, float(diameter_m))
        transverse, longitudinal = _normalized_observable_decay(
            model=model,
            times_seconds=times_seconds,
            offset_hz=float(args.offset_hz),
        )
        transverse_decays[float(diameter_m)] = transverse
        longitudinal_decays[float(diameter_m)] = longitudinal

    return WallRelaxationResult(
        sizes_m=sizes_m,
        selected_diameters_m=selected_m,
        times_seconds=times_seconds,
        temperature_kelvin=float(args.temperature_k),
        mass_amu=float(args.mass_amu),
        accommodation_probability=float(args.accommodation_probability),
        depolarization_probability=float(args.depolarization_probability),
        mean_speed_m_per_s=mean_speed,
        cylinder_aspect=float(args.cylinder_aspect),
        offset_hz=float(args.offset_hz),
        surface_to_volume=surface_to_volume,
        collision_rates_per_second=collision_rates,
        t1_seconds=t1_seconds,
        transverse_decays=transverse_decays,
        longitudinal_decays=longitudinal_decays,
    )


def _plot(plt, result: WallRelaxationResult):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)

    for name, sv in result.surface_to_volume.items():
        axes[0].loglog(
            sv,
            result.collision_rates_per_second[name],
            marker="o",
            label=name,
        )
    axes[0].invert_xaxis()
    axes[0].set_xlabel("surface-to-volume ratio S/V (1/m)")
    axes[0].set_ylabel("wall encounters per second")
    axes[0].set_title("Kinetic Wall-Collision Rate")
    axes[0].legend(fontsize=8)

    for name, t1_seconds in result.t1_seconds.items():
        axes[1].loglog(
            result.sizes_m * 1.0e3,
            t1_seconds / 3600.0,
            marker="o",
            label=name,
        )
    axes[1].set_xlabel("diameter or edge length (mm)")
    axes[1].set_ylabel("T1 = T2 (hours)")
    axes[1].set_title("Relaxation from Per-Collision Spin Loss")
    axes[1].legend(fontsize=8)

    for diameter_m, transverse in result.transverse_decays.items():
        model = WallCollisionRelaxationModel.from_geometry(
            0.5,
            surface_to_volume_per_m=float(sphere_surface_to_volume_per_m(diameter_m)),
            temperature_kelvin=result.temperature_kelvin,
            mass_amu=result.mass_amu,
            accommodation_probability=result.accommodation_probability,
            depolarization_probability=result.depolarization_probability,
        )
        axes[2].semilogy(
            result.times_seconds / 3600.0,
            transverse,
            label=(
                f"{diameter_m * 1e3:g} mm sphere, "
                f"T2={model.t2_seconds / 3600.0:.2g} h"
            ),
        )
    axes[2].set_xlabel("time (hours)")
    axes[2].set_ylabel("normalized transverse signal")
    axes[2].set_title("Liouville Spin-1/2 Decay")
    axes[2].legend(fontsize=8)

    equivalent_rho_um_per_s = (
        result.mean_speed_m_per_s
        * result.accommodation_probability
        * result.depolarization_probability
        * 0.25
        * 1.0e6
    )
    fig.suptitle(
        "Microscopic wall-induced 129Xe relaxation "
        f"(p={result.depolarization_probability:g}, "
        f"rho_eq={equivalent_rho_um_per_s:.3g} um/s)"
    )
    return fig


def main() -> None:
    args = _parse_args()
    if args.temperature_k <= 0.0:
        raise SystemExit("--temperature-k must be positive")
    if args.mass_amu <= 0.0:
        raise SystemExit("--mass-amu must be positive")
    if args.accommodation_probability < 0.0 or args.accommodation_probability > 1.0:
        raise SystemExit("--accommodation-probability must be in [0, 1]")
    if args.depolarization_probability < 0.0 or args.depolarization_probability > 1.0:
        raise SystemExit("--depolarization-probability must be in [0, 1]")
    if args.cylinder_aspect <= 0.0:
        raise SystemExit("--cylinder-aspect must be positive")
    if args.max_time_hours <= 0.0:
        raise SystemExit("--max-time-hours must be positive")
    if args.points < 2:
        raise SystemExit("--points must be at least two")
    if any(value <= 0.0 for value in args.sizes_mm):
        raise SystemExit("--sizes-mm values must be positive")
    if any(value <= 0.0 for value in args.selected_diameters_mm):
        raise SystemExit("--selected-diameters-mm values must be positive")

    plt = load_matplotlib(headless=args.output is not None)
    result = _simulate(args)
    equivalent_rho = (
        result.mean_speed_m_per_s
        * result.accommodation_probability
        * result.depolarization_probability
        * 0.25
    )

    print("Microscopic wall-induced 129Xe relaxation")
    print(f"gamma/2pi: {GAMMA_129XE_HZ_PER_T / 1e6:.6g} MHz/T")
    print(f"temperature: {result.temperature_kelvin:.6g} K")
    print(f"mean speed: {result.mean_speed_m_per_s:.6g} m/s")
    print(
        "per-collision depolarization probability: "
        f"{result.depolarization_probability:.6g}"
    )
    print(f"equivalent surface relaxivity: {equivalent_rho * 1e6:.6g} um/s")
    for diameter_m in result.selected_diameters_m:
        model = _model_for_sphere(args, float(diameter_m))
        sphere_sv = float(sphere_surface_to_volume_per_m(diameter_m))
        print(
            f"  {diameter_m * 1e3:6.2f} mm sphere: "
            f"S/V={sphere_sv:.6g} 1/m, "
            f"k_wall={model.collision_rate_per_second:.6g} 1/s, "
            f"T1=T2={model.t1_seconds / 3600.0:.6g} h"
        )

    fig = _plot(plt, result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
