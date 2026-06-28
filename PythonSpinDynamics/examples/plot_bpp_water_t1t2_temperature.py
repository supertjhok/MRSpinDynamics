"""Plot estimated proton T1/T2 versus temperature from a computed tau_c.

This is a principled companion to ``plot_redfield_water_cpmg.py``. Instead of
guessing a rotational correlation time, it derives ``tau_c(T)`` from the
Stokes-Einstein-Debye hydrodynamic helper, using a shear-viscosity correlation,
and feeds the result into the BPP spectral-density model to estimate ``T1`` and
``T2`` versus temperature.

Two liquids are contrasted: water (small, roughly spherical, stick boundary) and
decane (a larger, elongated, flexible chain). Decane's bigger hydrodynamic volume
gives a several-fold longer ``tau_c`` and therefore faster relaxation, while its
non-spherical, internally mobile shape needs a sub-unity slip factor so the
stick-limit SED estimate is not overstated.

The absolute coupling scale is the intramolecular proton-proton dipolar constant
from the dominant H-H distance, so the predicted relaxation times are
order-of-magnitude physical rather than fitted. The location of any ``T1``
minimum is cross-checked against ``tau_c_from_t1_minimum``, which depends only on
the Larmor frequency.

Run with ``--output bpp_t1t2.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.relaxation import (  # noqa: E402
    BPP_T1_MINIMUM_OMEGA_TAU,
    bpp_relaxation_rates,
    dipolar_coupling_hz,
    stokes_einstein_debye_correlation_time,
    tau_c_from_t1_minimum,
)

GAMMA_1H_HZ_PER_T = 42.57747892e6

# Vogel-Tammann-Fulcher correlation for liquid water shear viscosity, valid
# roughly 273-373 K: eta(T) = A * exp(B / (T - C)) in Pa.s. The constants
# reproduce ~0.89 mPa.s at 298 K.
WATER_VISCOSITY_A_PA_S = 2.939e-5
WATER_VISCOSITY_B_K = 507.88
WATER_VISCOSITY_C_K = 149.3

# Arrhenius correlation for liquid decane shear viscosity,
# eta(T) = A * exp(B / T) in Pa.s, reproducing ~0.85 mPa.s at 298 K with an
# ~11.6 kJ/mol flow activation energy.
DECANE_VISCOSITY_A_PA_S = 7.9e-6
DECANE_VISCOSITY_B_K = 1394.0


def water_viscosity_pa_s(temperature_kelvin: np.ndarray) -> np.ndarray:
    """Return liquid water shear viscosity from the VTF correlation."""

    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    return WATER_VISCOSITY_A_PA_S * np.exp(
        WATER_VISCOSITY_B_K / (temperature - WATER_VISCOSITY_C_K)
    )


def decane_viscosity_pa_s(temperature_kelvin: np.ndarray) -> np.ndarray:
    """Return liquid decane shear viscosity from the Arrhenius correlation."""

    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    return DECANE_VISCOSITY_A_PA_S * np.exp(DECANE_VISCOSITY_B_K / temperature)


@dataclass(frozen=True)
class Molecule:
    """Hydrodynamic and dipolar inputs for one liquid species."""

    name: str
    hydrodynamic_radius_angstrom: float
    hh_distance_angstrom: float
    slip_factor: float
    viscosity_pa_s: Callable[[np.ndarray], np.ndarray]
    # Lowest valid temperature for the viscosity correlation (VTF pole), or
    # ``None`` when the correlation has no pole.
    viscosity_floor_kelvin: float | None = None


WATER = Molecule(
    name="water",
    hydrodynamic_radius_angstrom=1.45,  # small, roughly spherical
    hh_distance_angstrom=1.52,  # intramolecular H-H
    slip_factor=1.0,  # stick limit works for a small molecule
    viscosity_pa_s=water_viscosity_pa_s,
    viscosity_floor_kelvin=WATER_VISCOSITY_C_K,
)

DECANE = Molecule(
    name="decane",
    hydrodynamic_radius_angstrom=4.3,  # equivalent-sphere radius from molar volume
    hh_distance_angstrom=1.78,  # geminal CH2 protons
    slip_factor=0.30,  # elongated + flexible: slip and segmental motion
    viscosity_pa_s=decane_viscosity_pa_s,
    viscosity_floor_kelvin=None,
)

PRESETS = {molecule.name: molecule for molecule in (WATER, DECANE)}


@dataclass(frozen=True)
class MoleculeRelaxationResult:
    """Temperature sweep of computed tau_c and BPP relaxation times."""

    name: str
    temperature_kelvin: np.ndarray
    viscosity_pa_s: np.ndarray
    tau_c_seconds: np.ndarray
    t1_seconds: np.ndarray
    t2_seconds: np.ndarray
    larmor_hz: float
    coupling_scale_per_second2: float
    tau_c_at_t1_min_seconds: float
    t1_minimum_temperature_kelvin: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecules",
        nargs="+",
        choices=sorted(PRESETS),
        default=["water", "decane"],
        help="Which preset liquids to contrast.",
    )
    parser.add_argument("--temp-min-k", type=float, default=275.0)
    parser.add_argument("--temp-max-k", type=float, default=360.0)
    parser.add_argument("--points", type=int, default=220)
    parser.add_argument("--larmor-mhz", type=float, default=20.0)
    parser.add_argument(
        "--radius-a",
        type=float,
        default=None,
        help="Override the hydrodynamic radius (angstrom) for every species.",
    )
    parser.add_argument(
        "--slip-factor",
        type=float,
        default=None,
        help="Override the microviscosity/slip factor in (0, 1] for every species.",
    )
    parser.add_argument(
        "--viscosity-pa-s",
        type=float,
        default=None,
        help="Override the viscosity correlation with a constant value.",
    )
    parser.add_argument(
        "--hh-distance-a",
        type=float,
        default=None,
        help="Override the dominant H-H distance (angstrom) for every species.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _coupling_scale(hh_distance_angstrom: float) -> float:
    """Return the like-spin dipolar BPP scale from the H-H distance.

    With the package spectral-density normalization ``J(w) = 2 tau /
    (1 + w^2 tau^2)`` and ``R1 ~ J(w0) + 4 J(2 w0)``, the intramolecular
    homonuclear dipolar prefactor is ``(3/10) d^2``, where ``d`` is the dipolar
    coupling in rad/s. This captures the dominant intramolecular pair only.
    """

    coupling_hz = dipolar_coupling_hz(
        hh_distance_angstrom,
        gamma_a_hz_per_t=GAMMA_1H_HZ_PER_T,
        gamma_b_hz_per_t=GAMMA_1H_HZ_PER_T,
    )
    coupling_rad_per_s = 2.0 * np.pi * coupling_hz
    return 0.3 * coupling_rad_per_s**2


def simulate_molecule(
    molecule: Molecule,
    args: argparse.Namespace,
) -> MoleculeRelaxationResult:
    """Return the temperature sweep of tau_c and BPP T1/T2 for one species."""

    temperatures = np.linspace(args.temp_min_k, args.temp_max_k, int(args.points))
    if args.viscosity_pa_s is not None:
        viscosity = np.full_like(temperatures, float(args.viscosity_pa_s))
    else:
        viscosity = np.asarray(molecule.viscosity_pa_s(temperatures), dtype=np.float64)

    radius_angstrom = (
        molecule.hydrodynamic_radius_angstrom
        if args.radius_a is None
        else float(args.radius_a)
    )
    slip_factor = (
        molecule.slip_factor if args.slip_factor is None else float(args.slip_factor)
    )
    hh_distance_angstrom = (
        molecule.hh_distance_angstrom
        if args.hh_distance_a is None
        else float(args.hh_distance_a)
    )

    tau_c = stokes_einstein_debye_correlation_time(
        radius_angstrom * 1.0e-10,
        viscosity,
        temperatures,
        slip_factor=slip_factor,
    )

    larmor_hz = float(args.larmor_mhz) * 1.0e6
    omega = 2.0 * np.pi * larmor_hz
    coupling_scale = _coupling_scale(hh_distance_angstrom)
    rates = bpp_relaxation_rates(
        angular_frequency_rad_per_s=omega,
        correlation_time_seconds=tau_c,
        temperature_kelvin=temperatures,
        coupling_scale_per_second2=coupling_scale,
    )

    tau_c_at_t1_min = tau_c_from_t1_minimum(omega)
    tau_c_array = np.asarray(tau_c, dtype=np.float64)
    # The BPP T1 turnover is only a real minimum when the swept tau_c brackets
    # the turnover value; otherwise the system stays on one side of it (both
    # liquids here are in extreme narrowing, w0 tau_c << 0.6158).
    if tau_c_array.min() <= tau_c_at_t1_min <= tau_c_array.max():
        t1_min_idx = int(np.argmin(rates.t1_seconds))
        t1_minimum_temperature = float(temperatures[t1_min_idx])
    else:
        t1_minimum_temperature = None

    return MoleculeRelaxationResult(
        name=molecule.name,
        temperature_kelvin=temperatures,
        viscosity_pa_s=np.asarray(viscosity, dtype=np.float64),
        tau_c_seconds=tau_c_array,
        t1_seconds=rates.t1_seconds,
        t2_seconds=rates.t2_seconds,
        larmor_hz=larmor_hz,
        coupling_scale_per_second2=coupling_scale,
        tau_c_at_t1_min_seconds=tau_c_at_t1_min,
        t1_minimum_temperature_kelvin=t1_minimum_temperature,
    )


def _plot(plt, results: list[MoleculeRelaxationResult]):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    larmor_hz = results[0].larmor_hz

    for index, result in enumerate(results):
        color = f"C{index}"
        axes[0].semilogy(
            result.temperature_kelvin,
            result.tau_c_seconds * 1e12,
            color=color,
            label=result.name,
        )
        axes[1].semilogy(
            result.temperature_kelvin,
            result.t1_seconds,
            color=color,
            label=f"{result.name} T1",
        )
        axes[1].semilogy(
            result.temperature_kelvin,
            result.t2_seconds,
            color=color,
            ls="--",
            label=f"{result.name} T2",
        )
        if result.t1_minimum_temperature_kelvin is not None:
            axes[1].axvline(
                result.t1_minimum_temperature_kelvin,
                color=color,
                ls=":",
                lw=1.0,
            )
        axes[2].plot(
            result.temperature_kelvin,
            2.0 * np.pi * larmor_hz * result.tau_c_seconds,
            color=color,
            label=result.name,
        )

    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("tau_c (ps)")
    axes[0].set_title("Stokes-Einstein-Debye tau_c")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Relaxation time (s)")
    axes[1].set_title(f"BPP T1 (solid) / T2 (dashed) at {larmor_hz / 1e6:g} MHz")
    axes[1].legend(fontsize=8)

    axes[2].axhline(
        BPP_T1_MINIMUM_OMEGA_TAU,
        color="0.5",
        ls="--",
        lw=1.0,
        label=f"T1-min turnover ({BPP_T1_MINIMUM_OMEGA_TAU:.3g})",
    )
    axes[2].set_xlabel("Temperature (K)")
    axes[2].set_ylabel("w0 tau_c")
    axes[2].set_title("Motional regime")
    axes[2].legend(fontsize=8)

    fig.suptitle("Proton BPP relaxation from a computed correlation time")
    return fig


def _report(result: MoleculeRelaxationResult) -> None:
    ref_idx = int(np.argmin(np.abs(result.temperature_kelvin - 298.15)))
    omega = 2.0 * np.pi * result.larmor_hz
    min_omega_tau = float(omega * result.tau_c_seconds.min())
    max_omega_tau = float(omega * result.tau_c_seconds.max())
    print(f"[{result.name}] coupling scale: {result.coupling_scale_per_second2:.6g} 1/s^2")
    print(
        f"  near 298 K: tau_c = {result.tau_c_seconds[ref_idx] * 1e12:.3g} ps, "
        f"T1 = {result.t1_seconds[ref_idx]:.3g} s, "
        f"T2 = {result.t2_seconds[ref_idx]:.3g} s"
    )
    if result.t1_minimum_temperature_kelvin is not None:
        t1_min_idx = int(np.argmin(result.t1_seconds))
        print(
            "  T1 minimum: "
            f"{result.t1_seconds[t1_min_idx]:.3g} s at "
            f"{result.t1_minimum_temperature_kelvin:.4g} K"
        )
    elif max_omega_tau < BPP_T1_MINIMUM_OMEGA_TAU:
        print(
            "  no T1 minimum in range: extreme narrowing, "
            f"max w0 tau_c = {max_omega_tau:.3g} < {BPP_T1_MINIMUM_OMEGA_TAU:.4g}"
        )
    else:
        print(
            "  no T1 minimum in range: slow motion, "
            f"min w0 tau_c = {min_omega_tau:.3g} > {BPP_T1_MINIMUM_OMEGA_TAU:.4g}"
        )


def main() -> None:
    args = _parse_args()
    if args.points <= 1:
        raise SystemExit("--points must be greater than one")
    if args.temp_max_k <= args.temp_min_k:
        raise SystemExit("--temp-max-k must exceed --temp-min-k")
    if args.slip_factor is not None and args.slip_factor <= 0.0:
        raise SystemExit("--slip-factor must be positive")

    # De-duplicate while preserving the requested order.
    selected = list(dict.fromkeys(args.molecules))
    molecules = [PRESETS[name] for name in selected]
    for molecule in molecules:
        floor = molecule.viscosity_floor_kelvin
        if (
            floor is not None
            and args.viscosity_pa_s is None
            and args.temp_min_k <= floor
        ):
            raise SystemExit(
                f"--temp-min-k must exceed the {molecule.name} viscosity pole "
                f"({floor:g} K)"
            )

    plt = load_matplotlib(headless=args.output is not None)
    results = [simulate_molecule(molecule, args) for molecule in molecules]

    print("Proton BPP relaxation from Stokes-Einstein-Debye tau_c")
    print(f"Larmor frequency: {results[0].larmor_hz / 1e6:.6g} MHz")
    print(
        "T1 turnover would need w0 tau_c = "
        f"{BPP_T1_MINIMUM_OMEGA_TAU:.4g}, i.e. tau_c = "
        f"{results[0].tau_c_at_t1_min_seconds * 1e12:.4g} ps"
    )
    for result in results:
        _report(result)

    fig = _plot(plt, results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
