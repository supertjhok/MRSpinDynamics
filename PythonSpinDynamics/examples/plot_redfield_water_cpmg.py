"""Plot proton CPMG decay in bulk water with shared Redfield relaxation.

This example treats one water proton as the observed spin-1/2 and the other
intramolecular proton as an isotropically tumbling dipolar bath source. The
ideal CPMG pulse train is propagated in the rotating frame while the Redfield
rates are built from the lab-frame Larmor splitting.

Run with ``--output redfield_water_cpmg.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling.evolution import propagator  # noqa: E402
from spin_dynamics.phase_cycling import cpmg_two_step_phase_cycle  # noqa: E402
from spin_dynamics.relaxation import (  # noqa: E402
    DipolarRelaxationSource,
    IsotropicLiquidMotionalAveraging,
    RedfieldDipolarRelaxationModel,
    liouville_hamiltonian,
    matrix_exponential,
    single_spin_matrices,
)


GAMMA_1H_HZ_PER_T = 42.57747892e6
WATER_HH_DISTANCE_ANGSTROM = 1.52


@dataclass(frozen=True)
class WaterCPMGResult:
    """Water proton CPMG echo trains for a tau_c sweep."""

    echo_times_seconds: np.ndarray
    tau_c_seconds: np.ndarray
    echo_trains: np.ndarray
    fitted_t2_seconds: np.ndarray
    larmor_hz: float
    echo_spacing_seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--larmor-mhz", type=float, default=20.0)
    parser.add_argument("--echo-spacing-ms", type=float, default=20.0)
    parser.add_argument(
        "--echo-spacing-seconds",
        type=float,
        default=None,
        help="CPMG echo spacing in seconds; overrides --echo-spacing-ms.",
    )
    parser.add_argument("--num-echoes", type=int, default=64)
    parser.add_argument(
        "--tau-c-ps",
        type=float,
        nargs="+",
        default=[1.5, 2.5, 5.0],
        help="Isotropic rotational correlation times to compare, in ps.",
    )
    parser.add_argument(
        "--correlation-time-seconds",
        type=float,
        nargs="+",
        default=None,
        help="Correlation times in seconds; overrides --tau-c-ps.",
    )
    parser.add_argument("--hh-distance-a", type=float, default=WATER_HH_DISTANCE_ANGSTROM)
    parser.add_argument(
        "--offset-hz",
        type=float,
        default=0.0,
        help="Rotating-frame resonance offset included in the CPMG sequence.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _fit_t2(echo_times: np.ndarray, amplitudes: np.ndarray) -> float:
    magnitude = np.abs(np.asarray(amplitudes, dtype=np.complex128))
    mask = magnitude > max(float(np.max(magnitude)) * 1.0e-8, 1.0e-300)
    if np.count_nonzero(mask) < 2:
        return np.inf
    slope, _ = np.polyfit(echo_times[mask], np.log(magnitude[mask]), 1)
    return -1.0 / slope if slope < 0.0 else np.inf


def _simulate_one(
    *,
    larmor_hz: float,
    echo_spacing_seconds: float,
    num_echoes: int,
    tau_c_seconds: float,
    hh_distance_angstrom: float,
    offset_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    ops = single_spin_matrices(0.5)
    lab_hamiltonian = 2.0 * np.pi * larmor_hz * ops.iz
    rotating_hamiltonian = 2.0 * np.pi * offset_hz * ops.iz
    source = DipolarRelaxationSource(
        vector_angstrom=(hh_distance_angstrom, 0.0, 0.0),
        gamma_target_hz_per_t=GAMMA_1H_HZ_PER_T,
        gamma_bath_hz_per_t=GAMMA_1H_HZ_PER_T,
        bath_spin=0.5,
    )
    relaxation = RedfieldDipolarRelaxationModel.from_dipolar_sources(
        0.5,
        (source,),
        motion=IsotropicLiquidMotionalAveraging(tau_c_seconds),
    )
    free_generator = (
        liouville_hamiltonian(rotating_hamiltonian)
        + relaxation.superoperator(lab_hamiltonian)
    )
    free = matrix_exponential(free_generator, 0.5 * echo_spacing_seconds)
    # Match the established ideal CPMG convention: a 90_y excitation prepares
    # +Ix coherence, then each refocusing pulse is 180_x.
    pi_x = propagator(ops.ix, np.pi)

    rho = ops.ix.copy()
    echo_times = (np.arange(num_echoes, dtype=np.float64) + 1.0) * echo_spacing_seconds
    echoes = np.empty(num_echoes, dtype=np.complex128)
    for idx in range(num_echoes):
        rho = (free @ rho.reshape(-1, order="F")).reshape(rho.shape, order="F")
        rho = pi_x @ rho @ pi_x.conj().T
        rho = (free @ rho.reshape(-1, order="F")).reshape(rho.shape, order="F")
        echoes[idx] = np.trace(rho @ (ops.ix + 1j * ops.iy))
    return echo_times, echoes, _fit_t2(echo_times, echoes)


def _simulate(args: argparse.Namespace) -> WaterCPMGResult:
    larmor_hz = float(args.larmor_mhz) * 1.0e6
    echo_spacing_seconds = (
        float(args.echo_spacing_seconds)
        if args.echo_spacing_seconds is not None
        else float(args.echo_spacing_ms) * 1.0e-3
    )
    tau_c = (
        np.asarray(args.correlation_time_seconds, dtype=np.float64)
        if args.correlation_time_seconds is not None
        else np.asarray(args.tau_c_ps, dtype=np.float64) * 1.0e-12
    )

    # Keep the example tied to the same public phase-cycle helper used by the
    # finite CPMG workflows, even though this homogeneous single-spin example
    # can be propagated as one ideal branch after the 90_y pulse.
    phase_cycle = cpmg_two_step_phase_cycle()
    if not np.isclose(phase_cycle.pulse_phases("excitation")[0], np.pi / 2.0):
        raise RuntimeError("unexpected default CPMG excitation phase")

    trains = []
    fitted = []
    echo_times = None
    for tau in tau_c:
        echo_times, echoes, t2 = _simulate_one(
            larmor_hz=larmor_hz,
            echo_spacing_seconds=echo_spacing_seconds,
            num_echoes=int(args.num_echoes),
            tau_c_seconds=float(tau),
            hh_distance_angstrom=float(args.hh_distance_a),
            offset_hz=float(args.offset_hz),
        )
        trains.append(np.abs(echoes))
        fitted.append(t2)

    trains_array = np.asarray(trains, dtype=np.float64)
    trains_array /= trains_array[:, :1]
    return WaterCPMGResult(
        echo_times_seconds=np.asarray(echo_times, dtype=np.float64),
        tau_c_seconds=tau_c,
        echo_trains=trains_array,
        fitted_t2_seconds=np.asarray(fitted, dtype=np.float64),
        larmor_hz=larmor_hz,
        echo_spacing_seconds=echo_spacing_seconds,
    )


def _plot(plt, result: WaterCPMGResult):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for tau_c, train, t2 in zip(
        result.tau_c_seconds,
        result.echo_trains,
        result.fitted_t2_seconds,
    ):
        axes[0].semilogy(
            result.echo_times_seconds,
            train,
            marker="o",
            ms=2.5,
            label=f"{tau_c * 1e12:g} ps, T2={t2:.2g} s",
        )
    axes[0].set_xlabel("echo time (s)")
    axes[0].set_ylabel("normalized echo magnitude")
    axes[0].set_title("Ideal CPMG Echo Train")
    axes[0].legend(fontsize=8)

    axes[1].plot(result.tau_c_seconds * 1e12, result.fitted_t2_seconds, marker="o")
    axes[1].set_xlabel("rotational tau_c (ps)")
    axes[1].set_ylabel("fitted T2 (s)")
    axes[1].set_title("Redfield T2 Sensitivity")

    omega = 2.0 * np.pi * result.larmor_hz
    spectral = 2.0 * result.tau_c_seconds / (1.0 + (omega * result.tau_c_seconds) ** 2)
    axes[2].plot(result.tau_c_seconds * 1e12, spectral * 1e12, marker="o")
    axes[2].set_xlabel("rotational tau_c (ps)")
    axes[2].set_ylabel("J(omega0) (ps)")
    axes[2].set_title(f"Spectral Density at {result.larmor_hz / 1e6:g} MHz")

    fig.suptitle(
        "Bulk-water proton CPMG from isotropic Redfield dipolar relaxation"
    )
    return fig


def main() -> None:
    args = _parse_args()
    if args.num_echoes <= 1:
        raise SystemExit("--num-echoes must be greater than one")
    echo_spacing_seconds = (
        args.echo_spacing_seconds
        if args.echo_spacing_seconds is not None
        else args.echo_spacing_ms * 1.0e-3
    )
    if echo_spacing_seconds <= 0:
        raise SystemExit("echo spacing must be positive")
    correlation_times = (
        args.correlation_time_seconds
        if args.correlation_time_seconds is not None
        else [value * 1.0e-12 for value in args.tau_c_ps]
    )
    if any(value <= 0.0 for value in correlation_times):
        raise SystemExit("correlation times must be positive")

    plt = load_matplotlib(headless=args.output is not None)
    result = _simulate(args)
    print("Bulk-water proton Redfield CPMG")
    print(f"Larmor frequency: {result.larmor_hz / 1e6:.6g} MHz")
    print(f"Echo spacing: {result.echo_spacing_seconds * 1e3:.6g} ms")
    for tau_c, t2 in zip(result.tau_c_seconds, result.fitted_t2_seconds):
        print(f"  tau_c = {tau_c * 1e12:5.2f} ps -> fitted T2 = {t2:.6g} s")

    fig = _plot(plt, result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
