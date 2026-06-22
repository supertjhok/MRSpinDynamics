"""Auto-select the NQR model from physics, then simulate with it.

``select_nqr_model`` reads the reduced-vs-full choice from the static
Hamiltonian and the RF matrix elements for the coil polarization. This example
wires it into a small driver: it picks the model, routes to the matching
simulator -- the reduced fictitious-spin-1/2 path for an isolated spin-1 line,
the full ``(2I+1)`` density matrix for spin-3/2 -- and plots the resulting FID.
A third case (the same spin-1 site under a broadband pulse) shows that the
choice follows the physics, not the spin.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    NQRRelaxationModel,
    QuadrupolarSite,
    SelectivePulse,
    apply_selective_pulse,
    diagonalize_site,
    equilibrium_density,
    select_nqr_model,
    simulate_full_fid,
    transition_signal,
)


@dataclass
class Scenario:
    name: str
    site: QuadrupolarSite
    target: str
    nutation_hz: float
    pulse_seconds: float
    b1: tuple[float, float, float]


def _reduced_fid(site, label, *, nutation_hz, pulse_seconds, b1, carrier_hz,
                 t2_seconds, times):
    """FID from the reduced two-level model: excite, then free-precess the
    coherence at its detuning with a T2 envelope."""

    eigensystem = diagonalize_site(site)
    transition = eigensystem.transition(label)
    rho = apply_selective_pulse(
        equilibrium_density(eigensystem.levels_hz),
        transition,
        SelectivePulse(label, duration_seconds=pulse_seconds,
                       nutation_hz=nutation_hz, rf_frequency_hz=carrier_hz),
        b1_direction_pas=b1,
    )
    amplitude = transition_signal(rho, transition, b1_direction_pas=b1)
    detuning = transition.frequency_hz - carrier_hz
    return amplitude * np.exp(-1j * 2 * np.pi * detuning * times) * np.exp(
        -times / t2_seconds
    )


def run_auto(scenario: Scenario, *, offset_hz, t2_seconds, times):
    """Select the model from physics, then simulate the FID with it."""

    choice = select_nqr_model(
        scenario.site, scenario.target,
        nutation_hz=scenario.nutation_hz,
        pulse_duration_seconds=scenario.pulse_seconds,
        b1_direction_pas=scenario.b1,
        linewidth_hz=0.0,
    )
    target_hz = choice.target_frequency_hz
    carrier_hz = target_hz + offset_hz
    if choice.recommended_model == "full":
        fid = simulate_full_fid(
            scenario.site, nutation_hz=scenario.nutation_hz,
            pulse_duration_seconds=scenario.pulse_seconds, times_seconds=times,
            rf_frequency_hz=carrier_hz, b1_direction_pas=scenario.b1,
            relaxation=NQRRelaxationModel(t2_seconds=t2_seconds),
        )
        signal = fid.signal
    else:
        signal = _reduced_fid(
            scenario.site, scenario.target, nutation_hz=scenario.nutation_hz,
            pulse_seconds=scenario.pulse_seconds, b1=scenario.b1,
            carrier_hz=carrier_hz, t2_seconds=t2_seconds, times=times,
        )
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return choice, signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-khz", type=float, default=8.0,
                        help="Carrier offset from the target line in kHz.")
    parser.add_argument("--t2-us", type=float, default=300.0,
                        help="Transverse relaxation time in microseconds.")
    parser.add_argument("--acq-us", type=float, default=800.0,
                        help="Acquisition window in microseconds.")
    parser.add_argument("--points", type=int, default=800, help="FID samples.")
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    spin1 = QuadrupolarSite(spin=1, isotope="14N",
                            quadrupole_frequency_hz=900e3, eta=0.3)
    spin32 = QuadrupolarSite(spin=1.5, isotope="35Cl",
                             quadrupole_frequency_hz=30e6, eta=0.1)
    label32 = diagonalize_site(spin32).transitions[0].label

    scenarios = [
        Scenario("spin-1, isolated line\n(weak narrowband pulse)", spin1, "x",
                 nutation_hz=2e3, pulse_seconds=60e-6, b1=(1, 1, 1)),
        Scenario("spin-1, same site\n(strong broadband pulse)", spin1, "x",
                 nutation_hz=100e3, pulse_seconds=2.5e-6, b1=(1, 1, 1)),
        Scenario("spin-3/2 (35Cl)\n(Kramers doublets)", spin32, label32,
                 nutation_hz=20e3, pulse_seconds=10e-6, b1=(1, 1, 1)),
    ]

    times = np.linspace(0.0, args.acq_us * 1e-6, args.points)
    t2 = args.t2_us * 1e-6
    offset = args.offset_khz * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    for ax, scenario in zip(axes, scenarios):
        choice, signal = run_auto(scenario, offset_hz=offset, t2_seconds=t2,
                                  times=times)
        color = "C0" if choice.recommended_model == "reduced" else "C3"
        print(f"### {scenario.name.splitlines()[0]}")
        print(choice.describe())
        print()
        ax.plot(times * 1e6, np.abs(signal), color=color, label="|signal|")
        ax.plot(times * 1e6, np.real(signal), color=color, alpha=0.4, label="Re")
        ax.set_title(
            f"{scenario.name}\n-> {choice.recommended_model.upper()} "
            f"(isolation x{choice.isolation_ratio:.0f})"
            if np.isfinite(choice.isolation_ratio)
            else f"{scenario.name}\n-> {choice.recommended_model.upper()} "
            f"(isolated by polarization)",
            fontsize=9,
        )
        ax.set_xlabel("time (us)")
        ax.set_ylabel("normalized FID")
        ax.legend(fontsize=8)
    fig.suptitle("Auto-selected NQR model and resulting FID", fontsize=12)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
