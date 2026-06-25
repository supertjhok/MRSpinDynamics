"""Phase-cycled stimulated echo in an inhomogeneous static field.

A three-pulse stimulated echo stores one dephased quadrature along z during the
mixing interval and reads it back with a third 90-degree pulse. In a broad B0
distribution the selected pathway refocuses at one evolution time after the read
pulse, while equilibrium recovery during storage can create a prompt FID from
the read pulse itself.

This example uses the public ``PhaseCycle`` table to compare:

1. a single uncycled scan,
2. a simple two-step readout cycle, and
3. the full three-pulse phase cycle selecting the stimulated-echo pathway.

The two-step cycle removes the readout phase sign, but it cannot distinguish the
stimulated echo from a last-pulse FID. The full table cycles all three 90-degree
pulses and uses the receiver signature ``-phi1 + phi2 + phi3`` to keep the
stimulated echo while rejecting the FID and other coherence pathways.

Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a three-pulse stimulated-echo phase cycle in an inhomogeneous "
            "B0 field, contrasting it with a single scan and a two-step cycle."
        )
    )
    parser.add_argument(
        "--num-offsets",
        type=int,
        default=801,
        help="Number of isochromats in the static B0 distribution.",
    )
    parser.add_argument(
        "--offset-span-hz",
        type=float,
        default=14000.0,
        help="Full B0 offset span across the synthetic sample (Hz).",
    )
    parser.add_argument(
        "--offset-sigma-hz",
        type=float,
        default=3500.0,
        help="Gaussian density width of the B0 offset distribution (Hz).",
    )
    parser.add_argument(
        "--tau-ms",
        type=float,
        default=1.0,
        help="Evolution time tau before and after storage (ms).",
    )
    parser.add_argument(
        "--storage-ms",
        type=float,
        default=4.0,
        help="Storage interval between the second and third 90-degree pulses (ms).",
    )
    parser.add_argument(
        "--t1-ms",
        type=float,
        default=13.9,
        help="Longitudinal relaxation time during storage (ms).",
    )
    parser.add_argument(
        "--t2-ms",
        type=float,
        default=0.5,
        help="Transverse relaxation time during storage (ms).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=401,
        help="Acquisition samples after the third pulse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path. If omitted, show the plot.",
    )
    return parser.parse_args()


def _rotate(magnetization: np.ndarray, angle_rad: float, phase_rad: float) -> np.ndarray:
    axis = np.array([np.cos(phase_rad), np.sin(phase_rad), 0.0])
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    projection = np.sum(magnetization * axis, axis=-1, keepdims=True)
    return (
        magnetization * cosine
        + np.cross(axis, magnetization) * sine
        + axis * projection * (1.0 - cosine)
    )


def _free_precession(
    magnetization: np.ndarray,
    offsets_rad_s: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    angle = offsets_rad_s * float(duration_s)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    out = magnetization.copy()
    mx = magnetization[:, 0]
    my = magnetization[:, 1]
    out[:, 0] = mx * cosine - my * sine
    out[:, 1] = mx * sine + my * cosine
    return out


def _field_distribution(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets_hz = np.linspace(
        -0.5 * float(args.offset_span_hz),
        0.5 * float(args.offset_span_hz),
        int(args.num_offsets),
    )
    sigma = max(float(args.offset_sigma_hz), np.finfo(float).eps)
    weights = np.exp(-0.5 * (offsets_hz / sigma) ** 2)
    weights /= np.sum(weights)
    return offsets_hz, 2.0 * np.pi * offsets_hz, weights


def _simulate_branch(
    *,
    pulse_phases: tuple[float, float, float],
    offsets_rad_s: np.ndarray,
    weights: np.ndarray,
    acquisition_times_s: np.ndarray,
    tau_s: float,
    storage_s: float,
    t1_s: float,
    t2_s: float,
) -> np.ndarray:
    phi1, phi2, phi3 = pulse_phases
    magnetization = np.zeros((offsets_rad_s.size, 3), dtype=np.float64)
    magnetization[:, 2] = 1.0

    magnetization = _rotate(magnetization, np.pi / 2.0, phi1)
    magnetization = _free_precession(magnetization, offsets_rad_s, tau_s)
    magnetization = _rotate(magnetization, np.pi / 2.0, phi2)
    magnetization = _free_precession(magnetization, offsets_rad_s, storage_s)
    e1 = 0.0 if t1_s <= 0.0 else np.exp(-storage_s / t1_s)
    e2 = 0.0 if t2_s <= 0.0 else np.exp(-storage_s / t2_s)
    magnetization[:, :2] *= e2
    magnetization[:, 2] = 1.0 + (magnetization[:, 2] - 1.0) * e1
    magnetization = _rotate(magnetization, np.pi / 2.0, phi3)

    initial_transverse = magnetization[:, 0] + 1j * magnetization[:, 1]
    precession = np.exp(
        1j * offsets_rad_s[:, np.newaxis] * acquisition_times_s[np.newaxis, :]
    )
    return weights @ (initial_transverse[:, np.newaxis] * precession)


def _stimulated_echo_cycle():
    from spin_dynamics.phase_cycling import PhaseCycle, PhaseStep

    phase_values = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
    steps = []
    for index, (phi1, phi2, phi3) in enumerate(product(phase_values, repeat=3)):
        steps.append(
            PhaseStep(
                {
                    "excitation_90": phi1,
                    "store_90": phi2,
                    "read_90": phi3,
                },
                receiver_phase_rad=-phi1 + phi2 + phi3,
                label=f"ste_{index}",
            )
        )
    return PhaseCycle(
        steps,
        pulse_names=("excitation_90", "store_90", "read_90"),
        name="three_pulse_stimulated_echo",
    )


def _two_step_readout_cycle():
    from spin_dynamics.phase_cycling import PhaseCycle, PhaseStep

    return PhaseCycle(
        (
            PhaseStep(
                {"excitation_90": 0.0, "store_90": 0.0, "read_90": 0.0},
                receiver_phase_rad=0.0,
                label="read_plus",
            ),
            PhaseStep(
                {"excitation_90": 0.0, "store_90": 0.0, "read_90": np.pi},
                receiver_phase_rad=np.pi,
                label="read_minus",
            ),
        ),
        pulse_names=("excitation_90", "store_90", "read_90"),
        name="two_step_readout",
    )


def _combine_cycle(
    cycle,
    *,
    offsets_rad_s: np.ndarray,
    weights: np.ndarray,
    acquisition_times_s: np.ndarray,
    tau_s: float,
    storage_s: float,
    t1_s: float,
    t2_s: float,
) -> np.ndarray:
    branches = [
        _simulate_branch(
            pulse_phases=(
                step.pulse_phase("excitation_90"),
                step.pulse_phase("store_90"),
                step.pulse_phase("read_90"),
            ),
            offsets_rad_s=offsets_rad_s,
            weights=weights,
            acquisition_times_s=acquisition_times_s,
            tau_s=tau_s,
            storage_s=storage_s,
            t1_s=t1_s,
            t2_s=t2_s,
        )
        for step in cycle.steps
    ]
    return cycle.combine(branches)


def _pathway_projection(
    receiver_phase,
    *,
    offsets_rad_s: np.ndarray,
    weights: np.ndarray,
    acquisition_times_s: np.ndarray,
    tau_s: float,
    storage_s: float,
    t1_s: float,
    t2_s: float,
):
    from spin_dynamics.phase_cycling import PhaseCycle, PhaseStep

    phase_values = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
    steps = []
    branches = []
    for index, (phi1, phi2, phi3) in enumerate(product(phase_values, repeat=3)):
        steps.append(
            PhaseStep(
                {
                    "excitation_90": phi1,
                    "store_90": phi2,
                    "read_90": phi3,
                },
                receiver_phase_rad=receiver_phase(phi1, phi2, phi3),
                label=f"path_{index}",
            )
        )
        branches.append(
            _simulate_branch(
                pulse_phases=(phi1, phi2, phi3),
                offsets_rad_s=offsets_rad_s,
                weights=weights,
                acquisition_times_s=acquisition_times_s,
                tau_s=tau_s,
                storage_s=storage_s,
                t1_s=t1_s,
                t2_s=t2_s,
            )
        )
    cycle = PhaseCycle(
        steps,
        pulse_names=("excitation_90", "store_90", "read_90"),
        name="pathway_projection",
    )
    return cycle.combine(branches)


def _run_example(args: argparse.Namespace):
    offsets_hz, offsets_rad_s, weights = _field_distribution(args)
    tau_s = float(args.tau_ms) * 1.0e-3
    storage_s = float(args.storage_ms) * 1.0e-3
    t1_s = float(args.t1_ms) * 1.0e-3
    t2_s = float(args.t2_ms) * 1.0e-3
    acquisition_times_s = np.linspace(0.0, 2.0 * tau_s, int(args.samples))

    single = _simulate_branch(
        pulse_phases=(0.0, 0.0, 0.0),
        offsets_rad_s=offsets_rad_s,
        weights=weights,
        acquisition_times_s=acquisition_times_s,
        tau_s=tau_s,
        storage_s=storage_s,
        t1_s=t1_s,
        t2_s=t2_s,
    )
    two_step = _combine_cycle(
        _two_step_readout_cycle(),
        offsets_rad_s=offsets_rad_s,
        weights=weights,
        acquisition_times_s=acquisition_times_s,
        tau_s=tau_s,
        storage_s=storage_s,
        t1_s=t1_s,
        t2_s=t2_s,
    )
    full_cycle = _stimulated_echo_cycle()
    selected = _combine_cycle(
        full_cycle,
        offsets_rad_s=offsets_rad_s,
        weights=weights,
        acquisition_times_s=acquisition_times_s,
        tau_s=tau_s,
        storage_s=storage_s,
        t1_s=t1_s,
        t2_s=t2_s,
    )
    fid_projection = _pathway_projection(
        lambda _phi1, _phi2, phi3: phi3,
        offsets_rad_s=offsets_rad_s,
        weights=weights,
        acquisition_times_s=acquisition_times_s,
        tau_s=tau_s,
        storage_s=storage_s,
        t1_s=t1_s,
        t2_s=t2_s,
    )
    anti_echo_projection = _pathway_projection(
        lambda phi1, phi2, phi3: phi1 - phi2 + phi3,
        offsets_rad_s=offsets_rad_s,
        weights=weights,
        acquisition_times_s=acquisition_times_s,
        tau_s=tau_s,
        storage_s=storage_s,
        t1_s=t1_s,
        t2_s=t2_s,
    )
    return {
        "offsets_hz": offsets_hz,
        "weights": weights,
        "times_s": acquisition_times_s,
        "storage_recovery_fraction": 1.0 - np.exp(-storage_s / t1_s)
        if t1_s > 0.0
        else 1.0,
        "single": single,
        "two_step": two_step,
        "selected": selected,
        "fid_projection": fid_projection,
        "anti_echo_projection": anti_echo_projection,
        "phase_cycle": full_cycle,
    }


def _plot(plt, args: argparse.Namespace, results: dict[str, np.ndarray]):
    times_ms = results["times_s"] * 1.0e3
    tau_ms = float(args.tau_ms)
    offsets_khz = results["offsets_hz"] / 1.0e3

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    axes = axes.ravel()

    axes[0].plot(offsets_khz, results["weights"], color="#3a3a3a")
    axes[0].set_xlabel("B0 offset (kHz)")
    axes[0].set_ylabel("isochromat weight")
    axes[0].set_title("Inhomogeneous static-field distribution")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        times_ms,
        np.abs(results["single"]),
        color="#7f7f7f",
        linewidth=1.2,
        label="single scan",
    )
    axes[1].plot(
        times_ms,
        np.abs(results["two_step"]),
        color="#1f77b4",
        linewidth=1.2,
        label="two-step readout cycle",
    )
    axes[1].plot(
        times_ms,
        np.abs(results["selected"]),
        color="#d62728",
        linewidth=1.5,
        label="full three-pulse cycle",
    )
    axes[1].axvline(tau_ms, color="k", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("time after third pulse (ms)")
    axes[1].set_ylabel("|signal| / M0")
    axes[1].set_title("Full phase cycle rejects prompt FID leakage")
    axes[1].legend(fontsize="small")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(
        times_ms,
        np.abs(results["selected"]),
        color="#d62728",
        linewidth=1.5,
        label="selected STE: -phi1 + phi2 + phi3",
    )
    axes[2].plot(
        times_ms,
        np.abs(results["fid_projection"]),
        color="#2ca02c",
        linewidth=1.1,
        label="last-pulse FID: phi3",
    )
    axes[2].plot(
        times_ms,
        np.abs(results["anti_echo_projection"]),
        color="#9467bd",
        linewidth=1.1,
        label="anti-echo-like: phi1 - phi2 + phi3",
    )
    axes[2].axvline(tau_ms, color="k", linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("time after third pulse (ms)")
    axes[2].set_ylabel("projected |signal| / M0")
    axes[2].set_title("Receiver phase acts as a pathway filter")
    axes[2].legend(fontsize="small")
    axes[2].grid(True, alpha=0.25)

    complex_signal = results["selected"]
    axes[3].plot(times_ms, complex_signal.real, color="#d62728", label="real")
    axes[3].plot(times_ms, complex_signal.imag, color="#1f77b4", label="imag")
    axes[3].axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
    axes[3].axvline(tau_ms, color="k", linestyle=":", linewidth=1.0)
    axes[3].set_xlabel("time after third pulse (ms)")
    axes[3].set_ylabel("phase-cycled signal")
    axes[3].set_title("Selected stimulated echo refocuses at tau")
    axes[3].legend(fontsize="small")
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(
        "Three-pulse stimulated-echo phase cycling in an inhomogeneous B0 field",
        y=0.995,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))
    results = _run_example(args)

    selected = results["selected"]
    fid = results["fid_projection"]
    echo_index = int(np.argmin(np.abs(results["times_s"] - args.tau_ms * 1.0e-3)))
    print(
        f"cycle {results['phase_cycle'].name}: "
        f"{results['phase_cycle'].num_steps} steps, "
        f"{len(results['phase_cycle'].pulse_names)} pulse columns"
    )
    print(
        f"selected echo at tau={args.tau_ms:.3g} ms: "
        f"|S|={abs(selected[echo_index]):.4f}; "
        f"prompt FID projection |S(0)|={abs(fid[0]):.4f}; "
        f"storage recovery={results['storage_recovery_fraction']:.3f}"
    )

    fig = _plot(plt, args, results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
