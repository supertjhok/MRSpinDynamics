"""Plot ESR T1/T2 relaxation effects in pulsed spin-1/2 simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    ESRRelaxationModel,
    ESRSpinSystem,
    equilibrium_density,
    flip_angle_duration,
    propagate_density_liouville,
    resonance_field_tesla,
    resonance_frequency_hz,
    simulate_fid,
    simulate_hahn_echo,
)


def _normalized(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return values / scale if scale > 0 else values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g", type=float, default=2.00231930436256)
    parser.add_argument("--microwave-ghz", type=float, default=9.5)
    parser.add_argument("--nutation-mhz", type=float, default=5.0)
    parser.add_argument("--t1-us", type=float, default=20.0)
    parser.add_argument("--t2-us", type=float, default=3.0)
    parser.add_argument("--fid-duration-us", type=float, default=12.0)
    parser.add_argument("--max-tau-us", type=float, default=6.0)
    parser.add_argument("--detuning-span-mhz", type=float, default=2.0)
    parser.add_argument("--isochromats", type=int, default=31)
    parser.add_argument("--points", type=int, default=301)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    system = ESRSpinSystem(g_tensor=args.g)
    microwave_hz = args.microwave_ghz * 1e9
    nutation_hz = args.nutation_mhz * 1e6
    t1 = args.t1_us * 1e-6
    t2 = args.t2_us * 1e-6

    b0_resonant = resonance_field_tesla(system, microwave_hz)
    b0_vector = np.array([0.0, 0.0, b0_resonant], dtype=np.float64)
    carrier = resonance_frequency_hz(system, b0_vector)
    t90 = flip_angle_duration(np.pi / 2.0, nutation_hz)
    t180 = flip_angle_duration(np.pi, nutation_hz)

    # The relaxation model acts inside the Liouville propagator.  Do not pass a
    # finite t2_seconds envelope here; that quick envelope is kept for simple
    # demos, while ESRRelaxationModel is the physical relaxation path.
    fid_times = np.linspace(0.0, args.fid_duration_us * 1e-6, args.points)
    fid_fast = simulate_fid(
        system,
        b0_vector,
        nutation_hz=nutation_hz,
        pulse_duration_seconds=t90,
        times_seconds=fid_times,
        rf_frequency_hz=carrier,
        relaxation=ESRRelaxationModel(t2_seconds=t2),
    )
    fid_slow = simulate_fid(
        system,
        b0_vector,
        nutation_hz=nutation_hz,
        pulse_duration_seconds=t90,
        times_seconds=fid_times,
        rf_frequency_hz=carrier,
        relaxation=ESRRelaxationModel(t2_seconds=2.0 * t2),
    )

    # A Hahn echo is visible after summing detuned isochromats.  At the echo
    # center, the static detuning is refocused, while T2 relaxation remains and
    # gives the familiar approximately exp(-2*tau/T2) envelope.
    offsets_hz = np.linspace(
        -0.5 * args.detuning_span_mhz * 1e6,
        0.5 * args.detuning_span_mhz * 1e6,
        int(args.isochromats),
    )
    tau_values = np.linspace(0.25e-6, args.max_tau_us * 1e-6, 18)
    echo_amplitudes = np.empty(tau_values.size, dtype=np.float64)
    for tau_idx, tau in enumerate(tau_values):
        echo_sum = 0.0 + 0.0j
        for offset_hz in offsets_hz:
            shifted_field = resonance_field_tesla(system, carrier + offset_hz)
            result = simulate_hahn_echo(
                system,
                [0.0, 0.0, shifted_field],
                nutation_hz=nutation_hz,
                excitation_duration_seconds=t90,
                refocus_duration_seconds=t180,
                tau_seconds=tau,
                times_seconds=[tau],
                rf_frequency_hz=carrier,
                relaxation=ESRRelaxationModel(t2_seconds=t2),
            )
            echo_sum += result.signal[0]
        echo_amplitudes[tau_idx] = abs(echo_sum)

    # T1 acts on the longitudinal population difference.  The density matrices
    # used here are high-temperature deviations, so relaxation drives the
    # trace-zero population difference toward zero rather than toward a full
    # thermal density matrix.
    eigensystem = fid_fast.eigensystem
    rho0 = equilibrium_density(eigensystem.levels_hz)
    zero_hamiltonian = np.zeros_like(rho0)
    recovery_times = np.linspace(0.0, 4.0 * t1, args.points)
    longitudinal = np.empty(recovery_times.size, dtype=np.float64)
    for idx, delay in enumerate(recovery_times):
        rho = propagate_density_liouville(
            rho0,
            zero_hamiltonian,
            delay,
            relaxation=ESRRelaxationModel(t1_seconds=t1),
        )
        longitudinal[idx] = float(np.real(rho[0, 0] - rho[1, 1]))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(
        1e6 * fid_times,
        np.abs(fid_fast.signal),
        label=f"T2={args.t2_us:g} us",
    )
    axes[0, 0].plot(
        1e6 * fid_times,
        np.abs(fid_slow.signal),
        label=f"T2={2.0 * args.t2_us:g} us",
    )
    axes[0, 0].set_xlabel("Time after 90-degree pulse (us)")
    axes[0, 0].set_ylabel("|FID|")
    axes[0, 0].set_title("Liouville T2 During FID")
    axes[0, 0].legend()

    axes[0, 1].plot(
        2.0e6 * tau_values,
        _normalized(echo_amplitudes),
        "o-",
        label="simulated echo peaks",
    )
    axes[0, 1].plot(
        2.0e6 * tau_values,
        np.exp(-2.0 * tau_values / t2),
        "--",
        label="exp(-2 tau / T2)",
    )
    axes[0, 1].set_xlabel("Echo evolution time 2 tau (us)")
    axes[0, 1].set_ylabel("Normalized echo amplitude")
    axes[0, 1].set_title("Hahn-Echo T2 Envelope")
    axes[0, 1].legend()

    axes[1, 0].plot(
        1e6 * recovery_times,
        _normalized(longitudinal),
        color="tab:green",
        label="population difference",
    )
    axes[1, 0].plot(
        1e6 * recovery_times,
        np.exp(-recovery_times / t1),
        "--",
        color="0.3",
        label="exp(-t / T1)",
    )
    axes[1, 0].set_xlabel("Delay (us)")
    axes[1, 0].set_ylabel("Normalized longitudinal deviation")
    axes[1, 0].set_title("T1 Population Relaxation")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    summary = (
        f"g = {args.g:.6g}\n"
        f"carrier = {args.microwave_ghz:g} GHz\n"
        f"B0 = {1e3 * b0_resonant:.3f} mT\n"
        f"nutation = {args.nutation_mhz:g} MHz\n"
        f"t90 = {1e9 * t90:.1f} ns\n"
        f"t180 = {1e9 * t180:.1f} ns\n"
        f"T1 = {args.t1_us:g} us\n"
        f"T2 = {args.t2_us:g} us"
    )
    axes[1, 1].text(0.0, 1.0, summary, va="top", family="monospace")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
