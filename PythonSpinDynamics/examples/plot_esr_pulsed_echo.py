"""Plot pulsed ESR FID and Hahn-echo simulations for a spin-1/2 electron."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    ESRSpinSystem,
    flip_angle_duration,
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
    parser.add_argument(
        "--nutation-mhz",
        type=float,
        default=5.0,
        help="On-resonance spin-1/2 Rabi rate in MHz.",
    )
    parser.add_argument("--tau-us", type=float, default=2.0)
    parser.add_argument("--t2-us", type=float, default=8.0)
    parser.add_argument(
        "--detuning-span-mhz",
        type=float,
        default=2.0,
        help="Full detuning span for the Hahn-echo isochromat ensemble.",
    )
    parser.add_argument("--isochromats", type=int, default=41)
    parser.add_argument("--points", type=int, default=501)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    system = ESRSpinSystem(g_tensor=args.g)
    microwave_hz = args.microwave_ghz * 1e9
    nutation_hz = args.nutation_mhz * 1e6
    tau = args.tau_us * 1e-6
    t2 = args.t2_us * 1e-6

    # Choose B0 so the single-crystal z-axis transition is on resonance with
    # the microwave carrier.  This gives clean pulse calibration while still
    # letting us introduce controlled offsets for the echo panel.
    b0_resonant = resonance_field_tesla(system, microwave_hz)
    b0_vector = np.array([0.0, 0.0, b0_resonant], dtype=np.float64)
    carrier = resonance_frequency_hz(system, b0_vector)

    t90 = flip_angle_duration(np.pi / 2.0, nutation_hz)
    t180 = flip_angle_duration(np.pi, nutation_hz)

    # A rectangular pulse rotates by theta = 2*pi*nutation_hz*pulse_duration.
    # The detected transverse magnetization therefore follows a sin(theta)
    # calibration curve for a resonant spin-1/2.
    pulse_durations = np.linspace(0.0, 2.0 * t180, 161)
    pulse_signal = np.empty(pulse_durations.size, dtype=np.complex128)
    for idx, duration in enumerate(pulse_durations):
        result = simulate_fid(
            system,
            b0_vector,
            nutation_hz=nutation_hz,
            pulse_duration_seconds=duration,
            times_seconds=[0.0],
            rf_frequency_hz=carrier,
        )
        pulse_signal[idx] = result.signal[0]

    fid_times = np.linspace(0.0, 5.0 * tau, args.points)
    fid = simulate_fid(
        system,
        b0_vector,
        nutation_hz=nutation_hz,
        pulse_duration_seconds=t90,
        times_seconds=fid_times,
        rf_frequency_hz=carrier,
        t2_seconds=t2,
    )

    # A single detuned spin keeps roughly constant magnitude in this simple
    # model; the familiar echo envelope appears when many detuned isochromats
    # dephase and then refocus after the 180-degree pulse.
    echo_times = np.linspace(0.0, 2.0 * tau, args.points)
    offsets_hz = np.linspace(
        -0.5 * args.detuning_span_mhz * 1e6,
        0.5 * args.detuning_span_mhz * 1e6,
        int(args.isochromats),
    )
    echo_signal = np.zeros(echo_times.size, dtype=np.complex128)
    individual_echoes = []
    for offset_hz in offsets_hz:
        shifted_field = resonance_field_tesla(system, carrier + offset_hz)
        result = simulate_hahn_echo(
            system,
            [0.0, 0.0, shifted_field],
            nutation_hz=nutation_hz,
            excitation_duration_seconds=t90,
            refocus_duration_seconds=t180,
            tau_seconds=tau,
            times_seconds=echo_times,
            rf_frequency_hz=carrier,
            t2_seconds=t2,
        )
        echo_signal += result.signal
        if len(individual_echoes) < 7:
            individual_echoes.append(result.signal)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(1e6 * pulse_durations, np.abs(pulse_signal), color="tab:blue")
    axes[0, 0].axvline(1e6 * t90, color="tab:orange", linestyle="--", label="90 deg")
    axes[0, 0].axvline(1e6 * t180, color="tab:green", linestyle="--", label="180 deg")
    axes[0, 0].set_xlabel("Pulse duration (us)")
    axes[0, 0].set_ylabel("|FID at t=0|")
    axes[0, 0].set_title("Rectangular Pulse Calibration")
    axes[0, 0].legend()

    axes[0, 1].plot(1e6 * fid.times_seconds, fid.signal.real, label="real")
    axes[0, 1].plot(1e6 * fid.times_seconds, fid.signal.imag, label="imag")
    axes[0, 1].plot(1e6 * fid.times_seconds, np.abs(fid.signal), label="magnitude")
    axes[0, 1].set_xlabel("Time after pulse (us)")
    axes[0, 1].set_ylabel("Baseband signal")
    axes[0, 1].set_title("On-Resonance 90-Degree FID")
    axes[0, 1].legend()

    for signal in individual_echoes:
        axes[1, 0].plot(
            1e6 * echo_times,
            np.real(signal),
            color="0.65",
            linewidth=0.8,
        )
    axes[1, 0].plot(
        1e6 * echo_times,
        np.real(echo_signal / max(len(offsets_hz), 1)),
        color="tab:purple",
        linewidth=2,
        label="ensemble average",
    )
    axes[1, 0].axvline(args.tau_us, color="tab:orange", linestyle="--")
    axes[1, 0].set_xlabel("Time after 180-degree pulse (us)")
    axes[1, 0].set_ylabel("Real signal")
    axes[1, 0].set_title("Isochromat Rephasing")
    axes[1, 0].legend()

    axes[1, 1].plot(
        1e6 * echo_times,
        np.abs(_normalized(echo_signal)),
        color="tab:red",
    )
    axes[1, 1].axvline(args.tau_us, color="tab:orange", linestyle="--", label="tau")
    axes[1, 1].set_xlabel("Time after 180-degree pulse (us)")
    axes[1, 1].set_ylabel("Normalized echo magnitude")
    axes[1, 1].set_title("Hahn Echo Envelope")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
