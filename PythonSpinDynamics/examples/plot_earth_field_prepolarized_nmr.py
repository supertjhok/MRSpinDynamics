"""Plot Earth's-field NMR after electromagnet prepolarization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.prepolarization import (  # noqa: E402
    longitudinal_recovery,
    prepolarized_state,
)
from spin_dynamics.radiation_damping import PROTON_GAMMA  # noqa: E402
from spin_dynamics.relaxation import BPPRelaxationModel  # noqa: E402


GAMMA_PROTON_HZ_PER_T = PROTON_GAMMA / (2.0 * np.pi)


@dataclass(frozen=True)
class EarthFieldNMRCase:
    """Prepared low-field FID and diagnostic sweep arrays."""

    time_seconds: np.ndarray
    baseband_signal: np.ndarray
    envelope: np.ndarray
    spectrum_offsets_hz: np.ndarray
    spectrum: np.ndarray
    prep_times_seconds: np.ndarray
    prepared_vs_time: np.ndarray
    transferred_vs_time: np.ndarray
    transfer_delays_seconds: np.ndarray
    transferred_vs_delay: np.ndarray
    earth_larmor_hz: float
    polarizer_larmor_hz: float
    t1_polarizer_seconds: float
    t1_earth_seconds: float
    t2_earth_seconds: float
    prepared_m0: float
    detected_m0: float
    full_prepolarized_m0: float


def _bpp_times_for_field(args: argparse.Namespace, field_tesla: float) -> tuple[float, float]:
    rates = BPPRelaxationModel(
        angular_frequency_rad_per_s=PROTON_GAMMA * field_tesla,
        tau_ref_seconds=args.tau_ref_ns * 1e-9,
        reference_temperature_kelvin=args.reference_temp_k,
        activation_energy_j_per_mol=args.activation_energy_kj_mol * 1e3,
        coupling_scale_per_second2=args.coupling_scale,
        baseline_r1_per_second=args.baseline_r1,
        baseline_r2_per_second=args.baseline_r2,
    ).rates(args.temperature_k)
    return float(rates.t1_seconds), float(rates.t2_seconds)


def build_earth_field_case(args: argparse.Namespace) -> EarthFieldNMRCase:
    """Build arrays for an electromagnet-prepolarized Earth's-field FID."""

    earth_field = args.earth_field_ut * 1e-6
    polarizing_field = args.prepolarizing_field_mt * 1e-3
    if earth_field <= 0.0 or polarizing_field <= 0.0:
        raise ValueError("fields must be positive")
    t1_polarizer, _t2_polarizer = _bpp_times_for_field(args, polarizing_field)
    t1_earth, t2_earth = _bpp_times_for_field(args, earth_field)

    prepared = prepolarized_state(
        polarizing_field,
        earth_field,
        args.prepolarization_time_s,
        t1_polarizer,
    )
    prepared_m0 = float(prepared.m0)
    detected_m0 = float(
        longitudinal_recovery(
            prepared_m0,
            1.0,
            args.transfer_delay_s,
            t1_earth,
        )
    )
    full_prepolarized_m0 = polarizing_field / earth_field

    time = np.linspace(0.0, args.acquisition_time_s, int(args.points))
    flip = np.deg2rad(args.flip_angle_degrees)
    sigma_hz = float(args.field_spread_hz)
    detuning_hz = float(args.detuning_hz)
    envelope = (
        detected_m0
        * np.sin(flip)
        * np.exp(-time / t2_earth)
        * np.exp(-0.5 * (2.0 * np.pi * sigma_hz * time) ** 2)
    )
    baseband = envelope * np.exp(1j * 2.0 * np.pi * detuning_hz * time)

    if time.size > 1:
        dwell = float(time[1] - time[0])
    else:
        dwell = args.acquisition_time_s
    window = np.hanning(time.size)
    spectrum = np.fft.fftshift(np.fft.fft(baseband * window))
    spectrum_offsets = np.fft.fftshift(np.fft.fftfreq(time.size, dwell))
    spectrum_abs = np.abs(spectrum)
    if np.max(spectrum_abs) > 0.0:
        spectrum_abs = spectrum_abs / np.max(spectrum_abs)

    prep_times = np.linspace(0.0, max(4.0 * t1_polarizer, args.prepolarization_time_s), 160)
    prepared_vs_time = prepolarized_state(
        polarizing_field,
        earth_field,
        prep_times,
        t1_polarizer,
    ).m0
    transferred_vs_time = longitudinal_recovery(
        prepared_vs_time,
        1.0,
        args.transfer_delay_s,
        t1_earth,
    )
    transfer_delays = np.linspace(0.0, max(4.0 * t1_earth, args.transfer_delay_s), 160)
    transferred_vs_delay = longitudinal_recovery(
        prepared_m0,
        1.0,
        transfer_delays,
        t1_earth,
    )

    return EarthFieldNMRCase(
        time_seconds=time,
        baseband_signal=baseband,
        envelope=envelope,
        spectrum_offsets_hz=spectrum_offsets,
        spectrum=spectrum_abs,
        prep_times_seconds=prep_times,
        prepared_vs_time=prepared_vs_time,
        transferred_vs_time=transferred_vs_time,
        transfer_delays_seconds=transfer_delays,
        transferred_vs_delay=transferred_vs_delay,
        earth_larmor_hz=GAMMA_PROTON_HZ_PER_T * earth_field,
        polarizer_larmor_hz=GAMMA_PROTON_HZ_PER_T * polarizing_field,
        t1_polarizer_seconds=t1_polarizer,
        t1_earth_seconds=t1_earth,
        t2_earth_seconds=t2_earth,
        prepared_m0=prepared_m0,
        detected_m0=detected_m0,
        full_prepolarized_m0=full_prepolarized_m0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earth-field-ut", type=float, default=50.0)
    parser.add_argument("--prepolarizing-field-mt", type=float, default=100.0)
    parser.add_argument("--prepolarization-time-s", type=float, default=5.0)
    parser.add_argument("--transfer-delay-s", type=float, default=0.25)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--reference-temp-k", type=float, default=300.0)
    parser.add_argument("--tau-ref-ns", type=float, default=8.0)
    parser.add_argument("--activation-energy-kj-mol", type=float, default=16.0)
    parser.add_argument("--coupling-scale", type=float, default=5.0e6)
    parser.add_argument("--baseline-r1", type=float, default=0.0)
    parser.add_argument("--baseline-r2", type=float, default=0.0)
    parser.add_argument("--acquisition-time-s", type=float, default=1.0)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--field-spread-hz", type=float, default=0.8)
    parser.add_argument("--detuning-hz", type=float, default=0.7)
    parser.add_argument("--flip-angle-degrees", type=float, default=90.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.points < 8:
        raise ValueError("points must be at least 8")
    if args.acquisition_time_s <= 0.0:
        raise ValueError("acquisition_time_s must be positive")
    if args.field_spread_hz < 0.0:
        raise ValueError("field_spread_hz must be non-negative")

    plt = load_matplotlib(headless=args.output is not None)
    case = build_earth_field_case(args)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    axes[0, 0].plot(
        case.prep_times_seconds,
        case.prepared_vs_time,
        label="after polarizer",
    )
    axes[0, 0].plot(
        case.prep_times_seconds,
        case.transferred_vs_time,
        label=f"after {args.transfer_delay_s:g} s transfer",
    )
    axes[0, 0].axhline(
        case.full_prepolarized_m0,
        linestyle="--",
        color="0.35",
        label="full polarizer equilibrium",
    )
    axes[0, 0].axvline(args.prepolarization_time_s, color="0.2", linewidth=1.0)
    axes[0, 0].set_xlabel("Prepolarization time (s)")
    axes[0, 0].set_ylabel("M0 in Earth's-field thermal units")
    axes[0, 0].set_title("Electromagnet Prepolarization")
    axes[0, 0].legend()

    axes[0, 1].plot(case.transfer_delays_seconds, case.transferred_vs_delay)
    axes[0, 1].axvline(args.transfer_delay_s, color="0.2", linewidth=1.0)
    axes[0, 1].axhline(1.0, linestyle="--", color="0.35", label="Earth thermal M0")
    axes[0, 1].set_xlabel("Transfer delay before pulse (s)")
    axes[0, 1].set_ylabel("M0 at detection")
    axes[0, 1].set_title("Relaxation During Field Switch / Transfer")
    axes[0, 1].legend()

    axes[1, 0].plot(
        1e3 * case.time_seconds,
        np.real(case.baseband_signal),
        label="real baseband",
    )
    axes[1, 0].plot(
        1e3 * case.time_seconds,
        case.envelope,
        color="tab:orange",
        label="envelope",
    )
    axes[1, 0].set_xlabel("Time after 90-degree pulse (ms)")
    axes[1, 0].set_ylabel("Demodulated signal")
    axes[1, 0].set_title("Earth's-Field FID")
    axes[1, 0].legend()

    axes[1, 1].plot(case.spectrum_offsets_hz, case.spectrum)
    axes[1, 1].set_xlim(-8.0 * max(args.field_spread_hz, 0.25), 8.0 * max(args.field_spread_hz, 0.25))
    axes[1, 1].set_xlabel("Frequency offset from Earth's-field Larmor (Hz)")
    axes[1, 1].set_ylabel("Normalized spectrum")
    axes[1, 1].set_title(f"Spectrum Around {case.earth_larmor_hz:.1f} Hz")

    print("Electromagnet-prepolarized Earth's-field NMR")
    print(f"Earth-field proton Larmor Hz: {case.earth_larmor_hz:.6g}")
    print(f"polarizer proton Larmor Hz: {case.polarizer_larmor_hz:.6g}")
    print(f"T1 in polarizer s: {case.t1_polarizer_seconds:.6g}")
    print(f"T1 in Earth field s: {case.t1_earth_seconds:.6g}")
    print(f"T2 in Earth field s: {case.t2_earth_seconds:.6g}")
    print(f"prepared M0 before transfer: {case.prepared_m0:.6g}")
    print(f"detected M0 after transfer: {case.detected_m0:.6g}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
