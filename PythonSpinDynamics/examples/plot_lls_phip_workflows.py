"""Plot a complete SLIC LLS workflow and hydrogenative PHIP spectra.

The left panel prepares singlet order, stores it with a measured ``T_S``, and
reconverts it for detection.  The right panel compares high-field PASADENA
with an explicitly trajectory-defined ALTADENA transport model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling import coupled_spin_system  # noqa: E402
from spin_dynamics.hyperpolarization import (  # noqa: E402
    PHIPFieldSegment,
    simulate_hydrogenative_phip,
    simulate_slic_lls,
)


def _spectrum(signal: np.ndarray, dwell_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    apodized = signal * np.exp(-np.arange(signal.size) * dwell_seconds * 2.0)
    spectrum = np.fft.fftshift(np.fft.fft(apodized))
    frequency = np.fft.fftshift(np.fft.fftfreq(signal.size, dwell_seconds))
    return frequency, np.real(spectrum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts", type=float, default=18.0, help="Singlet lifetime in seconds.")
    parser.add_argument("--para-fraction", type=float, default=0.90)
    parser.add_argument("--pairwise-fraction", type=float, default=0.72)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    pair = coupled_spin_system(
        offsets_hz=[-0.35, 0.35],
        couplings_hz=[[0.0, 7.0], [7.0, 0.0]],
        labels=["A", "B"],
    )
    storage_times = np.linspace(0.0, 3.0 * args.ts, 81)
    lls = simulate_slic_lls(
        pair,
        storage_times,
        singlet_lifetime_seconds=args.ts,
    )

    product = coupled_spin_system(
        offsets_hz=[-45.0, 45.0],
        couplings_hz=[[0.0, 7.2], [7.2, 0.0]],
        labels=["H_a", "H_b"],
    )
    dwell = 2.5e-4
    acquisition_times = np.arange(1024) * dwell
    shared = dict(
        para_fraction=args.para_fraction,
        pairwise_addition_fraction=args.pairwise_fraction,
        pulse_flip_angle_radians=np.pi / 4.0,
        t2_seconds=0.12,
    )
    pasadena = simulate_hydrogenative_phip(
        product,
        acquisition_times,
        protocol="pasadena",
        reaction_time_seconds=0.05,
        **shared,
    )
    ramp = [
        PHIPFieldSegment(scale, 1.5e-3)
        for scale in np.linspace(0.02, 1.0, 48)
    ]
    altadena = simulate_hydrogenative_phip(
        product,
        acquisition_times,
        protocol="altadena",
        reaction_time_seconds=0.05,
        field_trajectory=ramp,
        **shared,
    )
    frequency, pasadena_spectrum = _spectrum(pasadena.signal, dwell)
    _, altadena_spectrum = _spectrum(altadena.signal, dwell)
    scale = max(np.max(np.abs(pasadena_spectrum)), np.max(np.abs(altadena_spectrum)))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    axes[0].plot(storage_times, lls.normalized_signal, label="SLIC readout")
    axes[0].plot(
        storage_times,
        lls.normalized_signal[0] * np.exp(-storage_times / args.ts),
        "--",
        label=r"$\exp(-t/T_S)$",
    )
    axes[0].set(
        title="LLS preparation–storage–readout",
        xlabel="Storage time (s)",
        ylabel="Signal / initial transverse signal",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    window = np.abs(frequency) <= 120.0
    axes[1].plot(frequency[window], pasadena_spectrum[window] / scale, label="PASADENA")
    axes[1].plot(frequency[window], altadena_spectrum[window] / scale, label="ALTADENA")
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set(
        title="Hydrogenative PHIP product spectra",
        xlabel="Offset (Hz)",
        ylabel="Real spectrum (shared normalization)",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(
        rf"Dense two-spin reference workflows: $T_S$={args.ts:g} s, "
        rf"para-H$_2$={100 * args.para_fraction:.0f}%, "
        rf"pairwise={100 * args.pairwise_fraction:.0f}%"
    )

    print("LLS and hydrogenative PHIP workflow")
    print(f"  prepared singlet amplitude: {lls.prepared_singlet_amplitude:.6f}")
    print(f"  zero-storage readout: {lls.normalized_signal[0]:.6f}")
    print(f"  effective para excess: {args.pairwise_fraction * (args.para_fraction - 0.25):.6f}")
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
