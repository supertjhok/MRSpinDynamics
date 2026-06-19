"""Plot low-field heteronuclear J-editing mixture models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling import (  # noqa: E402
    carbon_detected_j_modulation,
    fit_known_j_spectrum,
    j_modulation_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=129, help="Encoding-time samples.")
    parser.add_argument("--max-time-ms", type=float, default=12.8, help="Maximum encoding time.")
    parser.add_argument("--cycles", type=int, default=1, help="J-encoding cycles.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    tau = np.linspace(0.0, args.max_time_ms * 1e-3, args.points)
    couplings = np.array([125.0, 160.0], dtype=np.float64)
    amplitudes = np.array([0.85, 0.15], dtype=np.float64)

    proton_signal = j_modulation_curve(
        tau,
        couplings,
        amplitudes,
        cycles=args.cycles,
    )
    carbon_ch2 = carbon_detected_j_modulation(
        tau,
        couplings,
        abundances=amplitudes,
        proton_counts=[2, 1],
        cycles=args.cycles,
    )
    carbon_ch3 = carbon_detected_j_modulation(
        tau,
        couplings,
        abundances=amplitudes,
        proton_counts=[3, 1],
        cycles=args.cycles,
    )
    fit = fit_known_j_spectrum(
        tau,
        proton_signal,
        couplings,
        cycles=args.cycles,
        include_background=False,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    tau_ms = 1e3 * tau

    axes[0, 0].plot(tau_ms, proton_signal, label="85% 125 Hz + 15% 160 Hz")
    axes[0, 0].plot(tau_ms, fit.fitted, "--", label="known-J fit")
    axes[0, 0].set_title("Proton-Detected J Modulation")
    axes[0, 0].set_xlabel("Encoding time N tau (ms)")
    axes[0, 0].set_ylabel("Normalized echo amplitude")
    axes[0, 0].legend()

    axes[0, 1].plot(tau_ms, carbon_ch2, label="aliphatic CH2 + aromatic CH")
    axes[0, 1].plot(tau_ms, carbon_ch3, label="aliphatic CH3 + aromatic CH")
    axes[0, 1].set_title("Carbon-Detected cos(...)^n Model")
    axes[0, 1].set_xlabel("Encoding time N tau (ms)")
    axes[0, 1].set_ylabel("Relative echo amplitude")
    axes[0, 1].legend()

    axes[1, 0].bar(["125 Hz", "160 Hz"], fit.amplitudes, color=["tab:blue", "tab:orange"])
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Recovered Known-J Spectrum")
    axes[1, 0].set_ylabel("Amplitude")

    residual = proton_signal - fit.fitted
    axes[1, 1].plot(tau_ms, residual, color="tab:green")
    axes[1, 1].axhline(0.0, color="0.4", linewidth=1)
    axes[1, 1].set_title(f"Fit Residual, ||r||={fit.residual_norm:.2e}")
    axes[1, 1].set_xlabel("Encoding time N tau (ms)")
    axes[1, 1].set_ylabel("Residual")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
