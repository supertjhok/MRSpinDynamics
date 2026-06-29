"""Pulsed dipolar/hyperfine ESR: ESEEM, HYSCORE, and ENDOR for an S=1/2,I=1/2 pair.

Shows the three-pulse ESEEM trace and its spectrum, a 2D HYSCORE spectrum with
its cross-peaks, and Davies vs Mims ENDOR (including a Mims blind spot) for a
single anisotropic hyperfine coupling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    HyperfineCoupling,
    cross_peak_positions,
    davies_endor_spectrum,
    eseem_spectrum,
    hyscore_signal,
    hyscore_spectrum,
    mims_endor_spectrum,
    nuclear_frequencies,
    three_pulse_eseem,
    three_pulse_eseem_quantum,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--larmor-mhz", type=float, default=14.5)
    parser.add_argument("--secular-mhz", type=float, default=3.0)
    parser.add_argument("--pseudosecular-mhz", type=float, default=2.5)
    parser.add_argument("--tau-ns", type=float, default=136.0)
    parser.add_argument("--eseem-points", type=int, default=400)
    parser.add_argument("--hyscore-points", type=int, default=72)
    parser.add_argument("--hyscore-step-ns", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    coupling = HyperfineCoupling(
        larmor_hz=args.larmor_mhz * 1e6,
        secular_hz=args.secular_mhz * 1e6,
        pseudosecular_hz=args.pseudosecular_mhz * 1e6,
    )
    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    print(
        f"nu_alpha={nu_alpha / 1e6:.3f} MHz, nu_beta={nu_beta / 1e6:.3f} MHz, "
        f"cross-peaks at {[(round(a / 1e6, 2), round(b / 1e6, 2)) for a, b in cross_peak_positions(coupling)]}"
    )

    # Three-pulse ESEEM (analytic vs density matrix) and its spectrum.
    T = np.linspace(0.0, 8.0e-6, args.eseem_points)
    tau = args.tau_ns * 1e-9
    eseem_analytic = three_pulse_eseem(T, coupling, tau_seconds=tau)
    eseem_quantum = three_pulse_eseem_quantum(T, coupling, tau_seconds=tau)
    freqs, spectrum = eseem_spectrum(T, eseem_analytic, zero_fill=8)

    # HYSCORE 2D.
    grid = np.arange(args.hyscore_points) * args.hyscore_step_ns * 1e-9
    signal = hyscore_signal(grid, grid, coupling, tau_seconds=tau)
    hys = hyscore_spectrum(grid, grid, signal, zero_fill=4)

    # ENDOR: Davies and a Mims trace at a non-blind tau.
    rf = np.linspace(8.0e6, 22.0e6, 2000)
    davies = davies_endor_spectrum(rf, coupling, linewidth_hz=1.5e5)
    mims = mims_endor_spectrum(rf, coupling, tau_seconds=tau, linewidth_hz=1.5e5)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(1e6 * T, eseem_analytic, label="analytic", color="tab:blue")
    axes[0, 0].plot(1e6 * T, eseem_quantum, "--", label="density matrix", color="tab:orange")
    axes[0, 0].set_xlabel("T (us)")
    axes[0, 0].set_ylabel("V(T)")
    axes[0, 0].set_title("Three-Pulse ESEEM")
    axes[0, 0].legend()

    axes[0, 1].plot(freqs / 1e6, spectrum, color="tab:green")
    for nu in (nu_alpha, nu_beta):
        axes[0, 1].axvline(nu / 1e6, color="tab:red", linestyle=":")
    axes[0, 1].set_xlim(0, 1.4 * max(nu_alpha, nu_beta) / 1e6)
    axes[0, 1].set_xlabel("Frequency (MHz)")
    axes[0, 1].set_title("ESEEM Spectrum")

    extent = [
        hys.frequencies2_hz[0] / 1e6,
        hys.frequencies2_hz[-1] / 1e6,
        hys.frequencies1_hz[0] / 1e6,
        hys.frequencies1_hz[-1] / 1e6,
    ]
    axes[1, 0].imshow(
        hys.spectrum, origin="lower", extent=extent, aspect="auto", cmap="inferno"
    )
    axes[1, 0].set_xlim(0, 1.4 * max(nu_alpha, nu_beta) / 1e6)
    axes[1, 0].set_ylim(0, 1.4 * max(nu_alpha, nu_beta) / 1e6)
    axes[1, 0].set_xlabel("nu2 (MHz)")
    axes[1, 0].set_ylabel("nu1 (MHz)")
    axes[1, 0].set_title("HYSCORE (cross-peaks)")

    axes[1, 1].plot(rf / 1e6, davies.spectrum, label="Davies", color="tab:blue")
    axes[1, 1].plot(rf / 1e6, mims.spectrum, label="Mims", color="tab:orange")
    axes[1, 1].set_xlabel("RF frequency (MHz)")
    axes[1, 1].set_ylabel("ENDOR intensity")
    axes[1, 1].set_title("Davies vs Mims ENDOR")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
