"""Quadrupolar-nucleus ESEEM/HYSCORE/ENDOR (e.g. 14N exact cancellation).

For an S=1/2 electron coupled to a quadrupolar nucleus (I=1 or 3/2) this shows
the three-pulse ESEEM trace and spectrum, a 2D HYSCORE spectrum, and Davies/Mims
ENDOR, with the per-manifold nuclear frequencies marked. The defaults reproduce
the classic 14N (I=1) exact-cancellation regime, in which one electron manifold
becomes a pure nuclear quadrupole interaction and contributes the three sharp NQR
lines nu_+/- = nu_Q(1 +/- eta/3) and nu_0 = (2/3) nu_Q eta.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    HyperfineCoupling,
    davies_endor_spectrum,
    endor_frequencies,
    eseem_spectrum,
    hyscore_signal,
    hyscore_spectrum,
    manifold_frequencies,
    mims_endor_spectrum,
    three_pulse_eseem_quantum,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuclear-spin", type=float, default=1.0, help="1 or 1.5.")
    parser.add_argument("--larmor-mhz", type=float, default=1.05, help="Nuclear Larmor (14N at X-band ~1.05).")
    parser.add_argument(
        "--secular-mhz",
        type=float,
        default=2.10,
        help="Secular hyperfine A; A = 2*larmor gives exact cancellation.",
    )
    parser.add_argument("--pseudosecular-mhz", type=float, default=0.15)
    parser.add_argument("--quadrupole-mhz", type=float, default=2.625, help="Quadrupole frequency nu_Q.")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--dwell-ns", type=float, default=50.0, help="ESEEM/HYSCORE dwell time.")
    parser.add_argument("--eseem-points", type=int, default=1024)
    parser.add_argument("--hyscore-points", type=int, default=72)
    parser.add_argument("--tau-ns", type=float, default=150.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    coupling = HyperfineCoupling(
        larmor_hz=args.larmor_mhz * 1e6,
        secular_hz=args.secular_mhz * 1e6,
        pseudosecular_hz=args.pseudosecular_mhz * 1e6,
        nuclear_spin=args.nuclear_spin,
        quadrupole_hz=args.quadrupole_mhz * 1e6,
        eta=args.eta,
    )
    alpha, beta = manifold_frequencies(coupling)
    tau = args.tau_ns * 1e-9
    dwell = args.dwell_ns * 1e-9

    print(
        f"I={args.nuclear_spin}: alpha-manifold {np.round(alpha / 1e6, 3)} MHz, "
        f"beta-manifold {np.round(beta / 1e6, 3)} MHz"
    )
    if np.isclose(args.nuclear_spin, 1.0):
        nu_q = args.quadrupole_mhz
        print(
            "expected NQR lines (MHz): "
            f"nu+={nu_q * (1 + args.eta / 3):.3f} nu-={nu_q * (1 - args.eta / 3):.3f} "
            f"nu0={(2 / 3) * nu_q * args.eta:.3f}"
        )

    # Three-pulse ESEEM and its spectrum.
    times = np.arange(args.eseem_points) * dwell
    trace = three_pulse_eseem_quantum(times, coupling, tau_seconds=tau)
    freqs, spectrum = eseem_spectrum(times, trace, zero_fill=4)

    # HYSCORE 2D.
    grid = np.arange(args.hyscore_points) * dwell
    signal = hyscore_signal(grid, grid, coupling, tau_seconds=tau)
    hys = hyscore_spectrum(grid, grid, signal, zero_fill=4)

    # ENDOR.
    fmax = 1.3 * float(max(alpha.max(), beta.max()))
    rf = np.linspace(0.0, fmax, 2000)
    davies = davies_endor_spectrum(rf, coupling, linewidth_hz=0.05e6)
    mims = mims_endor_spectrum(rf, coupling, tau_seconds=tau, linewidth_hz=0.05e6)
    lines = endor_frequencies(coupling)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(1e6 * times, trace, color="tab:blue")
    axes[0, 0].set_xlabel("T (us)")
    axes[0, 0].set_ylabel("V(T)")
    axes[0, 0].set_title("Three-Pulse ESEEM")

    axes[0, 1].plot(freqs / 1e6, spectrum, color="tab:green")
    for nu in alpha:
        axes[0, 1].axvline(nu / 1e6, color="tab:red", linestyle=":", linewidth=1)
    for nu in beta:
        axes[0, 1].axvline(nu / 1e6, color="tab:purple", linestyle="--", linewidth=1)
    axes[0, 1].set_xlim(0, fmax / 1e6)
    axes[0, 1].set_xlabel("Frequency (MHz)")
    axes[0, 1].set_ylabel("|FT|")
    axes[0, 1].set_title("ESEEM Spectrum (alpha :, beta --)")

    extent = [
        hys.frequencies2_hz[0] / 1e6,
        hys.frequencies2_hz[-1] / 1e6,
        hys.frequencies1_hz[0] / 1e6,
        hys.frequencies1_hz[-1] / 1e6,
    ]
    axes[1, 0].imshow(
        hys.spectrum, origin="lower", extent=extent, aspect="auto", cmap="inferno"
    )
    axes[1, 0].set_xlim(0, fmax / 1e6)
    axes[1, 0].set_ylim(0, fmax / 1e6)
    axes[1, 0].set_xlabel("nu2 (MHz)")
    axes[1, 0].set_ylabel("nu1 (MHz)")
    axes[1, 0].set_title("HYSCORE")

    axes[1, 1].plot(rf / 1e6, davies.spectrum, label="Davies", color="tab:blue")
    axes[1, 1].plot(rf / 1e6, mims.spectrum, label="Mims", color="tab:orange")
    for nu in lines:
        axes[1, 1].axvline(nu / 1e6, color="0.7", linestyle=":", linewidth=0.8)
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
