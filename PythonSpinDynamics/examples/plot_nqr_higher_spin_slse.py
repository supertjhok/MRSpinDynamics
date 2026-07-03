r"""Pulsed SLSE detection of higher-spin quadrupolar nuclei (27Al 5/2, 51V 7/2).

Spin-lock spin-echo (SLSE) is the workhorse pulsed-NQR detection train: a spin-lock
excitation followed by a comb of refocusing pulses that build a long-lived echo
train, so many echoes can be co-added for sensitivity. This example runs the full
``(2I+1)``-level density-matrix SLSE on the *fundamental* zero-field line of two
real higher-spin nuclei -- ``27Al`` (spin 5/2) and ``51V`` (spin 7/2) -- over a
powder, exactly as the spin-3/2 chlorine example does, showing that the pulsed
detection machinery now runs for spin > 3/2.

Panels:

1. Powder SLSE echo train (|echo| vs echo time) for 27Al and 51V, with the
   phenomenological ``T2e`` envelope overlaid -- the decay a real experiment
   co-adds against.
2. The single refocused echo shape (real part and magnitude) of the 27Al train,
   from ``simulate_full_echo``.

Both nuclei are built from the isotope presets (``quadrupolar_site``), so only the
fundamental line ``nu_Q`` and asymmetry ``eta`` are supplied. Run with
``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full density-matrix SLSE echo-train detection of higher-spin NQR "
            "nuclei (27Al 5/2, 51V 7/2) over a powder."
        )
    )
    parser.add_argument("--nu-al-mhz", type=float, default=0.5,
                        help="27Al fundamental line nu_Q (MHz).")
    parser.add_argument("--nu-v-mhz", type=float, default=0.4,
                        help="51V fundamental line nu_Q (MHz).")
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--nutation-khz", type=float, default=12.0)
    parser.add_argument("--echo-spacing-us", type=float, default=300.0)
    parser.add_argument("--num-echoes", type=int, default=20)
    parser.add_argument("--t2e-ms", type=float, default=4.0,
                        help="Phenomenological spin-locked decay time (ms).")
    parser.add_argument("--n-theta", type=int, default=6)
    parser.add_argument("--n-phi", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.nqr import (
        diagonalize_site,
        powder_average_grid,
        quadrupolar_site,
        simulate_full_slse,
    )

    nutation = args.nutation_khz * 1e3
    te = args.echo_spacing_us * 1e-6
    t2e = args.t2e_ms * 1e-3
    exc = 25e-6
    powder = powder_average_grid(args.n_theta, args.n_phi)

    nuclei = [
        ("27Al", "C2", args.nu_al_mhz * 1e6, 2.5),
        ("51V", "C3", args.nu_v_mhz * 1e6, 3.5),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    al_result = None
    for iso, col, nu_q, spin in nuclei:
        site = quadrupolar_site(iso, nu_q_hz=nu_q, eta=args.eta)
        nu1 = min(t.frequency_hz for t in diagonalize_site(site).transitions)
        slse = simulate_full_slse(
            site, nutation_hz=nutation, excitation_duration_seconds=exc,
            refocus_duration_seconds=2 * exc, echo_spacing_seconds=te,
            num_echoes=args.num_echoes, orientations=powder, rf_frequency_hz=nu1,
            t2e_seconds=t2e,
        )
        amp = np.abs(slse.echo_amplitudes)
        ax[0].plot(slse.echo_times * 1e3, amp / amp[0], col + "o-", ms=4,
                   label=f"{iso} (I={spin:g}), nu_Q={nu_q/1e6:.2f} MHz")
        print(f"{iso}: powder SLSE {args.num_echoes} echoes, "
              f"last/first = {amp[-1]/amp[0]:.3f}")
        if iso == "27Al":
            al_result = slse

    echo_time_axis = (np.arange(args.num_echoes) + 1.0) * te
    ax[0].plot(echo_time_axis * 1e3, np.exp(-echo_time_axis / t2e), "k--", lw=0.9,
               label=f"exp(-t/T2e), T2e={args.t2e_ms:g} ms")
    ax[0].set_xlabel("echo time (ms)")
    ax[0].set_ylabel("normalized |echo|")
    ax[0].set_title("Powder SLSE echo train")
    ax[0].legend(fontsize=8)

    # (b) How the powder average is built: the per-crystallite echo trains spread
    # around the orientation-weighted mean (the co-added detection signal).
    times_ms = al_result.echo_times * 1e3
    local = np.abs(np.asarray(al_result.local_echo_amplitudes))
    scale = np.abs(al_result.echo_amplitudes[0])
    for row in local:
        ax[1].plot(times_ms, row / scale, color="0.75", lw=0.6, alpha=0.6)
    ax[1].plot(times_ms, np.abs(al_result.echo_amplitudes) / scale, "C2o-", ms=4,
               label="powder average")
    ax[1].plot([], [], color="0.75", lw=0.6, label="per crystallite")
    ax[1].set_xlabel("echo time (ms)")
    ax[1].set_ylabel("normalized |echo|")
    ax[1].set_title("27Al: crystallite spread vs. powder average")
    ax[1].legend(fontsize=8)

    fig.suptitle("Higher-spin pulsed NQR detection: full-model SLSE on the fundamental line")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
