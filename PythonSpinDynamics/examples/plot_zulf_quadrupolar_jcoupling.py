"""Earth's-field J-coupled spectra with quadrupolar (14N) line broadening.

At Earth's field (~50 uT) chemical shifts vanish and the spectrum is a
J-coupled spectrum (JCS): heteronuclear and homonuclear scalar couplings split
lines, with all nuclei inside a ~2 kHz band. A quadrupolar nucleus such as 14N
(spin-1) relaxes quickly, and its relaxation rate R(14N) = R1 = R2 broadens the
splittings of any spin J-coupled to it -- resolved multiplet -> coalescence ->
self-decoupled singlet as R(14N) grows (Altenhof et al., J. Magn. Reson. 355
(2023) 107540).

This example reproduces that behaviour on a fictitious three-spin system:

* Heteronuclear panels: 1H-19F-14N. Sweeping R(14N) collapses the 19F triplet
  from 2J(F,N) and indirectly broadens the 1H region through 1H-19F coupling.
* Homonuclear panel: 1H-1H-19F. The mutual homonuclear J(H,H) reshapes the
  J-coupled spectrum (visible in ZULF because the protons are also coupled to
  the 19F reporter).
* Self-decoupling curve: the 19F peak amplitude versus R(14N) is non-monotonic,
  dipping through the coalescence regime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling import (  # noqa: E402
    multinuclear_system,
    simulate_zulf_spectrum,
)


def _region(frequencies: np.ndarray, spectrum: np.ndarray, lo: float, hi: float):
    mask = (frequencies >= lo) & (frequencies <= hi)
    return frequencies[mask], np.abs(spectrum[mask])


def _spectrum(system, r14n_hz, args, detect_indices):
    nspin = system.nspin
    r = np.full(nspin, args.spin_half_rate_hz, dtype=np.float64)
    # The quadrupolar nucleus is the last site in every system built below.
    r[-1] = float(r14n_hz)
    return simulate_zulf_spectrum(
        system,
        r1_per_second=r,
        r2_per_second=r,
        dwell_seconds=args.dwell_ms * 1e-3,
        n_points=int(args.n_points),
        detect_indices=detect_indices,
        apodization_hz=args.apodization_hz,
    )


def build_data(args: argparse.Namespace) -> dict:
    """Build heteronuclear, homonuclear, and self-decoupling arrays."""

    b0 = args.b0_ut * 1e-6

    # Heteronuclear 1H-19F-14N: strong 2J(F,N), weaker 1H couplings.
    het_j = np.zeros((3, 3))
    het_j[0, 1] = het_j[1, 0] = args.j_hf_hz
    het_j[0, 2] = het_j[2, 0] = args.j_hn_hz
    het_j[1, 2] = het_j[2, 1] = args.j_fn_hz
    het = multinuclear_system(["1H", "19F", "14N"], het_j, b0)
    f_idx = het.indices_for_isotope("19F")
    h_idx = het.indices_for_isotope("1H")

    het_f = []
    het_h = []
    for r14n in args.r14n_hz:
        sf = _spectrum(het, r14n, args, f_idx)
        sh = _spectrum(het, r14n, args, h_idx)
        het_f.append(_region(sf.frequencies_hz, sf.spectrum, *args.f_window))
        het_h.append(_region(sh.frequencies_hz, sh.spectrum, *args.h_window))

    # Homonuclear 1H-1H-19F: two protons mutually coupled and both to 19F.
    homo_curves = []
    for j_hh in args.j_hh_hz:
        homo_j = np.zeros((3, 3))
        homo_j[0, 1] = homo_j[1, 0] = j_hh
        homo_j[0, 2] = homo_j[2, 0] = args.j_hf_hz
        homo_j[1, 2] = homo_j[2, 1] = 0.5 * args.j_hf_hz
        homo = multinuclear_system(["1H", "1H", "19F"], homo_j, b0)
        # No quadrupolar nucleus here; give a slow common rate.
        spec = simulate_zulf_spectrum(
            homo,
            r1_per_second=args.spin_half_rate_hz,
            r2_per_second=args.spin_half_rate_hz,
            dwell_seconds=args.dwell_ms * 1e-3,
            n_points=int(args.n_points),
            detect_indices=homo.indices_for_isotope("1H"),
            apodization_hz=args.apodization_hz,
        )
        homo_curves.append(_region(spec.frequencies_hz, spec.spectrum, *args.h_window))

    # Self-decoupling: 19F peak amplitude vs R(14N) (log sweep).
    r_sweep = np.geomspace(args.sweep_min_hz, args.sweep_max_hz, int(args.sweep_points))
    amp = np.empty_like(r_sweep)
    width = np.empty_like(r_sweep)
    for idx, r14n in enumerate(r_sweep):
        s = _spectrum(het, r14n, args, f_idx)
        _, mag = _region(s.frequencies_hz, s.spectrum, *args.f_window)
        peak = float(mag.max())
        amp[idx] = peak
        # Effective width = integral / peak (df cancels in the ratio up to a bin).
        width[idx] = float(mag.sum()) / peak if peak > 0 else np.nan

    return {
        "r14n_hz": np.asarray(args.r14n_hz, dtype=np.float64),
        "het_f": het_f,
        "het_h": het_h,
        "j_hh_hz": np.asarray(args.j_hh_hz, dtype=np.float64),
        "homo": homo_curves,
        "sweep_r14n": r_sweep,
        "sweep_amp": amp,
        "sweep_width": width,
        "larmor": {iso: float(nu) for iso, nu in zip(het.isotopes, het.larmor_hz)},
    }


def _stack(ax, curves, labels, offsets_scale=1.15):
    step = 0.0
    for (freq, mag), label in zip(curves, labels):
        norm = mag / mag.max() if mag.max() > 0 else mag
        ax.plot(freq, norm + step, lw=1.0, label=label)
        step += offsets_scale
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--b0-ut", type=float, default=50.0)
    parser.add_argument("--j-hf-hz", type=float, default=8.0)
    parser.add_argument("--j-hn-hz", type=float, default=1.0)
    parser.add_argument("--j-fn-hz", type=float, default=37.0)
    parser.add_argument(
        "--r14n-hz",
        type=float,
        nargs="+",
        default=[0.0, 100.0, 250.0, 1600.0, 1.0e5],
        help="14N relaxation rates (Hz) for the stacked heteronuclear spectra",
    )
    parser.add_argument(
        "--j-hh-hz",
        type=float,
        nargs="+",
        default=[0.0, 6.0, 12.0],
        help="homonuclear J(H,H) values (Hz) for the homonuclear panel",
    )
    parser.add_argument("--spin-half-rate-hz", type=float, default=0.3)
    parser.add_argument("--dwell-ms", type=float, default=0.2)
    parser.add_argument("--n-points", type=int, default=32768)
    parser.add_argument("--apodization-hz", type=float, default=0.8)
    parser.add_argument("--sweep-min-hz", type=float, default=1.0)
    parser.add_argument("--sweep-max-hz", type=float, default=1.0e6)
    parser.add_argument("--sweep-points", type=int, default=40)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Detection windows (Hz) around the 19F and 1H Larmor frequencies at ~50 uT.
    nu_h = 42.5774806e6 * args.b0_ut * 1e-6
    nu_f = 40.0775017e6 * args.b0_ut * 1e-6
    args.f_window = (nu_f - 90.0, nu_f + 90.0)
    args.h_window = (nu_h - 60.0, nu_h + 60.0)

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), constrained_layout=True)

    labels = [f"R(14N)={r:g} Hz" for r in data["r14n_hz"]]
    _stack(axes[0, 0], data["het_f"], labels)
    axes[0, 0].set_title("Heteronuclear 19F JCS vs R(14N)")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].legend(fontsize=7, loc="upper right")

    _stack(axes[0, 1], data["het_h"], labels)
    axes[0, 1].set_title("Heteronuclear 1H JCS vs R(14N)")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].legend(fontsize=7, loc="upper right")

    homo_labels = [f"J(H,H)={j:g} Hz" for j in data["j_hh_hz"]]
    _stack(axes[1, 0], data["homo"], homo_labels)
    axes[1, 0].set_title("Homonuclear 1H JCS (1H-1H-19F) vs J(H,H)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].legend(fontsize=7, loc="upper right")

    amp_ax = axes[1, 1]
    amp_ax.semilogx(data["sweep_r14n"], data["sweep_amp"] / data["sweep_amp"].max(),
                    color="tab:red", label="19F peak amplitude")
    amp_ax.set_xlabel("R(14N) (Hz)")
    amp_ax.set_ylabel("Normalized 19F peak amplitude")
    amp_ax.set_title("Quadrupolar Self-Decoupling of 19F")
    amp_ax.grid(True, which="both", alpha=0.2)
    amp_ax.legend(fontsize=8)

    print("Earth's-field J-coupled spectra with 14N quadrupolar broadening")
    print(f"B0 = {args.b0_ut:g} uT   Larmor Hz: " + ", ".join(
        f"{iso}={nu:.1f}" for iso, nu in data["larmor"].items()))
    trough = int(np.argmin(data["sweep_amp"]))
    print(f"19F amplitude minimum near R(14N) = {data['sweep_r14n'][trough]:.4g} Hz")
    print(f"couplings: J(H,F)={args.j_hf_hz:g}  J(F,N)={args.j_fn_hz:g}  "
          f"J(H,N)={args.j_hn_hz:g} Hz")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
