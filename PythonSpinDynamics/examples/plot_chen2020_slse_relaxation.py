"""Reproduce the field-dependent SLSE relaxation of 35Cl in NaClO3.

Chen et al., J. Magn. Reson. 311 (2020) 106660 ("Single-shot spatially-localized
NQR") report that the SLSE transverse relaxation time T2,SLSE of 35Cl in NaClO3
powder *increases* strongly with a weak static field B0 (Table 1: 1.4 ms at 0 G
up to 28 ms at 41 G), while the linewidth (1/T2*) increases. The mechanism is
homonuclear 35Cl-35Cl flip-flop (cross-relaxation): at B0 = 0 every Cl pair is a
"like" pair (one NQR line) with strong cross-relaxation; a weak field splits the
spin-3/2 line into a Zeeman quadruplet (powder-broadened), so neighbour pairs
fall out of the flip-flop bandwidth and the cross-relaxation is quenched.

This script computes the field dependence *from the package's spin-3/2 Zeeman
diagonalization* and fits a two-parameter relaxation model:

    1/T2,SLSE(B0) = R_floor + R_xrelax * [ Delta_f(0) / Delta_f(B0) ],

where Delta_f(B0) is the powder-averaged 35Cl Zeeman NQR linewidth (the
intensity-weighted RMS transition shift from `simulate_weak_b0_spectrum`, in
quadrature with the measured B0=0 linewidth). The 1/Delta_f scaling is the
standard secular-dipolar cross-relaxation result (flip-flop rate proportional to
the spectral overlap at zero frequency difference). The two fitted constants are
physical: R_xrelax is the B0=0 flip-flop rate and R_floor is the residual,
field-independent (T1-limited) rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import QuadrupolarSite, simulate_weak_b0_spectrum  # noqa: E402
from spin_dynamics.nqr.orientations import b0_b1_powder_average_grid  # noqa: E402


# NaClO3 35Cl: nu_Q ~ 30.656 MHz, axial EFG (eta ~ 0), gamma = 4.1717 MHz/T.
SITE = QuadrupolarSite(spin=1.5, isotope="35Cl", quadrupole_frequency_hz=30.656e6,
                       eta=0.0, gamma_hz_per_t=4.1717e6)

# Table 1 (room temperature), B0 in gauss -> T2,SLSE in ms.
B0_TABLE_G = np.array([0.0, 8.0, 16.8, 25.0, 33.0, 41.0])
T2_TABLE_MS = np.array([1.4, 12.0, 17.0, 21.3, 24.9, 28.0])
T2STAR0_S = 373e-6  # measured B0=0 SLSE T2* -> intrinsic linewidth


def zeeman_linewidth_hz(b0_gauss: float, grid, sigma0_hz: float) -> float:
    """Powder 35Cl Zeeman NQR linewidth: intensity-weighted RMS transition shift
    in quadrature with the intrinsic B0=0 width."""

    result = simulate_weak_b0_spectrum(
        SITE, b0_gauss * 1e-4, orientations=grid, broadening_hz=10.0,
        points=16, weak_ratio_action="ignore",
    )
    centers = np.array([t.frequency_hz - result.reference_frequency_hz
                        for t in result.transitions])
    weights = np.array([t.intensity for t in result.transitions])
    weights = weights / weights.sum()
    mean = float((weights * centers).sum())
    variance = float((weights * (centers - mean) ** 2).sum())
    return float(np.sqrt(variance + sigma0_hz ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-theta", type=int, default=18, help="Powder polar samples.")
    parser.add_argument("--n-phi", type=int, default=36, help="Powder azimuthal samples.")
    parser.add_argument("--max-field-g", type=float, default=45.0,
                        help="Maximum B0 for the model curve, in gauss.")
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    # B0=0 Gaussian sigma from the measured SLSE T2* (FWHM = 1/(pi T2*)).
    sigma0 = 1.0 / (np.pi * T2STAR0_S) / (2 * np.sqrt(2 * np.log(2)))
    grid = b0_b1_powder_average_grid(args.n_theta, args.n_phi, n_chi=1,
                                     b1_b0_angle=0.0)

    # Field dependence from the package: S(B0) = Delta_f(0) / Delta_f(B0).
    lw_table = np.array([zeeman_linewidth_hz(b, grid, sigma0) for b in B0_TABLE_G])
    s_table = lw_table[0] / lw_table

    # Two-parameter linear least-squares fit of 1/T2,SLSE = R_floor + R_x * S.
    rate = 1.0 / (T2_TABLE_MS * 1e-3)
    design = np.column_stack([np.ones_like(s_table), s_table])
    (r_floor, r_x), *_ = np.linalg.lstsq(design, rate, rcond=None)
    predicted_ms = 1e3 / (r_floor + r_x * s_table)
    rms = float(np.sqrt(np.mean((rate - design @ [r_floor, r_x]) ** 2)))

    print(f"R_floor   = {r_floor:6.1f} s^-1  (T2 floor {1e3/r_floor:.0f} ms; "
          f"cf. measured T1 ~ 35-50 ms)")
    print(f"R_xrelax  = {r_x:6.1f} s^-1  (B0=0 flip-flop cross-relaxation rate)")
    print(f"RMS resid = {rms:6.1f} s^-1  over a 714 -> 36 s^-1 range")
    print(f"linewidth Delta_f: {lw_table[0]:.0f} Hz (0 G) -> {lw_table[-1]:.0f} Hz "
          f"(41 G); gamma*B0 at 41 G = {SITE.gamma_hz_per_t*41e-4:.0f} Hz")
    print(f"\n{'B0(G)':>6} {'T2 model(ms)':>13} {'T2 meas(ms)':>12} {'ratio':>6}")
    for b, pm, mm in zip(B0_TABLE_G, predicted_ms, T2_TABLE_MS):
        print(f"{b:>6} {pm:>13.1f} {mm:>12.1f} {pm/mm:>6.2f}")

    # Smooth model curve.
    fields = np.linspace(0.0, args.max_field_g, 46)
    lw_curve = np.array([zeeman_linewidth_hz(b, grid, sigma0) for b in fields])
    t2_curve_ms = 1e3 / (r_floor + r_x * (lw_table[0] / lw_curve))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].plot(fields, t2_curve_ms, "C0-",
                 label="model: R_floor + R_x * Delta_f(0)/Delta_f(B0)")
    axes[0].plot(B0_TABLE_G, T2_TABLE_MS, "ko", label="Chen 2020, Table 1")
    axes[0].axhline(1e3 / r_floor, color="0.6", ls="--", lw=1,
                    label=f"floor {1e3/r_floor:.0f} ms ~ T1")
    axes[0].set_xlabel("static field B0 (G)")
    axes[0].set_ylabel("T2,SLSE (ms)")
    axes[0].set_title("Field-dependent SLSE relaxation of 35Cl (NaClO3)")
    axes[0].legend(fontsize=8)

    # Flip-flop survival S(B0): the dimensionless factor that quenches the
    # cross-relaxation. Compare the package model with the value implied by the
    # measured rates, S = (1/T2,meas - R_floor) / R_xrelax.
    s_curve = lw_table[0] / lw_curve
    s_implied = (rate - r_floor) / r_x
    axes[1].semilogy(fields, s_curve, "C0-",
                     label="model S = Delta_f(0)/Delta_f(B0)")
    axes[1].semilogy(B0_TABLE_G, s_implied, "ko",
                     label="implied by Table 1 rates")
    axes[1].set_xlabel("static field B0 (G)")
    axes[1].set_ylabel("flip-flop survival S(B0)")
    axes[1].set_title("Zeeman splitting quenches 35Cl-35Cl cross-relaxation")
    axes[1].legend(fontsize=8)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"\nsaved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
