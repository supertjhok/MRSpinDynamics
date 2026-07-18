"""Performance of the PEEC solver: surface-impedance backend and graded meshing.

The volume PEEC solver must tile the cross-section finer than the skin depth, so its cost
(``O(K^2)`` in the cell count) explodes at high frequency where deep skin needs many cells.
Two accelerations fix this:

1. **Surface-impedance backend** (`extract_impedance_surface`) -- for deep skin, place a ring
   of filaments around the cross-section boundary with the analytic surface impedance
   ``Z_s = (1+j) rho/delta``. Cost is independent of ``a/delta`` and the accuracy *improves*
   with frequency, the opposite of the volume solver.
2. **Graded meshing** (`grading > 1`) -- concentrate volume cells toward the surface so the
   skin layer is resolved with far fewer cells in the moderate-skin regime.

The script prints three studies and, with ``--save out.png``, plots them:
(a) AC resistance vs frequency -- volume (fixed coarse mesh, diverges) vs surface-impedance
    (tracks the exact Kelvin curve); (b) resistance error vs cell count -- uniform vs graded;
(c) the ``O(K^2)`` build-time scaling that motivates each acceleration.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from scipy.special import bei, beip, ber, berp

from spin_dynamics.fields.coil_peec import Conductor, extract_impedance, extract_impedance_surface
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER
from spin_dynamics.fields.magnetostatics import MU0

RHO = ANNEALED_COPPER.resistivity
A = 1e-3           # wire radius / half-side (m)
LEN = 0.1          # conductor length (m)
SIDE = 1e-3        # square side (m)


def kelvin_r(a, freq):
    """Exact AC resistance (ohm) of a round wire (Kelvin functions)."""
    delta = np.sqrt(2 * RHO / (2 * np.pi * freq * MU0))
    q = np.sqrt(2) * a / delta
    ratio = (q / 2) * (ber(q) * beip(q) - bei(q) * berp(q)) / (berp(q) ** 2 + beip(q) ** 2)
    return ratio * RHO * LEN / (np.pi * a**2)


def _round_wire(**kw):
    return Conductor(np.array([[0, 0, 0], [0, 0, LEN]]), wire_radius=A, material=ANNEALED_COPPER, **kw)


def _square_bar(grading=1.0, n=8):
    return Conductor(np.array([[0, 0, 0], [0, 0, LEN]]), material=ANNEALED_COPPER,
                     cross_section="rect", width=SIDE, height=SIDE, n_width=n, n_height=n, grading=grading)


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    # (a) R vs frequency: coarse volume mesh vs surface-impedance vs exact Kelvin.
    freqs = np.geomspace(1e5, 1e8, 10)
    r_kelvin = np.array([kelvin_r(A, f) for f in freqs])
    r_vol = np.array([extract_impedance(_round_wire(n_radial=6, n_angular=6), [f]).resistance[0] for f in freqs])
    r_sibc = np.array([extract_impedance_surface(_round_wire(), [f], n_perimeter=48).resistance[0] for f in freqs])
    print("AC resistance vs frequency (1 mm round copper wire, 100 mm):")
    print("  %8s %8s %12s %12s %12s" % ("f (MHz)", "a/delta", "Kelvin mO", "volume K=~28", "SIBC K=48"))
    for f, rk, rv, rs in zip(freqs, r_kelvin, r_vol, r_sibc):
        d = np.sqrt(2 * RHO / (2 * np.pi * f * MU0))
        print("  %8.1f %8.0f %12.2f %12.2f %12.2f" % (f / 1e6, A / d, rk * 1e3, rv * 1e3, rs * 1e3))
    print("  -> the coarse volume mesh collapses in deep skin; SIBC tracks Kelvin (better at high f).")

    # (b) resistance error vs cell count: uniform vs graded (1 MHz square bar).
    f_g = 1e6
    a_eq = np.sqrt(SIDE * SIDE / np.pi)
    kelvin_ratio = kelvin_r(a_eq, f_g) / (RHO * LEN / (np.pi * a_eq**2))
    ns = [4, 6, 8, 12, 16]
    err_u, err_g, ks = [], [], []
    for n in ns:
        iu = extract_impedance(_square_bar(1.0, n), [f_g])
        ig = extract_impedance(_square_bar(2.5, n), [f_g])
        err_u.append(abs(iu.resistance[0] / iu.dc_resistance - kelvin_ratio) / kelvin_ratio)
        err_g.append(abs(ig.resistance[0] / ig.dc_resistance - kelvin_ratio) / kelvin_ratio)
        ks.append(n * n)
    print("\nResistance error vs cell count, 1 MHz square bar (graded concentrates cells at the surface):")
    print("  %6s %12s %12s" % ("K", "uniform err", "graded err"))
    for k, eu, eg in zip(ks, err_u, err_g):
        print("  %6d %11.1f%% %11.1f%%" % (k, eu * 100, eg * 100))

    # (c) build-time scaling with cell count (motivates FMM for very large K).
    print("\nChain-matrix build time vs cell count (O(K^2)); SIBC/graded keep K small:")
    build_k, build_t = [], []
    for n in (4, 8, 12):
        c = _square_bar(1.0, n)
        t0 = time.time()
        extract_impedance(c, [1e5])
        dt = time.time() - t0
        build_k.append(n * n)
        build_t.append(dt)
        print("  K=%4d: %.2f s" % (n * n, dt))

    plt = load_matplotlib(required=True, headless=args.save is not None)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fm = freqs / 1e6
    ax[0].loglog(fm, r_kelvin * 1e3, "k-", lw=2, label="Kelvin (exact)")
    ax[0].loglog(fm, r_vol * 1e3, "s--", color="C3", label="volume, K~28 (fixed)")
    ax[0].loglog(fm, r_sibc * 1e3, "o-", color="C0", label="surface impedance, K=48")
    ax[0].set_xlabel("frequency (MHz)")
    ax[0].set_ylabel("AC resistance (mOhm)")
    ax[0].set_title("Deep-skin resistance: SIBC vs coarse volume")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.2)

    ax[1].plot(ks, np.array(err_u) * 100, "s-", color="C3", label="uniform")
    ax[1].plot(ks, np.array(err_g) * 100, "o-", color="C2", label="graded (surface)")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_xlabel("cross-section cells K")
    ax[1].set_ylabel("resistance error (%)")
    ax[1].set_title("Graded vs uniform mesh (1 MHz)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.2)

    ax[2].loglog(build_k, build_t, "o-", color="C4", label="measured build")
    ref = build_t[0] * (np.array(build_k) / build_k[0]) ** 2
    ax[2].loglog(build_k, ref, "k--", lw=1, label="O(K^2)")
    ax[2].set_xlabel("cross-section cells K")
    ax[2].set_ylabel("build time (s)")
    ax[2].set_title("Cost scaling (motivates FMM at large K)")
    ax[2].legend(fontsize=8)
    ax[2].grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"\n  saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
