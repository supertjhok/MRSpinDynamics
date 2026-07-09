"""Effect of a metal shield box on an RF coil -- a 14N NQR probe (2-3 MHz).

A single-layer solenoid is wound on a Teflon (PTFE) former (a hollow cylinder) and mounted
inside a grounded rectangular aluminium box; one end of the coil and the box are at ground.
The example extracts the coil properties versus frequency, with and without the box, using
the PEEC solver:

* **L(f), R(f)** -- the surface-impedance backend (`extract_impedance_surface`) with the
  **full per-segment formulation**; at 2-3 MHz the copper wire is already several skin
  depths thick (deep skin), where a volume mesh would need thousands of cells. The full
  formulation resolves the turn-to-turn proximity loss from first principles (validated
  against continuum FEM -- see docs/coil_peec.md), so no empirical proximity factor is
  applied.
* **Self-capacitance and self-resonance** -- `self_capacitance`, with the grounded box added
  as image charges (`GroundedBox`) and the Teflon former as an effective permittivity. Both
  raise C and lower the SRF; the design target is SRF > 10 MHz so the 2-3 MHz working band is
  well below resonance.
* **Shield loss** -- eddy currents induced in the box walls couple back into the coil as an
  added series resistance (`fields.quasistatic.reflected_resistance`), lowering Q.

The plot (``--save out.png``) shows L, unloaded Q, and the capacitance / self-resonance for
the free coil versus the shielded coil.

Two loss subtleties (see docs/coil_peec.md, "Loss modelling"):

* **Proximity** -- the chain (reduced) SIBC solve gives the skin resistance only; the full
  per-segment solve adds the turn-to-turn proximity crowding. The example plots both so the
  proximity contribution is visible, and prints the resulting proximity factor next to
  Medhurst's empirical table value for reference (the table reads ~20% high against
  continuum-FEM references for tight winding; the full-PEEC number is the field-solver
  answer). Either way this is a *theoretical* Q; a measured coil is typically another
  ~1.3-2x lower (dielectric former, solder/lead resistance, radiation).
* **Shield eddy loss** uses the surface-resistance model (loss in a skin depth of the wall,
  R ~ sqrt f), which is correct for a solid metal box -- unlike the full-penetration
  first-order eddy model, which would over-predict it by orders of magnitude at these
  frequencies.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.coil_peec import (
    Conductor,
    GroundedBox,
    extract_impedance_surface,
    radiation_resistance,
    self_capacitance,
)
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER, medhurst_proximity_factor
from spin_dynamics.fields.magnetostatics import MU0, biot_savart

SIGMA_AL = 3.5e7  # aluminium conductivity (S/m)
EPS_TEFLON = 2.1  # PTFE relative permittivity


def wall_field_integral(coil_segments, half, n=28):
    """Integral of |H_tan/I|^2 over the six box faces (per unit coil current), 1/m^2 * m^2.

    Sampled from the coil's Biot-Savart field. With the surface resistance
    ``R_s = sqrt(pi f mu0 / sigma)`` (ohm/square), the shield eddy loss couples back as a
    reflected series resistance ``R_box(f) = 2 R_s(f) * integral`` -- the high-frequency
    surface-impedance model (loss confined to a skin depth in the wall, so ``R_box ~ sqrt f``),
    which is correct for a solid metal box unlike the full-penetration first-order eddy model.
    """

    total = 0.0
    for axis in range(3):
        u, v = [i for i in range(3) if i != axis]
        gu = np.linspace(-half[u], half[u], n)
        gv = np.linspace(-half[v], half[v], n)
        du, dv = gu[1] - gu[0], gv[1] - gv[0]
        uu, vv = np.meshgrid(gu, gv, indexing="ij")
        for sgn in (-1.0, 1.0):
            pts = np.zeros((n * n, 3))
            pts[:, axis] = sgn * half[axis]
            pts[:, u] = uu.ravel()
            pts[:, v] = vv.ravel()
            h = biot_savart(pts, coil_segments, 1.0) / MU0  # H per unit current (A/m)
            total += float(np.sum(h[:, u] ** 2 + h[:, v] ** 2) * du * dv)
    return total


def solenoid_path(diameter, length, turns, n_per_turn):
    th = np.linspace(0.0, 2.0 * np.pi * turns, turns * n_per_turn + 1)
    z = np.linspace(-length / 2, length / 2, th.size)
    return np.column_stack([(diameter / 2) * np.cos(th), (diameter / 2) * np.sin(th), z])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-inch", type=float, default=1.5, help="Coil inner diameter (inches).")
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--length-mm", type=float, default=60.0)
    parser.add_argument("--wire-mm", type=float, default=1.6, help="Wire diameter.")
    parser.add_argument("--box-mm", type=float, default=70.0, help="Box inner side (square cross-section).")
    parser.add_argument("--box-len-mm", type=float, default=100.0)
    parser.add_argument("--teflon-fill", type=float, default=0.4, help="Former dielectric filling factor.")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    d_wire = args.wire_mm * 1e-3
    diameter = args.id_inch * 25.4e-3 + d_wire  # conductor-centre diameter
    length = args.length_mm * 1e-3
    # n_per_turn=10 keeps the full per-segment SIBC system tractable (M*K unknowns).
    path = solenoid_path(diameter, length, int(args.turns), n_per_turn=10)
    coil = Conductor(path, wire_radius=d_wire / 2, material=ANNEALED_COPPER)

    # Grounded aluminium box centred on the coil (square cross-section along the coil axis z).
    half = np.array([args.box_mm / 2, args.box_mm / 2, args.box_len_mm / 2]) * 1e-3
    box = GroundedBox(center=(0.0, 0.0, 0.0), half_extents=tuple(half), n_orders=1)
    eps_eff = 1.0 + (EPS_TEFLON - 1.0) * args.teflon_fill  # partial-fill effective permittivity

    # --- capacitance / self-resonance for the three cases ---
    c_free = self_capacitance(coil)
    c_teflon = self_capacitance(coil, relative_permittivity=eps_eff)
    c_box = self_capacitance(coil, shield=box, relative_permittivity=eps_eff)

    # --- L(f), R(f) from the surface-impedance backend; box eddy loss adds series R ---
    freqs = np.linspace(1.0e6, 12.0e6, 12)
    # chain: skin only (isolated-wire crowding); full: + per-segment turn-to-turn proximity,
    # resolved from first principles (no empirical factor) -- docs/coil_peec.md.
    imp_skin = extract_impedance_surface(coil, freqs, n_perimeter=24)
    imp = extract_impedance_surface(coil, freqs, n_perimeter=24, formulation="full")
    ind = imp.inductance
    r_skin = imp_skin.resistance
    r_coil = imp.resistance
    # The realized proximity factor, with Medhurst's empirical table as a reference point
    # (it reads ~20% high vs continuum-FEM references for tight winding).
    phi_peec = float(np.interp(2.5e6, freqs, r_coil / r_skin))
    phi_medhurst = medhurst_proximity_factor(length / diameter, (length / args.turns) / d_wire)

    # box wall eddy loss via the surface-impedance model: R_box(f) = 2 R_s(f) * field integral
    coil_segments = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    field_int = wall_field_integral(coil_segments, half)
    r_s = np.sqrt(np.pi * freqs * MU0 / SIGMA_AL)  # surface resistance (ohm/square) ~ sqrt(f)
    r_box = 2.0 * r_s * field_int
    r_total = r_coil + r_box

    q_skin = 2 * np.pi * freqs * ind / r_skin       # skin only (proximity ignored)
    q_free = 2 * np.pi * freqs * ind / r_coil        # + turn-to-turn proximity
    q_box = 2 * np.pi * freqs * ind / r_total        # + proximity + shield eddy loss

    def srf(c):
        return 1.0 / (2 * np.pi * np.sqrt(imp.dc_inductance * c))

    print(f"14N NQR coil: ID {args.id_inch:.2f}\" ({diameter * 1e3:.1f} mm), {args.turns} turns, "
          f"{args.length_mm:.0f} mm, {args.wire_mm:.1f} mm wire")
    print(f"  L (2.5 MHz)        = {np.interp(2.5e6, freqs, ind) * 1e6:.2f} uH")
    print(f"  self-C free        = {c_free * 1e12:.2f} pF   -> SRF {srf(c_free) / 1e6:.1f} MHz")
    print(f"  + Teflon former    = {c_teflon * 1e12:.2f} pF   -> SRF {srf(c_teflon) / 1e6:.1f} MHz")
    print(f"  + grounded box     = {c_box * 1e12:.2f} pF   -> SRF {srf(c_box) / 1e6:.1f} MHz")
    print(f"  R (2.5 MHz): skin {np.interp(2.5e6, freqs, r_skin) * 1e3:.0f} mOhm  "
          f"-> full solve {np.interp(2.5e6, freqs, r_coil) * 1e3:.0f} mOhm "
          f"(proximity factor {phi_peec:.2f}; Medhurst table {phi_medhurst:.2f} for reference)  "
          f"+ box eddy {np.interp(2.5e6, freqs, r_box) * 1e3:.1f} mOhm")
    r_rad = radiation_resistance(coil, 2.5e6)
    print(f"  radiation (magnetic-dipole, first-order): {r_rad * 1e9:.1f} nOhm "
          "-- negligible at NQR frequencies, as expected (R_rad ~ f^4)")
    print(f"  Q (2.5 MHz): skin-only {np.interp(2.5e6, freqs, q_skin):.0f}  ->  "
          f"+proximity {np.interp(2.5e6, freqs, q_free):.0f}  ->  +shield {np.interp(2.5e6, freqs, q_box):.0f}")
    print("  NOTE: this is the theoretical (idealized-copper) Q; a measured coil is typically")
    print("        another ~1.3-2x lower (dielectric former, solder/lead resistance, radiation).")
    print(f"  SRF > 10 MHz target: {'MET' if srf(c_box) > 10e6 else 'NOT MET (box too tight)'}")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, ax3 = plt.subplots(1, 3, figsize=(16, 4.6))
        fm = freqs / 1e6
        band = (2.0, 3.0)

        ax3[0].plot(fm, ind * 1e6, "o-", color="C0")
        ax3[0].axvspan(*band, color="orange", alpha=0.15, label="14N NQR band")
        ax3[0].set_xlabel("frequency (MHz)")
        ax3[0].set_ylabel("L (uH)")
        ax3[0].set_title("Inductance (geometry-set, shield ~unchanged)")
        ax3[0].legend(fontsize=8)

        ax3[1].plot(fm, q_skin, "^:", color="0.6", label="skin only (chain)")
        ax3[1].plot(fm, q_free, "o-", color="C2", label="+ proximity (full)")
        ax3[1].plot(fm, q_box, "s-", color="C3", label="+ proximity + box")
        ax3[1].axvspan(*band, color="orange", alpha=0.15)
        ax3[1].set_xlabel("frequency (MHz)")
        ax3[1].set_ylabel("unloaded Q")
        ax3[1].set_title("Q: proximity + shield loss (theoretical upper bound)")
        ax3[1].legend(fontsize=8)

        labels = ["free", "+Teflon", "+box"]
        caps = np.array([c_free, c_teflon, c_box]) * 1e12
        srfs = np.array([srf(c_free), srf(c_teflon), srf(c_box)]) / 1e6
        xb = np.arange(3)
        b = ax3[2].bar(xb, caps, color=["C0", "C4", "C3"], alpha=0.8)
        ax3[2].set_xticks(xb)
        ax3[2].set_xticklabels(labels)
        ax3[2].set_ylabel("self-capacitance (pF)")
        axr = ax3[2].twinx()
        axr.plot(xb, srfs, "kD--", label="SRF")
        axr.axhline(10, color="red", ls=":", lw=1, label="SRF target 10 MHz")
        axr.set_ylabel("self-resonance (MHz)")
        ax3[2].set_title("Capacitance & SRF: former + shield")
        for rect, mhz in zip(b, srfs):
            axr.annotate(f"{mhz:.0f} MHz", (rect.get_x() + rect.get_width() / 2, mhz),
                         ha="center", va="bottom", fontsize=8)
        axr.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"  saved: {args.save}")


if __name__ == "__main__":
    main()
