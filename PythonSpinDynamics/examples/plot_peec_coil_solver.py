"""PEEC solver for arbitrary coils: RF properties beyond the single-layer solenoid.

The QOIL model (``coil_properties.solenoid_properties``) only handles single-layer air-core
solenoids. The partial-element (PEEC) solver in ``fields.coil_peec`` extracts inductance,
AC resistance, Q, self-capacitance and self-resonance for **any wire path**, resolving the
skin and proximity effects from first principles. This example shows two geometries the
analytic model cannot express and one figure of the current crowding that drives the AC
resistance:

1. **Two-layer RF solenoid** (wind up, then back down at a larger radius) -- L, R and Q vs
   frequency plus the self-resonant frequency, and a cross-section current-density map
   showing the skin/proximity crowding.
2. **Gradient Maxwell pair** (two opposed coaxial loops) -- L and R from PEEC, the gradient
   efficiency ``dB_z/dz`` per amp from Biot-Savart, and the shield eddy time constants from
   ``eddy_modes`` -- demonstrating the same solver spans RF and gradient coils.

Both geometries put multiple conductors close together, so the resistance uses the **full
(per-segment) formulation** -- FastHenry's actual system -- which resolves the turn-to-turn
proximity loss that the fast reduced ``chain`` formulation under-captures (see
``docs/coil_peec.md``, "Loss modelling"). The R(f) table switches backend with skin depth:
a graded volume solve at low frequency (``a/delta`` < ~3) and the surface-impedance (SIBC)
solve above, both per-segment. The current-density map is taken at one mid-coil segment,
where the crowding toward the neighbouring turns -- the proximity effect itself -- is
visible; a chain map could only show the path-averaged (symmetric) pattern.

Run with ``--save out.png`` for the figure; otherwise it prints tables.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import coils
from spin_dynamics.fields.coil_peec import (
    Conductor,
    current_distribution,
    extract_impedance,
    extract_impedance_surface,
    self_capacitance,
)
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER
from spin_dynamics.fields.eddy_modes import EddyModes
from spin_dynamics.fields.magnetostatics import biot_savart


def two_layer_solenoid_path(diameter, length, turns_per_layer, layer_gap, n_per_turn):
    """A connected two-layer helix: up the inner layer, down the outer layer."""

    r_in = diameter / 2.0
    r_out = r_in + layer_gap
    n = turns_per_layer * n_per_turn
    th1 = np.linspace(0.0, 2 * np.pi * turns_per_layer, n + 1)
    z1 = np.linspace(-length / 2, length / 2, th1.size)
    up = np.column_stack([r_in * np.cos(th1), r_in * np.sin(th1), z1])
    # Return layer keeps the SAME angular sense (series-aiding) while descending, so the
    # two layers reinforce (L ~ N^2). Theta continues increasing; z runs back down.
    th2 = np.linspace(2 * np.pi * turns_per_layer, 4 * np.pi * turns_per_layer, n + 1)
    z2 = np.linspace(length / 2, -length / 2, th2.size)
    down = np.column_stack([r_out * np.cos(th2), r_out * np.sin(th2), z2])
    return np.vstack([up, down[1:]])


def maxwell_pair_path(radius, separation, n_per_turn):
    """A connected pair of opposed coaxial loops (a z-gradient coil)."""

    th = np.linspace(0.0, 2 * np.pi, n_per_turn + 1)
    top = np.column_stack([radius * np.cos(th), radius * np.sin(th), np.full(th.size, separation / 2)])
    # second loop wound in reverse so the current opposes -> linear dB_z/dz
    bot = np.column_stack([radius * np.cos(th[::-1]), radius * np.sin(th[::-1]),
                           np.full(th.size, -separation / 2)])
    return np.vstack([top, bot[1:]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diameter-mm", type=float, default=20.0)
    parser.add_argument("--length-mm", type=float, default=30.0)
    parser.add_argument("--turns-per-layer", type=int, default=6)
    parser.add_argument("--layer-gap-mm", type=float, default=1.5)
    parser.add_argument("--wire-mm", type=float, default=1.0)
    parser.add_argument("--map-frequency-mhz", type=float, default=2.0)
    parser.add_argument("--grad-radius-mm", type=float, default=40.0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    d = args.wire_mm * 1e-3
    # --- Two-layer RF solenoid ---
    # n_per_turn=10 keeps the full (per-segment) system tractable: M*K unknowns.
    pts = two_layer_solenoid_path(args.diameter_mm * 1e-3, args.length_mm * 1e-3,
                                  args.turns_per_layer, args.layer_gap_mm * 1e-3, n_per_turn=10)
    coil = Conductor(pts, wire_radius=d / 2, material=ANNEALED_COPPER, n_radial=4, n_angular=6)
    freqs = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 2.0]) * 1e6
    # Hybrid full solve: graded volume below a/delta ~ 3, SIBC above -- both per-segment, so
    # the turn-to-turn proximity loss is resolved at every frequency (docs/coil_peec.md).
    rho = ANNEALED_COPPER.resistivity
    MU0_ = 4e-7 * np.pi
    a_over_delta = (d / 2) / np.sqrt(rho / (np.pi * freqs * MU0_))
    lo = freqs[a_over_delta < 3.0]
    hi = freqs[a_over_delta >= 3.0]
    imp_lo = extract_impedance(coil, lo, formulation="full") if lo.size else None
    imp_hi = extract_impedance_surface(coil, hi, n_perimeter=32, formulation="full") if hi.size else None
    ind = np.concatenate([x.inductance for x in (imp_lo, imp_hi) if x is not None])
    res = np.concatenate([x.resistance for x in (imp_lo, imp_hi) if x is not None])
    l_dc = (imp_lo or imp_hi).dc_inductance
    cap = self_capacitance(coil)
    f_srf = 1.0 / (2 * np.pi * np.sqrt(l_dc * cap))

    print("Two-layer RF solenoid (not expressible by the single-layer analytic model)")
    print(f"  {2 * args.turns_per_layer} turns, D={args.diameter_mm:.0f} mm inner, "
          f"layer gap {args.layer_gap_mm:.1f} mm, wire {args.wire_mm:.1f} mm")
    print(f"  self-capacitance {cap * 1e12:.2f} pF, self-resonance {f_srf / 1e6:.0f} MHz")
    print(f"  {'f (MHz)':>8}{'L (uH)':>10}{'R (ohm)':>10}{'Q':>8}   (full formulation: skin + proximity)")
    for f, ll, rr in zip(freqs, ind, res):
        print(f"  {f / 1e6:8.2f}{ll * 1e6:10.3f}{rr:10.4f}{2 * np.pi * f * ll / rr:8.0f}")

    # --- Gradient Maxwell pair ---
    a = args.grad_radius_mm * 1e-3
    sep = a * np.sqrt(3.0)
    gpath = maxwell_pair_path(a, sep, n_per_turn=48)
    gcoil = Conductor(gpath, wire_radius=d / 2, material=ANNEALED_COPPER, n_radial=3, n_angular=6)
    gimp = extract_impedance(gcoil, [10e3], formulation="full")
    # Gradient efficiency dB_z/dz per amp (Biot-Savart, opposed loops).
    grad_segs = coils.maxwell_pair(radius=a, separation=sep, axis="z")
    dz = 1e-4
    bz_p = biot_savart(np.array([[0, 0, dz]]), grad_segs, 1.0)[0, 2]
    bz_m = biot_savart(np.array([[0, 0, -dz]]), grad_segs, 1.0)[0, 2]
    eff = (bz_p - bz_m) / (2 * dz)  # T/m per A
    # Shield eddy time constants (a conducting bore around the gradient).
    radii, centers = coils.cylindrical_shield(radius=1.5 * a, length=4 * a, n_rings=12)
    eddy = EddyModes(radii, centers, wire_radius=2e-3, resistivity=1.7e-8)
    terms = eddy.eddy_terms(grad_segs, max_terms=3)
    print("\nGradient Maxwell pair (same solver, gradient coil)")
    print(f"  radius {args.grad_radius_mm:.0f} mm, separation a*sqrt(3): "
          f"L={gimp.inductance[0] * 1e6:.2f} uH, R(10kHz)={gimp.resistance[0] * 1e3:.2f} mOhm")
    print(f"  gradient efficiency dB_z/dz = {eff * 1e3:.3f} mT/m per A")
    print("  dominant shield eddy modes (alpha, tau): "
          + ", ".join(f"({a_:.3f}, {t_ * 1e3:.1f} ms)" for a_, t_ in terms))

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        fmhz = freqs / 1e6
        q = 2 * np.pi * freqs * ind / res

        ax = axes[0]
        ax.plot(fmhz, ind * 1e6, "o-", color="C0", label="L (uH)")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("L (uH)", color="C0")
        ax.set_title(f"Two-layer solenoid L, R  (f_res={f_srf / 1e6:.0f} MHz)")
        axr = ax.twinx()
        axr.plot(fmhz, res, "s-", color="C3", label="R (ohm)")
        axr.set_ylabel("R (ohm)", color="C3")

        axes[1].plot(fmhz, q, "o-", color="C4")
        axes[1].set_xlabel("frequency (MHz)")
        axes[1].set_ylabel("Q")
        axes[1].set_title("Unloaded Q (full formulation: skin + proximity)")

        # current-density map at one mid-coil segment (full solve): the asymmetric crowding
        # toward the neighbouring turns IS the proximity effect.
        offs, cur = current_distribution(coil, args.map_frequency_mhz * 1e6, formulation="full")
        dens = cur  # normalized magnitude
        sc = axes[2].scatter(offs[:, 0] * 1e3, offs[:, 1] * 1e3, c=dens, s=120, cmap="inferno")
        axes[2].set_aspect("equal")
        axes[2].set_xlabel("x (mm)")
        axes[2].set_ylabel("y (mm)")
        axes[2].set_title(f"Mid-coil segment current crowding @ {args.map_frequency_mhz:.0f} MHz")
        fig.colorbar(sc, ax=axes[2], fraction=0.046, label="|I| (norm.)")
        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"\n  saved: {args.save}")


if __name__ == "__main__":
    main()
