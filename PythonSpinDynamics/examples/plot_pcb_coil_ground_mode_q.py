"""Single-ended vs differential Q of a planar PCB coil over a ground plane.

A planar spiral etched on a PCB sits a board-thickness above a copper ground plane. How you
*drive* it changes its Q, even though the geometry and the magnetic field are identical:

* **Single-ended (unbalanced):** one terminal is grounded, the other driven. The winding
  potential ramps 0 -> V from the grounded end to the driven end, so the whole coil swings
  against the ground plane. The coil-to-ground capacitance is fully charged through the lossy
  board dielectric (FR4 ``tan d``), and the displacement current returns through the ground
  plane -- a large capacitive (E-field) loss.
* **Differential (balanced):** the coil is driven push-pull, antisymmetric about a virtual
  ground at the winding centre (-V/2 -> +V/2). The two halves' displacement currents to the
  ground plane are equal and opposite, so the *net* common-mode coupling to ground -- and its
  dielectric loss -- is largely nulled.

The magnetic behaviour (L, the copper skin/proximity resistance, and the eddy loss induced in
the ground plane) is set by the coil *current* and is therefore identical in both modes. The
whole single-ended/differential Q difference lives in the **capacitive coupling to ground**,
which this script isolates with the PEEC electrostatic solver:

    C_g(mode) = C_eff(mode, with ground plane) - C_eff(mode, free space)

computed for the single-ended potential ramp and the differential antisymmetric profile via
``self_capacitance(coil, shield=GroundPlane(...), potential=...)``. The lossy board turns that
capacitance into an equivalent series resistance ``R_diel = tan d * w^3 * L^2 * C_g`` (a
first-order lumped model: below self-resonance the coil is inductive, the winding voltage is
~ wL*I, and the dielectric dissipates ``w tan d`` of the electric energy ``1/2 C_g V^2`` per
radian). The Q is then ``wL / (R_copper + R_groundeddy + R_diel(mode))``.

**Why the ground coupling drops ~7x, not ~2x.** The single-ended ramp ``v = s`` is a uniform
common-mode offset plus exactly the differential profile: ``v = 1/2 + (s - 1/2)``. Two things
follow. (i) The common-mode offset carries *three quarters* of the ramp's mean-square voltage,
not half: ``<v^2> = int_0^1 s^2 ds = 1/3``, of which the offset is ``(1/2)^2 = 1/4`` and the
zero-mean part only ``int_0^1 (s-1/2)^2 ds = 1/12``. So dropping the common mode (going
differential) already cuts the energy 4x, not 2x. (ii) Because the differential drive carries
~zero net charge, its cross-term with the uniform mode vanishes and the single-ended ground
capacitance splits cleanly:

    C_g(single-ended) = (1/4) * C_g(uniform) + C_g(differential)
    ratio = C_g(SE) / C_g(diff) = 1 + (1/4) * C_g(uniform) / C_g(differential)

where ``C_g(uniform)`` is the whole coil held at *one* potential over the plane -- the spiral as
a capacitor plate. That plate (a monopole: net charge, slowly-decaying coupling) couples far
more strongly to the plane than the differential mode (a zero-net-charge multipole, whose field
closes locally and cancels in the image), so ``C_g(uniform)/C_g(diff) ~ 26`` here and the ratio
lands near 7x rather than 4x. The same asymmetry makes the ratio *grow* as the plane recedes
(the monopole coupling decays slowly, the multipole fast). The script prints this decomposition
so the identity can be checked against the two directly-computed mode capacitances.

Outputs (``--save out.png`` for the figure, otherwise tables):
1. Q vs frequency, single-ended vs differential, with the common (mode-independent) copper +
   ground-eddy loss floor.
2. Q vs board thickness: as the ground plane approaches, single-ended Q collapses (C_g grows)
   while differential Q barely moves -- the balanced drive's headline advantage.
3. The loss budget at the design frequency.

The inductance includes the ground plane's flux-exclusion image (``extract_impedance(...,
ground_plane=...)``), which lowers L by the standard ``L = L_free - M(2h)``. This image is
set by the winding current, so it is **the same for both drive modes** -- the ground plane's
effect on L is mode-independent (validated against the analytic loop-over-plane reduction).
The mode difference is therefore purely capacitive, which is the point of the demo.

Scope / first-order caveats: the coil-to-ground capacitance uses an effective uniform
permittivity ``(eps_r+1)/2`` (trace at the air/FR4 interface); the ground plane's magnetic
image is a lossless PEC mirror (its resistive eddy loss is the separate surface integral
above); and any single-ended *galvanic-return* inductance/loss (current routed back through
the plane to a remote source-ground bond) is layout-dependent -- for the natural local ground
bond it coincides with the flux-exclusion eddy, so it is not a separate intrinsic term and is
not added here.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.coil_peec import (
    Conductor,
    GroundPlane,
    capacitance_to_ground,
    extract_impedance,
    self_capacitance,
)
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER
from spin_dynamics.fields.magnetostatics import MU0, biot_savart

SIGMA_CU = 5.8e7  # copper conductivity (S/m)


def spiral_corners(n_turns, d_out, pitch):
    """Corner vertices ``(K, 2)`` of a centred inward square spiral (a PCB coil).

    ``d_out`` is the outer side (m), ``pitch`` the turn-to-turn centre spacing (m).
    """

    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    x, y = 0.0, 0.0
    corners = [(x, y)]
    length = d_out
    for k in range(4 * n_turns):
        dx, dy = dirs[k % 4]
        x += dx * length
        y += dy * length
        corners.append((x, y))
        if k % 2 == 1:  # shrink the arm every half-turn -> turns spaced by `pitch`
            length -= pitch
    corners = np.array(corners)
    return corners - corners.mean(axis=0)  # centre it


def densify(corners, z, *, seg_per_side=None, target_len=None):
    """Turn corner vertices into an ``(M+1, 3)`` path at plane ``z``.

    Either a fixed ``seg_per_side`` per arm (coarse, for the inductance solve, where the
    segment mutuals are exact) or a ``target_len`` maximum segment length (fine, for the
    electrostatic solve, whose thin-wire point-image approximation needs segments no longer
    than the coil-to-plane spacing ``2h`` to stay accurate as the ground plane approaches).
    """

    pts = [corners[0]]
    for a, b in zip(corners[:-1], corners[1:]):
        arm = np.linalg.norm(b - a)
        n = seg_per_side if seg_per_side is not None else max(1, int(np.ceil(arm / target_len)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            pts.append(a + t * (b - a))
    xy = np.array(pts)
    return np.column_stack([xy[:, 0], xy[:, 1], np.full(xy.shape[0], z)])


def ground_eddy_resistance(coil_segments, freq, half_span, z_plane=0.0, n=40):
    """Mode-independent eddy loss induced in the ground plane (ohm, series).

    Surface-resistance model for a good conductor: the wall surface current is
    ``K = |H_tan| = 2 |H_tan,inc|`` (the image doubles the tangential field), the loss per
    area is ``(1/2) R_s |K|^2``, and equating to ``(1/2) I^2 R`` gives the series resistance
    ``R = 4 R_s * integral(|H_tan,inc/I|^2 dA)`` with ``R_s = sqrt(pi f mu0 / sigma)``. Set by
    the coil current, so identical single-ended vs differential -- the common loss floor.
    """

    gu = np.linspace(-half_span, half_span, n)
    du = gu[1] - gu[0]
    uu, vv = np.meshgrid(gu, gu, indexing="ij")
    pts = np.column_stack([uu.ravel(), vv.ravel(), np.full(uu.size, z_plane)])
    h = biot_savart(pts, coil_segments, 1.0) / MU0  # incident H per unit current (A/m)
    field_int = float(np.sum(h[:, 0] ** 2 + h[:, 1] ** 2) * du * du)  # integral of |H_tan,inc/I|^2
    r_s = np.sqrt(np.pi * freq * MU0 / SIGMA_CU)
    return 4.0 * r_s * field_int  # good-conductor image factor (see docstring)


def coil_to_ground_capacitance(coil, gp, eps_eff, potential):
    """``(C_g, C_eff)``: coil-to-ground capacitance (F) and total effective C for a mode.

    ``C_g = C_eff(with ground) - C_eff(free)`` is the part of the winding's field that
    terminates on the ground plane -- what the lossy dielectric dissipates. ``potential=None``
    is the single-ended ramp; ``potential=lambda s: s-0.5`` is the differential profile.
    """

    c_gnd = self_capacitance(coil, shield=gp, relative_permittivity=eps_eff, potential=potential)
    c_free = self_capacitance(coil, relative_permittivity=eps_eff, potential=potential)
    return c_gnd - c_free, c_gnd


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--outer-mm", type=float, default=25.0)
    p.add_argument("--pitch-mm", type=float, default=1.2, help="Turn-to-turn centre spacing.")
    p.add_argument("--trace-mm", type=float, default=0.6, help="Trace width.")
    p.add_argument("--copper-um", type=float, default=35.0, help="Trace (and plane) copper thickness.")
    p.add_argument("--board-mm", type=float, default=0.8, help="FR4 thickness = coil height over the plane.")
    p.add_argument("--eps-r", type=float, default=4.3, help="FR4 relative permittivity.")
    p.add_argument("--tan-delta", type=float, default=0.02, help="FR4 dielectric loss tangent.")
    p.add_argument("--f0-mhz", type=float, default=63.8, help="Design frequency (1.5 T proton).")
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    w_tr = args.trace_mm * 1e-3
    t_cu = args.copper_um * 1e-6
    eps_eff = 0.5 * (args.eps_r + 1.0)  # trace at the air/FR4 interface, first-order
    tand = args.tan_delta

    corners = spiral_corners(args.turns, args.outer_mm * 1e-3, args.pitch_mm * 1e-3)

    def build_imp(height):
        # coarse path for the inductance/resistance solve (segment mutuals are exact)
        path = densify(corners, height, seg_per_side=6)
        return Conductor(path, cross_section="rect", width=w_tr, height=t_cu,
                         n_width=10, n_height=2, material=ANNEALED_COPPER)

    def build_electro(height):
        # fine path (segments <= ~0.3 mm) so the thin-wire point-image stays accurate as the
        # ground plane approaches; electrostatics is a cheap N x N solve so this is affordable
        path = densify(corners, height, target_len=0.3e-3)
        return Conductor(path, cross_section="rect", width=w_tr, height=t_cu,
                         material=ANNEALED_COPPER)

    h0 = args.board_mm * 1e-3
    coil = build_imp(h0)
    coil_e = build_electro(h0)
    gp = GroundPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
    segs = [(coil.path_points[i], coil.path_points[i + 1]) for i in range(len(coil.path_points) - 1)]
    half_span = args.outer_mm * 1e-3  # integrate the eddy field over 2x the coil footprint

    MODES = {"single-ended": None, "differential": (lambda s: s - 0.5)}

    print(f"Planar PCB spiral: {args.turns} turns, {args.outer_mm:.0f} mm outer, "
          f"{args.trace_mm:.1f} mm trace, {args.copper_um:.0f} um Cu")
    print(f"  on {args.board_mm:.1f} mm FR4 (eps_r {args.eps_r}, tan d {tand}) over a ground plane\n")

    # --- (1) frequency sweep at the design height ---
    freqs = np.linspace(10e6, 120e6, 12)
    L = np.empty(freqs.size)
    r_cu = np.empty(freqs.size)
    r_eddy = np.empty(freqs.size)
    q = {m: np.empty(freqs.size) for m in MODES}
    cg = {m: coil_to_ground_capacitance(coil_e, gp, eps_eff, prof)[0] for m, prof in MODES.items()}
    for i, f in enumerate(freqs):
        # L and R include the ground plane's flux-exclusion image (lowers L). This is set by
        # the winding current, so it is IDENTICAL for both drive modes -- the ground plane's
        # effect on inductance is mode-independent (only the capacitance splits by mode).
        imp = extract_impedance(coil, [f], formulation="full", ground_plane=gp)
        L[i] = imp.inductance[0]
        r_cu[i] = imp.resistance[0]
        r_eddy[i] = ground_eddy_resistance(segs, f, half_span)
        w = 2 * np.pi * f
        for m in MODES:
            r_diel = tand * w**3 * L[i] ** 2 * cg[m]
            q[m][i] = w * L[i] / (r_cu[i] + r_eddy[i] + r_diel)

    l_free = extract_impedance(coil, [args.f0_mhz * 1e6], formulation="full").inductance[0]
    l_gnd = np.interp(args.f0_mhz * 1e6, freqs, L)
    print(f"  inductance at {args.f0_mhz:.0f} MHz: {l_free * 1e9:.0f} nH free -> {l_gnd * 1e9:.0f} nH over the plane "
          f"({100 * (l_gnd - l_free) / l_free:+.0f}%, flux-exclusion image; same for both modes)\n")
    print(f"  coil-to-ground C: single-ended {cg['single-ended'] * 1e12:.3f} pF, "
          f"differential {cg['differential'] * 1e12:.3f} pF "
          f"({cg['single-ended'] / cg['differential']:.1f}x less common-mode coupling)")
    # Why ~7x and not ~2x: single-ended = 1/4 common-mode (whole-coil plate) + differential.
    cg_uniform = (capacitance_to_ground(coil_e, shield=gp, relative_permittivity=eps_eff)
                  - capacitance_to_ground(coil_e, relative_permittivity=eps_eff))
    predicted = 0.25 * cg_uniform + cg['differential']
    print(f"    decomposition: C_g(uniform plate) {cg_uniform * 1e12:.3f} pF; "
          f"1/4*plate + C_g(diff) = {predicted * 1e12:.3f} pF  (= C_g(SE) {cg['single-ended'] * 1e12:.3f} pF)")
    print(f"    ratio = 1 + 1/4*(plate/diff) = {1 + 0.25 * cg_uniform / cg['differential']:.1f}x "
          "(the plane over-weights the common/monopole mode vs the differential multipole)\n")
    print(f"  {'f (MHz)':>8}{'L (nH)':>9}{'R_cu':>8}{'R_eddy':>8}"
          f"{'Q_SE':>8}{'Q_diff':>9}{'gain':>7}")
    for i, f in enumerate(freqs):
        print(f"  {f / 1e6:8.0f}{L[i] * 1e9:9.0f}{r_cu[i] * 1e3:7.1f}m{r_eddy[i] * 1e3:7.1f}m"
              f"{q['single-ended'][i]:8.0f}{q['differential'][i]:9.0f}"
              f"{q['differential'][i] / q['single-ended'][i]:6.2f}x")

    # loss budget at f0
    i0 = int(np.argmin(np.abs(freqs - args.f0_mhz * 1e6)))
    f0 = freqs[i0]
    w0 = 2 * np.pi * f0
    print(f"\n  Loss budget at {f0 / 1e6:.0f} MHz (mOhm):")
    print(f"    copper (skin+proximity)  {r_cu[i0] * 1e3:7.1f}  [both modes]")
    print(f"    ground-plane eddy        {r_eddy[i0] * 1e3:7.1f}  [both modes]")
    for m in MODES:
        print(f"    dielectric, {m:<12} {tand * w0**3 * L[i0] ** 2 * cg[m] * 1e3:7.1f}")

    # --- (2) height sweep at f0 ---
    # Both L (flux-exclusion image, recomputed per height) and C_g change as the plane nears;
    # L is mode-independent, C_g is not -- so the mode split is still purely capacitive.
    heights = np.array([0.4, 0.8, 1.6, 3.2]) * 1e-3
    qh = {m: np.empty(heights.size) for m in MODES}
    cgh = {m: np.empty(heights.size) for m in MODES}
    lh = np.empty(heights.size)
    for j, hgt in enumerate(heights):
        coil_h = build_imp(hgt)
        imp_h = extract_impedance(coil_h, [f0], formulation="full", ground_plane=gp)
        lh[j], r_cu_h = imp_h.inductance[0], imp_h.resistance[0]
        segs_h = [(coil_h.path_points[i], coil_h.path_points[i + 1]) for i in range(len(coil_h.path_points) - 1)]
        r_eddy_h = ground_eddy_resistance(segs_h, f0, half_span, z_plane=0.0)
        ch = build_electro(hgt)
        for m, prof in MODES.items():
            c_g, _ = coil_to_ground_capacitance(ch, gp, eps_eff, prof)
            cgh[m][j] = c_g
            r_diel = tand * w0**3 * lh[j] ** 2 * c_g
            qh[m][j] = w0 * lh[j] / (r_cu_h + r_eddy_h + r_diel)
    print(f"\n  Q vs board thickness at {f0 / 1e6:.0f} MHz (L mode-independent, C_g splits by mode):")
    print(f"  {'h (mm)':>8}{'L (nH)':>9}{'C_g,SE':>10}{'C_g,diff':>10}{'Q_SE':>8}{'Q_diff':>9}{'gain':>7}")
    for j, hgt in enumerate(heights):
        print(f"  {hgt * 1e3:8.2f}{lh[j] * 1e9:9.0f}{cgh['single-ended'][j] * 1e12:9.3f}p{cgh['differential'][j] * 1e12:9.3f}p"
              f"{qh['single-ended'][j]:8.0f}{qh['differential'][j]:9.0f}"
              f"{qh['differential'][j] / qh['single-ended'][j]:6.2f}x")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        fm = freqs / 1e6

        ax[0].plot(fm, q["single-ended"], "o-", color="C3", label="single-ended")
        ax[0].plot(fm, q["differential"], "s-", color="C0", label="differential")
        ax[0].set_xlabel("frequency (MHz)")
        ax[0].set_ylabel("unloaded Q")
        ax[0].set_title("Q vs frequency (coil over ground plane)")
        ax[0].legend(fontsize=8)
        ax[0].grid(True, alpha=0.2)

        ax[1].plot(heights * 1e3, qh["single-ended"], "o-", color="C3", label="Q single-ended")
        ax[1].plot(heights * 1e3, qh["differential"], "s-", color="C0", label="Q differential")
        ax[1].axvline(args.board_mm, color="gray", ls=":", lw=1, label=f"{args.board_mm:.1f} mm board")
        ax[1].set_xlabel("coil height above ground plane (mm)")
        ax[1].set_ylabel(f"unloaded Q at {f0 / 1e6:.0f} MHz")
        ax[1].set_title("Approaching plane: L collapses (both modes), Q falls")
        ax[1].legend(fontsize=8, loc="upper left")
        ax[1].grid(True, alpha=0.2)
        ax1r = ax[1].twinx()  # the mode-independent inductance collapse behind the Q trend
        ax1r.plot(heights * 1e3, lh * 1e9, "^--", color="0.5", label="L (both modes)")
        ax1r.set_ylabel("L (nH)", color="0.5")
        ax1r.legend(fontsize=8, loc="lower right")

        # loss budget bars at f0
        labels = ["copper", "gnd eddy", "dielectric"]
        se = [r_cu[i0], r_eddy[i0], tand * w0**3 * L[i0] ** 2 * cg["single-ended"]]
        di = [r_cu[i0], r_eddy[i0], tand * w0**3 * L[i0] ** 2 * cg["differential"]]
        xb = np.arange(3)
        ax[2].bar(xb - 0.19, np.array(se) * 1e3, 0.38, color="C3", label="single-ended")
        ax[2].bar(xb + 0.19, np.array(di) * 1e3, 0.38, color="C0", label="differential")
        ax[2].set_xticks(xb)
        ax[2].set_xticklabels(labels)
        ax[2].set_ylabel(f"series resistance (mOhm) at {f0 / 1e6:.0f} MHz")
        ax[2].set_title("Loss budget: dielectric is the mode-dependent term")
        ax[2].legend(fontsize=8)
        ax[2].grid(True, axis="y", alpha=0.2)

        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"\n  saved: {args.save}")


if __name__ == "__main__":
    main()
