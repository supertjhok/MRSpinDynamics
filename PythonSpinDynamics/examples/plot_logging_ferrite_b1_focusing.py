"""Ferrite-focused RF coil for a Jasper-Jackson NMR logging tool.

An inside-out well-logging tool (two opposing SmCo magnets with a low-gradient
"sweet spot" shell in the formation, as in ``plot_logging_cpmg_multinuclear.py``)
transmits and receives with a solenoid coil on the tool. Placing a
high-permeability RF ferrite core *behind* the coil (radially inward, on the
non-formation side) mirrors flux outward and raises the coil's ``B1`` efficiency
at the sweet spot. This is a magnetic-material effect that free-space
Biot-Savart cannot capture; it is computed here with the axisymmetric nonlinear
magnetostatic solver
(:class:`spin_dynamics.fields.AxisymmetricMagnetostatics`).

Two quantified benefits follow from the efficiency gain
``eta = B1_perp / I`` at the sweet spot:

* **RF power** for a fixed ``B1`` (fixed nutation) scales as
  ``P ~ I^2 = (B1/eta)^2``, so it drops by ``(eta_air / eta_ferrite)^2``.
* **Received signal** couples by reciprocity as ``eta`` (same transmit and
  receive field per unit current), so at fixed noise the SNR rises by
  ``eta_ferrite / eta_air``.

The ferrite operates far below saturation at logging ``B1`` levels, so it is
modelled as linear high-permeability; the benefit is limited by the core's
demagnetizing factor (its aspect ratio), not by the permeability -- a design
trade-off the sweep panel makes explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import coils  # noqa: E402
from spin_dynamics.fields.magnetostatics import (  # noqa: E402
    FiniteMagnetRod,
    finite_magnet_array_b0,
)
from spin_dynamics.fields.nonlinear_magnetostatics import (  # noqa: E402
    AxisymmetricMagnetostatics,
    air,
    rf_ferrite,
)
from spin_dynamics.fields.quasistatic import coil_inductance, coil_loading  # noqa: E402

GAMMA_H = 2.6752218744e8  # 1H gyromagnetic ratio (rad/s/T)
KB = 1.380649e-23


def _jj_sweet_spot(args):
    """Return the midplane sweet-spot radius, field, and RF frequency."""

    length = args.magnet_length_cm * 1e-2
    radius = args.magnet_radius_cm * 1e-2
    gap = args.gap_cm * 1e-2
    zc = gap / 2.0 + length / 2.0
    rods = [
        FiniteMagnetRod(center=(0, 0, zc), length=length, br=(0, 0, -args.remanence),
                        shape="cylinder", radius=radius),
        FiniteMagnetRod(center=(0, 0, -zc), length=length, br=(0, 0, args.remanence),
                        shape="cylinder", radius=radius),
    ]
    r_line = np.linspace(args.borehole_radius_cm * 1e-2, 0.30, 200)
    pts = np.stack([r_line, np.zeros_like(r_line), np.zeros_like(r_line)], axis=-1)
    b0 = np.linalg.norm(finite_magnet_array_b0(pts, rods), axis=-1)
    i_sweet = int(np.argmax(b0))
    r_sweet = float(r_line[i_sweet])
    b_sweet = float(b0[i_sweet])
    return r_sweet, b_sweet, GAMMA_H * b_sweet / (2.0 * np.pi), r_line, b0


def _b1_efficiency(args, core_outer_m: float | None):
    """Solve the axisymmetric coil (+optional ferrite core) and return B1(r, z=0).

    The coil carries a total of ``coil_turns`` ampere-turns (i.e. 1 A per turn),
    so ``|B_z|`` at ``z = 0`` is the per-amp transverse ``B1`` efficiency (the
    radial sweet-spot ``B0`` makes the axial ``B_z`` the perpendicular
    component). Returns the solution and the ``B1(r)`` profile at ``z = 0``.
    """

    r_coil = args.coil_radius_cm * 1e-2
    length = args.coil_length_cm * 1e-2
    rmax, zmax = args.box_cm * 1e-2, args.box_cm * 1e-2
    nr, nz = int(args.nr), int(args.nz)
    prob = AxisymmetricMagnetostatics(np.linspace(0, rmax, nr), np.linspace(-zmax, zmax, nz))
    dr = rmax / (nr - 1)
    prob.add_material(
        prob.rectangle((r_coil, r_coil + 2 * dr), (-length / 2, length / 2)),
        air(),
        current_total_a=float(args.coil_turns),
    )
    if core_outer_m is not None:
        prob.add_material(
            prob.rectangle((0.0, core_outer_m), (-args.core_length_cm * 1e-2 / 2,
                                                 args.core_length_cm * 1e-2 / 2)),
            rf_ferrite(mu_r=args.ferrite_mu_r),
        )
    sol = prob.solve()
    r_line = np.linspace(0.0, rmax, nr)
    b1_line = np.abs(sol.sample_bz(r_line, np.zeros_like(r_line)))
    return prob, sol, r_line, b1_line


def _coil_resistance(args, f_rf):
    """Return the loaded coil resistance (wire + reflected brine) at ``f_rf``."""

    r_coil = args.coil_radius_cm * 1e-2
    r_bh = args.borehole_radius_cm * 1e-2
    turns = int(args.coil_turns)
    length = args.coil_length_cm * 1e-2
    radii = np.full(turns, r_coil)
    centers = np.zeros((turns, 3))
    centers[:, 2] = np.linspace(-length / 2, length / 2, turns)
    inductance = coil_inductance(radii, centers, wire_radius=1e-3)
    coil = coils.solenoid(radius=r_coil, length=length, turns=turns, axis="z", n_segments=48)
    ng = 21
    axg = np.linspace(-r_bh * 1.1, r_bh * 1.1, ng)
    azg = np.linspace(-0.2, 0.2, ng)
    gx, gy, gz = np.meshgrid(axg, axg, azg, indexing="ij")
    rho = np.hypot(gx, gy)
    brine = (rho >= r_coil) & (rho <= r_bh) & (np.abs(gz) <= 0.18)
    r_wire = 0.5 * np.sqrt(f_rf / 1e6)
    loading = coil_loading(
        np.stack([gx, gy, gz], axis=-1), coil, conductivity=args.brine_conductivity,
        mask=brine, spacing=(axg[1] - axg[0], axg[1] - axg[0], azg[1] - azg[0]),
        frequencies=[f_rf], inductance=inductance, coil_resistance=r_wire,
    )
    return float(loading.reflected_resistance[0] + r_wire)


def build_data(args) -> dict:
    """Build the B0 sweet spot, the air/ferrite B1 maps, and the quantified gain."""

    r_sweet, b_sweet, f_rf, r0_line, b0_line = _jj_sweet_spot(args)
    core_outer = args.core_radius_cm * 1e-2

    prob_air, sol_air, r_line, b1_air = _b1_efficiency(args, None)
    prob_fer, sol_fer, _, b1_fer = _b1_efficiency(args, core_outer)

    eta_air = float(np.interp(r_sweet, r_line, b1_air))  # T per amp
    eta_fer = float(np.interp(r_sweet, r_line, b1_fer))
    gain = eta_fer / eta_air
    power_ratio = (eta_air / eta_fer) ** 2

    # Absolute operating point: fixed B1 from the hardware pulse length.
    b1_target = (np.pi / 2.0) / (args.pulse90_us * 1e-6) / GAMMA_H
    i_air = b1_target / eta_air
    i_fer = b1_target / eta_fer
    r_total = _coil_resistance(args, f_rf)
    p_air = 0.5 * i_air**2 * r_total
    p_fer = 0.5 * i_fer**2 * r_total

    # Design sweep: efficiency gain and RF-power fraction vs ferrite core radius.
    thick = np.linspace(args.sweep_min_cm, args.coil_radius_cm - 0.5, int(args.sweep_points))
    sweep_gain = np.empty_like(thick)
    for idx, rc_cm in enumerate(thick):
        _, _, rl, b1 = _b1_efficiency(args, rc_cm * 1e-2)
        sweep_gain[idx] = float(np.interp(r_sweet, rl, b1)) / eta_air

    # B1 map (uT/A) for the ferrite case over the r-z plane.
    b1_map = np.abs(sol_fer.b_z)
    return {
        "r_sweet": r_sweet, "b_sweet": b_sweet, "f_rf": f_rf,
        "r0_line": r0_line, "b0_line": b0_line,
        "r_line": r_line, "b1_air": b1_air, "b1_fer": b1_fer,
        "eta_air": eta_air, "eta_fer": eta_fer, "gain": gain, "power_ratio": power_ratio,
        "b1_target": b1_target, "i_air": i_air, "i_fer": i_fer,
        "p_air": p_air, "p_fer": p_fer, "r_total": r_total,
        "sweep_thick_cm": thick, "sweep_gain": sweep_gain,
        "grid_r": sol_fer.r_m, "grid_z": sol_fer.z_m, "b1_map": b1_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--remanence", type=float, default=1.05)
    parser.add_argument("--magnet-length-cm", type=float, default=30.0)
    parser.add_argument("--magnet-radius-cm", type=float, default=8.0)
    parser.add_argument("--gap-cm", type=float, default=34.0)
    parser.add_argument("--borehole-radius-cm", type=float, default=11.0)
    parser.add_argument("--coil-radius-cm", type=float, default=9.0)
    parser.add_argument("--coil-length-cm", type=float, default=30.0)
    parser.add_argument("--coil-turns", type=int, default=30)
    parser.add_argument("--core-radius-cm", type=float, default=7.5)
    parser.add_argument("--core-length-cm", type=float, default=30.0)
    parser.add_argument("--ferrite-mu-r", type=float, default=1000.0)
    parser.add_argument("--pulse90-us", type=float, default=25.0)
    parser.add_argument("--brine-conductivity", type=float, default=10.0)
    parser.add_argument("--box-cm", type=float, default=60.0)
    parser.add_argument("--nr", type=int, default=181)
    parser.add_argument("--nz", type=int, default=241)
    parser.add_argument("--sweep-min-cm", type=float, default=2.0)
    parser.add_argument("--sweep-points", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), constrained_layout=True)

    # (0,0) B1 efficiency map (ferrite case).
    r_cm = data["grid_r"] * 100.0
    z_cm = data["grid_z"] * 100.0
    show = r_cm <= 30.0
    im = axes[0, 0].pcolormesh(
        r_cm[show], z_cm, data["b1_map"][show, :].T * 1e6, shading="auto", cmap="viridis"
    )
    axes[0, 0].axvline(args.coil_radius_cm, color="w", ls="--", lw=1, label="coil")
    axes[0, 0].axvline(args.core_radius_cm, color="orange", ls="--", lw=1, label="ferrite edge")
    axes[0, 0].axvline(data["r_sweet"] * 100, color="r", ls=":", lw=1.2, label="sweet spot")
    axes[0, 0].set_xlabel("radius (cm)")
    axes[0, 0].set_ylabel("z (cm)")
    axes[0, 0].set_title("B1 efficiency map, ferrite core (uT/A)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=axes[0, 0])

    # (0,1) B1(r) at z=0: air vs ferrite.
    axes[0, 1].plot(data["r_line"] * 100, data["b1_air"] * 1e6, label="air-cored coil")
    axes[0, 1].plot(data["r_line"] * 100, data["b1_fer"] * 1e6, label="ferrite-cored coil")
    axes[0, 1].axvline(data["r_sweet"] * 100, color="r", ls=":", lw=1.2, label="sweet spot")
    axes[0, 1].axvspan(0, args.coil_radius_cm, color="gray", alpha=0.08)
    axes[0, 1].set_xlim(0, 30)
    axes[0, 1].set_xlabel("radius (cm)")
    axes[0, 1].set_ylabel("B1 efficiency |B_z| (uT/A)")
    axes[0, 1].set_title("Focused B1 in the formation")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.2)

    # (1,0) design sweep.
    ax = axes[1, 0]
    axp = ax.twinx()
    ax.plot(data["sweep_thick_cm"], data["sweep_gain"], "o-", color="tab:blue",
            label="B1 gain")
    axp.plot(data["sweep_thick_cm"], 100.0 / data["sweep_gain"] ** 2, "s--",
             color="tab:red", label="RF power (% of air)")
    ax.axvline(args.core_radius_cm, color="gray", ls=":", lw=1)
    ax.set_xlabel("Ferrite core radius (cm)")
    ax.set_ylabel("B1 gain (x)", color="tab:blue")
    axp.set_ylabel("RF power for same B1 (% of air)", color="tab:red")
    ax.set_title("Bigger core -> more focusing (demag-limited)")
    ax.grid(True, alpha=0.2)

    # (1,1) B0 context + quantified summary.
    axb = axes[1, 1]
    axb.plot(data["r0_line"] * 100, data["b0_line"], color="tab:purple")
    axb.axvline(data["r_sweet"] * 100, color="r", ls=":", lw=1.2)
    axb.set_xlabel("radius (cm)")
    axb.set_ylabel("|B0| at z=0 (T)")
    axb.set_title("JJ sweet spot and quantified gain")
    summary = (
        f"sweet spot: r={data['r_sweet']*100:.1f} cm, f_RF={data['f_rf']/1e6:.2f} MHz\n"
        f"B1 efficiency: {data['eta_air']*1e6:.1f} -> {data['eta_fer']*1e6:.1f} uT/A "
        f"(x{data['gain']:.2f})\n"
        f"RF power for same B1: {100*data['power_ratio']:.0f}% "
        f"({1/data['power_ratio']:.1f}x reduction)\n"
        f"received-signal SNR (reciprocity): x{data['gain']:.2f}\n"
        f"coil current: {data['i_air']:.1f} -> {data['i_fer']:.1f} A;  "
        f"RF power: {data['p_air']/1e3:.1f} -> {data['p_fer']/1e3:.1f} kW"
    )
    axb.text(0.03, 0.97, summary, transform=axb.transAxes, va="top", ha="left",
             fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    print("Jasper-Jackson logging tool: ferrite-focused RF coil")
    print(f"  sweet spot: r={data['r_sweet']*100:.1f} cm, |B0|={data['b_sweet']:.4f} T, "
          f"f_RF={data['f_rf']/1e6:.3f} MHz")
    print(f"  B1 efficiency: air {data['eta_air']*1e6:.2f} uT/A -> ferrite "
          f"{data['eta_fer']*1e6:.2f} uT/A  (x{data['gain']:.2f})")
    print(f"  RF power for same B1: {100*data['power_ratio']:.1f}% of air "
          f"({1/data['power_ratio']:.2f}x reduction)")
    print(f"  received-signal SNR gain (reciprocity): x{data['gain']:.2f}")
    print(f"  coil current {data['i_air']:.1f} A -> {data['i_fer']:.1f} A, "
          f"RF power {data['p_air']/1e3:.2f} kW -> {data['p_fer']/1e3:.2f} kW "
          f"(R_total={data['r_total']:.2f} ohm)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
