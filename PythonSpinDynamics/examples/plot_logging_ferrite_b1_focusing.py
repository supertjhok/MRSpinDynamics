"""Optimizing the RF coil of a Jasper-Jackson logging tool: ferrite core, turn
spacing, and honest SNR with RF losses.

An inside-out well-logging tool (two opposing SmCo magnets, low-gradient sweet
spot shell in the formation) transmits and receives with a solenoid coil. Three
design levers set the per-echo signal-to-noise ratio at the sweet spot, and this
example computes all of them with the axisymmetric nonlinear magnetostatic solver
(:class:`spin_dynamics.fields.AxisymmetricMagnetostatics`) coupled to the same
asymptotic-CPMG echo model as ``plot_logging_cpmg_multinuclear.py``:

1. **Ferrite core** behind the coil (high-permeability, far below saturation ->
   linear) mirrors flux outward and raises the B1 efficiency at the sweet spot.
   The gain is limited by the core's demagnetizing factor (aspect ratio), so its
   height and thickness are swept.
2. **Turn spacing.** Concentrating turns toward the z = 0 midplane (where the
   sweet spot sits) is a strong efficiency lever. B1 homogeneity over the shell
   degrades, but the CPMG echo is inherently tolerant of B1 inhomogeneity, so the
   region-integrated echo signal keeps rising -- the optimum is as concentrated
   as practical constraints (turn size, voltage) allow, not an interior maximum.
3. **RF losses.** The coil is brine-loss dominated. Both levers boost the coil's
   coupling to the lossy borehole brine as well as to the sample, so the
   reflected resistance (from the solved vector potential) rises with the field,
   partly offsetting the signal gain. The ferrite core's own RF loss (a
   loss-tangent model) is included and is negligible next to the brine.

The bottom line is the **net** per-echo SNR (region-integrated echo signal
divided by the Johnson noise of the loaded coil) and the RF power for a fixed
sweet-spot B1 -- not the naive efficiency ratio, which ignores the loss that
grows with the field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from plot_logging_cpmg_multinuclear import (  # noqa: E402
    GAMMA_H,
    KB,
    _curie_m0,
    _echo_table,
    _jj_magnet,
)
from spin_dynamics.fields.magnetostatics import finite_magnet_array_b0  # noqa: E402
from spin_dynamics.fields.nonlinear_magnetostatics import (  # noqa: E402
    MU0,
    AxisymmetricMagnetostatics,
    rf_ferrite,
)

N_H = 98.6e3 * 6.02214076e23  # saturated-brine 1H number density (m^-3)


def _context(args):
    """Build the JJ B0 sweet spot, sensitive-region grid, and the echo table."""

    rods = _jj_magnet(args.remanence, args.magnet_length_cm * 1e-2,
                      args.magnet_radius_cm * 1e-2, args.gap_cm * 1e-2)
    rr = np.linspace(args.borehole_radius_cm * 1e-2, 0.26, int(args.region_nr))
    zz = np.linspace(-0.10, 0.10, int(args.region_nz))
    grid_r, grid_z = np.meshgrid(rr, zz, indexing="ij")
    pts = np.stack([grid_r, np.zeros_like(grid_r), grid_z], axis=-1)
    b0 = np.linalg.norm(finite_magnet_array_b0(pts, rods), axis=-1)
    mid = int(np.argmin(np.abs(zz)))
    i_sweet = int(np.argmax(b0[:, mid]))
    r_sweet, b_sweet = float(rr[i_sweet]), float(b0[i_sweet, mid])
    f_rf = GAMMA_H * b_sweet / (2.0 * np.pi)

    t90 = args.pulse90_us * 1e-6
    t180 = 2.0 * t90
    w1_cal = (np.pi / 2.0) / t90
    tau_half = max(0.5 * args.echo_spacing_us * 1e-6 - 0.5 * t180, 0.0)
    table = _echo_table(0.5, t90, t180, tau_half,
                        np.linspace(-10 * w1_cal, 10 * w1_cal, 101),
                        np.linspace(0.0, 3.0 * w1_cal, 51))
    m0 = _curie_m0(N_H, GAMMA_H, 0.5, b0, args.temperature_k)
    dv = 2.0 * np.pi * grid_r * (rr[1] - rr[0]) * (zz[1] - zz[0])
    return dict(
        grid_r=grid_r, grid_z=grid_z, b0=b0, mid=mid, i_sweet=i_sweet,
        r_sweet=r_sweet, b_sweet=b_sweet, f_rf=f_rf, w_rf=2.0 * np.pi * f_rf,
        w1_cal=w1_cal, table=table, m0=m0, dv=dv,
        offset=GAMMA_H * (b0 - b_sweet),
    )


def _solve(args, conc_cm: float, core_radius_cm: float | None, core_length_cm: float):
    """Axisymmetric solve for a coil turn distribution and optional ferrite core."""

    r_coil = args.coil_radius_cm * 1e-2
    rmax = args.box_cm * 1e-2
    nr, nz = int(args.nr), int(args.nz)
    prob = AxisymmetricMagnetostatics(np.linspace(0, rmax, nr), np.linspace(-rmax, rmax, nz))
    dr = rmax / (nr - 1)
    coil = prob.rectangle((r_coil, r_coil + 2 * dr),
                          (-args.coil_length_cm * 1e-2 / 2, args.coil_length_cm * 1e-2 / 2))
    profile = np.exp(-(prob._gz / (conc_cm * 1e-2)) ** 2) * coil
    prob.jphi[coil] = (
        float(args.coil_turns) / (np.sum(profile[coil]) * prob.hr * prob.hz)
    ) * profile[coil]
    core_mask = None
    if core_radius_cm is not None:
        core_mask = prob.rectangle((0.0, core_radius_cm * 1e-2),
                                   (-core_length_cm * 1e-2 / 2, core_length_cm * 1e-2 / 2))
        prob.add_material(core_mask, rf_ferrite(mu_r=args.ferrite_mu_r))
    return prob, prob.solve(), core_mask


def _evaluate(args, ctx, prob, sol, core_mask):
    """Return efficiency, echo signal, loss breakdown, SNR, and homogeneity."""

    b1 = np.abs(sol.sample_bz(ctx["grid_r"].ravel(), ctx["grid_z"].ravel())
                ).reshape(ctx["grid_r"].shape)  # T per amp
    eta_sweet = float(b1[ctx["i_sweet"], ctx["mid"]])
    coil_current = ctx["w1_cal"] / (GAMMA_H * eta_sweet)

    omega1 = GAMMA_H * b1 * coil_current
    echo = ctx["table"](np.stack([ctx["offset"], omega1], axis=-1))
    signal = float(np.sum(ctx["w_rf"] * b1 * ctx["m0"] * echo * ctx["dv"]))

    # Loaded coil resistance. R_wire (skin-effect model); R_brine = reflected
    # resistance from the solved vector potential over the borehole annulus (it
    # scales with the field, so it grows as the coil is focused); R_ferrite from
    # a loss-tangent model over the core.
    r_coil, r_bh = args.coil_radius_cm * 1e-2, args.borehole_radius_cm * 1e-2
    r_wire = 0.5 * np.sqrt(ctx["f_rf"] / 1e6)
    r_grid = prob.r[:, None]
    dv_ax = np.broadcast_to(2.0 * np.pi * r_grid * prob.hr * prob.hz, sol.a_phi.shape)
    brine = (prob._gr >= r_coil) & (prob._gr <= r_bh) & (np.abs(prob._gz) <= 0.18)
    r_brine = ctx["w_rf"] ** 2 * args.brine_conductivity * float(
        np.sum((sol.a_phi**2 * dv_ax)[brine])
    )
    r_ferrite = 0.0
    if core_mask is not None:
        b2 = sol.b_r**2 + sol.b_z**2
        r_ferrite = (ctx["w_rf"] * args.ferrite_loss_tangent) / (MU0 * args.ferrite_mu_r) * float(
            np.sum((b2 * dv_ax)[core_mask])
        )
    r_total = r_wire + r_brine + r_ferrite
    noise = np.sqrt(4.0 * KB * args.temperature_k * r_total * args.noise_bandwidth_hz)
    snr = signal / noise if noise > 0 else 0.0

    shell = (np.abs(ctx["grid_r"] - ctx["r_sweet"]) < 0.02) & (np.abs(ctx["grid_z"]) < 0.03)
    homog = float(np.ptp(b1[shell]) / np.mean(b1[shell]))
    rf_power = 0.5 * coil_current**2 * r_total
    return dict(b1=b1, eta=eta_sweet, current=coil_current, signal=signal,
                r_wire=r_wire, r_brine=r_brine, r_ferrite=r_ferrite, r_total=r_total,
                snr=snr, homog=homog, rf_power=rf_power)


def build_data(args) -> dict:
    ctx = _context(args)
    uniform_cm = args.coil_length_cm  # sigma >= length is effectively uniform

    pb, sb, mb = _solve(args, uniform_cm, None, args.core_length_cm)
    base = _evaluate(args, ctx, pb, sb, mb)
    po, so, mo = _solve(args, args.turn_concentration_cm, args.core_radius_cm, args.core_length_cm)
    opt = _evaluate(args, ctx, po, so, mo)

    # Ferrite core-radius sweep (uniform coil, isolates the ferrite lever).
    core_radii = np.linspace(args.sweep_core_min_cm, args.coil_radius_cm - 0.5,
                             int(args.sweep_points))
    fer_eta = np.empty_like(core_radii)
    for idx, cr in enumerate(core_radii):
        p, s, m = _solve(args, uniform_cm, cr, args.core_length_cm)
        fer_eta[idx] = _evaluate(args, ctx, p, s, m)["eta"]

    # Turn-concentration sweep (with the ferrite core in place).
    concs = np.linspace(args.sweep_conc_min_cm, uniform_cm, int(args.sweep_points))[::-1]
    turn_snr = np.empty_like(concs)
    turn_homog = np.empty_like(concs)
    for idx, c in enumerate(concs):
        p, s, m = _solve(args, c, args.core_radius_cm, args.core_length_cm)
        ev = _evaluate(args, ctx, p, s, m)
        turn_snr[idx] = ev["snr"]
        turn_homog[idx] = ev["homog"]

    return dict(ctx=ctx, base=base, opt=opt, opt_prob=po, opt_sol=so,
                core_radii_cm=core_radii, fer_eta=fer_eta, eta_air=base["eta"],
                concs_cm=concs, turn_snr=turn_snr, turn_homog=turn_homog)


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
    parser.add_argument("--turn-concentration-cm", type=float, default=5.0,
                        help="Gaussian sigma of the turn density (small = concentrated)")
    parser.add_argument("--core-radius-cm", type=float, default=8.5)
    parser.add_argument("--core-length-cm", type=float, default=20.0)
    parser.add_argument("--ferrite-mu-r", type=float, default=1000.0)
    parser.add_argument("--ferrite-loss-tangent", type=float, default=0.05)
    parser.add_argument("--pulse90-us", type=float, default=25.0)
    parser.add_argument("--echo-spacing-us", type=float, default=600.0)
    parser.add_argument("--temperature-k", type=float, default=350.0)
    parser.add_argument("--brine-conductivity", type=float, default=10.0)
    parser.add_argument("--noise-bandwidth-hz", type=float, default=5000.0)
    parser.add_argument("--box-cm", type=float, default=60.0)
    parser.add_argument("--nr", type=int, default=161)
    parser.add_argument("--nz", type=int, default=221)
    parser.add_argument("--region-nr", type=int, default=70)
    parser.add_argument("--region-nz", type=int, default=70)
    parser.add_argument("--sweep-core-min-cm", type=float, default=3.0)
    parser.add_argument("--sweep-conc-min-cm", type=float, default=3.0)
    parser.add_argument("--sweep-points", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)
    ctx, base, opt = data["ctx"], data["base"], data["opt"]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), constrained_layout=True)

    # (0,0) B1 map of the optimized coil.
    r_cm = data["opt_prob"].r * 100.0
    z_cm = data["opt_prob"].z * 100.0
    show = r_cm <= 26.0
    im = axes[0, 0].pcolormesh(r_cm[show], z_cm,
                               np.abs(data["opt_sol"].b_z)[show, :].T * 1e6,
                               shading="auto", cmap="viridis")
    axes[0, 0].axvline(args.coil_radius_cm, color="w", ls="--", lw=1, label="coil")
    axes[0, 0].axvline(args.core_radius_cm, color="orange", ls="--", lw=1, label="ferrite")
    axes[0, 0].axvline(ctx["r_sweet"] * 100, color="r", ls=":", lw=1.2, label="sweet spot")
    axes[0, 0].set_xlabel("radius (cm)")
    axes[0, 0].set_ylabel("z (cm)")
    axes[0, 0].set_title("Optimized B1 efficiency (uT/A)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=axes[0, 0])

    # (0,1) turn-concentration sweep: SNR (monotone) vs homogeneity (degrades).
    ax = axes[0, 1]
    axh = ax.twinx()
    snr_rel = data["turn_snr"] / base["snr"]
    ax.plot(data["concs_cm"], snr_rel, "o-", color="tab:blue", label="net SNR")
    axh.plot(data["concs_cm"], 100 * data["turn_homog"], "s--", color="tab:red",
             label="B1 spread over shell")
    ax.axvline(args.turn_concentration_cm, color="gray", ls=":", lw=1)
    ax.set_xlabel("Turn concentration sigma (cm)  <- more concentrated")
    ax.set_ylabel("net SNR (x baseline)", color="tab:blue")
    axh.set_ylabel("B1 spread over shell (%)", color="tab:red")
    ax.set_title("Turn spacing: SNR rises, CPMG tolerates B1 spread")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.2)

    # (1,0) ferrite core-radius sweep.
    axes[1, 0].plot(data["core_radii_cm"], data["fer_eta"] / data["eta_air"], "o-",
                    color="tab:green")
    axes[1, 0].axvline(args.core_radius_cm, color="gray", ls=":", lw=1)
    axes[1, 0].set_xlabel("Ferrite core radius (cm)")
    axes[1, 0].set_ylabel("B1 efficiency gain (x)")
    axes[1, 0].set_title(f"Ferrite core (L={args.core_length_cm:g} cm), demag-limited")
    axes[1, 0].grid(True, alpha=0.2)

    # (1,1) honest summary.
    axb = axes[1, 1]
    axb.axis("off")
    naive_power = (base["eta"] / opt["eta"]) ** 2
    net_power = opt["rf_power"] / base["rf_power"]

    def row(label, col_a, col_b):
        return f"{label:<20s}{col_a:>14s}{col_b:>14s}"

    lines = [
        "Jasper-Jackson RF coil optimization",
        f"sweet spot r={ctx['r_sweet']*100:.1f} cm, f_RF={ctx['f_rf']/1e6:.2f} MHz",
        "",
        row("", "baseline", "optimized"),
        row("  coil", "uniform", "concentrated"),
        row("  core", "air", "ferrite"),
        row("  B1 eff (uT/A)", f"{base['eta']*1e6:.1f}", f"{opt['eta']*1e6:.1f}"),
        row("  R_total (ohm)", f"{base['r_total']:.1f}", f"{opt['r_total']:.1f}"),
        row("  wire/brine/ferr",
            f"{base['r_wire']:.1f}/{base['r_brine']:.0f}/0",
            f"{opt['r_wire']:.1f}/{opt['r_brine']:.0f}/{opt['r_ferrite']:.2f}"),
        row("  B1 spread (shell)", f"{100*base['homog']:.0f}%", f"{100*opt['homog']:.0f}%"),
        "",
        f"net per-echo SNR gain:  x{opt['snr']/base['snr']:.2f}",
        f"RF power for same B1:   {100*net_power:.0f}% ({1/net_power:.1f}x reduction)",
        f"  (naive, ignoring loss: {100*naive_power:.0f}% / {1/naive_power:.1f}x)",
    ]
    axb.text(0.0, 1.0, "\n".join(lines), transform=axb.transAxes, va="top", ha="left",
             family="monospace", fontsize=8.5)

    print("Jasper-Jackson logging coil: ferrite core + turn-spacing optimization")
    print(f"  sweet spot r={ctx['r_sweet']*100:.1f} cm, f_RF={ctx['f_rf']/1e6:.3f} MHz")
    print(f"  B1 efficiency:  baseline {base['eta']*1e6:.2f} -> optimized "
          f"{opt['eta']*1e6:.2f} uT/A  (x{opt['eta']/base['eta']:.2f})")
    print(f"  R_total: baseline {base['r_total']:.1f} ohm "
          f"(brine {base['r_brine']:.1f}) -> optimized {opt['r_total']:.1f} ohm "
          f"(brine {opt['r_brine']:.1f}, ferrite loss {opt['r_ferrite']:.3f})")
    print(f"  net per-echo SNR gain: x{opt['snr']/base['snr']:.2f} "
          f"(naive reciprocity x{opt['eta']/base['eta']:.2f})")
    print(f"  RF power for same B1: {100*net_power:.1f}% of baseline "
          f"({1/net_power:.2f}x reduction; naive {1/naive_power:.2f}x)")
    print(f"  B1 spread over shell: {100*base['homog']:.0f}% -> {100*opt['homog']:.0f}% "
          "(CPMG-tolerated)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
