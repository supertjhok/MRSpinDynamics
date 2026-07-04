"""Sample loading of an NMR well-logging coil by a conductive brine borehole.

Eddy-current loss in a conductive sample couples back into the coil (by
reciprocity) as an added series resistance ``R_reflected(omega) = omega^2 *
integral sigma |A_unit|^2 dV``. This degrades the loaded Q and raises the
Johnson-noise floor -- and it grows as omega^2, so there is a real signal-vs-loss
trade-off with operating frequency.

The demonstration is an inside-out NMR well-logging tool (Jasper-Jackson-style):
a circular-loop / solenoid transmit-receive coil on the tool axis inside a
cylindrical borehole filled with saturated brine (a strong conductor). The brine
in the borehole is the dominant near-field loss, so the coil is heavily loaded.
Sweeping 0.5-2 MHz shows R_reflected rising as f^2 and overtaking the coil's own
resistance, collapsing Q and inflating the noise floor.

First-order (Born) limitation (docs/field_solvers_quasistatic.md): no skin-effect
shielding, so the loss is over-estimated once the brine skin depth drops below the
borehole size -- the example prints the skin depth so the validity edge is clear.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import coils
from spin_dynamics.fields.magnetostatics import MU0
from spin_dynamics.fields.quasistatic import coil_inductance, coil_loading, eddy_currents


def _skin_depth(sigma, freq):
    return np.sqrt(2.0 / (MU0 * sigma * 2.0 * np.pi * freq))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coil-radius-mm", type=float, default=45.0)
    parser.add_argument("--coil-length-mm", type=float, default=100.0)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--tool-radius-mm", type=float, default=50.0, help="Brine annulus inner radius.")
    parser.add_argument("--borehole-radius-mm", type=float, default=110.0)
    parser.add_argument("--brine-conductivity", type=float, default=10.0, help="Saturated brine (S/m).")
    parser.add_argument("--coil-resistance-1mhz", type=float, default=0.5, help="Coil resistance at 1 MHz (ohm).")
    parser.add_argument("--wire-radius-mm", type=float, default=1.0)
    parser.add_argument("--fmin-mhz", type=float, default=0.5)
    parser.add_argument("--fmax-mhz", type=float, default=2.0)
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--grid-xy", type=int, default=41)
    parser.add_argument("--grid-z", type=int, default=45)
    parser.add_argument("--map-frequency-mhz", type=float, default=1.0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    a = args.coil_radius_mm * 1e-3
    length = args.coil_length_mm * 1e-3
    r_tool = args.tool_radius_mm * 1e-3
    r_bh = args.borehole_radius_mm * 1e-3
    sigma = args.brine_conductivity

    coil = coils.solenoid(radius=a, length=length, turns=int(args.turns), axis="z", n_segments=48)
    radii = np.full(int(args.turns), a)
    centers = np.zeros((int(args.turns), 3))
    centers[:, 2] = np.linspace(-length / 2, length / 2, int(args.turns))
    inductance = coil_inductance(radii, centers, wire_radius=args.wire_radius_mm * 1e-3)

    # Brine annulus (tool surface -> borehole wall) on a Cartesian grid.
    ext_xy = r_bh * 1.1
    ext_z = length * 1.6
    axx = np.linspace(-ext_xy, ext_xy, int(args.grid_xy))
    axz = np.linspace(-ext_z, ext_z, int(args.grid_z))
    dxy = axx[1] - axx[0]
    dz = axz[1] - axz[0]
    X, Y, Z = np.meshgrid(axx, axx, axz, indexing="ij")
    rho = np.sqrt(X**2 + Y**2)
    brine = (rho >= r_tool) & (rho <= r_bh) & (np.abs(Z) <= ext_z * 0.85)
    grid = np.stack([X, Y, Z], axis=-1)

    freqs = np.linspace(args.fmin_mhz * 1e6, args.fmax_mhz * 1e6, int(args.points))
    # Coil resistance with a sqrt(f) skin-effect model (referenced at 1 MHz).
    r_coil = args.coil_resistance_1mhz * np.sqrt(freqs / 1e6)
    loading = coil_loading(
        grid, coil, conductivity=sigma, mask=brine, spacing=(dxy, dxy, dz),
        frequencies=freqs, inductance=inductance, coil_resistance=r_coil,
    )

    print("NMR well-logging coil loaded by a conductive brine borehole")
    print(f"  coil: {args.turns}-turn solenoid r={args.coil_radius_mm:.0f} mm, L={inductance * 1e6:.2f} uH")
    print(f"  borehole: brine sigma={sigma:.1f} S/m, annulus {args.tool_radius_mm:.0f}-{args.borehole_radius_mm:.0f} mm")
    print(f"  {'f (MHz)':>8s}{'R_coil':>9s}{'R_refl':>9s}{'Q_unl':>7s}{'Q_load':>7s}{'noise x':>9s}{'skin (cm)':>11s}")
    for i, f in enumerate(freqs):
        if i % max(1, len(freqs) // 8) and i != len(freqs) - 1:
            continue
        print(f"  {f / 1e6:8.2f}{r_coil[i]:9.3f}{loading.reflected_resistance[i]:9.3f}"
              f"{loading.q_unloaded[i]:7.0f}{loading.q_loaded[i]:7.0f}"
              f"{loading.noise_penalty[i]:9.2f}{_skin_depth(sigma, f) * 100:11.1f}")

    # crossover: where reflected resistance overtakes the coil resistance
    over = np.where(loading.reflected_resistance > r_coil)[0]
    if over.size:
        print(f"  sample-noise dominated (R_refl > R_coil) above {freqs[over[0]] / 1e6:.2f} MHz")
    delta_max = _skin_depth(sigma, freqs[-1])
    if delta_max < r_bh:
        print(f"  NOTE: skin depth {delta_max * 100:.1f} cm < borehole radius {r_bh * 100:.1f} cm at the top of "
              "the band -- first-order model over-estimates loss there.")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fmhz = freqs / 1e6

        axes[0, 0].loglog(fmhz, loading.reflected_resistance, "o-", lw=1.4, label="R_reflected (~f^2)")
        axes[0, 0].loglog(fmhz, r_coil, "s-", lw=1.4, label="R_coil (~sqrt f)")
        axes[0, 0].set_title("Series resistance vs frequency")
        axes[0, 0].set_xlabel("frequency (MHz)")
        axes[0, 0].set_ylabel("resistance (ohm)")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, which="both", alpha=0.2)

        axes[0, 1].plot(fmhz, loading.q_unloaded, "s-", lw=1.4, label="Q unloaded")
        axes[0, 1].plot(fmhz, loading.q_loaded, "o-", lw=1.4, label="Q loaded (brine)")
        axes[0, 1].set_title("Quality factor collapse")
        axes[0, 1].set_xlabel("frequency (MHz)")
        axes[0, 1].set_ylabel("Q")
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].plot(fmhz, loading.noise_penalty, "o-", color="C3", lw=1.4)
        axes[1, 0].axhline(1.0, ls="--", color="gray", lw=1)
        axes[1, 0].set_title("Noise-floor penalty  sqrt(R_total / R_coil)")
        axes[1, 0].set_xlabel("frequency (MHz)")
        axes[1, 0].set_ylabel("SNR degradation (x)")

        # eddy Joule-loss map in the borehole (x-z plane) at map-frequency
        omega = 2.0 * np.pi * args.map_frequency_mhz * 1e6
        res = eddy_currents(grid, coil, omega, conductivity=sigma, mask=brine,
                            spacing=(dxy, dxy, dz), charge_correction=False)
        joule = sigma * np.sum(np.abs(res.e_field) ** 2, axis=-1)
        mid = joule.shape[1] // 2
        extent = [-ext_xy * 1e3, ext_xy * 1e3, -ext_z * 1e3, ext_z * 1e3]
        im = axes[1, 1].imshow(joule[:, mid, :].T, origin="lower", extent=extent,
                               aspect="auto", cmap="inferno")
        axes[1, 1].set_title(f"Brine eddy loss sigma|E|^2 at {args.map_frequency_mhz:.1f} MHz (W/m^3)")
        axes[1, 1].set_xlabel("x (mm)")
        axes[1, 1].set_ylabel("z (mm)")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"  saved: {args.save}")


if __name__ == "__main__":
    main()
