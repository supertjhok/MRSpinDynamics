"""RF properties of a single-layer solenoid, and how they drive probe SNR.

Turns a solenoid's geometry into the numbers a probe designer needs -- inductance,
AC (skin + proximity) resistance, unloaded Q and self-resonant frequency -- with
:func:`spin_dynamics.fields.coil_properties.solenoid_properties` (a port of the QOIL
sheath-helix model). Two things are demonstrated:

1. **Frequency sweep.** The effective (RF) inductance ``L_eff`` tracks the
   frequency-independent current-sheet value ``L_s`` at low frequency and rises toward
   the first (quarter-wave) self-resonance as transmission-line effects kick in. The
   solver-independent field-based inductance (Biot-Savart / Neumann,
   :func:`solenoid_field_inductance`) is overlaid as a cross-check of ``L_s``. The AC
   resistance grows ~sqrt(f) and the unloaded ``Q = omega L / R`` peaks near a cubical
   form factor.

2. **Design loop -> SNR.** At a chosen operating frequency the extracted ``(L, R)`` and
   the tuning capacitance ``C = 1/(omega^2 L)`` are fed to
   :func:`spin_dynamics.fields.quasistatic.coil_loading` against a conductive sample,
   closing the geometry -> lumped-parameters -> loaded-Q / noise-penalty loop that used
   to require hand-typed coil values.

Run with ``--save out.png`` to write the four-panel figure; otherwise it prints tables.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import coils
from spin_dynamics.fields.coil_properties import (
    ANNEALED_COPPER,
    ConductorMaterial,
    solenoid_field_inductance,
    solenoid_properties,
)
from spin_dynamics.fields.quasistatic import coil_inductance, coil_loading


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diameter-mm", type=float, default=20.0, help="Mean coil diameter.")
    parser.add_argument("--length-mm", type=float, default=30.0, help="Coil length.")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--wire-mm", type=float, default=1.0, help="Wire diameter.")
    parser.add_argument("--fmin-mhz", type=float, default=1.0)
    parser.add_argument("--fmax-mhz", type=float, default=60.0)
    parser.add_argument("--points", type=int, default=40)
    parser.add_argument("--design-mhz", type=float, default=10.0, help="Operating frequency for the SNR demo.")
    parser.add_argument("--sample-conductivity", type=float, default=0.7, help="Aqueous sample (S/m).")
    parser.add_argument("--cryo-temperature-k", type=float, default=77.0, help="Cold temperature for the cryo-Q comparison.")
    parser.add_argument("--cryo-rrr", type=float, default=100.0, help="Residual-resistivity ratio of the cold conductor.")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    D = args.diameter_mm * 1e-3
    length = args.length_mm * 1e-3
    d = args.wire_mm * 1e-3
    turns = int(args.turns)

    # Field-based DC inductance (frequency independent) as an independent cross-check.
    l_field = solenoid_field_inductance(diameter=D, length=length, turns=turns, wire_diameter=d)

    # --- Frequency sweep of the extracted properties --------------------------------
    freqs = np.linspace(args.fmin_mhz * 1e6, args.fmax_mhz * 1e6, int(args.points))
    l_eff = np.empty_like(freqs)
    l_s = np.empty_like(freqs)
    r_ac = np.empty_like(freqs)
    q = np.empty_like(freqs)
    f_res = np.nan
    for i, f in enumerate(freqs):
        cp = solenoid_properties(diameter=D, length=length, turns=turns, wire_diameter=d, frequency=f)
        l_eff[i] = cp.inductance_effective
        l_s[i] = cp.inductance_currentsheet
        r_ac[i] = cp.ac_resistance
        q[i] = cp.q_factor
        f_res = cp.self_resonant_frequency

    print("Single-layer solenoid RF properties")
    print(f"  geometry: {turns}-turn, D={args.diameter_mm:.1f} mm, l={args.length_mm:.1f} mm, "
          f"wire d={args.wire_mm:.2f} mm  (form factor l/D={length / D:.2f})")
    print(f"  current-sheet L_s = {l_s[0] * 1e6:.3f} uH   field-based L = {l_field * 1e6:.3f} uH "
          f"(differ {abs(l_field - l_s[0]) / l_s[0] * 100:.1f}%)")
    print(f"  self-resonant frequency f_res = {f_res / 1e6:.1f} MHz")
    print(f"  {'f (MHz)':>8s}{'L_eff (uH)':>12s}{'R_ac (ohm)':>12s}{'Q':>8s}")
    for i, f in enumerate(freqs):
        if i % max(1, len(freqs) // 8) and i != len(freqs) - 1:
            continue
        print(f"  {f / 1e6:8.1f}{l_eff[i] * 1e6:12.3f}{r_ac[i]:12.4f}{q[i]:8.0f}")

    # --- Design loop: extracted (L, R, C) -> loaded Q / SNR penalty ------------------
    f0 = args.design_mhz * 1e6
    cp0 = solenoid_properties(diameter=D, length=length, turns=turns, wire_diameter=d, frequency=f0)
    c_tune = cp0.tuning_capacitance()
    print(f"\nProbe design at {args.design_mhz:.1f} MHz")
    print(f"  L = {cp0.inductance * 1e6:.3f} uH   R = {cp0.ac_resistance:.4f} ohm   "
          f"Q_unloaded = {cp0.q_factor:.0f}")
    print(f"  tuning capacitance C = {c_tune * 1e12:.1f} pF  (resonates the tank at {args.design_mhz:.1f} MHz)")

    # Conductive aqueous sample filling the bore, on a Cartesian grid.
    a = D / 2.0
    ext_xy = a * 1.2
    ext_z = length * 0.9
    n_xy, n_z = 41, 41
    axx = np.linspace(-ext_xy, ext_xy, n_xy)
    axz = np.linspace(-ext_z, ext_z, n_z)
    dxy = axx[1] - axx[0]
    dz = axz[1] - axz[0]
    X, Y, Z = np.meshgrid(axx, axx, axz, indexing="ij")
    rho = np.sqrt(X**2 + Y**2)
    sample = (rho <= 0.8 * a) & (np.abs(Z) <= 0.8 * ext_z)
    grid = np.stack([X, Y, Z], axis=-1)

    coil = coils.solenoid(radius=a, length=length, turns=turns, axis="z", n_segments=48)
    radii = np.full(turns, a)
    centers = np.zeros((turns, 3))
    centers[:, 2] = np.linspace(-length / 2, length / 2, turns)
    inductance = coil_inductance(radii, centers, wire_radius=d / 2.0)
    loading = coil_loading(
        grid, coil, conductivity=args.sample_conductivity, mask=sample,
        spacing=(dxy, dxy, dz), frequencies=[f0], inductance=inductance,
        coil_resistance=cp0.ac_resistance,
    )
    print(f"  loaded by sigma={args.sample_conductivity:.2f} S/m sample: "
          f"R_reflected={loading.reflected_resistance[0]:.4f} ohm  "
          f"Q_loaded={loading.q_loaded[0]:.0f}  noise penalty x{loading.noise_penalty[0]:.2f}")

    # Cooling the coil cuts the conductor resistivity, so R drops and the unloaded Q
    # rises -- the cryoprobe gain. RRR sets the residual (defect-scattering) floor.
    cold = ConductorMaterial(
        f"{ANNEALED_COPPER.name} (RRR={args.cryo_rrr:g})",
        ANNEALED_COPPER.resistivity, ANNEALED_COPPER.mu_r,
        temp_coefficient=ANNEALED_COPPER.temp_coefficient,
        reference_temperature=ANNEALED_COPPER.reference_temperature,
        residual_resistivity_ratio=args.cryo_rrr,
    )
    cp_cold = solenoid_properties(
        diameter=D, length=length, turns=turns, wire_diameter=d, frequency=f0,
        material=cold, temperature=args.cryo_temperature_k,
    )
    print(f"  cooling {cp0.temperature:.0f} K -> {cp_cold.temperature:.0f} K "
          f"(RRR={args.cryo_rrr:g}): R {cp0.ac_resistance:.4f} -> {cp_cold.ac_resistance:.4f} ohm, "
          f"Q_unloaded {cp0.q_factor:.0f} -> {cp_cold.q_factor:.0f} "
          f"(x{cp_cold.q_factor / cp0.q_factor:.2f})")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fmhz = freqs / 1e6
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(fmhz, l_eff * 1e6, "o-", ms=3, lw=1.4, label="L_eff (sheath helix)")
        axes[0, 0].plot(fmhz, l_s * 1e6, "-", lw=1.4, label="L_s (current sheet)")
        axes[0, 0].axhline(l_field * 1e6, ls=":", color="C2", lw=1.4, label="L field-based (DC)")
        axes[0, 0].axvline(f_res / 1e6, ls="--", color="gray", lw=1, label=f"f_res={f_res / 1e6:.0f} MHz")
        axes[0, 0].set_title("Inductance: RF vs quasistatic")
        axes[0, 0].set_xlabel("frequency (MHz)")
        axes[0, 0].set_ylabel("inductance (uH)")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.2)

        axes[0, 1].plot(fmhz, r_ac, "o-", ms=3, color="C3", lw=1.4)
        axes[0, 1].set_title("AC resistance (skin + proximity) ~ sqrt(f)")
        axes[0, 1].set_xlabel("frequency (MHz)")
        axes[0, 1].set_ylabel("R_ac (ohm)")
        axes[0, 1].grid(True, alpha=0.2)

        axes[1, 0].plot(fmhz, q, "o-", ms=3, color="C4", lw=1.4)
        axes[1, 0].axvline(f_res / 1e6, ls="--", color="gray", lw=1)
        axes[1, 0].set_title("Unloaded quality factor  Q = omega L / R")
        axes[1, 0].set_xlabel("frequency (MHz)")
        axes[1, 0].set_ylabel("Q")
        axes[1, 0].grid(True, alpha=0.2)

        # Q vs form factor at the design frequency (the classic cubical-coil optimum).
        # Restrict to lengths where the winding pitch still clears the wire diameter.
        ratio_min = 1.05 * turns * d / D
        ratios = np.linspace(max(0.4, ratio_min), 3.0, 30)
        q_ratio = []
        for ratio in ratios:
            ll = ratio * D
            cpr = solenoid_properties(diameter=D, length=ll, turns=turns, wire_diameter=d, frequency=f0)
            q_ratio.append(cpr.q_factor)
        axes[1, 1].plot(ratios, q_ratio, "o-", ms=3, color="C5", lw=1.4)
        axes[1, 1].axvline(1.0, ls="--", color="gray", lw=1, label="cubical (l = D)")
        axes[1, 1].set_title(f"Q vs form factor at {args.design_mhz:.0f} MHz")
        axes[1, 1].set_xlabel("length / diameter")
        axes[1, 1].set_ylabel("Q")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"  saved: {args.save}")


if __name__ == "__main__":
    main()
