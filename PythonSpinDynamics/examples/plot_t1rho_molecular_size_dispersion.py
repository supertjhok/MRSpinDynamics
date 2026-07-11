"""Prepolarized T1rho relaxation dispersion for molecules of different sizes.

A spin-lock (T1rho) experiment reports the spectral density of molecular motion
at the spin-lock nutation frequency ``gamma B1`` (0.1-3 kHz is typical), rather
than at zero frequency like ``T2``. The rotational correlation time ``tau_c`` of
a tumbling molecule grows with its hydrodynamic size (Stokes-Einstein-Debye), so
sweeping molecule size sweeps ``tau_c`` and moves the T1rho dispersion:

* small, fast tumblers (short ``tau_c``): ``R1rho`` is small and flat -- no
  dispersion in the accessible spin-lock band, and ``R1rho ~ R2``;
* large, slow tumblers (long ``tau_c``): ``R1rho`` is large at weak lock (where
  it equals ``R2``) and disperses down toward a plateau once the lock frequency
  ``gamma B1`` climbs past ``1/tau_c``.

Because slow motions require ``tau_c`` in the microsecond-millisecond range to
disperse in the kHz spin-lock band, the default medium here is viscous (like a
crowded intracellular / concentrated-macromolecule environment); tune it with
``--viscosity-mpa-s``. The sequence is prepolarized: the sample first polarizes
in ``--prepolarizing-field-t`` before a ``(pi/2)`` pulse tips it onto the
spin-lock axis, so the readout also reflects the lab-frame ``T1`` build-up.

References: Kowalewski, Chpt 4.3; Stanford RAD226b Lecture 11 ("Relaxation in
the Rotating Frame").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.relaxation import (  # noqa: E402
    bpp_relaxation_rates,
    simulate_prepolarized_t1rho,
    stokes_einstein_debye_correlation_time,
    t1rho_relaxation_rate,
)


def build_size_dispersion(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Build prepolarized T1rho dispersion arrays over molecule size."""

    radii_m = np.asarray(args.radii_nm, dtype=np.float64) * 1.0e-9
    if radii_m.size == 0 or np.any(radii_m <= 0.0):
        raise ValueError("radii-nm must be positive")
    viscosity_pa_s = args.viscosity_mpa_s * 1.0e-3
    tau_c = stokes_einstein_debye_correlation_time(
        radii_m,
        viscosity_pa_s,
        args.temperature_k,
        slip_factor=args.slip_factor,
    )

    spin_lock_hz = np.geomspace(
        args.spin_lock_min_hz,
        args.spin_lock_max_hz,
        int(args.spin_locks),
    )
    omega1 = 2.0 * np.pi * spin_lock_hz
    omega0 = 2.0 * np.pi * args.larmor_mhz * 1.0e6

    r1rho = t1rho_relaxation_rate(
        omega1[np.newaxis, :],
        omega0,
        tau_c[:, np.newaxis],
        coupling_scale_per_second2=args.coupling_scale,
    )
    t1rho = np.divide(
        1.0,
        r1rho,
        out=np.full_like(r1rho, np.inf, dtype=np.float64),
        where=r1rho > 0.0,
    )

    # Lab-frame T1/T2 at the same correlation times drive prepolarization and
    # set the R2 (weak-lock) asymptote each dispersion curve relaxes from.
    lab = bpp_relaxation_rates(
        angular_frequency_rad_per_s=omega0,
        correlation_time_seconds=tau_c,
        coupling_scale_per_second2=args.coupling_scale,
    )

    experiment = simulate_prepolarized_t1rho(
        spin_lock_angular_rad_per_s=omega1,
        larmor_angular_rad_per_s=omega0,
        correlation_time_seconds=tau_c,
        spin_lock_time_seconds=args.spin_lock_time_s,
        coupling_scale_per_second2=args.coupling_scale,
        polarizing_field_tesla=args.prepolarizing_field_t,
        detection_field_tesla=args.detection_field_t,
        prepolarization_time_seconds=args.prepolarization_time_s,
        t1_seconds=lab.t1_seconds,
    )

    return {
        "radii_nm": radii_m * 1.0e9,
        "tau_c": tau_c,
        "spin_lock_hz": spin_lock_hz,
        "r1rho": r1rho,
        "t1rho": t1rho,
        "t1_seconds": lab.t1_seconds,
        "t2_seconds": lab.t2_seconds,
        "prepared_m0": experiment.prepared_m0,
        "locked_signal": experiment.locked_signal,
        "corner_hz": 1.0 / (2.0 * np.pi * tau_c),
    }


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--radii-nm",
        type=float,
        nargs="+",
        default=[0.5, 2.0, 6.0, 20.0],
        help="hydrodynamic radii of the tumbling molecules (nm)",
    )
    parser.add_argument("--viscosity-mpa-s", type=float, default=100.0)
    parser.add_argument("--temperature-k", type=float, default=310.0)
    parser.add_argument("--slip-factor", type=float, default=1.0)
    parser.add_argument("--spin-lock-min-hz", type=float, default=20.0)
    parser.add_argument("--spin-lock-max-hz", type=float, default=20000.0)
    parser.add_argument("--spin-locks", type=int, default=160)
    parser.add_argument("--spin-lock-time-s", type=float, default=0.03)
    parser.add_argument("--lock-marker-hz-low", type=float, default=100.0)
    parser.add_argument("--lock-marker-hz-high", type=float, default=2000.0)
    parser.add_argument("--larmor-mhz", type=float, default=20.0)
    parser.add_argument("--coupling-scale", type=float, default=2.0e6)
    parser.add_argument("--prepolarizing-field-t", type=float, default=0.3)
    parser.add_argument("--detection-field-t", type=float, default=0.02)
    parser.add_argument("--prepolarization-time-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.spin_lock_min_hz <= 0.0 or args.spin_lock_max_hz <= args.spin_lock_min_hz:
        raise ValueError("spin-lock range must be positive and increasing")

    plt = load_matplotlib(headless=args.output is not None)
    data = build_size_dispersion(args)
    radii_nm = data["radii_nm"]
    spin_lock_hz = data["spin_lock_hz"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, radii_nm.size))

    disp_ax = axes[0, 0]
    for idx, radius in enumerate(radii_nm):
        disp_ax.loglog(
            spin_lock_hz,
            data["r1rho"][idx],
            color=colors[idx],
            label=f"{radius:g} nm (tau_c={data['tau_c'][idx] * 1e6:.3g} us)",
        )
        corner = data["corner_hz"][idx]
        if spin_lock_hz[0] <= corner <= spin_lock_hz[-1]:
            disp_ax.axvline(corner, color=colors[idx], ls=":", lw=0.8, alpha=0.6)
    disp_ax.set_xlabel("Spin-lock frequency gamma B1 / 2pi (Hz)")
    disp_ax.set_ylabel("R1rho (1/s)")
    disp_ax.set_title("T1rho Dispersion vs Molecule Size")
    disp_ax.legend(fontsize=8)
    disp_ax.grid(True, which="both", alpha=0.2)

    t1rho_ax = axes[0, 1]
    for idx, radius in enumerate(radii_nm):
        t1rho_ax.loglog(
            spin_lock_hz,
            data["t1rho"][idx] * 1.0e3,
            color=colors[idx],
            label=f"{radius:g} nm",
        )
    t1rho_ax.set_xlabel("Spin-lock frequency gamma B1 / 2pi (Hz)")
    t1rho_ax.set_ylabel("T1rho (ms)")
    t1rho_ax.set_title("Spin-Lock Relaxation Time")
    t1rho_ax.legend(fontsize=8)
    t1rho_ax.grid(True, which="both", alpha=0.2)

    # R1rho and R2 versus correlation time at a fixed lock frequency: the
    # dispersion "turns over" the monotonic R2 rise (lecture slide 17).
    peak_ax = axes[1, 0]
    tau_grid = np.geomspace(
        data["tau_c"].min() * 0.2,
        data["tau_c"].max() * 5.0,
        200,
    )
    omega0 = 2.0 * np.pi * args.larmor_mhz * 1.0e6
    r2_grid = bpp_relaxation_rates(
        angular_frequency_rad_per_s=omega0,
        correlation_time_seconds=tau_grid,
        coupling_scale_per_second2=args.coupling_scale,
    ).r2_per_second
    peak_ax.loglog(tau_grid * 1e6, r2_grid, "k--", label="R2 (weak lock)")
    for lock_hz in (args.lock_marker_hz_low, args.lock_marker_hz_high):
        r1rho_grid = t1rho_relaxation_rate(
            2.0 * np.pi * lock_hz,
            omega0,
            tau_grid,
            coupling_scale_per_second2=args.coupling_scale,
        )
        peak_ax.loglog(tau_grid * 1e6, r1rho_grid, label=f"R1rho @ {lock_hz:g} Hz")
    peak_ax.set_xlabel("Correlation time tau_c (us)")
    peak_ax.set_ylabel("Relaxation rate (1/s)")
    peak_ax.set_title("Dispersion Turns Over the R2 Rise")
    peak_ax.legend(fontsize=8)
    peak_ax.grid(True, which="both", alpha=0.2)

    # Signal fraction surviving the spin lock, exp(-TSL * R1rho): the T1rho
    # contrast the sequence generates. Absolute readout is this times the
    # prepared M0 (shown per size in the legend), which spans many decades
    # because slow molecules also have long lab-frame T1 and polarize poorly.
    sig_ax = axes[1, 1]
    fraction = np.exp(-args.spin_lock_time_s * data["r1rho"])
    for idx, radius in enumerate(radii_nm):
        sig_ax.semilogx(
            spin_lock_hz,
            fraction[idx],
            color=colors[idx],
            label=f"{radius:g} nm (M0={data['prepared_m0'][idx]:.2g})",
        )
    sig_ax.set_xlabel("Spin-lock frequency gamma B1 / 2pi (Hz)")
    sig_ax.set_ylabel(f"Signal fraction exp(-TSL*R1rho), TSL={args.spin_lock_time_s * 1e3:g} ms")
    sig_ax.set_title("Prepolarized T1rho Contrast")
    sig_ax.set_ylim(-0.05, 1.05)
    sig_ax.legend(fontsize=8)
    sig_ax.grid(True, which="both", alpha=0.2)

    print("Prepolarized T1rho molecular-size dispersion")
    print(f"viscosity mPa*s: {args.viscosity_mpa_s:g}")
    print(f"temperature K: {args.temperature_k:g}")
    for idx, radius in enumerate(radii_nm):
        print(
            f"  r={radius:g} nm  tau_c={data['tau_c'][idx] * 1e6:.4g} us"
            f"  corner={data['corner_hz'][idx]:.4g} Hz"
            f"  R1rho[{spin_lock_hz[0]:.0f}Hz]={data['r1rho'][idx, 0]:.4g}/s"
            f"  R1rho[{spin_lock_hz[-1]:.0f}Hz]={data['r1rho'][idx, -1]:.4g}/s"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
