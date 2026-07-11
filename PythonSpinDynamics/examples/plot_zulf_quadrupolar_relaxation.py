"""Physically derived 14N quadrupolar relaxation and its effect on Earth's-field JCS.

The companion example ``plot_zulf_quadrupolar_jcoupling.py`` sweeps the 14N
relaxation rate R(14N) directly. Here R(14N) is instead *derived* from the
nitrogen quadrupole coupling constant C_Q and the rotational correlation time
tau_c of the molecule, using the single-correlation-time quadrupolar model
(Abragam Ch. VIII; Kowalewski & Maler Ch. 4):

    R1 = R2 -> (3 pi^2 / 10) * (2I+3)/(I^2(2I-1)) * (1 + eta^2/3) * C_Q^2 * tau_c

in the extreme-narrowing limit that holds at Earth's field. Faster tumbling
(shorter tau_c, e.g. smaller molecule or lower viscosity) gives *slower*
quadrupolar relaxation, so the molecular dynamics control how strongly the 14N
collapses the J-coupled spectrum of the spins bonded to it.

Panels: (left) R1/R2(14N) versus tau_c for the chosen C_Q, marking the
coalescence rate; (right) the resulting 19F J-coupled spectrum for a few tau_c
values, showing the collapse driven by molecular tumbling rather than an
ad-hoc rate. Reproduces the mechanism behind Altenhof et al., J. Magn. Reson.
355 (2023) 107540.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling import (  # noqa: E402
    multinuclear_quadrupolar_rates,
    multinuclear_system,
    simulate_zulf_spectrum,
)
from spin_dynamics.relaxation import quadrupolar_relaxation_rates  # noqa: E402


def build_data(args: argparse.Namespace) -> dict:
    """Build the R(14N)-vs-tau_c dispersion and tau_c-resolved 19F spectra."""

    b0 = args.b0_ut * 1e-6
    cq_hz = args.cq_mhz * 1e6
    nu_n = 3.0766e6 * b0  # 14N Larmor at this field

    tau_grid = np.geomspace(args.tau_min_ps, args.tau_max_ps, int(args.tau_points)) * 1e-12
    rates = quadrupolar_relaxation_rates(
        cq_hz,
        1.0,
        tau_grid,
        asymmetry=args.eta,
        larmor_angular_rad_per_s=2.0 * np.pi * nu_n,
    )

    # Three-spin 1H-19F-14N with strong 2J(F,N).
    j = np.zeros((3, 3))
    j[0, 1] = j[1, 0] = args.j_hf_hz
    j[0, 2] = j[2, 0] = args.j_hn_hz
    j[1, 2] = j[2, 1] = args.j_fn_hz
    system = multinuclear_system(["1H", "19F", "14N"], j, b0)
    f_idx = system.indices_for_isotope("19F")
    nu_f = float(system.larmor_hz[1])
    window = (nu_f - 90.0, nu_f + 90.0)

    spectra = []
    derived_r = []
    for tau_ps in args.tau_c_ps:
        r1, r2 = multinuclear_quadrupolar_rates(
            system,
            correlation_time_seconds=tau_ps * 1e-12,
            quadrupole_coupling_hz=cq_hz,
            asymmetry=args.eta,
            spin_half_rate_hz=args.spin_half_rate_hz,
        )
        derived_r.append(float(r1[2]))
        spec = simulate_zulf_spectrum(
            system,
            r1_per_second=r1,
            r2_per_second=r2,
            dwell_seconds=args.dwell_ms * 1e-3,
            n_points=int(args.n_points),
            detect_indices=f_idx,
            apodization_hz=args.apodization_hz,
        )
        mask = (spec.frequencies_hz >= window[0]) & (spec.frequencies_hz <= window[1])
        spectra.append((spec.frequencies_hz[mask], np.abs(spec.spectrum[mask])))

    return {
        "tau_grid_ps": tau_grid * 1e12,
        "r1": rates.r1_per_second,
        "r2": rates.r2_per_second,
        "tau_c_ps": np.asarray(args.tau_c_ps, dtype=np.float64),
        "derived_r": np.asarray(derived_r, dtype=np.float64),
        "spectra": spectra,
        "larmor_n": nu_n,
    }


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--b0-ut", type=float, default=50.0)
    parser.add_argument("--cq-mhz", type=float, default=4.0)
    parser.add_argument("--eta", type=float, default=0.4)
    parser.add_argument("--j-hf-hz", type=float, default=8.0)
    parser.add_argument("--j-hn-hz", type=float, default=1.0)
    parser.add_argument("--j-fn-hz", type=float, default=37.0)
    parser.add_argument("--tau-min-ps", type=float, default=0.1)
    parser.add_argument("--tau-max-ps", type=float, default=100.0)
    parser.add_argument("--tau-points", type=int, default=120)
    parser.add_argument(
        "--tau-c-ps",
        type=float,
        nargs="+",
        default=[0.3, 1.0, 3.0, 10.0, 40.0],
        help="correlation times (ps) for the tau_c-resolved 19F spectra",
    )
    parser.add_argument("--spin-half-rate-hz", type=float, default=0.3)
    parser.add_argument("--dwell-ms", type=float, default=0.2)
    parser.add_argument("--n-points", type=int, default=32768)
    parser.add_argument("--apodization-hz", type=float, default=0.8)
    parser.add_argument("--coalescence-hz", type=float, default=250.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)

    disp = axes[0]
    disp.loglog(data["tau_grid_ps"], data["r1"], label="R1(14N)")
    disp.loglog(data["tau_grid_ps"], data["r2"], "--", label="R2(14N)")
    disp.axhline(args.coalescence_hz, color="gray", ls=":", lw=1.0,
                 label=f"coalescence ~{args.coalescence_hz:g} Hz")
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, data["tau_c_ps"].size))
    for tau_ps, r_val, color in zip(data["tau_c_ps"], data["derived_r"], colors):
        disp.plot(tau_ps, r_val, "o", color=color, ms=7)
    disp.set_xlabel("Correlation time tau_c (ps)")
    disp.set_ylabel("14N relaxation rate (1/s)")
    disp.set_title(f"Derived R(14N): C_Q={args.cq_mhz:g} MHz, eta={args.eta:g}")
    disp.legend(fontsize=8)
    disp.grid(True, which="both", alpha=0.2)

    spec_ax = axes[1]
    step = 0.0
    for (freq, mag), tau_ps, r_val, color in zip(
        data["spectra"], data["tau_c_ps"], data["derived_r"], colors
    ):
        norm = mag / mag.max() if mag.max() > 0 else mag
        spec_ax.plot(freq, norm + step, color=color, lw=1.0,
                     label=f"tau_c={tau_ps:g} ps -> R(14N)={r_val:.0f} Hz")
        step += 1.15
    spec_ax.set_yticks([])
    spec_ax.set_xlabel("Frequency (Hz)")
    spec_ax.set_title("19F JCS driven by molecular tumbling")
    spec_ax.legend(fontsize=7, loc="upper right")

    print("Earth's-field 19F JCS with 14N relaxation derived from C_Q + tau_c")
    print(f"B0={args.b0_ut:g} uT  C_Q(14N)={args.cq_mhz:g} MHz  eta={args.eta:g}  "
          f"nu(14N)={data['larmor_n']:.1f} Hz")
    for tau_ps, r_val in zip(data["tau_c_ps"], data["derived_r"]):
        print(f"  tau_c={tau_ps:6g} ps -> R(14N)={r_val:8.1f} Hz")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
