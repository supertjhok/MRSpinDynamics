"""Compare an unrelaxed vs a relaxed NaNO2 14N finite-temperature EFG run.

Reads two results JSON files written by ``efg_temperature.py collect --out-json``
(one for the unrelaxed starter geometry, one for the relaxed geometry) and prints
a side-by-side comparison against the measured NaNO2 14N reference.  Also writes a
Markdown report.

The point of the comparison: relaxing the structure to an energy minimum should
move the equilibrium asymmetry ``eta`` toward the experimental ~0.38 and split the
two strong lines toward the measured 3.60 / 4.64 MHz, while the temperature slope
keeps the right (negative, Bayer) sign.

Run (from the QuadrupolarDFT root):
    python3 examples/abinit/compare_relaxation.py \
        runs/nano2_relax_study/unrelaxed.json \
        runs/nano2_relax_study/relaxed.json \
        --out runs/nano2_relax_study/relaxation_comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured NaNO2 14N reference (ordered phase), from the NQR database series used
# in integration/examples/nano2_temperature_coefficients.py:
#   nu_plus : 4929 kHz @ 77 K -> 4637 kHz @ 300 K
#   nu_minus: 3757 kHz @ 77 K -> 3601 kHz @ 300 K
# For spin 1: nu_0 = nu_+ - nu_-, C_Q = (2/3)(nu_+ + nu_-), eta = 3(nu_+ - nu_-)/(nu_+ + nu_-).
MEASURED = {
    77.0: {"nu_plus": 4.929, "nu_minus": 3.757},
    300.0: {"nu_plus": 4.637, "nu_minus": 3.601},
}


def _measured_at(temperature_k):
    """Return (lines_mhz, cq_mhz, eta) for a measured temperature, or None."""

    row = MEASURED.get(temperature_k)
    if row is None:
        return None
    nu_p, nu_m = row["nu_plus"], row["nu_minus"]
    nu_0 = nu_p - nu_m
    cq = (2.0 / 3.0) * (nu_p + nu_m)
    eta = 3.0 * (nu_p - nu_m) / (nu_p + nu_m)
    return sorted([nu_0, nu_m, nu_p]), cq, eta


def _measured_dnu_dt():
    """Measured per-line dnu/dT (kHz/K) over the 77-300 K span: [nu_0, nu_-, nu_+]."""

    lo = _measured_at(77.0)[0]
    hi = _measured_at(300.0)[0]
    span = 300.0 - 77.0
    return [(hi[i] - lo[i]) / span * 1e3 for i in range(3)]


def _fmt_lines(lines):
    return ", ".join(f"{x:.4f}" for x in lines)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _point_at(results, temperature_k):
    for point in results["points"]:
        if abs(point["temperature_k"] - temperature_k) < 1e-6:
            return point
    return None


def build_report(unrelaxed, relaxed):
    lines = []

    def out(text=""):
        lines.append(text)

    out("# NaNO2 14N: relaxed vs unrelaxed finite-temperature EFG")
    out()
    out(f"- Unrelaxed: `{unrelaxed['label']}`  ({unrelaxed['n_modes']} modes, "
        f"target atom index {unrelaxed['target_atom_index']})")
    out(f"- Relaxed:   `{relaxed['label']}`  ({relaxed['n_modes']} modes, "
        f"target atom index {relaxed['target_atom_index']})")
    out(f"- Spin {relaxed['spin']:.0f}, Q = {relaxed['quadmom_barns']} barns.")
    out()
    out("Measured reference (ordered phase, NQR database): at 300 K the lines are "
        "~1.036 / 3.601 / 4.637 MHz, eta ~ 0.38, C_Q ~ 5.49 MHz.")
    out()

    # ---- Static (0 K, fixed geometry) -------------------------------------
    out("## Static EFG (0 K, no vibration)")
    out()
    out("| quantity | unrelaxed | relaxed | measured (300 K) |")
    out("|---|---|---|---|")
    us, rs = unrelaxed["static"], relaxed["static"]
    meas = _measured_at(300.0)
    out(f"| C_Q (MHz) | {us['cq_mhz']:+.4f} | {rs['cq_mhz']:+.4f} | {meas[1]:.2f} |")
    out(f"| eta | {us['eta']:.4f} | {rs['eta']:.4f} | {meas[2]:.2f} |")
    out(f"| lines (MHz) | {_fmt_lines(us['lines_mhz'])} | "
        f"{_fmt_lines(rs['lines_mhz'])} | {_fmt_lines(meas[0])} |")
    out()

    # ---- Temperature sweep -------------------------------------------------
    out("## Temperature sweep")
    out()
    out("| T (K) | C_Q unrelax | C_Q relax | eta unrelax | eta relax | "
        "lines unrelax (MHz) | lines relax (MHz) | lines measured (MHz) |")
    out("|---|---|---|---|---|---|---|---|")
    temps = sorted({p["temperature_k"] for p in unrelaxed["points"]}
                   | {p["temperature_k"] for p in relaxed["points"]})
    for t in temps:
        up = _point_at(unrelaxed, t)
        rp = _point_at(relaxed, t)
        meas = _measured_at(t)
        cq_u = f"{up['cq_mhz']:+.4f}" if up else "--"
        cq_r = f"{rp['cq_mhz']:+.4f}" if rp else "--"
        eta_u = f"{up['eta']:.4f}" if up else "--"
        eta_r = f"{rp['eta']:.4f}" if rp else "--"
        ln_u = _fmt_lines(up["lines_mhz"]) if up else "--"
        ln_r = _fmt_lines(rp["lines_mhz"]) if rp else "--"
        ln_m = _fmt_lines(meas[0]) if meas else "--"
        out(f"| {t:.0f} | {cq_u} | {cq_r} | {eta_u} | {eta_r} | "
            f"{ln_u} | {ln_r} | {ln_m} |")
    out()

    # ---- dnu/dT ------------------------------------------------------------
    out("## Temperature coefficients dnu/dT (kHz/K)")
    out()
    meas_slopes = _measured_dnu_dt()
    u_slopes = unrelaxed.get("dnu_dt_khz_per_k", [])
    r_slopes = relaxed.get("dnu_dt_khz_per_k", [])
    out("| line | unrelaxed | relaxed | measured (77-300 K) |")
    out("|---|---|---|---|")
    n = max(len(u_slopes), len(r_slopes), len(meas_slopes))
    for i in range(n):
        u = f"{u_slopes[i]['slope_khz_per_k']:+.2f}" if i < len(u_slopes) else "--"
        r = f"{r_slopes[i]['slope_khz_per_k']:+.2f}" if i < len(r_slopes) else "--"
        m = f"{meas_slopes[i]:+.2f}" if i < len(meas_slopes) else "--"
        label = (f"{r_slopes[i]['line_mhz']:.3f} MHz" if i < len(r_slopes)
                 else f"{u_slopes[i]['line_mhz']:.3f} MHz" if i < len(u_slopes)
                 else f"line {i}")
        out(f"| {label} | {u} | {r} | {m} |")
    out()

    # ---- Verdict -----------------------------------------------------------
    out("## Reading the result")
    out()
    d_eta_u = abs(us["eta"] - 0.38)
    d_eta_r = abs(rs["eta"] - 0.38)
    moved = "toward" if d_eta_r < d_eta_u else "away from"
    out(f"- Static eta moved {moved} the experimental 0.38 after relaxation "
        f"({us['eta']:.3f} -> {rs['eta']:.3f}; |error| {d_eta_u:.3f} -> {d_eta_r:.3f}).")
    split_u = us["lines_mhz"][-1] - us["lines_mhz"][-2]
    split_r = rs["lines_mhz"][-1] - rs["lines_mhz"][-2]
    out(f"- Splitting of the two strong lines: {split_u:.3f} MHz (unrelaxed) -> "
        f"{split_r:.3f} MHz (relaxed); measured ~1.04 MHz.")
    wn_u = unrelaxed.get("mode_wavenumbers_cm_inv") or []
    wn_r = relaxed.get("mode_wavenumbers_cm_inv") or []
    if wn_u and wn_r:
        out(f"- Lowest libration used: {min(wn_u):.1f} cm^-1 (unrelaxed) vs "
            f"{min(wn_r):.1f} cm^-1 (relaxed), over {unrelaxed['n_modes']} vs "
            f"{relaxed['n_modes']} modes. A softer lowest mode gives a larger thermal "
            "amplitude and a stronger dnu/dT -- physical, not a problem. Whether the "
            "relaxation reached a true minimum is decided by the absence of imaginary "
            "modes in the phonon run (anaddb.out), not by this frequency: the workflow "
            "drops imaginary/acoustic modes, so they never reach this comparison.")
    else:
        out(f"- Mode counts: {unrelaxed['n_modes']} (unrelaxed) vs "
            f"{relaxed['n_modes']} (relaxed).")
    out()
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unrelaxed_json", help="collect --out-json for the starter geometry")
    parser.add_argument("relaxed_json", help="collect --out-json for the relaxed geometry")
    parser.add_argument("--out", help="write the Markdown report to this path")
    args = parser.parse_args()

    unrelaxed = _load(args.unrelaxed_json)
    relaxed = _load(args.relaxed_json)
    report = build_report(unrelaxed, relaxed)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
