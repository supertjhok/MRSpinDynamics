"""OPM-detected 14N NQR: tuned atomic magnetometer vs SQUID and coil.

Radio-frequency (Mx) optically pumped magnetometers detect nuclear quadrupole
resonance at ultra-low field without cryogenics. Savukov, Seltzer & Romalis and
US 7,521,928 demonstrated a potassium RF magnetometer at ~0.24 fT/rtHz tuned to
423 kHz to detect the 14N NQR of ammonium nitrate; SQUIDs have detected the
3.31 MHz 14N line of HMT. This example uses the field-referred detector layer
(``spin_dynamics.detection``) to compare, on a chosen 14N line:

* an **RF/Mx OPM** tuned to the line (band-pass, sub-fT floor on resonance),
* a **SERF OPM** (zero-field low-pass) -- which is *rolled off* at MHz and so
  cannot detect a MHz NQR line, the defining OPM caveat,
* a **SQUID** (flat 1 fT/rtHz, but cryogenic), and
* an **ideal Faraday coil** (rising 1/f field noise).

Three things are shown (panels with ``--save``):

* **Noise floor across the line.** The RF-OPM band-pass dips to its floor at the
  carrier; SERF sits orders of magnitude higher at MHz; the coil and SQUID are
  ~flat across the narrow line.
* **Detected SNR** on the line for each detector (relative units).
* **Bandwidth matching.** As the NQR linewidth grows past the OPM bandwidth, the
  RF-OPM SNR degrades -- the sensor's atomic bandwidth must cover the line.

The 14N line frequency/linewidth default to literature values (see ``--compound``);
signal amplitudes are relative -- the physics shown is the detector comparison and
the atomic-bandwidth roll-off, not an absolute SNR.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.detection import (  # noqa: E402
    IdealFaradayCoil,
    OPMMagnetometer,
    SQUIDMagnetometer,
    detected_field_snr,
)

# 14N NQR lines: (frequency Hz, HWHM Hz). Frequencies are zero-field NQR lines;
# linewidths are representative narrow-line values for the detection comparison.
COMPOUNDS = {
    "ammonium-nitrate": (423.0e3, 100.0),  # Savukov RF-OPM demonstration
    "hmt": (3.308e6, 150.0),               # SQUID-detected 14N line (HMT)
    "rdx": (5.19e6, 300.0),                # explosive 14N line (illustrative)
}


def nqr_line(f_center, hwhm, *, n=8001, span=60.0):
    """Unit-area Lorentzian 14N NQR line at ``f_center`` with half-width ``hwhm``."""

    fg = np.linspace(f_center - span * hwhm, f_center + span * hwhm, n)
    s = (hwhm / np.pi) / ((fg - f_center) ** 2 + hwhm**2)
    return fg, s


def build_detectors(f_center, *, opm_bandwidth_hz):
    return {
        "OPM (RF, tuned)": OPMMagnetometer.rf(
            f_center, field_noise_T_per_rtHz=0.24e-15, bandwidth_hz=opm_bandwidth_hz
        ),
        "OPM (SERF, DC)": OPMMagnetometer.serf(
            field_noise_T_per_rtHz=0.16e-15, bandwidth_hz=200.0
        ),
        "SQUID": SQUIDMagnetometer.berkeley_2007(),
        "Faraday coil": IdealFaradayCoil(
            field_noise_T_per_rtHz_ref=1e-15, f_ref_hz=1e6
        ),
    }


def run(*, f_center, hwhm, opm_bandwidth_hz):
    fg, s = nqr_line(f_center, hwhm)
    detectors = build_detectors(f_center, opm_bandwidth_hz=opm_bandwidth_hz)
    snr = {name: detected_field_snr(s, fg, det).snr for name, det in detectors.items()}

    # bandwidth matching: RF-OPM SNR vs line HWHM / sensor bandwidth
    rf = detectors["OPM (RF, tuned)"]
    lw_over_bw = np.logspace(-1.0, 1.5, 40)
    hwhms = lw_over_bw * opm_bandwidth_hz
    bw_snr = []
    for w in hwhms:
        fgi, si = nqr_line(f_center, w)
        bw_snr.append(detected_field_snr(si, fgi, rf).snr)
    return {
        "fg": fg,
        "s": s,
        "detectors": detectors,
        "snr": snr,
        "lw_over_bw": lw_over_bw,
        "bw_snr": np.asarray(bw_snr),
        "f_center": f_center,
        "hwhm": hwhm,
        "opm_bandwidth_hz": opm_bandwidth_hz,
    }


def print_summary(res, *, compound) -> None:
    fc = res["f_center"]
    rf = res["detectors"]["OPM (RF, tuned)"]
    serf = res["detectors"]["OPM (SERF, DC)"]
    floor_rf = rf.field_noise_amplitude(np.array([fc]))[0]
    serf_penalty = (
        serf.field_noise_amplitude(np.array([fc]))[0]
        / serf.field_noise_amplitude(np.array([0.0]))[0]
    )
    print(f"OPM-detected 14N NQR: {compound} line at {fc / 1e6:.3f} MHz")
    print(f"  RF-OPM floor on resonance   : {floor_rf * 1e15:.2f} fT/rtHz")
    print(f"  SERF roll-off at the line   : {serf_penalty:.0f}x its DC floor "
          f"(~ f/df = {fc / serf.bandwidth_hz:.0f})")
    ref = res["snr"]["Faraday coil"]
    print("  detected SNR (relative to Faraday coil):")
    for name, value in res["snr"].items():
        print(f"    {name:18s}: {value / ref:6.2f}x")


def make_figure(plt, res):
    fc = res["f_center"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) noise floor across the line
    ax = axes[0, 0]
    band = np.linspace(fc - 30 * res["hwhm"], fc + 30 * res["hwhm"], 2000)
    for name, det in res["detectors"].items():
        ax.plot((band - fc), det.field_noise_amplitude(band) * 1e15, label=name)
    ax.axvline(0.0, color="0.6", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("offset from line center (Hz)")
    ax.set_ylabel("field noise (fT/rtHz)")
    ax.set_title(f"(a) Noise floor across the {fc / 1e6:.3f} MHz line")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # (b) detected SNR bar chart (relative to coil)
    ax = axes[0, 1]
    ref = res["snr"]["Faraday coil"]
    names = list(res["snr"])
    vals = [res["snr"][n] / ref for n in names]
    ax.bar(range(len(names)), vals, color=["C0", "C1", "C2", "C3"])
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.axhline(1.0, color="0.6", ls="--", lw=1)
    ax.set_ylabel("detected SNR / coil")
    ax.set_title("(b) Detected SNR on the line")
    ax.grid(True, which="both", axis="y", alpha=0.25)

    # (c) bandwidth matching
    ax = axes[1, 0]
    ax.loglog(res["lw_over_bw"], res["bw_snr"] / res["bw_snr"][0])
    ax.axvline(1.0, color="0.6", ls="--", lw=1)
    ax.set_xlabel("line HWHM / OPM bandwidth")
    ax.set_ylabel("RF-OPM SNR (normalized)")
    ax.set_title("(c) OPM bandwidth must cover the line")
    ax.grid(True, which="both", alpha=0.25)

    # (d) wide-frequency response shape
    ax = axes[1, 1]
    wide = np.logspace(1, 7.5, 3000)
    for name in ("OPM (RF, tuned)", "OPM (SERF, DC)"):
        det = res["detectors"][name]
        ax.loglog(wide, det.field_noise_amplitude(wide) * 1e15, label=name)
    ax.axvline(fc, color="0.6", ls=":", lw=1)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("field noise (fT/rtHz)")
    ax.set_title("(d) RF band-pass vs SERF low-pass")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("OPM-detected 14N NQR: tuned atomic magnetometer vs SQUID and coil")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compound", choices=sorted(COMPOUNDS), default="ammonium-nitrate",
                        help="14N NQR line preset (default ammonium-nitrate, 423 kHz)")
    parser.add_argument("--frequency-hz", type=float, default=None,
                        help="override the line frequency (Hz)")
    parser.add_argument("--linewidth-hz", type=float, default=None,
                        help="override the line HWHM (Hz)")
    parser.add_argument("--opm-bandwidth-hz", type=float, default=300.0,
                        help="RF-OPM atomic bandwidth / HWHM in Hz (default 300)")
    parser.add_argument("--save", type=str, default=None,
                        help="path to save the figure; if omitted, only prints a summary")
    args = parser.parse_args()

    f_center, hwhm = COMPOUNDS[args.compound]
    if args.frequency_hz is not None:
        f_center = args.frequency_hz
    if args.linewidth_hz is not None:
        hwhm = args.linewidth_hz

    res = run(f_center=f_center, hwhm=hwhm, opm_bandwidth_hz=args.opm_bandwidth_hz)
    print_summary(res, compound=args.compound)

    if args.save:
        plt = load_matplotlib(required=True, headless=True)
        fig = make_figure(plt, res)
        fig.savefig(args.save, dpi=150)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
