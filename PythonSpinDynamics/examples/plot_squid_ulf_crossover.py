"""SQUID vs Faraday at ultra-low field: the microtesla detection crossover.

Reproduces the core quantitative claims of Clarke, Hatridge & Moessle,
"SQUID-Detected MRI in Microtesla Fields," *Annu. Rev. Biomed. Eng.* **9**, 389
(2007), using the field-referred detector layer (``spin_dynamics.detection``).

The whole argument rests on how detector *field* sensitivity scales with
frequency. A Faraday coil measures ``dPhi/dt``, so its field-referred noise rises
as ``1/f`` toward low frequency; an untuned SQUID flux transformer measures the
field itself, so its noise floor is flat. The two therefore cross at a few MHz,
and below the crossover the SQUID wins -- by a factor that grows without limit as
the field is lowered.

Three consequences are shown, each a panel with ``--save``:

* **Noise-floor crossover.** ``N(f)`` for a SQUID (flat, anchored at 1 fT/rtHz)
  and an ideal Faraday coil (``1/f``); the crossover frequency is marked.
* **Detected SNR vs B0, prepolarized sample.** With the moment set by ``Bp``
  (not ``B0``) and the line narrowing as ``Delta-nu ~ B0`` down to a homogeneous
  ``T2`` floor, the SQUID SNR *rises* as ``B0`` falls (``~ B0^-1/2``) while the
  Faraday SNR falls (``~ B0^+1/2``) -- the Clarke Summary Pt. 2/3 scalings.
* **Line narrowing (Fig. 4 analog).** The same line at several ``B0`` values,
  showing the ``B0``-proportional narrowing that concentrates the fixed spectral
  area into a taller peak.

A summary (crossover frequency, floor at 5.6 kHz / 132 uT, fitted scaling
exponents, SQUID/Faraday advantage at 132 uT) prints always; ``--save`` writes
the figure. Signal amplitudes are in relative units -- the physics shown is the
*scaling* and the crossover, not an absolute SNR.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.detection import (  # noqa: E402
    IdealFaradayCoil,
    SQUIDMagnetometer,
    detected_field_snr,
)

GAMMA_H_HZ_PER_T = 42.5774e6  # proton gyromagnetic ratio / 2pi


def line_spectrum(
    b0_t: float,
    *,
    frac_inhomogeneity: float,
    t2_s: float,
    area: float = 1.0,
    n: int = 4001,
    span_factor: float = 20.0,
):
    """A unit-area Lorentzian line at ``f0 = gamma*B0`` (prepolarized moment).

    The linewidth is the inhomogeneous width ``frac*f0`` floored by the
    homogeneous width ``1/(pi*T2)`` -- i.e. narrowing with ``B0`` stops once the
    homogeneous limit is reached (Clarke Summary Pt. 3).
    """

    f0 = GAMMA_H_HZ_PER_T * b0_t
    inhom = frac_inhomogeneity * f0
    homog = 1.0 / (np.pi * t2_s)
    hwhm = max(inhom, homog)
    fg = np.linspace(f0 - span_factor * hwhm, f0 + span_factor * hwhm, n)
    lorentzian = (hwhm / np.pi) / ((fg - f0) ** 2 + hwhm**2)  # unit area
    return fg, area * lorentzian, f0, hwhm


def _fit_exponent(b0_values, snr_values):
    logb = np.log(np.asarray(b0_values))
    logs = np.log(np.asarray(snr_values))
    return float(np.polyfit(logb, logs, 1)[0])


def run(
    *,
    squid_noise_ft: float,
    faraday_ref_ft: float,
    faraday_ref_mhz: float,
    frac_inhomogeneity: float,
    t2_s: float,
):
    squid = SQUIDMagnetometer(field_noise_T_per_rtHz=squid_noise_ft * 1e-15)
    faraday = IdealFaradayCoil(
        field_noise_T_per_rtHz_ref=faraday_ref_ft * 1e-15,
        f_ref_hz=faraday_ref_mhz * 1e6,
    )

    # Noise-floor crossover.
    freqs = np.logspace(2, 8, 4000)
    n_squid = squid.field_noise_amplitude(freqs)
    n_faraday = faraday.field_noise_amplitude(freqs)
    crossover_hz = float(freqs[np.argmin(np.abs(n_squid - n_faraday))])

    # Detected SNR vs B0 for a prepolarized sample.
    b0_values = np.logspace(np.log10(1.0e-6), np.log10(5.0e-3), 40)
    snr_squid, snr_faraday, f0s, widths = [], [], [], []
    for b0 in b0_values:
        fg, s, f0, hwhm = line_spectrum(
            b0, frac_inhomogeneity=frac_inhomogeneity, t2_s=t2_s
        )
        snr_squid.append(detected_field_snr(s, fg, squid).snr)
        snr_faraday.append(detected_field_snr(s, fg, faraday).snr)
        f0s.append(f0)
        widths.append(hwhm)
    snr_squid = np.asarray(snr_squid)
    snr_faraday = np.asarray(snr_faraday)

    return {
        "squid": squid,
        "faraday": faraday,
        "freqs": freqs,
        "n_squid": n_squid,
        "n_faraday": n_faraday,
        "crossover_hz": crossover_hz,
        "b0_values": b0_values,
        "f0s": np.asarray(f0s),
        "widths": np.asarray(widths),
        "snr_squid": snr_squid,
        "snr_faraday": snr_faraday,
    }


def print_summary(res, *, frac_inhomogeneity, t2_s) -> None:
    squid = res["squid"]
    b0 = res["b0_values"]
    exp_squid = _fit_exponent(b0, res["snr_squid"])
    exp_faraday = _fit_exponent(b0, res["snr_faraday"])
    floor_5k6 = squid.field_noise_amplitude(np.array([5.6e3]))[0]
    # advantage at the Clarke imaging field, 132 uT (5.6 kHz)
    i132 = int(np.argmin(np.abs(b0 - 1.32e-4)))
    ratio_132 = res["snr_squid"][i132] / res["snr_faraday"][i132]

    print("SQUID vs Faraday, ultra-low-field detection (Clarke 2007)")
    print(f"  SQUID floor @ 5.6 kHz         : {floor_5k6 * 1e15:.2f} fT/rtHz")
    print(f"  noise-floor crossover         : {res['crossover_hz'] / 1e6:.3f} MHz")
    print(f"  SNR(SQUID)    ~ B0^{exp_squid:+.2f}   (expect -0.50)")
    print(f"  SNR(Faraday)  ~ B0^{exp_faraday:+.2f}   (expect +0.50)")
    print(f"  SQUID/Faraday advantage @132 uT: {ratio_132:.0f}x")
    print(f"  (frac inhomogeneity {frac_inhomogeneity:g}, T2 {t2_s:g} s)")


def make_figure(plt, res):
    squid, faraday = res["squid"], res["faraday"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.loglog(res["freqs"], res["n_squid"] * 1e15, label=squid.name)
    ax.loglog(res["freqs"], res["n_faraday"] * 1e15, label=faraday.name)
    ax.axvline(res["crossover_hz"], color="0.5", ls="--", lw=1)
    ax.annotate(
        f"crossover {res['crossover_hz'] / 1e6:.2f} MHz",
        xy=(res["crossover_hz"], 1.0),
        xytext=(0.05, 0.1),
        textcoords="axes fraction",
        fontsize=9,
    )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("field-referred noise (fT/rtHz)")
    ax.set_title("(a) Detector noise floor")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    b0_ut = res["b0_values"] * 1e6
    ax.loglog(b0_ut, res["snr_squid"], label="SQUID")
    ax.loglog(b0_ut, res["snr_faraday"], label="Faraday coil")
    ax.axvline(132.0, color="0.5", ls=":", lw=1)
    ax.set_xlabel("precession field B0 (uT)")
    ax.set_ylabel("detected SNR (relative)")
    ax.set_title("(b) Prepolarized detected SNR vs B0")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ratio = res["snr_squid"] / res["snr_faraday"]
    ax.loglog(b0_ut, ratio)
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("precession field B0 (uT)")
    ax.set_ylabel("SNR(SQUID) / SNR(Faraday)")
    ax.set_title("(c) SQUID advantage grows as B0 falls")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 1]
    for b0 in (1.8e-3, 1.32e-4, 1.8e-5):
        fg, s, f0, _ = line_spectrum(b0, frac_inhomogeneity=0.01, t2_s=1.0)
        ax.plot((fg - f0), s / s.max(), label=f"{b0 * 1e6:.0f} uT")
    ax.set_xlabel("frequency offset from f0 (Hz)")
    ax.set_ylabel("normalized line")
    ax.set_title("(d) Linewidth narrows as B0 falls")
    ax.set_xlim(-2000, 2000)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, title="B0")

    fig.suptitle("Ultra-low-field NMR: SQUID vs Faraday detection (Clarke 2007)")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squid-noise-ft", type=float, default=1.0,
                        help="SQUID white field-noise floor in fT/rtHz (default 1.0)")
    parser.add_argument("--faraday-ref-ft", type=float, default=1.0,
                        help="Faraday field-noise at the reference frequency, fT/rtHz")
    parser.add_argument("--faraday-ref-mhz", type=float, default=1.0,
                        help="reference frequency for the Faraday 1/f floor (MHz)")
    parser.add_argument("--frac-inhomogeneity", type=float, default=0.01,
                        help="fractional B0 inhomogeneity dB0/B0 (default 0.01)")
    parser.add_argument("--t2", type=float, default=1.0,
                        help="homogeneous T2 in s, flooring the linewidth (default 1.0)")
    parser.add_argument("--save", type=str, default=None,
                        help="path to save the figure; if omitted, only prints a summary")
    args = parser.parse_args()

    res = run(
        squid_noise_ft=args.squid_noise_ft,
        faraday_ref_ft=args.faraday_ref_ft,
        faraday_ref_mhz=args.faraday_ref_mhz,
        frac_inhomogeneity=args.frac_inhomogeneity,
        t2_s=args.t2,
    )
    print_summary(res, frac_inhomogeneity=args.frac_inhomogeneity, t2_s=args.t2)

    if args.save:
        plt = load_matplotlib(required=True, headless=True)
        fig = make_figure(plt, res)
        fig.savefig(args.save, dpi=150)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
