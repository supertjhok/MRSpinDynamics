"""Compare predicted NaNO2 14N dnu/dT against the database.

Closes the temperature loop: predict each line's temperature coefficient from a
finite-temperature model, then check it against the measured ``dnu_dt`` stored
in the NQR database.

Here the prediction comes from an analytic Bayer fit to the measured nu(T)
series (a stand-in for a finite-displacement DFT temperature sweep, which would
plug in via ``slopes_from_temperature_points``). The point is the validation
harness: matching predicted lines to measured coefficients by frequency.

Run:
    python integration/examples/nano2_temperature_coefficients.py
"""

from __future__ import annotations

from quadrupolar_dft import fit_bayer_single_mode

from mr_integration import compare_temperature_coefficients

# NaNO2 14N nu_+ and nu_- series, ordered phase (NQR database, kHz).
SERIES = {
    "nu_plus": ([77.0, 80.0, 293.0, 300.0], [4929.0, 4929.0, 4647.0, 4637.0]),
    "nu_minus": ([77.0, 80.0, 293.0, 300.0], [3757.0, 3755.0, 3608.0, 3601.0]),
}
REFERENCE_T_K = 296.0


def main() -> None:
    predicted: list[tuple[float, float]] = []
    print("Predicted temperature coefficients (Bayer fit per line):")
    for label, (temps, freqs_khz) in SERIES.items():
        freqs_hz = [f * 1e3 for f in freqs_khz]
        fit = fit_bayer_single_mode(temps, freqs_hz)
        slope = fit.slope_hz_per_k(REFERENCE_T_K)
        frequency = fit.frequency(REFERENCE_T_K)
        predicted.append((frequency, slope))
        print(
            f"  {label}: nu({REFERENCE_T_K:.0f} K)={frequency / 1e6:.4f} MHz  "
            f"omega={fit.wavenumber_cm_inv:.0f} cm^-1  "
            f"dnu/dT={slope / 1e3:+.2f} kHz/K"
        )

    print()
    comparison = compare_temperature_coefficients(
        compound="Nitrous acid sodium salt",
        isotope="14N",
        temperature_k=293.0,
        predicted=predicted,
    )
    if not comparison.matches:
        print("No measured temperature coefficients found in the database.")
        return
    print(comparison.format_table())
    print()
    print(f"  signs agree: {comparison.signs_agree}")
    print(
        "  (the model under-predicts the magnitude because NaNO2 softens toward\n"
        "   its ferroelectric transition; a DFT/AIMD sweep would slot in here\n"
        "   via slopes_from_temperature_points.)"
    )


if __name__ == "__main__":
    main()
