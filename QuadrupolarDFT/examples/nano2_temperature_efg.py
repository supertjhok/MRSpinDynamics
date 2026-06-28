"""Finite-temperature NQR of NaNO2 14N: harmonic averaging vs measured dnu/dT.

NaNO2 14N is the worked example for the whole MRSpinDynamics workspace, and its
nu_+ line has a measured temperature series in the NQR database, so it is a real
test case for the 0 K -> finite-T correction.

Two parts:

A. Validate the temperature *functional form* against the measured line. The
   database has nu_+(T) in the ordered phase plus a reported dnu/dT. Fitting the
   single-mode Bayer-Kushida model nu(T) = nu0 (1 - a coth(hbar omega/2kT))
   recovers a physical librational frequency and reproduces the sign and rough
   magnitude of the measured slope.

B. Demonstrate the harmonic-averaging *tensor* pipeline that a DFT phonon
   calculation would feed: an equilibrium EFG plus one librational mode curvature
   -> C_Q(T), eta(T), nu(T). (The curvature here is a calibrated stand-in for a
   finite-displacement DFT result; the machinery is identical.)

Run:
    python examples/nano2_temperature_efg.py
"""

from __future__ import annotations

import numpy as np

from quadrupolar_dft import (
    EFGTensor,
    VibrationalMode,
    coupling_constant_hz,
    efg_temperature_sweep,
    fit_bayer_single_mode,
    mean_square_normal_coordinate,
    nqr_frequencies_hz,
    wavenumber_to_angular_frequency,
)
from quadrupolar_dft.constants import (
    BARN_M2,
    ELEMENTARY_CHARGE_C,
    NITROGEN_14_QUADRUPOLE_MOMENT_BARN,
    PLANCK_CONSTANT_J_S,
)

# --- Measured NaNO2 14N nu_+ line, ordered phase (NQR database, kHz) ----------
# Below the ~437 K ferroelectric transition; the 436 K point is omitted because
# the single harmonic mode cannot capture the transition softening.
MEAS_T_K = np.array([77.0, 80.0, 293.0, 300.0])
MEAS_NU_PLUS_HZ = np.array([4929.0, 4929.0, 4647.0, 4637.0]) * 1e3
MEASURED_DNU_DT_HZ_PER_K = -2.199e3  # database value near room temperature

Q_BARN = NITROGEN_14_QUADRUPOLE_MOMENT_BARN


def vzz_from_cq(cq_hz: float) -> float:
    """Invert C_Q = e Q Vzz / h for Vzz in V/m^2."""

    return cq_hz * PLANCK_CONSTANT_J_S / (
        ELEMENTARY_CHARGE_C * Q_BARN * BARN_M2
    )


def equilibrium_tensor(cq_hz: float, eta: float) -> EFGTensor:
    """Build a principal-axis EFG tensor from (C_Q, eta)."""

    vzz = vzz_from_cq(cq_hz)
    vxx = -vzz * (1.0 - eta) / 2.0
    vyy = -vzz * (1.0 + eta) / 2.0
    return EFGTensor.from_components(np.diag([vxx, vyy, vzz]), unit="si")


def part_a_bayer_fit() -> float:
    print("=" * 70)
    print("A. Single-mode Bayer fit to measured NaNO2 14N nu_+(T)")
    print("=" * 70)
    fit = fit_bayer_single_mode(MEAS_T_K, MEAS_NU_PLUS_HZ)
    print(f"  fitted librational wavenumber : {fit.wavenumber_cm_inv:.0f} cm^-1")
    print(f"  nu0 (0 K intercept)           : {fit.nu0_hz / 1e6:.4f} MHz")
    print(f"  librational amplitude a       : {fit.amplitude:.4f}")
    print(f"  fit RMS                       : {fit.rms_hz / 1e3:.2f} kHz")
    print("  model vs measured:")
    for temperature, measured in zip(MEAS_T_K, MEAS_NU_PLUS_HZ):
        model = fit.frequency(temperature)
        print(
            f"    T={temperature:6.1f} K  meas={measured / 1e6:.4f}  "
            f"model={model / 1e6:.4f} MHz  "
            f"resid={(model - measured) / 1e3:+.2f} kHz"
        )
    slope = fit.slope_hz_per_k(296.0)
    print(
        f"  dnu/dT near 296 K: model={slope / 1e3:+.2f} kHz/K  "
        f"measured={MEASURED_DNU_DT_HZ_PER_K / 1e3:+.2f} kHz/K"
    )
    print(
        "  (right sign and order of magnitude; the model under-predicts the\n"
        "   slope because NaNO2 softens approaching its ferroelectric\n"
        "   transition -- an anharmonic effect beyond a single harmonic mode.)"
    )
    return fit.wavenumber_cm_inv


def part_b_tensor_sweep(wavenumber_cm_inv: float) -> None:
    print()
    print("=" * 70)
    print("B. Harmonic tensor pipeline (DFT-style equilibrium + one libration)")
    print("=" * 70)
    # NaNO2-like equilibrium (literature C_Q, eta); a DFT run would supply this.
    cq_eq_hz = 5.497e6
    eta_eq = 0.38
    equilibrium = equilibrium_tensor(cq_eq_hz, eta_eq)

    # Calibrate one librational mode's EFG curvature to a ~6% V_zz reduction by
    # 300 K (a finite-displacement DFT calculation would supply this tensor).
    omega = wavenumber_to_angular_frequency(wavenumber_cm_inv)
    q2_300 = mean_square_normal_coordinate(omega, 300.0)
    vzz_eq = equilibrium.vzz_si
    target_fraction = 0.06
    czz = -2.0 * target_fraction * vzz_eq / q2_300
    # Split the (positive) trace partner unevenly into xx/yy so eta also shifts.
    curvature = np.diag([-0.65 * czz, -0.35 * czz, czz])
    mode = VibrationalMode(
        wavenumber_cm_inv=wavenumber_cm_inv,
        efg_curvature_si=curvature,
        label="NO2- libration",
    )

    temperatures = [0.0, 77.0, 150.0, 250.0, 300.0, 400.0]
    points = efg_temperature_sweep(
        equilibrium,
        [mode],
        temperatures,
        spin=1.0,
        quadrupole_moment_barns=Q_BARN,
    )
    cq0 = coupling_constant_hz(equilibrium.vzz_si, Q_BARN)
    static_lines = np.sort(nqr_frequencies_hz(spin=1.0, cq_hz=cq0, eta=eta_eq))
    print(
        f"  static (no vibration): C_Q={cq0 / 1e6:.4f} MHz  eta={eta_eq:.4f}  "
        f"nu_+={static_lines[-1] / 1e6:.4f} MHz"
    )
    print("    T(K)   C_Q(MHz)   eta      nu_+(MHz)")
    for point in points:
        nu_plus = np.sort(point.frequencies_hz)[-1]
        print(
            f"   {point.temperature_k:5.0f}   {point.cq_hz / 1e6:7.4f}  "
            f"{point.eta:6.4f}   {nu_plus / 1e6:8.4f}"
        )
    print(
        "  Note eta drifts with T -- the tensor is averaged in the crystal\n"
        "  frame and only then diagonalized; averaging C_Q or eta directly\n"
        "  would miss this. Even the 0 K row is below the static value\n"
        "  (zero-point motion)."
    )


def main() -> None:
    wavenumber = part_a_bayer_fit()
    part_b_tensor_sweep(wavenumber)


if __name__ == "__main__":
    main()
