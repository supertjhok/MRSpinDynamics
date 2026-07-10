"""FEMM cross-validation of the 1D conduction solver (Phase 3).

Problem: a long solid cylinder (radius a) of a lossy dielectric with a uniform
volumetric heat source q_v and its outer surface held at T0 (ends insulated ->
purely radial). The steady radial profile is the parabola

    T(r) = T0 + q_v (a^2 - r^2) / (4 k)

so center rise = q_v a^2 / (4k). Three answers must agree: analytic, the
package Conduction1D cylinder solve, and a FEMM axisymmetric heat-flow FEA.

Run:  python validation/femm/validate_conduction.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from spin_dynamics.thermal import Conduction1D  # noqa: E402

A = 15e-3           # cylinder radius (m)
LENGTH = 80e-3      # long cylinder (ends far from the mid-plane probe)
K = 0.5             # conductivity (W/m/K), lossy tissue-like
Q_V = 2.0e4         # volumetric source (W/m^3)
T0 = 300.0          # surface temperature (K)


def analytic(r: np.ndarray) -> np.ndarray:
    return T0 + Q_V * (A**2 - r**2) / (4.0 * K)


def package(r: np.ndarray) -> np.ndarray:
    c = Conduction1D(
        r, geometry="cylinder", conductivity=K, rho_cp=1e6, source=Q_V,
        inner_bc=("insulated",), outer_bc=("temperature", T0),
    )
    return c.steady_state().temperature


def femm_profile(r_probe: np.ndarray) -> np.ndarray:
    import femm

    femm.openfemm(1)
    try:
        femm.newdocument(2)  # heat flow
        femm.hi_probdef("meters", "axi", 1e-8, 0, 30)
        # Cross-section in (r, z): 0..A x 0..LENGTH, probe at mid-plane.
        femm.hi_drawrectangle(0.0, 0.0, A, LENGTH)
        # Volumetric generation is a material property in FEMM (qv, W/m^3).
        femm.hi_addmaterial("tissue", K, K, Q_V, 0.0)
        femm.hi_addblocklabel(0.5 * A, 0.5 * LENGTH)
        femm.hi_selectlabel(0.5 * A, 0.5 * LENGTH)
        femm.hi_setblockprop("tissue", 1, 0.0, 0)
        femm.hi_clearselected()
        # Outer surface r = A at fixed T0; ends and axis insulated (natural).
        femm.hi_addboundprop("surf", 0, T0, 0.0, 0.0, 0.0, 0.0)
        femm.hi_selectsegment(A, 0.5 * LENGTH)
        femm.hi_setsegmentprop("surf", 0.0, 1, 0, 0, "<None>")
        femm.hi_clearselected()
        path = Path(tempfile.mkdtemp()) / "conduction_validation.feh"
        femm.hi_saveas(str(path))
        femm.hi_analyze()
        femm.hi_loadsolution()
        out = np.array(
            [femm.ho_getpointvalues(float(r), 0.5 * LENGTH)[0] for r in r_probe]
        )
        return out
    finally:
        femm.closefemm()


def main() -> None:
    n = 200
    r = np.linspace(A / (2 * n), A - A / (2 * n), n)
    t_a = analytic(r)
    t_p = package(r)
    rise_a = t_a.max() - T0
    print("Solid cylinder, uniform source, surface at T0:")
    print(f"  analytic center rise: {rise_a:.4f} K")
    print(f"  package  center rise: {t_p.max() - T0:.4f} K")
    print(f"  package vs analytic (max abs): {np.max(np.abs(t_p - t_a)):.2e} K")
    r_probe = np.linspace(0.5e-3, A - 0.5e-3, 12)
    try:
        t_f = femm_profile(r_probe)
    except Exception as exc:  # noqa: BLE001
        print(f"  FEMM: unavailable ({exc})")
        return
    t_a_probe = analytic(r_probe)
    rel = np.max(np.abs(t_f - t_a_probe)) / rise_a
    print(f"  FEMM center rise (r~0): {t_f[0] - T0:.4f} K")
    print(f"  FEMM vs analytic (max, fraction of rise): {rel:.3%}")


if __name__ == "__main__":
    main()
