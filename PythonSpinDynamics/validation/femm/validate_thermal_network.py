"""FEMM cross-validation of the lumped thermal network (Phase 1).

Problem: a PTFE cylindrical shell (r1..r2, length L, ends insulated) with a
prescribed total heat input P on the inner surface and convection (h, T_inf)
on the outer surface. With insulated ends this is exactly the 1D radial
problem, so three answers must agree:

- analytic:  T1 = T_inf + P/(h*2*pi*r2*L) + P*ln(r2/r1)/(2*pi*k*L)
- network:   bath --(h*A2)-- outer --(2*pi*k*L/ln(r2/r1))-- inner
- FEMM:      axisymmetric heat-flow FEA (requires FEMM 4.2 + pyfemm)

Run:  python validation/femm/validate_thermal_network.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from spin_dynamics.thermal import (  # noqa: E402
    ThermalLink,
    ThermalNetwork,
    ThermalNode,
    convection_conductance,
    cylindrical_shell_conductance,
)

R1, R2, LENGTH = 5e-3, 15e-3, 40e-3
K_SHELL = 0.25          # PTFE, W/m/K
H_FILM = 25.0           # W/m^2/K
T_INF = 293.15          # K
POWER = 0.5             # W


def analytic() -> tuple[float, float]:
    a2 = 2.0 * np.pi * R2 * LENGTH
    t_outer = T_INF + POWER / (H_FILM * a2)
    t_inner = t_outer + POWER * np.log(R2 / R1) / (2.0 * np.pi * K_SHELL * LENGTH)
    return t_inner, t_outer


def network() -> tuple[float, float]:
    nodes = [
        ThermalNode("inner", heat_capacity=1.0),
        ThermalNode("outer", heat_capacity=1.0),
        ThermalNode("ambient", heat_capacity=None, initial_temperature=T_INF),
    ]
    links = [
        ThermalLink(
            "inner", "outer",
            conductance=cylindrical_shell_conductance(K_SHELL, LENGTH, R1, R2),
        ),
        ThermalLink(
            "outer", "ambient",
            conductance=convection_conductance(H_FILM, 2.0 * np.pi * R2 * LENGTH),
        ),
    ]
    steady = ThermalNetwork(nodes, links, sources={"inner": POWER}).steady_state()
    return steady["inner"], steady["outer"]


def femm_model() -> tuple[float, float]:
    import femm

    femm.openfemm(1)
    try:
        femm.newdocument(2)  # heat-flow problem
        femm.hi_probdef("meters", "axi", 1e-8, 0, 30)
        # Shell cross-section in the (r, z) plane.
        femm.hi_drawrectangle(R1, 0.0, R2, LENGTH)
        femm.hi_addmaterial("shell", K_SHELL, K_SHELL, 0.0, 0.0)
        femm.hi_addblocklabel(0.5 * (R1 + R2), 0.5 * LENGTH)
        femm.hi_selectlabel(0.5 * (R1 + R2), 0.5 * LENGTH)
        femm.hi_setblockprop("shell", 1, 0.0, 0)
        femm.hi_clearselected()
        # Inner surface: prescribed heat flux totalling POWER. FEMM's qs is
        # outward-directed (positive extracts heat), so inject with qs < 0.
        qs = -POWER / (2.0 * np.pi * R1 * LENGTH)
        femm.hi_addboundprop("inner_flux", 1, 0.0, qs, 0.0, 0.0, 0.0)
        femm.hi_selectsegment(R1, 0.5 * LENGTH)
        femm.hi_setsegmentprop("inner_flux", 0.0, 1, 0, 0, "<None>")
        femm.hi_clearselected()
        # Outer surface: convection to T_inf. Ends: natural (insulated).
        femm.hi_addboundprop("outer_conv", 2, 0.0, 0.0, T_INF, H_FILM, 0.0)
        femm.hi_selectsegment(R2, 0.5 * LENGTH)
        femm.hi_setsegmentprop("outer_conv", 0.0, 1, 0, 0, "<None>")
        femm.hi_clearselected()
        path = Path(tempfile.mkdtemp()) / "thermal_network_validation.feh"
        femm.hi_saveas(str(path))
        femm.hi_analyze()
        femm.hi_loadsolution()
        t_inner = femm.ho_getpointvalues(R1 + 1e-5, 0.5 * LENGTH)[0]
        t_outer = femm.ho_getpointvalues(R2 - 1e-5, 0.5 * LENGTH)[0]
        return float(t_inner), float(t_outer)
    finally:
        femm.closefemm()


def main() -> None:
    t1_a, t2_a = analytic()
    t1_n, t2_n = network()
    print("Coaxial shell, P on inner surface, convection on outer:")
    print(f"  analytic: T_inner = {t1_a:.4f} K, T_outer = {t2_a:.4f} K")
    print(f"  network : T_inner = {t1_n:.4f} K, T_outer = {t2_n:.4f} K")
    print(f"  network vs analytic: {abs(t1_n - t1_a):.2e} K / {abs(t2_n - t2_a):.2e} K")
    try:
        t1_f, t2_f = femm_model()
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  FEMM    : unavailable ({exc})")
        return
    print(f"  FEMM    : T_inner = {t1_f:.4f} K, T_outer = {t2_f:.4f} K")
    rel1 = abs(t1_f - t1_a) / (t1_a - T_INF)
    rel2 = abs(t2_f - t2_a) / (t2_a - T_INF)
    print(f"  FEMM vs analytic (fraction of rise): {rel1:.3%} / {rel2:.3%}")


if __name__ == "__main__":
    main()
