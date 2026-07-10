"""Inline process-monitoring NMR on a flowing sample in a pipe.

A process stream flows through an upstream prepolarizing magnet and then a
detection coil that runs a CPMG relaxometry train. Flow changes the measurement
in three ways, each modeled by ``spin_dynamics.flow`` and the thermal solvers
(see ``docs/flow_modeling.md``):

* **Washout during acquisition.** Excited spins leave the sensitive volume and
  fresh spins enter, so the CPMG train decays faster than the intrinsic ``T2``.
  Plug and laminar flow give distinct decay shapes (linear cutoff vs a slow
  near-wall tail) even though they share the same initial extra rate ``1/tau``.
* **Transit-time polarization.** The signal amplitude is set by how much
  longitudinal magnetization the spins built up during their transit through
  the prepolarizer -- rising as the flow slows.
* **Advective heat removal.** The flowing sample carries away heat deposited by
  the RF, clamping the sample temperature.

Panels (``--output results/flowing_pipe_nmr.png`` to save):

1. CPMG decay at fixed flow: static vs plug vs laminar washout.
2. Apparent ``T2`` and detected signal amplitude vs volumetric flow rate.
3. Steady sample temperature vs flow rate (RF heating removed by flow).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.flow import (  # noqa: E402
    FlowModel,
    apply_washout,
    inflow_polarization,
)
from spin_dynamics.thermal import (  # noqa: E402
    WATER_THERMAL,
    ThermalLink,
    ThermalNetwork,
    ThermalNode,
    flow_conductance,
)

# Pipe / probe scenario.
PIPE_RADIUS = 4e-3          # m
SENSITIVE_LENGTH = 25e-3    # m (detection coil length)
PREPOLARIZER_LENGTH = 0.15  # m
B_POL, B_DET = 0.5, 0.05    # T
T1, T2 = 1.2, 0.4           # s
RF_POWER = 0.8              # W deposited in the sample by the CPMG train
T_INLET = 293.15            # K


def _flow(q, regime):
    return FlowModel(
        volumetric_flow_rate=q, pipe_radius=PIPE_RADIUS,
        sensitive_length=SENSITIVE_LENGTH, regime=regime,
    )


def apparent_t2(t, signal):
    """Initial-slope apparent T2 from a log-linear fit over the first decade."""

    good = signal > 0.3 * signal[0]
    coeffs = np.polyfit(t[good], np.log(signal[good]), 1)
    return -1.0 / coeffs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib()

    # Panel 1: CPMG decay at a fixed flow rate chosen so tau ~ T2 (washout
    # clearly visible against the intrinsic decay).
    q_fixed = 3e-6  # m^3/s ~ 3 mL/s
    techo = np.linspace(0.0, 1.2, 300)
    static = np.exp(-techo / T2)
    plug = apply_washout(_flow(q_fixed, "plug"), techo, static)
    laminar = apply_washout(_flow(q_fixed, "laminar"), techo, static)
    tau_fixed = _flow(q_fixed, "plug").mean_residence_time
    print("Inline flowing-pipe NMR")
    print(f"  fixed flow: Q = {q_fixed*1e6:.2f} mL/s, tau = {tau_fixed:.3f} s, T2 = {T2:.2f} s")

    # Panel 2: apparent T2 and detected amplitude vs flow rate.
    q_grid = np.logspace(-8, -5.3, 40)  # m^3/s
    app_t2 = {"plug": [], "laminar": []}
    amplitude = {"plug": [], "laminar": []}
    for q in q_grid:
        for regime in ("plug", "laminar"):
            f = _flow(q, regime)
            sig = apply_washout(f, techo, static)
            app_t2[regime].append(apparent_t2(techo, sig))
            pol = inflow_polarization(
                f, polarizing_field_tesla=B_POL, detection_field_tesla=B_DET,
                prepolarizer_length_meters=PREPOLARIZER_LENGTH, t1_seconds=T1,
            )
            amplitude[regime].append(pol)

    # Panel 3: steady sample temperature vs flow rate (advective cooling).
    temps = []
    for q in q_grid:
        g_flow = flow_conductance(WATER_THERMAL.density, WATER_THERMAL.specific_heat, q)
        nodes = [
            ThermalNode("sample", heat_capacity=5.0, initial_temperature=T_INLET),
            ThermalNode("inlet", heat_capacity=None, initial_temperature=T_INLET),
        ]
        # Also a weak conduction path to the room so the zero-flow limit is finite.
        links = [
            ThermalLink("sample", "inlet", conductance=g_flow + 0.05),
        ]
        net = ThermalNetwork(nodes, links, sources={"sample": RF_POWER})
        temps.append(net.steady_state()["sample"] - 273.15)

    print(f"  apparent T2 (plug) at slowest/fastest flow: "
          f"{app_t2['plug'][0]:.3f} / {app_t2['plug'][-1]:.3f} s")
    print(f"  inflow polarization (laminar) slow/fast: "
          f"{amplitude['laminar'][0]:.2f} / {amplitude['laminar'][-1]:.2f}")
    print(f"  sample temperature rise slow/fast flow: "
          f"{temps[0]-(T_INLET-273.15):.2f} / {temps[-1]-(T_INLET-273.15):.2f} K")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(techo, static, "k--", label="static (no flow)")
    ax.plot(techo, plug, label="plug")
    ax.plot(techo, laminar, label="laminar")
    ax.set_xlabel("echo time (s)")
    ax.set_ylabel("CPMG signal")
    ax.set_title(f"Washout at Q = {q_fixed*1e6:.1f} mL/s")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx(q_grid * 1e6, app_t2["plug"], label="apparent T2 (plug)")
    ax.semilogx(q_grid * 1e6, app_t2["laminar"], label="apparent T2 (laminar)")
    ax.axhline(T2, color="k", ls=":", lw=0.8, label="intrinsic T2")
    ax.set_xlabel("flow rate (mL/s)")
    ax.set_ylabel("apparent T2 (s)")
    ax.set_title("Flow-enhanced relaxation")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.semilogx(q_grid * 1e6, amplitude["laminar"], color="C3", lw=1.0, alpha=0.7)
    ax2.set_ylabel("inflow polarization", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    ax = axes[2]
    ax.semilogx(q_grid * 1e6, temps, color="C1")
    ax.set_xlabel("flow rate (mL/s)")
    ax.set_ylabel("sample temperature (C)")
    ax.set_title(f"Advective cooling ({RF_POWER:.1f} W RF)")
    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
