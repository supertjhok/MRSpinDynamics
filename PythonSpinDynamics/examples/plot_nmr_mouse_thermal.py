"""Coupled thermal modeling of a single-sided (NMR-MOUSE) exam on tissue.

A single-sided probe pressed against skin runs a long CPMG relaxometry train.
Two heating paths act on the tissue and the surface RF coil:

* **RF coil self-heating.** The surface coil dissipates ``I^2 R / 2`` at the
  CPMG duty cycle; its copper resistance rises with temperature, and it leaks
  heat into the skin it is pressed against.
* **Sample SAR.** The RF E-field deposits power in the conductive tissue,
  strongest near the surface where B1 is largest; blood perfusion carries heat
  away and clamps the deep tissue toward arterial temperature.

The script combines all three thermal tools:

1. ``ThermalCoupling`` (lumped) for the coupled steady state of the coil and a
   representative tissue node -- coil temperature, R(T), and the sample
   temperature that feeds the Curie-law magnetization and the two-temperature
   noise model.
2. ``Conduction1D`` (slab + Pennes perfusion) for the depth profile of the
   temperature rise in tissue under a surface-peaked SAR density -- the
   hot-spot the lumped node cannot resolve, and the safety-relevant number.
3. ``ThermalNetwork.transient`` for the warm-up curve over the exam, showing
   how long until the coil and reported SNR stabilize.

Everything is opt-in and built from the sample/coil separation introduced with
the spin-noise models (``docs/spin_noise.md``, ``docs/thermal_modeling.md``):
the coil and the tissue sit at different temperatures throughout.

Run with ``--output results/nmr_mouse_thermal.png`` to save the figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.coil_properties import ANNEALED_COPPER  # noqa: E402
from spin_dynamics.sample import cuboid_geometry, water_sample  # noqa: E402
from spin_dynamics.thermal import (  # noqa: E402
    MUSCLE_TISSUE,
    Conduction1D,
    CoupledCoilDrive,
    CoupledSAR,
    PerfusionModel,
    ThermalCoupling,
    ThermalLink,
    ThermalNetwork,
    ThermalNode,
)

# --- Probe / sequence scenario ------------------------------------------------
T_BODY = 310.15          # 37 C arterial/tissue baseline (K)
T_ROOM = 296.15          # 23 C ambient / coil bath (K)
COIL_CURRENT = 6.0       # CPMG 180-pulse current amplitude (A)
COIL_R_REF = 0.4         # coil resistance at room temperature (ohm)
DUTY = 0.08              # CPMG RF duty cycle (echo-dense but sub-SAR-limit)
SENSITIVE_DEPTH = 3e-3   # NMR-MOUSE sensitive slice depth (m)
SAR_SURFACE = 0.35       # peak deposited power in the modeled tissue slab (W)
TISSUE_THICKNESS = 12e-3


def coupled_steady_state():
    """Lumped coupled steady state: coil + tissue node."""

    def factory(sources):
        nodes = [
            ThermalNode("coil", heat_capacity=2.0, initial_temperature=T_ROOM),
            ThermalNode("tissue", heat_capacity=8.0, initial_temperature=T_BODY),
            ThermalNode("ambient", heat_capacity=None, initial_temperature=T_ROOM),
            ThermalNode("core", heat_capacity=None, initial_temperature=T_BODY),
        ]
        links = [
            ThermalLink("coil", "ambient", conductance=0.15),   # coil to air/housing
            ThermalLink("coil", "tissue", conductance=0.05),    # contact conduction
            ThermalLink("tissue", "core", conductance=0.4),     # perfusion to body core
        ]
        return ThermalNetwork(nodes, links, sources)

    drive = CoupledCoilDrive(
        node="coil",
        current=COIL_CURRENT,
        duty_cycle=DUTY,
        material=ANNEALED_COPPER,
        reference_resistance=COIL_R_REF,
        reference_temperature=T_ROOM,
        resistance_exponent=0.5,  # skin-limited RF coil
    )
    sar = CoupledSAR(
        node="tissue",
        reference_power=SAR_SURFACE,
        reference_temperature=T_BODY,
        tempco=0.02,  # tissue conductivity ~ +2 %/K
    )
    sample = water_sample(
        cuboid_geometry(10e-3, 10e-3, TISSUE_THICKNESS),
        temperature=T_BODY,
        t2=0.05,
    )
    coupling = ThermalCoupling(
        factory, drive, sar=sar, sample=sample, sample_node="tissue"
    )
    return coupling.fixed_point()


def _tissue_solver(depth, q_v):
    perfusion = PerfusionModel(
        blood_perfusion=3.0,  # well-perfused muscle, kg/m^3/s
        blood_specific_heat=3617.0,
        arterial_temperature=T_BODY,
    )
    return Conduction1D(
        depth,
        geometry="slab",
        conductivity=MUSCLE_TISSUE.conductivity,
        rho_cp=MUSCLE_TISSUE.volumetric_heat_capacity,
        source=q_v,
        perfusion=perfusion,
        inner_bc=("convection", 60.0, T_ROOM),   # skin surface under the coil
        outer_bc=("temperature", T_BODY),        # deep tissue at body core
    )


def tissue_depth_profile():
    """Depth profile of the SAR-attributable tissue heating with perfusion.

    The skin surface sits against a cooler coil/room, so the *baseline* tissue
    profile already runs below body-core temperature near the surface. The
    safety-relevant quantity is the extra heating the exam causes: the
    difference between the profiles solved with and without the SAR source.
    SAR density falls off with depth as the surface coil's B1^2 decays; the
    perfusion sink pulls the deep tissue back to arterial temperature.
    """

    n = 200
    thickness = TISSUE_THICKNESS
    depth = np.linspace(thickness / (2 * n), thickness - thickness / (2 * n), n)
    # Surface-peaked deposition: q_v(z) = q0 exp(-z / d) with d ~ coil radius.
    decay = 4e-3
    q0 = 6e4  # W/m^3 at the surface
    q_v = q0 * np.exp(-depth / decay)
    baseline = _tissue_solver(depth, np.zeros_like(q_v)).steady_state().temperature
    heated = _tissue_solver(depth, q_v).steady_state().temperature
    return depth, heated, baseline, q_v


def warmup_transient():
    """Coil + tissue warm-up over the exam duration."""

    steady = coupled_steady_state()
    coil_power = steady.coil_power
    sample_power = steady.sample_power
    nodes = [
        ThermalNode("coil", heat_capacity=2.0, initial_temperature=T_ROOM),
        ThermalNode("tissue", heat_capacity=8.0, initial_temperature=T_BODY),
        ThermalNode("ambient", heat_capacity=None, initial_temperature=T_ROOM),
        ThermalNode("core", heat_capacity=None, initial_temperature=T_BODY),
    ]
    links = [
        ThermalLink("coil", "ambient", conductance=0.15),
        ThermalLink("coil", "tissue", conductance=0.05),
        ThermalLink("tissue", "core", conductance=0.4),
    ]
    net = ThermalNetwork(
        nodes, links, sources={"coil": coil_power, "tissue": sample_power}
    )
    times = np.linspace(0.0, 600.0, 400)  # 10-minute exam
    return net.transient(times), steady


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib()

    steady = coupled_steady_state()
    depth, heated, baseline, q_v = tissue_depth_profile()
    sar_rise = heated - baseline
    transient, _ = warmup_transient()

    print("Single-sided NMR-MOUSE thermal exam")
    print(f"  coil steady temperature: {steady.temperatures['coil'] - 273.15:.2f} C")
    print(f"  coil resistance: {steady.coil_resistance*1e3:.1f} mOhm "
          f"(cold {COIL_R_REF*1e3:.1f} mOhm)")
    print(f"  coil average power: {steady.coil_power*1e3:.1f} mW")
    print(f"  tissue node steady temperature: {steady.temperatures['tissue'] - 273.15:.2f} C")
    print(f"  tissue SAR (with tempco): {steady.sample_power*1e3:.1f} mW")
    print(f"  peak SAR-attributable tissue rise: +{sar_rise.max():.2f} K (at surface)")
    i_slice = np.argmin(np.abs(depth - SENSITIVE_DEPTH))
    print(f"  SAR rise at {SENSITIVE_DEPTH*1e3:.1f} mm sensitive slice: "
          f"+{sar_rise[i_slice]:.2f} K")
    m0_cold = water_sample(
        cuboid_geometry(10e-3, 10e-3, TISSUE_THICKNESS), temperature=T_BODY
    ).magnetization_density(0.5)
    m0_warm = steady.sample.magnetization_density(0.5)
    print(f"  Curie-law M0 change from heating: {(m0_warm/m0_cold - 1)*100:.2f} %")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(depth * 1e3, sar_rise, color="C3")
    ax.axvline(SENSITIVE_DEPTH * 1e3, color="k", ls="--", lw=0.8,
               label=f"sensitive slice ({SENSITIVE_DEPTH*1e3:.0f} mm)")
    ax.set_xlabel("depth into tissue (mm)")
    ax.set_ylabel("SAR-attributable rise (K)")
    ax.set_title("Tissue heating (SAR vs perfusion)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(depth * 1e3, q_v / 1e3, color="C1")
    ax.set_xlabel("depth into tissue (mm)")
    ax.set_ylabel("SAR density (kW/m$^3$)")
    ax.set_title("Surface-peaked deposition")

    ax = axes[2]
    ax.plot(transient.times, transient.temperatures["coil"] - 273.15,
            label="coil", color="C0")
    ax.plot(transient.times, transient.temperatures["tissue"] - 273.15,
            label="tissue node", color="C3")
    ax.set_xlabel("exam time (s)")
    ax.set_ylabel("temperature (C)")
    ax.set_title("Warm-up over a 10-min exam")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
