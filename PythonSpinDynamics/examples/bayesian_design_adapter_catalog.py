"""Inspect the four Phase 2 experiment-design adapters without scoring them."""
# Follow the example from physical inputs through simulation to reported observables.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spin_dynamics.design import (  # noqa: E402
    CPMGIRAdapter,
    CPMGIRDesign,
    ESRDelayDesign,
    ESRHahnAdapter,
    NQRFIDAdapter,
    NQRFrequencyDesign,
    PGSEAdapter,
    PGSEDesign,
)
from spin_dynamics.esr import ESRSpinSystem  # noqa: E402
from spin_dynamics.experiment import (  # noqa: E402
    Acquisition,
    CPMGIRTrain,
    ESRHahnEcho,
    Experiment,
    Hardware,
    NQRFID,
    PGSE,
    Sample,
    UniformB0,
)
from spin_dynamics.nqr import QuadrupolarSite  # noqa: E402


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    entries = [
        (
            "CPMG-IR",
            CPMGIRAdapter(
                Experiment(
                    CPMGIRTrain(num_echoes=2),
                    Sample(t1_seconds=0.2, t2_seconds=0.08),
                    acquisition=Acquisition(rephase_action="ignore"),
                ),
                {"t1_seconds": 0.2, "t2_seconds": 0.08},
            ),
            CPMGIRDesign(0.1),
        ),
        (
            "PGSE",
            PGSEAdapter(
                Experiment(
                    PGSE(num_echoes=1),
                    Sample(t2_seconds=0.1, diffusion_coefficient=1e-9),
                ),
                {"diffusion_coefficient": 1e-9, "t2_seconds": 0.1},
            ),
            PGSEDesign(0.05, 2e-3, 20e-3),
        ),
        (
            "NQR FID",
            NQRFIDAdapter(
                Experiment(
                    NQRFID(10e3, 25e-6, 100e-6, num_points=32),
                    Sample(site=QuadrupolarSite(quadrupole_frequency_hz=900e3)),
                ),
                {"quadrupole_frequency_hz": 900e3, "eta": 0.0},
            ),
            NQRFrequencyDesign(900e3),
        ),
        (
            "ESR Hahn",
            ESRHahnAdapter(
                Experiment(
                    ESRHahnEcho(25e6, 10e-9, 200e-9, num_points=32),
                    Sample(esr_system=ESRSpinSystem()),
                    Hardware(b0=UniformB0(field_tesla=0.35)),
                ),
                {"t2_seconds": 1e-6, "g_factor": 2.0023},
            ),
            ESRDelayDesign(200e-9),
        ),
    ]

    for name, adapter, action in entries:
        plan = adapter.plan(action)
        print(
            f"{name:10s} plan={'ok' if plan.ok else 'invalid':7s} "
            f"physical_cost={adapter.physical_seconds(action):.6g} s "
            f"workflow={plan.workflow}"
        )


if __name__ == "__main__":
    main()
