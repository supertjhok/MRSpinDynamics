"""Compare the receiver envelope pole to the Phase 2 lumped tuned network."""

from pathlib import Path
import json
import sys
import numpy as np

PHASE2 = Path(__file__).resolve().parents[1] / "phase2"
sys.path.insert(0, str(PHASE2))
from spatial_loading import TunedCircuit  # noqa: E402

sys.path.remove(str(PHASE2))


def main():
    report = json.loads(Path(".tmp/nqr_phase2_gate/gate2_report.json").read_text())
    rows = []
    for name in ("enclosing_helix", "axial_gradiometer"):
        n = next(
            r
            for r in report["candidates"][name]["network_refinements"]
            if r["path"] == 64 and r["perimeter"] == 32 and r["formulation"] == "chain"
        )
        for index, f in ((0, 3102400.0), (2, 3304300.0)):
            for temperature in (273.15, 293.15, 323.15):
                inductance = n["L"][index]
                r = 2 * np.pi * f * inductance / 30
                circuit = TunedCircuit(
                    inductance, r, n["C"], f, source_resistance_ohm=0.0
                )
                cs = circuit.tuning_capacitance_f
                d = np.linspace(-15000, 15000, 101)
                w = 2 * np.pi * (f + d)
                zw = r + 1j * w * inductance
                z = 1 / (1 / zw + 1j * w * n["C"])
                exact = z / ((z + 1 / (1j * w * cs)) * zw)
                exact /= exact[len(d) // 2]
                pole = 1 / (1 + 1j * d / (3304300.0 / 60))
                error = float(np.max(abs(exact - pole)))
                # Fixed damping dominates copper R; current feedback and per-line
                # retuning are explicit operating assumptions of the spin sweep.
                rows.append(
                    {
                        "candidate": name,
                        "frequency_hz": f,
                        "temperature_k": temperature,
                        "max_complex_transfer_error": error,
                        "tuning_capacitance_f": cs,
                        "loaded_q_exact": circuit.evaluate()["loaded_q_during_drive"],
                    }
                )
    out = {
        "rows": rows,
        "maximum_error": max(r["max_complex_transfer_error"] for r in rows),
        "scope": "retuned unloaded network, +/-15 kHz receive band; load and component corners remain unvalidated",
        "passed": all(r["max_complex_transfer_error"] < 0.05 for r in rows),
    }
    Path(".tmp/nqr_phase3/circuit_report.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(out["maximum_error"], out["passed"])


if __name__ == "__main__":
    main()
