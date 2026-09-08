"""Finite pulse-history tree; no permutation of fixed-history templates."""

from pathlib import Path  # noqa: E402
import hashlib
import json
import sys
import numpy as np

PHASE3 = Path(__file__).resolve().parents[1] / "phase3"
sys.path.insert(0, str(PHASE3))
from acquisition import Config, simulate, phase2_inputs, Receiver, lowpass  # noqa: E402

sys.path.remove(str(PHASE3))

ACTIONS = (("SLSE", "x"), ("FID", "y"), ("SORC", "x"))
TEMPERATURES = (273.15, 293.15, 323.15)


def filtered(x):
    c = Receiver()
    return lowpass(
        lowpass(lowpass(x, c.resonator_bandwidth_hz, c.fs_hz), c.bandwidth_hz, c.fs_hz),
        c.bandwidth_hz,
        c.fs_hz,
    )


def build(root=Path(".tmp/nqr_phase2_gate"), output=Path(".tmp/nqr_phase4"), depth=3):
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve()
    sources = [
        here,
        PHASE3 / "acquisition.py",
        PHASE3.parent / "phase1/pulsed_study.py",
        root / "gate2_report.json",
    ]
    fingerprint = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    target = output / "tree.npz"
    manifest = output / "tree.json"
    if manifest.exists() and target.exists():
        old = json.loads(manifest.read_text())
        if old["source_sha256"] == fingerprint and old["depth"] == depth:
            return old, dict(np.load(target))
    profile, network = phase2_inputs(root, "enclosing_helix")
    t = np.arange(5000) / 200000
    nodes = []
    waves = []
    masks = []
    sources = []
    leaks = []
    # Preorder layout is identical for each temperature. Only histories within
    # a single temperature share a density ensemble.
    for ti, temperature in enumerate(TEMPERATURES):
        this_waves = []
        this_sources = []
        this_leaks = []

        def visit(path, history, elapsed, parent_wave, parent_leak):
            for action, (sequence, line) in enumerate(ACTIONS):
                new_path = path + (action,)
                gap = 0.001 if path else 0.0
                start = 0.010 + elapsed + gap
                cfg = Config(
                    sequence=sequence,
                    line_id=line,
                    echoes=4,
                    motion_mode="stop_transfer_stop",
                    coil_dwell_s=0.025,
                    preparation_s=1.0,
                    settling_s=0.010,
                    prepolarization=not path,
                    temperature_k=temperature,
                    offset_points=81,
                    dt_s=10e-6,
                )
                index = 2 if line == "x" else 0
                electrical = (
                    network["L"][index],
                    network["R"][index] * (1 + 0.00393 * (temperature - 293.15)),
                    network["C"],
                )
                row, times, v, valid, detail = simulate(
                    cfg,
                    return_details=True,
                    initial_state=history,
                    recovery_s=gap,
                    field_profile=profile,
                    electrical=electrical,
                )
                leakage = parent_leak.copy()
                tau = row["rf_envelope_time_constant_s"]
                voltage = row["coil_reactive_voltage_peak_v"] * 10 ** (
                    -Receiver().isolation_db / 20
                )
                for a, b in detail["rf_events_s"] + start:
                    on = (t >= a) & (t < b)
                    tail = (t >= b) & (t <= b + 8 * tau)
                    leakage[on] += voltage * (1 - np.exp(-(t[on] - a) / tau))
                    leakage[tail] += (
                        voltage
                        * (1 - np.exp(-(b - a) / tau))
                        * np.exp(-(t[tail] - b) / tau)
                    )
                wave = parent_wave.copy()
                times = times + start
                splits = np.r_[
                    0, np.flatnonzero(np.diff(times) > 15e-6) + 1, len(times)
                ]
                for lo, hi in zip(splits[:-1], splits[1:]):
                    select = (t >= times[lo]) & (t <= times[hi - 1])
                    wave[select] = np.interp(t[select], times[lo:hi], v[lo:hi])
                mask = np.zeros(len(t), bool)
                for end, width in zip(times, detail["exposure_s"]):
                    mask |= (t > end - width) & (t <= end)
                node = len(this_waves)
                if ti == 0:
                    nodes.append(
                        {
                            "id": node,
                            "path": list(new_path),
                            "action": action,
                            "start_s": start,
                            "end_s": start + row["rf_train_s"],
                            "cost_s": gap + row["rf_train_s"],
                            "full_cycle_s": row["cycle_s"],
                            "source_trace_error": row["max_density_trace_error"],
                            "current_voltage_feasible": row[
                                "within_provisional_current_voltage_limits"
                            ],
                            "zeeman_feasible": row["zeeman_spacing_constraint"][
                                "passes"
                            ],
                        }
                    )
                    masks.append(mask)
                this_waves.append(filtered(wave))
                this_sources.append(wave)
                this_leaks.append(leakage)
                if len(new_path) < depth:
                    visit(
                        new_path,
                        detail["history"],
                        elapsed + gap + row["rf_train_s"],
                        wave,
                        leakage,
                    )

        visit((), None, 0.0, np.zeros(len(t), complex), np.zeros(len(t), complex))
        waves.append(this_waves)
        sources.append(this_sources)
        leaks.append(this_leaks)
        print("history tree temperature", temperature, "complete", flush=True)
    waves = np.array(waves)
    projectors = []
    for node in nodes:
        i = node["id"]
        selected = np.flatnonzero(masks[i])
        halves = np.array_split(selected, 2)
        for indices in halves:
            basis = np.zeros(len(t), complex)
            basis[indices] = waves[1, i, indices].conj()
            norm = np.linalg.norm(basis)
            if norm <= 1e-30:
                raise ValueError("empty matched-filter basis")
            projectors.append(basis / norm)
    projectors = np.array(projectors).reshape(len(nodes), 2, len(t))
    means = np.einsum("nkt,ant->ank", projectors, waves)
    info = {
        "depth": depth,
        "nodes": nodes,
        "temperatures_k": TEMPERATURES,
        "source_sha256": fingerprint,
        "fixed_path": [0, 1, 2],
        "sunk_and_exit_cost_s": nodes[0]["full_cycle_s"] - 0.025 + 0.012,
        "physical_cost_description": "0.332 s total entry/transfer/exit + 1 s magnet dwell + 2 s handling + 10 ms settling/calibration + 2 ms final guard; RF and inter-block recovery added once",
        "prepolarization": True,
        "coil_dwell_max_s": 0.025,
    }
    leak_nominal = np.array(leaks[1])
    blanking_mean = np.einsum("nkt,nt->nk", projectors, filtered(leak_nominal))
    arrays = {
        "time_s": t,
        "projectors": projectors,
        "means_v": means,
        "source_v": np.array(sources),
        "leakage_input_v": leak_nominal,
        "blanking_mean_v": blanking_mean,
    }
    np.savez_compressed(target, **arrays)
    manifest.write_text(json.dumps(info, indent=2) + "\n")
    return info, arrays


if __name__ == "__main__":
    build()
