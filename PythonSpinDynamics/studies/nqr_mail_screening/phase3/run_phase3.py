"""Reproducible Phase 3 numerical audit; no measured-site performance claims."""

from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from acquisition import schedule, Receiver, receiver, lowpass
from environment import cancellation_trial, thermal_trial


def run(root, output, quick=False):
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    records = []
    configurations = (
        [("enclosing_helix", 293.15, False)]
        if quick
        else [
            ("enclosing_helix", 273.15, False),
            ("enclosing_helix", 293.15, False),
            ("enclosing_helix", 323.15, False),
            ("enclosing_helix", 293.15, True),
            ("axial_gradiometer", 293.15, False),
        ]
    )
    prepared_records = [
        schedule(root, *configuration) for configuration in configurations
    ]
    if not quick:
        nominal = prepared_records[1]
        packet = dict(nominal)
        data = np.load(output / "finite_packet.npz")
        np.testing.assert_array_equal(data["time_s"], nominal["time_s"])
        packet["signal_v"] = data["signal_v"]
        packet["label"] = "finite_packet_27_voxels"
        prepared_records.append(packet)
        mixture = dict(nominal)
        mixture["signal_v"] = sum(
            weight * prepared_records[i]["signal_v"]
            for i, weight in enumerate((0.25, 0.5, 0.25))
        )
        mixture["temperature_k"] = 323.15  # ambient upper bound for receiver/thermal
        mixture["label"] = "temperature_mixture_0_20_50C"
        prepared_records.append(mixture)
    for record in prepared_records:
        candidate, temperature = record["candidate"], record["temperature_k"]
        prep = record["blocks"][0]["config"]["prepolarization"]
        thermal, trajectory = thermal_trial(record, root)
        key = record.get("label", f"{candidate}_{temperature:g}K_prepol{int(prep)}")
        trials = []
        for family in ("stationary", "nearby_source", "drift", "overload"):
            for method in (
                "none",
                "passive_20db",
                "single_reference",
                "multi_reference",
                "protected_adaptive",
            ):
                summary, arrays = cancellation_trial(record, family, method)
                summary["thermal"] = thermal
                trials.append(summary)
                np.savez_compressed(
                    output / f"{key}_{family}_{method}.npz",
                    time_s=record["time_s"],
                    signal_v=record["signal_v"],
                    receive_mask=record["receive_mask"],
                    fit_mask=record["fit_mask"],
                    thermal_trajectory=trajectory,
                    **arrays,
                )
        summary = {
            "id": key,
            "candidate": candidate,
            "temperature_k": temperature,
            "prepolarization": prep,
            "blocks": record["blocks"],
            "trials": trials,
            "thermal": thermal,
            "samples": len(record["time_s"]),
        }
        summaries.append(summary)
        records.append(record)
        print(key, "completed", flush=True)
    # Discrete stationary noise variance has an exact filter norm oracle.
    cfg = Receiver()
    impulse = np.zeros(10000)
    impulse[0] = 1
    h = lowpass(
        lowpass(
            lowpass(impulse, cfg.resonator_bandwidth_hz, cfg.fs_hz),
            cfg.bandwidth_hz,
            cfg.fs_hz,
        ),
        cfg.bandwidth_hz,
        cfg.fs_hz,
    )
    rng = np.random.default_rng(751)
    n = (rng.normal(size=200000) + 1j * rng.normal(size=200000)) / np.sqrt(2)
    filtered = lowpass(
        lowpass(
            lowpass(n, cfg.resonator_bandwidth_hz, cfg.fs_hz),
            cfg.bandwidth_hz,
            cfg.fs_hz,
        ),
        cfg.bandwidth_hz,
        cfg.fs_hz,
    )[1000:]
    noise_error = abs(np.mean(abs(filtered) ** 2) / np.sum(h * h) - 1)
    # Receiver rail recovery is independently checked with a deliberate impulse.
    pulse = np.zeros(1000, complex)
    pulse[20:30] = 1
    _, mask, margin = receiver(pulse, np.ones(1000, bool), cfg)
    overload_test = margin["adc_clipped_samples"] > 0 and not mask[30] and mask[-1]
    all_trials = [t for s in summaries for t in s["trials"]]
    checks = {
        "time_monotonic": all(np.all(np.diff(r["time_s"]) > 0) for r in records),
        "protected_windows": all(
            t["fit_signal_overlap_samples"] == 0 for t in all_trials
        ),
        "injection_preservation": all(
            t["protected_injection_relative_error"] < 1e-9 for t in all_trials
        ),
        "rf_energy_conservation": all(
            s["thermal"]["rf_energy_balance_relative_error"] < 1e-10 for s in summaries
        ),
        "noise_filter_normalization": bool(noise_error < 0.02),
        "overload_recovery": bool(overload_test),
        "trace_conservation": all(
            b["max_density_trace_error"] < 1e-16 for s in summaries for b in s["blocks"]
        ),
        "state_history_exercised": all(
            s["blocks"][-1]["history_deviation_from_equilibrium"] > 1e-10
            for s in summaries
        ),
        "margins_reported": all(
            "receiver_rail_margin_v" in t and "component_margin_k" in t["thermal"]
            for t in all_trials
        ),
    }
    here = Path(__file__).resolve().parent
    files = list(here.glob("*.py")) + [here.parent / "phase1/pulsed_study.py"]
    resolution_path, circuit_path = (
        output / "resolution_report.json",
        output / "circuit_report.json",
    )
    resolution = (
        json.loads(resolution_path.read_text()) if resolution_path.exists() else {}
    )
    circuit = json.loads(circuit_path.read_text()) if circuit_path.exists() else {}
    gate_passed = bool(
        not quick
        and all(checks.values())
        and resolution.get("passed")
        and circuit.get("passed")
    )
    report = {
        "phase": 3,
        "numerical_checks_passed": bool(all(checks.values())),
        "gate3_passed": gate_passed,
        "resolution_validation": resolution,
        "circuit_validation": circuit,
        "checks": checks,
        "noise_variance_relative_error": float(noise_error),
        "records": summaries,
        "receiver": cfg.__dict__,
        "site_data_status": "synthetic, unmeasured",
        "supported_scope": "Synthetic retuned Q=30 candidates; central moving 1 g packet, 0-50 C, x/y full-density schedules; overload and leaky-reference scenarios are rejection tests.",
        "outside_validated_scope": [
            "Hardware selection, empirical ROC/AUC and detection mass claims.",
            "Installed site spectra, reference pickup, shielding-induced coil changes and actual thermal contacts.",
            "Loaded/tolerance receiver corners without per-line retuning and current regulation.",
            "Vector gradiometer and arbitrary whole-envelope packet-placement convergence beyond the stated tests.",
            "Measured microscopic SLSE/SORC relaxation and a complete two-line prepolarization population model.",
        ],
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                root / "gate2_report.json",
                root / "enclosing_helix_converged_map.npz",
                root / "axial_gradiometer_converged_map.npz",
                resolution_path,
                circuit_path,
            ]
            if p.exists()
        },
        "source_sha256": {
            str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    (output / "phase3_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not quick:
        (here / "phase3_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", type=Path, default=Path(".tmp/nqr_phase2_gate"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/nqr_phase3"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    report = run(args.phase2, args.output, args.quick)
    print(json.dumps(report["checks"], indent=2))
    if not report["numerical_checks_passed"] or (
        not args.quick and not report["gate3_passed"]
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
